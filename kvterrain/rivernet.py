"""
kvterrain.rivernet
==================

The river network as POLYLINES — the product that rasterisation destroys.

After a river is rasterised into a surface field, a river and a long thin lake are
indistinguishable. What is lost is: the direction of flow, the connectivity of the
channel, the ramp profile as a curve, and the connections between rivers and lakes.
The runtime burn needs all four, so this module keeps them.

WHERE THE Z COMES FROM
----------------------
Elvenett polylines are 2D. There is no Z in the source to "keep" — it has to be
constructed. The bed comes from the carved terrain; the surface comes from the
profile that drove that carve:

    z(v)     = carved leaf DTM sampled beneath vertex v   <- the BED
    level(v) = shaped centreline profile                  <- the water SURFACE

Two values per vertex, deliberately, because they have different consumers:

  * `z` is the BED — the floor of the trench `bathymetry.carve_river_beds` cut —
    and it is what the burn runs its running-minimum over. Burning against the
    surface instead would cut ~1-11 m too shallow and leave the DEM dams in place.
  * `level` is the authored water surface, and it is what a river pins to when it
    is not disturbed. `riverbed` caps the raw centreline ground at dry banks,
    makes it downstream-only, smooths its staircase and ties it to confluences
    and lakes. The raster is spread from those same values. Reading a nearest
    raster pixel back here would lose that sub-pixel profile and can introduce
    rises, so the shaped array is passed directly. `z + carve_depth` remains the
    compatibility fallback when shaping is disabled.

This is NOT a fitted profile. An earlier revision rasterised the centreline,
fitted a monotone-descending profile (PAVA) and widened it back out; wherever the
chord between two sparse Elvenett vertices cut across a bank, the fit pooled the
surface metres above the bed and the widen smeared that inflated value onto lower
neighbours, so the runtime read depth >> 0 and the river spiked.

The defence here is DENSIFICATION plus bank-safe profile shaping, not fitting.
Vertices are resampled to the leaf spacing before Z is sampled, so the polyline
follows the real channel instead of cutting chords across it. This module REPORTS
non-monotonic descent and never corrects `z`; descent is enforced on the terrain before the carve, by
`riverbed` (the downhill-only bed), and `z` is sampled from that carved bed. The
numbers here sample bilinearly, so they also read the banks beside a trench and
do not reach zero even when the channel's own samples never rise.

FLOW DIRECTION
--------------
Source vertex order is trusted except for an unmistakable, entirely open-channel
error: a run whose DTM endpoint climbs by more than 25 m is reversed before links
are built. ELVIS is designed to encode flow direction, but Lierne contains one
such 58.6 m reversed run; trusting it carved the low endpoint through a hillside.
Smaller disagreements remain diagnostics because on flat reaches, through weirs,
across lakes and over DTM artefacts Z is the noisier signal.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from . import core
from . import water as kvwater
from . import waterid as kvid
from .bathymetry import DEFAULT_RIVER_DEPTH_BY_ORDER, depth_for_order
from .watersurface import _fill_nan_1d, _bresenham_path

RIVERS_BIN_FILE = "rivers.bin"
RIVERS_GEOJSON_FILE = "rivers.geojson"
LAKES_FILE = "lakes.json"
JUNCTIONS_FILE = "junctions.json"

BIN_MAGIC = b"KVRIVER2"
BIN_VERSION = 2
NO_HIERARCHY_NODE = 0xFFFFFFFF

# The visual width per vertex, beside rivers.bin rather than inside it: rivers.bin
# v2 stays byte-for-byte what the runtime reads today, and a runtime that does not
# know this file simply keeps stamping the hydraulic width.
RIVER_WIDTHS_FILE = "river_widths.bin"
WIDTHS_MAGIC = b"KVRWIDT1"
WIDTHS_VERSION = 1

# Modelled mean-annual flow.  One metre/year is 31.688 L/s/km², a deliberately
# explicit effective-runoff assumption rather than an unlabeled rainfall knob.
DEFAULT_EFFECTIVE_RUNOFF_M_PER_YEAR = 1.0
SECONDS_PER_MEAN_YEAR = 365.25 * 24.0 * 60.0 * 60.0

# Hydraulic geometry shared with the runtime: full width = clamp(a * Q^b).
DEFAULT_WIDTH_COEFFICIENT = 8.0
DEFAULT_WIDTH_EXPONENT = 0.5
DEFAULT_MIN_WIDTH_M = 2.0
DEFAULT_MAX_WIDTH_M = 80.0

JUNCTION_INFLOW = "Inflow"
JUNCTION_OUTFLOW = "Outflow"

# A vertex is "the same point" as another within this fraction of leaf spacing,
# for endpoint matching during connectivity building.
ENDPOINT_TOL_CELLS = 1.0
# Descent tolerance: a rise smaller than this is DTM noise, not a real reversal.
DESCENT_TOL_M = 0.05
# Source order remains authoritative except for an unmistakable non-lake error.
# Lierne contains one segment whose coordinates climb 58.6 m from start to end;
# trusting it makes the downhill running minimum carve that entire hillside to
# the low endpoint. Smaller disagreements can be dams, falls, or DTM noise.
CLEAR_REVERSE_RISE_M = 25.0

# The ELVIS query uses esriSpatialRelIntersects, so any feature TOUCHING the crop
# comes back in full — including the part outside it. There is no terrain there:
# world_to_grid/_bilinear/_nearest_idx all clamp, so an outside vertex silently
# receives the z, the level and the lake id of the nearest EDGE pixel. Those are
# fabricated values, and the runtime burns them. Vertices outside the region are
# therefore dropped, and a segment that leaves and re-enters is split, because
# within this export those two pieces genuinely are not connected.
MIN_RUN_VERTICES = 2

# Lake spans are detected by snapping each densified vertex to the nearest lake
# mask pixel, so a channel running ALONG a shoreline flickers in and out of the
# mask and shatters into short spurious spans. Each one costs three things: the
# runtime skips the burn inside a span (a gap in a burned channel is a dam), the
# level is pinned to a pool the vertex is not in, and _detect_junctions emits a
# spurious Inflow/Outflow pair. Close small gaps, then drop short runs.
LAKE_SPAN_CLOSE_GAP = 2      # vertices; bridge mask dropouts up to this long
LAKE_SPAN_MIN_VERTICES = 3   # runs shorter than this are snapping noise


# --------------------------------------------------------------------------- #
# Grid <-> world                                                               #
# --------------------------------------------------------------------------- #

def world_to_grid(plan: core.GridPlan, x, y):
    """World (UTM metres) -> fractional (row, col) on the leaf sample lattice.
    Row 0 is the northmost sample, matching the north-up assembled array."""
    SY = plan.samples_y
    col = (np.asarray(x, dtype=np.float64) - plan.origin_x) / plan.spacing_m
    row = ((plan.origin_y + (SY - 1) * plan.spacing_m
            - np.asarray(y, dtype=np.float64)) / plan.spacing_m)
    return row, col


def _bilinear(grid: np.ndarray, row, col) -> np.ndarray:
    """NaN-aware bilinear sample. Returns NaN only if all four corners are NaN."""
    SY, SX = grid.shape
    r = np.clip(np.asarray(row, dtype=np.float64), 0, SY - 1)
    c = np.clip(np.asarray(col, dtype=np.float64), 0, SX - 1)
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    r1 = np.minimum(r0 + 1, SY - 1)
    c1 = np.minimum(c0 + 1, SX - 1)
    fr = r - r0
    fc = c - c0

    vals = np.stack([grid[r0, c0], grid[r0, c1], grid[r1, c0], grid[r1, c1]]
                    ).astype(np.float64)
    wts = np.stack([(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc])
    ok = np.isfinite(vals)
    wsum = np.where(ok, wts, 0.0).sum(axis=0)
    vsum = np.where(ok, np.nan_to_num(vals) * wts, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, vsum / wsum, np.nan)


def _nearest_idx(plan: core.GridPlan, xy: np.ndarray):
    """Clamped integer (row, col) of the sample nearest each vertex.

    The clamp is only safe for vertices already known to be inside the region —
    see in_region() and MIN_RUN_VERTICES. Outside vertices clamp to an edge pixel
    and silently pick up its height and lake id.
    """
    row, col = world_to_grid(plan, xy[:, 0], xy[:, 1])
    r = np.clip(np.rint(row), 0, plan.samples_y - 1).astype(np.intp)
    c = np.clip(np.rint(col), 0, plan.samples_x - 1).astype(np.intp)
    return r, c


def in_region(plan: core.GridPlan, xy: np.ndarray) -> np.ndarray:
    """Boolean mask: which vertices lie inside the export's sample lattice.

    The bound is the SAMPLE extent (origin .. origin + cells*spacing), matching
    what world_to_grid maps onto [0, samples-1]; a vertex on the boundary is in.
    """
    x0, y0, x1, y1 = plan.bbox_utm
    x = np.asarray(xy[:, 0], dtype=np.float64)
    y = np.asarray(xy[:, 1], dtype=np.float64)
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _contiguous_runs(mask: np.ndarray, min_len: int) -> list:
    """[(start, end)] inclusive index ranges of consecutive True, length >= min_len."""
    out = []
    n = int(np.size(mask))
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if (j - i + 1) >= min_len:
            out.append((i, j))
        i = j + 1
    return out


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #

def densify_polyline(xy: np.ndarray, stride_m: float) -> np.ndarray:
    """
    Resample a polyline so no gap between consecutive vertices exceeds `stride_m`.
    Original vertices are preserved exactly; new ones are linearly interpolated.

    This is the whole defence against the old spike failure. Elvenett vertices can
    be hundreds of metres apart, and a straight chord between two of them happily
    crosses ridges and banks. Sampling Z on that chord reads the bank, not the
    channel. Densifying to leaf spacing means every sampled Z sits on a cell the
    channel actually passes through.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return xy.copy()
    stride = max(float(stride_m), 1e-6)

    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if d <= 0.0:
            continue
        n = max(1, int(np.ceil(d / stride)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out, dtype=np.float64)


def geometry_hash(xy: np.ndarray) -> str:
    """
    Deterministic 16-byte id from the geometry itself, as a cross-check on (and
    fallback for) the native ELVIS keys. Coordinates are quantised to 1 cm before
    hashing so that a re-fetch with different `geometryPrecision` still hashes the
    same. Direction-sensitive on purpose: a reversed segment is a different thing.
    """
    q = np.rint(np.asarray(xy, dtype=np.float64) * 100.0).astype(np.int64)
    return hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()


def orient_clear_uphill_run(xy: np.ndarray, ground: np.ndarray,
                            lake_spans: list,
                            min_rise_m: float = CLEAR_REVERSE_RISE_M
                            ) -> tuple[np.ndarray, bool]:
    """Reverse only a clearly backward, entirely open-channel source run."""
    p = np.asarray(xy, dtype=np.float64)
    g = np.asarray(ground, dtype=np.float64)
    if (p.shape[0] < 2 or g.size != p.shape[0] or lake_spans or
            not np.isfinite(g[[0, -1]]).all() or
            g[-1] <= g[0] + float(min_rise_m)):
        return p, False
    return p[::-1].copy(), True


def _segment_pixel_path(plan: core.GridPlan, xy: np.ndarray) -> list:
    """8-connected pixel path covering the polyline, clipped to the grid."""
    r, c = _nearest_idx(plan, xy)
    rc = list(zip(r.tolist(), c.tolist()))
    if not rc:
        return []
    path = _bresenham_path(rc)
    SY, SX = plan.samples_y, plan.samples_x
    return [(rr, cc) for rr, cc in path if 0 <= rr < SY and 0 <= cc < SX]


def _order_along_path(weight_raw: np.ndarray, path: list, fallback: int) -> int:
    """
    Recover a segment's stream order by reading the RASTERISED order grid along
    its own pixel path, taking the most common non-zero value.

    This is how the hovedelv upgrade reaches the polylines without hovedelv
    geometry ever entering the polyline set. The rasteriser already burned main
    rivers last, so those pixels carry the upgraded order; sampling them back
    guarantees the polyline's order equals the raster's order at every pixel it
    covers, which a spatial join between two overlapping layers could not.

    `weight_raw` is the PRE-lake-mask grid, so a lake-through-line still reports
    its true order instead of the 0 the display grid would give.
    """
    if not path:
        return int(fallback)
    rr = np.fromiter((p[0] for p in path), dtype=np.intp, count=len(path))
    cc = np.fromiter((p[1] for p in path), dtype=np.intp, count=len(path))
    vals = weight_raw[rr, cc]
    vals = vals[vals > 0]
    if vals.size == 0:
        return int(fallback)
    counts = np.bincount(vals.astype(np.intp))
    return int(np.argmax(counts))


# --------------------------------------------------------------------------- #
# Network container                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class RiverSegment:
    seg_id: int
    geom_hash: str
    order: int
    xy: np.ndarray                  # (N,2) float64, densified, downstream order
    z: np.ndarray                   # (N,)  float32 — sampled BED
    level: np.ndarray               # (N,)  float32 — water SURFACE
    strekn_lnr: Optional[int] = None
    elvid: Optional[str] = None
    vassdragsnr: Optional[str] = None
    vatnlnr: Optional[int] = None
    regine_area_km2: Optional[float] = None
    upstream: list = field(default_factory=list)     # seg_ids
    downstream: Optional[int] = None                 # seg_id
    lake_spans: list = field(default_factory=list)   # (start_idx, end_idx, local_lake_id)
    discharge_m3_s: float = 0.0
    width_m: Optional[float] = None
    # The VISUAL full width per vertex (`riverwidth`): N50 where it knows, else
    # width_m. None until set. `width_m` stays the hydraulic width in rivers.bin.
    vertex_width_m: Optional[np.ndarray] = None
    node_id: int = NO_HIERARCHY_NODE
    local_catchment_m2: float = 0.0
    accumulated_catchment_m2: float = 0.0

    @property
    def n(self) -> int:
        return int(self.xy.shape[0])


@dataclass
class Junction:
    segment_id: int
    strekn_lnr: Optional[int]
    lake_id: int                    # local lake id
    vatn_lnr: Optional[int]
    kind: str                       # Inflow | Outflow
    position: tuple                 # (x, y, z)


@dataclass
class RiverNetwork:
    segments: list = field(default_factory=list)
    junctions: list = field(default_factory=list)
    lakes: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    vertex_stride_m: float = 0.0

    def total_vertices(self) -> int:
        return int(sum(s.n for s in self.segments))


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class TracedNetwork:
    """
    The polyline network with its GEOMETRY settled and nothing sampled yet:
    densified, clipped to the region, ordered, lake spans found, links built.

    Tracing and sampling used to be one pass at the very end of the pipeline. They
    are two now because the downhill river bed (`riverbed`) needs the linked,
    densified centrelines BEFORE the river carve, while `z` must still be sampled
    AFTER it. Both stages read this one object, so the bed that was enforced and
    the polylines that ship are the same segments, vertex for vertex.
    """
    segments: list                  # RiverSegment, z/level still None
    stride_m: float
    clip: dict
    span_stats: dict


def trace_river_network(
    plan: core.GridPlan,
    feats: kvwater.WaterFeatures,
    wg: kvwater.WaterGrid,
    height_nodata: np.ndarray,
    *,
    vertex_stride_m: Optional[float] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> TracedNetwork:
    """
    Everything `build_river_network` does that does not depend on heights.

    `height_nodata` is only consulted for WHERE it is NaN: a run whose every bed
    sample would be nodata is dropped here, before segment ids and links are
    handed out, exactly as the single pass used to drop it. The carves never change
    that NaN mask before `leaf_bed_uncarved` is snapshotted, so tracing on the
    post-snap bed and sampling on the river-carved one agree about which segments
    exist.
    """
    stride = float(vertex_stride_m) if vertex_stride_m else float(plan.spacing_m)
    weight_raw = wg.weight_raw if wg.weight_raw is not None else wg.weight

    SY, SX = plan.samples_y, plan.samples_x
    if height_nodata.shape != (SY, SX):
        raise ValueError(f"height {height_nodata.shape} != grid {(SY, SX)}")

    segments: list = []
    clip = {"features": 0, "clipped": 0, "split": 0, "reversed_clear_uphill": 0,
            "vertices_in": 0, "vertices_dropped": 0, "features_dropped": 0}
    span_stats = {"raw": 0, "merged": 0, "dropped": 0, "kept": 0}

    # A national-scale crop is tens of thousands of features and this loop is the
    # longest single thing the pipeline does, so it says where it is. A bar that
    # cannot tell "working" from "hung" is the reason a slow run reads as a crash.
    n_features = len(feats.rivers)
    tick = max(1, n_features // 200)

    for i_feat, seg in enumerate(feats.rivers):
        if progress and i_feat % tick == 0:
            progress(i_feat / max(n_features, 1),
                     f"{i_feat:,}/{n_features:,} features, "
                     f"{len(segments):,} segments")
        if seg.xy.shape[0] < 2:
            continue

        path = _segment_pixel_path(plan, seg.xy)
        order = _order_along_path(weight_raw, path, seg.order)

        xy_full = densify_polyline(seg.xy, stride)

        # CLIP TO THE REGION before anything samples the grid. Everything below
        # (_bilinear, _nearest_idx, water_surface lookup) clamps out-of-range
        # coordinates to an edge pixel, which would hand an outside vertex a
        # fabricated bed, level and lake id — values the runtime then burns.
        # A feature that leaves and re-enters becomes two segments: inside this
        # export they are not connected, and pretending otherwise would splice a
        # channel across terrain that was never fetched.
        clip["features"] += 1
        clip["vertices_in"] += int(xy_full.shape[0])
        inside = in_region(plan, xy_full)
        runs = _contiguous_runs(inside, MIN_RUN_VERTICES)
        clip["vertices_dropped"] += int(xy_full.shape[0]) - int(inside.sum())
        if not runs:
            clip["features_dropped"] += 1
            continue
        if not bool(inside.all()):
            clip["clipped"] += 1
        if len(runs) > 1:
            clip["split"] += len(runs) - 1

        for (v0, v1) in runs:
            xy = xy_full[v0:v1 + 1]

            row, col = world_to_grid(plan, xy[:, 0], xy[:, 1])
            if not np.isfinite(_bilinear(height_nodata, row, col)).any():
                continue

            ri, ci = _nearest_idx(plan, xy)
            lake_at = wg.lake_id[ri, ci].astype(np.int64)
            spans, st = _runs_of_lake(lake_at)
            for k in span_stats:
                span_stats[k] += st[k]

            row, col = world_to_grid(plan, xy[:, 0], xy[:, 1])
            sampled_ground = _fill_nan_1d(_bilinear(height_nodata, row, col))
            xy, reversed_clear = orient_clear_uphill_run(
                xy, sampled_ground, spans)
            if reversed_clear:
                clip["reversed_clear_uphill"] += 1

            segments.append(RiverSegment(
                seg_id=len(segments),
                # Hash the CLIPPED run, not the source feature: after a split the
                # source geometry no longer identifies one emitted segment, and two
                # pieces sharing one hash would collide as keys.
                geom_hash=geometry_hash(xy),
                order=int(order),
                xy=xy,
                z=None,
                level=None,
                strekn_lnr=seg.strekn_lnr,
                elvid=seg.elvid,
                vassdragsnr=seg.vassdragsnr,
                vatnlnr=seg.vatnlnr,
                regine_area_km2=seg.regine_area_km2,
                lake_spans=spans,
            ))

    if progress:
        progress(1.0, f"linking {len(segments):,} segments")
    _link_segments(segments, tol_m=ENDPOINT_TOL_CELLS * plan.spacing_m)
    return TracedNetwork(segments=segments, stride_m=stride, clip=clip,
                         span_stats=span_stats)


def width_from_discharge(discharge_m3_s, *,
                         coefficient: float = DEFAULT_WIDTH_COEFFICIENT,
                         exponent: float = DEFAULT_WIDTH_EXPONENT,
                         min_width_m: float = DEFAULT_MIN_WIDTH_M,
                         max_width_m: float = DEFAULT_MAX_WIDTH_M,
                         scale: float = 1.0):
    """Full modelled channel width in metres from discharge in m³/s."""
    q = np.maximum(np.asarray(discharge_m3_s, dtype=np.float64), 0.0)
    width = float(coefficient) * np.power(q, float(exponent))
    width = np.clip(width, float(min_width_m), float(max_width_m)) * float(scale)
    return width


def assign_discharge_and_width(
    traced: TracedNetwork,
    *,
    effective_runoff_m_per_year: float = DEFAULT_EFFECTIVE_RUNOFF_M_PER_YEAR,
    width_coefficient: float = DEFAULT_WIDTH_COEFFICIENT,
    width_exponent: float = DEFAULT_WIDTH_EXPONENT,
    min_width_m: float = DEFAULT_MIN_WIDTH_M,
    max_width_m: float = DEFAULT_MAX_WIDTH_M,
    width_scale: float = 1.0,
) -> dict:
    """Put modelled mean-annual Q and its matching hydraulic width on the graph.

    ELVIS carries each local REGINE unit's area, not accumulated segment
    catchments.  A unit's area is therefore apportioned once across its emitted
    segments by polyline length (uniform lateral contribution), then accumulated
    through the authored upstream/downstream graph.  This preserves mass across
    forks and clipped components without pretending the source supplies gauges.
    """
    segments = traced.segments
    if effective_runoff_m_per_year < 0:
        raise ValueError("effective runoff must be non-negative")

    groups: dict[str, list] = {}
    missing = []
    for s in segments:
        if (s.vassdragsnr and s.regine_area_km2 is not None and
                np.isfinite(s.regine_area_km2) and s.regine_area_km2 > 0):
            groups.setdefault(s.vassdragsnr, []).append(s)
        else:
            missing.append(s)

    represented_area_m2 = 0.0
    inconsistent_units = 0
    for unit, members in groups.items():
        source_areas = np.asarray(
            [float(s.regine_area_km2) for s in members], dtype=np.float64)
        area_km2 = float(np.median(source_areas))
        if float(source_areas.max() - source_areas.min()) > 1e-6:
            inconsistent_units += 1
        represented_area_m2 += area_km2 * 1e6
        lengths = np.asarray([
            float(np.hypot(np.diff(s.xy[:, 0]), np.diff(s.xy[:, 1])).sum())
            for s in members], dtype=np.float64)
        if not np.isfinite(lengths).all() or float(lengths.sum()) <= 0:
            lengths[:] = 1.0
        local = area_km2 * 1e6 * lengths / float(lengths.sum())
        for s, a in zip(members, local):
            s.local_catchment_m2 = float(a)

    for s in missing:
        s.local_catchment_m2 = 0.0

    by_id = {s.seg_id: s for s in segments}
    indegree = {s.seg_id: sum(1 for u in s.upstream if u in by_id)
                for s in segments}
    ready = sorted(sid for sid, degree in indegree.items() if degree == 0)
    accumulated = {s.seg_id: float(s.local_catchment_m2) for s in segments}
    visited = []
    while ready:
        sid = ready.pop(0)
        visited.append(sid)
        s = by_id[sid]
        d = s.downstream
        if d is not None and d in by_id:
            accumulated[d] += accumulated[sid]
            indegree[d] -= 1
            if indegree[d] == 0:
                ready.append(d)
                ready.sort()

    cyclic = set(by_id) - set(visited)
    cycle_components = []
    unseen = set(cyclic)
    while unseen:
        start = min(unseen)
        component = []
        sid = start
        while sid in unseen:
            unseen.remove(sid)
            component.append(sid)
            nxt = by_id[sid].downstream
            if nxt is None or nxt not in cyclic:
                break
            sid = nxt
        total = float(sum(accumulated[v] for v in component))
        for v in component:
            accumulated[v] = total
        cycle_components.append((component, total))
    q_scale = float(effective_runoff_m_per_year) / SECONDS_PER_MEAN_YEAR
    for s in segments:
        area = max(float(accumulated[s.seg_id]), 0.0)
        s.accumulated_catchment_m2 = area
        # Q is a per-segment f32 in rivers.bin. Derive width from THAT packed
        # value so the runtime reproduces the shipped width bit-for-bit.
        s.discharge_m3_s = float(np.float32(area * q_scale))
        s.width_m = float(np.float32(width_from_discharge(
            s.discharge_m3_s, coefficient=width_coefficient,
            exponent=width_exponent, min_width_m=min_width_m,
            max_width_m=max_width_m, scale=width_scale)))

    sinks = [s for s in segments if s.downstream is None or s.downstream not in by_id]
    sink_area = float(sum(accumulated[s.seg_id] for s in sinks) +
                      sum(total for _component, total in cycle_components))
    qs = np.asarray([s.discharge_m3_s for s in segments], dtype=np.float64)
    widths = np.asarray([s.width_m for s in segments], dtype=np.float64)
    return {
        "kind": "modelled_mean_annual_discharge",
        "units": "m3/s",
        "reference_period": "mean annual",
        "gauged": False,
        "effective_runoff_m_per_year": float(effective_runoff_m_per_year),
        "effective_runoff_l_s_km2": float(
            effective_runoff_m_per_year * 1e6 * 1000.0 / SECONDS_PER_MEAN_YEAR),
        "seconds_per_year": float(SECONDS_PER_MEAN_YEAR),
        "regine_units": len(groups),
        "regine_units_with_inconsistent_source_area": int(inconsistent_units),
        "segments_without_regine_area": len(missing),
        "represented_regine_area_m2": represented_area_m2,
        "sink_accumulated_area_m2": sink_area,
        "graph_cycle_segments": len(cyclic),
        "graph_cycle_components": len(cycle_components),
        "discharge_min_m3_s": float(qs.min()) if qs.size else 0.0,
        "discharge_median_m3_s": float(np.median(qs)) if qs.size else 0.0,
        "discharge_max_m3_s": float(qs.max()) if qs.size else 0.0,
        "width_min_m": float(widths.min()) if widths.size else 0.0,
        "width_median_m": float(np.median(widths)) if widths.size else 0.0,
        "width_max_m": float(widths.max()) if widths.size else 0.0,
        "method": "For each vassdragsnr, the median finite positive NVE REGINE "
                  "unit area is apportioned once across "
                  "its emitted ELVIS segments by in-export polyline length, then "
                  "accumulated downstream on the authored graph. Q = accumulated "
                  "area * uniform effective runoff / seconds_per_year.",
        "source_field": "arealregineenhet_km2",
        "source_field_semantics": "local REGINE-unit area, not upstream area",
    }


def attach_hierarchy_nodes(plan: core.GridPlan, net: RiverNetwork, hier) -> dict:
    """Attach each segment to the leaf node at its downstream-most vertex."""
    root = 0
    counts = {"root": 0, "leaf": 0, "missing": 0}
    for s in net.segments:
        if s.n == 0:
            s.node_id = NO_HIERARCHY_NODE
            counts["missing"] += 1
            continue
        r, c = _nearest_idx(plan, s.xy[-1:])
        node = int(hier.labels[int(r[0]), int(c[0])])
        s.node_id = node
        counts["root" if node == root else "leaf"] += 1
    return counts


def build_river_network(
    plan: core.GridPlan,
    feats: kvwater.WaterFeatures,
    wg: kvwater.WaterGrid,
    height_leaf: np.ndarray,
    *,
    water_surface: Optional[np.ndarray] = None,
    vertex_stride_m: Optional[float] = None,
    depth_by_order: Optional[dict] = None,
    depth_scale: float = 1.0,
    traced: Optional[TracedNetwork] = None,
    shaped_profiles: Optional[dict] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> RiverNetwork:
    """
    Turn the elvenett polylines into the serialisable network: densified, Z-sampled,
    connected, clipped at lake boundaries, validated.

    `traced` is the geometry from `trace_river_network`; without it the trace is
    run here, which is what every caller did before the two were split.

    `height_leaf` must be the LEAF DTM the surface raster was built against, so the
    polyline `z` (the bed) matches the bed the raster used.

    `shaped_profiles`, when supplied, is the per-segment output of
    `riverbed.downhill_profiles`. It is the authority for open-channel `level`:
    the raster was spread FROM these vertices, so reading a nearest raster pixel
    back would reintroduce the sub-pixel disagreement that shaping removed.
    Authored lake spans are still pinned to their lake level.

    `water_surface` remains the compatibility source when no shaped profile is
    supplied. In that path each vertex reads the combined raster verbatim.

    Without `water_surface` the function falls back to `bed + carve depth` plus
    lake-span pinning, which is correct in the interior and wrong only at those
    boundaries.
    """
    depth_by_order = depth_by_order or DEFAULT_RIVER_DEPTH_BY_ORDER

    SY, SX = plan.samples_y, plan.samples_x
    if height_leaf.shape != (SY, SX):
        raise ValueError(f"height_leaf {height_leaf.shape} != grid {(SY, SX)}")

    if traced is None:
        traced = trace_river_network(plan, feats, wg, height_leaf,
                                     vertex_stride_m=vertex_stride_m,
                                     progress=progress)
    segments = traced.segments
    stride, clip, span_stats = traced.stride_m, traced.clip, traced.span_stats

    # Resolved level, not the raw NVE field: a polyline crossing a lake whose level
    # was estimated must be pinned to that same level, or the polyline and the
    # surface raster disagree at exactly the inlet/outlet the burn hands off at.
    lake_hoyde = {lid: wg.lake_level(info)
                  for lid, info in (wg.lake_table or {}).items()}

    ws = None if water_surface is None else np.asarray(water_surface)

    for s in segments:
        xy = s.xy
        row, col = world_to_grid(plan, xy[:, 0], xy[:, 1])
        z = _bilinear(height_leaf, row, col)
        # Nodata bed samples along the channel are interpolated from their finite
        # neighbours rather than dropped — a hole in the middle of a polyline would
        # otherwise break the burn's running minimum.
        z = _fill_nan_1d(z)

        # Fallback surface: the trench floor plus the depth that was carved to
        # make it, i.e. the ground the channel was cut from. Overwritten below
        # wherever the raster actually carries a surface at this vertex.
        level = z + depth_for_order(s.order, depth_by_order, depth_scale)

        if shaped_profiles is not None and s.seg_id in shaped_profiles:
            shaped = np.asarray(shaped_profiles[s.seg_id][2], dtype=np.float64)
            if shaped.shape != level.shape:
                raise ValueError(
                    f"shaped profile for segment {s.seg_id} has {shaped.shape}, "
                    f"expected {level.shape}")
            take = np.isfinite(shaped)
            level[take] = shaped[take]

        # Inside a lake, the authored lake surface wins over bed+raise: the lake's
        # NVE `hoyde` is real data and the raise is a synthetic constant, so a
        # through-line must not disagree with the pool it crosses.
        for a, b, lid in s.lake_spans:
            h = lake_hoyde.get(int(lid))
            if h is not None and np.isfinite(h):
                level[a:b + 1] = float(h)

        # Compatibility path without shaped profiles: wherever the raster carries
        # water at this vertex's pixel, take its value verbatim, including the old
        # adjacent-to-lake tie-in.
        if ws is not None and shaped_profiles is None:
            # INDEX FIRST, convert after. `np.asarray(ws, dtype=np.float64)`
            # here would rebuild the ENTIRE surface grid as float64 — 537 MB
            # on an 8193² export — allocate it, convert it, read a few hundred
            # values out of it, and throw it away. Once per segment. On Lierne
            # (87k segments) that one conversion was 90% of the whole
            # post-processing run: 1075 s of the 1185 s total. Selecting the
            # vertices first and converting those is identical arithmetic on
            # a few hundred elements instead of 67 million.
            ri, ci = _nearest_idx(plan, xy)
            ras = ws[ri, ci].astype(np.float64)
            take = np.isfinite(ras)
            level[take] = ras[take]

        s.z = z.astype(np.float32)
        s.level = level.astype(np.float32)

    junctions = _detect_junctions(segments, wg)
    lakes = build_lake_records(plan, wg)

    report = {
        "segments": len(segments),
        "vertices": int(sum(s.n for s in segments)),
        "vertex_stride_m": stride,
        "junctions": len(junctions),
        "lakes": len(lakes),
        # Region clipping. A large dropped fraction is expected and healthy — the
        # ELVIS query returns whole features that merely intersect the crop — but
        # it is reported because it is also the signal that the crop bisects the
        # network, and because these vertices used to be kept with edge-clamped
        # values.
        "clip_source_features": clip["features"],
        "clip_features_clipped": clip["clipped"],
        "clip_features_dropped": clip["features_dropped"],
        "clip_segments_split": clip["split"],
        "clip_vertices_dropped": clip["vertices_dropped"],
        "clear_uphill_segments_reversed": clip.get("reversed_clear_uphill", 0),
        "clear_uphill_reverse_threshold_m": float(CLEAR_REVERSE_RISE_M),
        "clip_vertices_dropped_pct": (
            100.0 * clip["vertices_dropped"] / clip["vertices_in"]
        ) if clip["vertices_in"] else 0.0,
        "clip_note": "vertices outside the export region are dropped and a feature "
                     "that re-enters is split; sampling them would clamp to an edge "
                     "pixel and fabricate bed, level and lake id.",
        # Lake-span cleaning.
        "lake_spans_raw": span_stats["raw"],
        "lake_spans_merged": span_stats["merged"],
        "lake_spans_dropped_short": span_stats["dropped"],
        "lake_spans_kept": span_stats["kept"],
        "lake_span_min_vertices": int(LAKE_SPAN_MIN_VERTICES),
        "lake_span_close_gap": int(LAKE_SPAN_CLOSE_GAP),
    }
    report.update(validate_descent(segments))
    report.update(validate_flow_direction(segments))
    report.update(validate_connectivity(segments))

    return RiverNetwork(segments=segments, junctions=junctions, lakes=lakes,
                        report=report, vertex_stride_m=stride)


def _runs_of_lake(
    lake_at: np.ndarray,
    *,
    close_gap: int = LAKE_SPAN_CLOSE_GAP,
    min_vertices: int = LAKE_SPAN_MIN_VERTICES,
) -> tuple:
    """
    Contiguous vertex runs sharing one non-zero lake id -> ([(start, end, id)], stats).

    Raw runs come straight off the nearest-pixel lake id, which is noisy where a
    channel runs along a shoreline: the mask flickers, so one real crossing can
    arrive as several short runs and a channel that merely grazes a lake can
    produce a 1-vertex run that is pure snapping artefact.

    Two cleanups, in order:
      1. CLOSE — two runs of the SAME id separated by at most `close_gap`
         vertices are one crossing that the mask dropped out of; merge them.
      2. DROP  — a surviving run shorter than `min_vertices` is noise, not a
         crossing. Dropping it restores the burn over those vertices, leaves the
         level as bed+raise instead of the pool's surface, and removes the
         spurious Inflow/Outflow pair _detect_junctions would have emitted.

    Both are reported (never silently applied) via the returned stats dict.
    """
    raw = []
    n = int(lake_at.size)
    i = 0
    while i < n:
        v = int(lake_at[i])
        if v == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and int(lake_at[j + 1]) == v:
            j += 1
        raw.append([i, j, v])
        i = j + 1

    closed = []
    for run in raw:
        if closed and closed[-1][2] == run[2] and (run[0] - closed[-1][1] - 1) <= close_gap:
            closed[-1][1] = run[1]
        else:
            closed.append(list(run))

    kept = [(a, b, v) for (a, b, v) in closed if (b - a + 1) >= min_vertices]

    stats = {
        "raw": len(raw),
        "merged": len(raw) - len(closed),
        "dropped": len(closed) - len(kept),
        "kept": len(kept),
    }
    return kept, stats


def _link_segments(segments: list, tol_m: float) -> None:
    """
    Build upstream/downstream links by matching each segment's LAST vertex to
    another's FIRST vertex. Endpoints are quantised to a `tol_m` lattice and the
    3x3 neighbourhood of keys is probed, so a segment whose endpoint lands a
    fraction either side of a cell boundary still matches.

    Where several candidates share an endpoint (a confluence), the highest-order
    one is taken as the downstream continuation — the trunk, not a sibling
    tributary.
    """
    if not segments:
        return
    tol = max(float(tol_m), 1e-6)

    starts: dict = {}
    for s in segments:
        k = (int(np.floor(s.xy[0, 0] / tol)), int(np.floor(s.xy[0, 1] / tol)))
        starts.setdefault(k, []).append(s.seg_id)

    by_id = {s.seg_id: s for s in segments}
    for s in segments:
        ex, ey = float(s.xy[-1, 0]), float(s.xy[-1, 1])
        kx, ky = int(np.floor(ex / tol)), int(np.floor(ey / tol))
        best = None
        best_order = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cid in starts.get((kx + dx, ky + dy), ()):
                    if cid == s.seg_id:
                        continue
                    cand = by_id[cid]
                    d = np.hypot(cand.xy[0, 0] - ex, cand.xy[0, 1] - ey)
                    if d <= tol and cand.order > best_order:
                        best, best_order = cid, cand.order
        if best is not None:
            s.downstream = best
            by_id[best].upstream.append(s.seg_id)


def _detect_junctions(segments: list, wg: kvwater.WaterGrid) -> list:
    """
    A junction is where a polyline crosses a lake polygon boundary. Detected on
    the rasterised lake mask at vertex resolution, which after densification is
    the leaf spacing — the same resolution the polygon was rasterised at, so a
    finer geometric clip would be false precision.

    Downstream direction gives the type directly: land -> lake is an Inflow, lake
    -> land is an Outflow. A segment lying wholly inside a lake (an Elvenett
    lake-through-line) produces no junction of its own; its neighbours supply the
    inflow and outflow, and the through-line is what keeps the network connected
    across the pool.
    """
    lake_lopenr = {lid: (info or {}).get("lopenr")
                   for lid, info in (wg.lake_table or {}).items()}
    out: list = []
    for s in segments:
        if not s.lake_spans:
            continue
        n = s.n
        for (a, b, lid) in s.lake_spans:
            if a > 0:                       # entered the lake at a
                out.append(Junction(
                    segment_id=s.seg_id, strekn_lnr=s.strekn_lnr,
                    lake_id=int(lid), vatn_lnr=lake_lopenr.get(int(lid)),
                    kind=JUNCTION_INFLOW,
                    position=(float(s.xy[a, 0]), float(s.xy[a, 1]), float(s.z[a])),
                ))
            if b < n - 1:                   # left the lake at b
                out.append(Junction(
                    segment_id=s.seg_id, strekn_lnr=s.strekn_lnr,
                    lake_id=int(lid), vatn_lnr=lake_lopenr.get(int(lid)),
                    kind=JUNCTION_OUTFLOW,
                    position=(float(s.xy[b, 0]), float(s.xy[b, 1]), float(s.z[b])),
                ))
    return out


def build_lake_records(plan: core.GridPlan, wg: kvwater.WaterGrid) -> list:
    """
    One record per authored lake. Geometry is NOT emitted — the spec allows a
    raster mask reference, and `water_id.atlas` already carries the per-pixel
    identity, so a few hundred lakes cost a few KB of scalars instead of a polygon
    streaming problem.
    """
    from scipy.ndimage import find_objects

    lake_id = wg.lake_id
    if lake_id.size == 0 or int(lake_id.max()) == 0:
        return []

    slices = find_objects(lake_id.astype(np.intp))
    SY = plan.samples_y
    out = []
    for local_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        info = (wg.lake_table or {}).get(local_id, {}) or {}
        sub = lake_id[sl] == local_id
        px = int(sub.sum())
        if px == 0:
            continue
        r0, r1 = sl[0].start, sl[0].stop - 1
        c0, c1 = sl[1].start, sl[1].stop - 1
        # rows run north->south, so the max row is the SOUTH edge
        x_min = plan.origin_x + c0 * plan.spacing_m
        x_max = plan.origin_x + c1 * plan.spacing_m
        y_max = plan.origin_y + (SY - 1 - r0) * plan.spacing_m
        y_min = plan.origin_y + (SY - 1 - r1) * plan.spacing_m
        out.append({
            "lake_id": local_id,
            "water_id_code": int(kvid.LAKE_ID_BASE + local_id),
            "vatn_lnr": info.get("lopenr"),
            "navn": info.get("navn"),
            "area_m2": info.get("area_m2"),
            # `hoyde_moh` stays the RAW NVE field — null when NVE has none — so the
            # source data is still legible. `authored_level_m` is the level the
            # export actually used, estimated or not, because that is what the
            # runtime pins on and it must match the surface raster exactly.
            # `level_source` is the only thing distinguishing the two, and nothing
            # downstream is required to read it.
            "hoyde_moh": info.get("hoyde_moh"),
            "authored_level_m": wg.lake_level(info),
            "level_source": info.get("level_source"),
            "pixel_count": px,
            "bbox_utm": [x_min, y_min, x_max, y_max],
        })
    return out


# --------------------------------------------------------------------------- #
# Validation — report only, never correct                                      #
# --------------------------------------------------------------------------- #

def validate_descent(segments: list, tol_m: float = DESCENT_TOL_M) -> dict:
    """
    Count vertices whose bed rises going downstream. Reported, NOT fixed here:
    `z` is sampled, never smoothed. Descent is enforced on the terrain itself by
    `riverbed` before the carve; bilinear sampling still reads the banks beside a
    narrow trench, so this count does not reach zero.
    """
    rising = 0
    total = 0
    worst = 0.0
    bad_segments = 0
    in_tot = in_ris = 0
    for s in segments:
        z = np.asarray(s.z, dtype=np.float64)
        if z.size < 2:
            continue
        d = np.diff(z)
        up = d > tol_m
        total += int(d.size)
        k = int(up.sum())
        rising += k
        if k:
            bad_segments += 1
            worst = max(worst, float(d[up].max()))
        # Split by lake-span membership. A rise inside a span is the polyline
        # crossing a pool, where bed shape carries no downstream signal and the
        # runtime does not burn anyway; a rise on open channel is the number that
        # actually bears on the burn. Reporting them together hides both.
        if s.lake_spans:
            m = np.zeros(z.size, dtype=bool)
            for (a, b, _lid) in s.lake_spans:
                m[a:b + 1] = True
            step_in = m[:-1] | m[1:]
            in_tot += int(step_in.sum())
            in_ris += int((up & step_in).sum())
    out_tot = total - in_tot
    out_ris = rising - in_ris
    return {
        "descent_vertices_checked": total,
        "descent_rising_vertices": rising,
        "descent_rising_pct": (100.0 * rising / total) if total else 0.0,
        "descent_segments_with_rise": bad_segments,
        "descent_worst_rise_m": worst,
        "descent_rising_pct_in_lake_span": (
            100.0 * in_ris / in_tot) if in_tot else 0.0,
        "descent_rising_pct_open_channel": (
            100.0 * out_ris / out_tot) if out_tot else 0.0,
        "descent_note": "reported from bilinearly sampled z. With the downhill "
                        "river bed on, descent is enforced on the terrain along each "
                        "polyline's own samples (water_surface.river_bathymetry."
                        "downhill_bed.pixel_*); the bilinear samples also read the "
                        "banks, so this does not reach zero.",
    }


def validate_flow_direction(segments: list, tol_m: float = 0.5) -> dict:
    """
    Compare the SOURCE vertex order against a Z-derived direction, and report how
    often they disagree — split by whether the segment touches a lake.

    Apart from the clear-uphill correction made during tracing, this is a
    diagnostic, not a correction: a diffuse low disagreement rate is ordinary DTM
    noise on flat reaches, whereas a high rate, or one concentrated in
    lake-touching segments, points at something real.
    """
    dis = touch = dis_touch = n = vatn = 0
    for s in segments:
        z = np.asarray(s.z, dtype=np.float64)
        if z.size < 2 or not np.isfinite(z).any():
            continue
        n += 1
        # "Touches a lake" means the GEOMETRY enters one, i.e. it has a lake span.
        # `vatnlnr` must NOT be part of this test: ELVIS populates it on ~94% of
        # segments (it references the watercourse's lake, not a crossing), so
        # including it put almost the whole network in one bucket and left the
        # non-lake rate resting on a few hundred segments — which is exactly the
        # comparison this check exists to make. Reported separately below.
        is_touch = bool(s.lake_spans)
        if is_touch:
            touch += 1
        if s.vatnlnr is not None:
            vatn += 1
        if float(z[-1]) > float(z[0]) + tol_m:      # ends higher than it starts
            dis += 1
            if is_touch:
                dis_touch += 1
    non_touch = n - touch
    dis_non = dis - dis_touch
    return {
        "flowdir_segments_checked": n,
        "flowdir_disagreements": dis,
        "flowdir_disagreement_pct": (100.0 * dis / n) if n else 0.0,
        "flowdir_lake_touching_segments": touch,
        "flowdir_vatnlnr_segments": vatn,
        "flowdir_disagreement_pct_lake_touching": (
            100.0 * dis_touch / touch) if touch else 0.0,
        "flowdir_disagreement_pct_non_lake": (
            100.0 * dis_non / non_touch) if non_touch else 0.0,
        "flowdir_note": "source vertex order is authoritative except that a non-lake "
                        "run climbing more than 25 m endpoint-to-endpoint is reversed "
                        "before linking; remaining Z disagreement is QA only.",
    }


def validate_connectivity(segments: list) -> dict:
    """Orphans (no link either way) and terminal counts."""
    orphan = sinks = sources = 0
    for s in segments:
        has_up = bool(s.upstream)
        has_down = s.downstream is not None
        if not has_up and not has_down:
            orphan += 1
        if not has_down:
            sinks += 1
        if not has_up:
            sources += 1
    return {
        "connectivity_orphan_segments": orphan,
        "connectivity_sink_segments": sinks,
        "connectivity_source_segments": sources,
        "connectivity_note": "orphans are segments with neither an upstream nor a "
                             "downstream neighbour; a few at the region border are "
                             "expected, many in the interior are not.",
    }


# --------------------------------------------------------------------------- #
# Serialisation                                                                #
# --------------------------------------------------------------------------- #
#
# rivers.bin is a flat binary blob, read by one sequential BinaryReader loop.
#
# The polylines do NOT stream and do not need an order or a hierarchy: they are an
# input to the burn, which is a one-time tier-0 operation over the whole world, so
# they are loaded once at startup and stay resident for as long as tier 0 does.
# The renderer never touches them — it keeps reading surface.atlas per frame. Two
# consumers, two formats, no conflict.
#
# Header (little-endian):
#   8s  magic "KVRIVER2"
#   I   version
#   I   epsg
#   d   origin_x            <- vertex x/y are stored RELATIVE to this
#   d   origin_y
#   f   vertex_stride_m
#   I   segment_count
#   I   total_vertex_count
#   I   reserved (0)
#
# Per segment:
#   I   seg_id
#   q   strekn_lnr        (-1 = absent)
#   q   vatn_lnr          (-1 = absent)
#   H   order
#   i   downstream_id     (-1 = none)
#   f   discharge_m3_s    (modelled mean annual discharge)
#   f   width_m            (full hydraulic width from manifest function)
#   I   node_id            (leaf label at downstream vertex; 0=root, UINT32_MAX=none)
#   H   upstream_count, then I * upstream_count
#   B   elvid_len,       then bytes (ascii)
#   B   vassdragsnr_len, then bytes (ascii)
#   H   lake_span_count, then (I start, I end, H lake_id) * count
#   16s geometry hash
#   I   vertex_count, then f32 * 4 * vertex_count  (x, y, z, level)
# --------------------------------------------------------------------------- #

def _pack_str(s: Optional[str]) -> bytes:
    b = (s or "").encode("ascii", "replace")[:255]
    return struct.pack("<B", len(b)) + b


def write_rivers_bin(net: RiverNetwork, plan: core.GridPlan, path: str) -> dict:
    with open(path, "wb") as fh:
        fh.write(struct.pack(
            "<8sIIddfIII", BIN_MAGIC, BIN_VERSION, int(plan.epsg),
            float(plan.origin_x), float(plan.origin_y),
            float(net.vertex_stride_m), len(net.segments),
            net.total_vertices(), 0))

        for s in net.segments:
            fh.write(struct.pack(
                "<IqqHiffI", s.seg_id,
                -1 if s.strekn_lnr is None else int(s.strekn_lnr),
                -1 if s.vatnlnr is None else int(s.vatnlnr),
                int(s.order),
                -1 if s.downstream is None else int(s.downstream),
                float(s.discharge_m3_s), float(s.width_m), int(s.node_id)))
            fh.write(struct.pack("<H", len(s.upstream)))
            if s.upstream:
                fh.write(np.asarray(s.upstream, dtype="<u4").tobytes())
            fh.write(_pack_str(s.elvid))
            fh.write(_pack_str(s.vassdragsnr))
            fh.write(struct.pack("<H", len(s.lake_spans)))
            for (a, b, lid) in s.lake_spans:
                fh.write(struct.pack("<IIH", int(a), int(b), int(lid)))
            fh.write(bytes.fromhex(s.geom_hash))
            fh.write(struct.pack("<I", s.n))
            v = np.empty((s.n, 4), dtype="<f4")
            v[:, 0] = (s.xy[:, 0] - plan.origin_x).astype(np.float32)
            v[:, 1] = (s.xy[:, 1] - plan.origin_y).astype(np.float32)
            v[:, 2] = s.z
            v[:, 3] = s.level
            fh.write(v.tobytes())

    return {
        "file": os.path.basename(path),
        "format": "kvterrain-rivers-bin/2",
        "magic": BIN_MAGIC.decode("ascii"),
        "version": BIN_VERSION,
        "byte_order": "little_endian",
        "header_bytes": 48,
        "header_layout": "<8sIIddfIII",
        "bytes": os.path.getsize(path),
        "segments": len(net.segments),
        "vertices": net.total_vertices(),
        "segment_prefix_bytes": 38,
        "segment_prefix_layout": "<IqqHiffI",
        "segment_prefix_fields": [
            {"name": "seg_id", "dtype": "u32le", "offset": 0},
            {"name": "strekn_lnr", "dtype": "i64le", "offset": 4,
             "none": -1},
            {"name": "vatn_lnr", "dtype": "i64le", "offset": 12,
             "none": -1},
            {"name": "order", "dtype": "u16le", "offset": 20},
            {"name": "downstream_id", "dtype": "i32le", "offset": 22,
             "none": -1},
            {"name": "discharge_m3_s", "dtype": "f32le", "offset": 26},
            {"name": "width_m", "dtype": "f32le", "offset": 30},
            {"name": "node_id", "dtype": "u32le", "offset": 34,
             "none": NO_HIERARCHY_NODE},
        ],
        "segment_tail": "u16 upstream_count; upstream_count*u32 ids; u8+ASCII "
                        "elvid; u8+ASCII vassdragsnr; u16 lake_span_count; "
                        "lake_span_count*(u32 start,u32 end,u16 lake_id); "
                        "16-byte geometry hash; u32 vertex_count; "
                        "vertex_count*(4*f32)",
        "node_id": "level-0 hierarchy leaf label at the nearest sample to the "
                   "segment's downstream-most vertex; 0 drains directly off-map; "
                   "4294967295 means unavailable",
        "vertex_fields": ["x_rel_origin_m", "y_rel_origin_m", "z_bed_m", "level_surface_m"],
        "vertex_order": "upstream_to_downstream",
        "note": ("z is the sampled DTM BED — the floor of the carved channel — burn "
                 "against this. level is the shaped water SURFACE: capped at dry "
                 "banks, downstream-only, smoothed, accordant at confluences and "
                 "graded onto lakes (authored lake level inside a span) — pin to "
                 "this. Loaded once, resident; not streamed."),
    }


def write_rivers_geojson(net: RiverNetwork, plan: core.GridPlan, path: str) -> dict:
    """
    Optional debugging sidecar in absolute CRS coordinates with Z, for QGIS. Not
    the primary format: Unity has no built-in GeoJSON reader, and parsing a few
    hundred thousand coordinate pairs through a JSON reader at startup is seconds
    of work and heavy GC churn for data that is a flat float array in disguise.
    But the first time descent validation reports something odd, you will want to
    look at where.
    """
    feats = []
    for s in net.segments:
        coords = [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                  for x, y, z in zip(s.xy[:, 0], s.xy[:, 1], s.z)]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "seg_id": s.seg_id,
                "strekn_lnr": s.strekn_lnr,
                "elvid": s.elvid,
                "vassdragsnr": s.vassdragsnr,
                "vatn_lnr": s.vatnlnr,
                "geom_hash": s.geom_hash,
                "order": s.order,
                "downstream_id": s.downstream,
                "upstream_ids": list(s.upstream),
                "lake_spans": [[int(a), int(b), int(l)] for a, b, l in s.lake_spans],
                "discharge_m3_s": float(s.discharge_m3_s),
                "width_m": float(s.width_m),
                "node_id": int(s.node_id),
                "level_m": [round(float(v), 2) for v in s.level],
            },
        })
    doc = {
        "type": "FeatureCollection",
        "crs": {"type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{plan.epsg}"}},
        "features": feats,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path),
            "note": "debug sidecar; rivers.bin is the runtime format"}


def write_lakes_json(net: RiverNetwork, plan: core.GridPlan, path: str) -> dict:
    doc = {
        "format": "kvterrain-lakes/1",
        "crs": f"EPSG:{plan.epsg}",
        "lake_id_base": kvid.LAKE_ID_BASE,
        "mask_reference": {
            "file": kvid.WATER_ID_FILE,
            "rule": f"a pixel belongs to lake_id n where water_id == "
                    f"{kvid.LAKE_ID_BASE} + n",
        },
        "authored_level_field": "authored_level_m",
        "policy": "AuthoredWins",
        "level_source_field": "level_source",
        # Keyed off the constants, not off copies of their values: the strings are
        # a serialisation contract, and a rename that updated one of the four
        # places they used to be spelled out would have gone unnoticed.
        "level_sources": {
            kvwater.LEVEL_SOURCE_NVE: "the lake's authored NVE `hoyde`",
            kvwater.LEVEL_SOURCE_ESTIMATED:
                "read off the LiDAR water surface inside the polygon (the low band "
                "of it), capped at the land ringing the lake, because NVE publishes "
                "no hoyde for this lake. Pin it exactly like an NVE one — it is the "
                "level the surface raster carries.",
        },
        "note": "no polygon geometry is emitted; the per-pixel mask in "
                "water_id.atlas is the geometry reference. `hoyde_moh` is the raw "
                "NVE field and may be null; `authored_level_m` is always the level "
                "this export actually used.",
        "lakes": net.lakes,
    }
    # Only described when used, so an export with the correction off stays
    # byte-identical to one made before the correction existed.
    if any(l.get("level_source") == kvwater.LEVEL_SOURCE_CORRECTED for l in net.lakes):
        doc["level_sources"][kvwater.LEVEL_SOURCE_CORRECTED] = (
            "NVE publishes a hoyde, but it stands more than the configured margin "
            "above practically all the land around the lake, so it was replaced by "
            "the same DTM estimate. `hoyde_moh` still holds the published value.")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return {"file": os.path.basename(path), "count": len(net.lakes),
            "with_authored_level": sum(
                1 for l in net.lakes if l.get("authored_level_m") is not None),
            "with_estimated_level": sum(
                1 for l in net.lakes
                if l.get("level_source") == kvwater.LEVEL_SOURCE_ESTIMATED),
            "with_corrected_level": sum(
                1 for l in net.lakes
                if l.get("level_source") == kvwater.LEVEL_SOURCE_CORRECTED)}


def write_junctions_json(net: RiverNetwork, plan: core.GridPlan, path: str) -> dict:
    doc = {
        "format": "kvterrain-junctions/1",
        "crs": f"EPSG:{plan.epsg}",
        "types": [JUNCTION_INFLOW, JUNCTION_OUTFLOW],
        "note": "Inflow: the river segment enters the lake at this point (it "
                "becomes a source on the lake's basin node). Outflow: the segment "
                "leaves the lake here (it starts at the lake's spill point).",
        "junctions": [
            {
                "segment_id": j.segment_id,
                "strekn_lnr": j.strekn_lnr,
                "lake_id": j.lake_id,
                "vatn_lnr": j.vatn_lnr,
                "type": j.kind,
                "position": [round(j.position[0], 2), round(j.position[1], 2),
                             round(j.position[2], 2)],
            }
            for j in net.junctions
        ],
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    inflow = sum(1 for j in net.junctions if j.kind == JUNCTION_INFLOW)
    return {"file": os.path.basename(path), "count": len(net.junctions),
            "inflow": inflow, "outflow": len(net.junctions) - inflow}


def write_river_widths_bin(net: RiverNetwork, path: str) -> dict:
    """
    The visual full width at every vertex, in rivers.bin's segment and vertex
    order: header `<8sIII` (magic, version, segment_count, vertex_count), then per
    segment `<II` (seg_id, vertex_count) and vertex_count f32 widths in metres. A
    segment without visual widths repeats its hydraulic width_m.
    """
    total = 0
    with open(path, "wb") as fh:
        fh.write(struct.pack("<8sIII", WIDTHS_MAGIC, WIDTHS_VERSION,
                             len(net.segments), net.total_vertices()))
        for s in net.segments:
            vw = getattr(s, "vertex_width_m", None)
            if vw is None or len(vw) != s.n:
                vw = np.full(s.n, float(s.width_m or 0.0))
            fh.write(struct.pack("<II", int(s.seg_id), int(s.n)))
            fh.write(np.asarray(vw, dtype="<f4").tobytes())
            total += s.n
    return {
        "file": os.path.basename(path),
        "format": "kvterrain-river-widths/1",
        "magic": WIDTHS_MAGIC.decode("ascii"),
        "version": WIDTHS_VERSION,
        "byte_order": "little_endian",
        "header_layout": "<8sIII (magic, version, segment_count, vertex_count)",
        "segment_layout": "<II (seg_id, vertex_count), then vertex_count * f32 width_m",
        "segments": len(net.segments),
        "vertices": int(total),
        "bytes": os.path.getsize(path),
        "quantity": "full VISUAL channel width in metres at each rivers.bin vertex, "
                    "same segment and vertex order",
        "note": "What the export rasterised, carved and levelled the river to: "
                "Kartverket N50's river areas and width classes where they exist, "
                "the hydraulic width (rivers.bin width_m, from discharge) elsewhere. "
                "width_m stays the hydraulic width for the solver.",
    }


def export_river_network(
    net: RiverNetwork, plan: core.GridPlan, out_dir: str, *,
    emit_geojson: bool = False,
) -> dict:
    """Write every polyline-derived product and return a manifest fragment."""
    os.makedirs(out_dir, exist_ok=True)
    man = {
        "rivers": write_rivers_bin(net, plan, os.path.join(out_dir, RIVERS_BIN_FILE)),
        "lakes": write_lakes_json(net, plan, os.path.join(out_dir, LAKES_FILE)),
        "junctions": write_junctions_json(
            net, plan, os.path.join(out_dir, JUNCTIONS_FILE)),
        "validation": net.report,
    }
    if any(getattr(s, "vertex_width_m", None) is not None for s in net.segments):
        man["river_widths"] = write_river_widths_bin(
            net, os.path.join(out_dir, RIVER_WIDTHS_FILE))
    hydro = net.report.get("hydrology")
    if hydro:
        man["rivers"]["discharge"] = hydro
        man["rivers"]["width_model"] = net.report["width_model"]
        man["rivers"]["hierarchy_attachment"] = net.report.get(
            "hierarchy_attachment", {})
    if emit_geojson:
        man["rivers_geojson"] = write_rivers_geojson(
            net, plan, os.path.join(out_dir, RIVERS_GEOJSON_FILE))
    return man


def _tool_version() -> str:
    """
    Best-effort generator version.

    Deliberately defensive. `from . import __version__` fails outright if the
    package directory has no `__init__.py` — Python then treats `kvterrain` as a
    NAMESPACE package, which imports fine (so `from kvterrain import core` works
    and a whole build runs to completion) but carries no module-level attributes.
    The symptom is a late `ImportError: cannot import name '__version__' from
    'kvterrain' (unknown location)` that throws away a finished export at the very
    last step, over a provenance string.

    Provenance is worth recording, not worth losing an export for. If the version
    cannot be determined the manifest says so and the build completes.
    """
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return str(version("kvterrain"))
    except Exception:
        pass
    return "unknown (kvterrain/__init__.py missing or not importable)"


def generator_metadata(plan: core.GridPlan, params: dict) -> dict:
    """Provenance block: who made this file, when, from what, with which knobs."""
    return {
        "name": "kvterrain",
        "version": _tool_version(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crs": f"EPSG:{plan.epsg}",
        "transform": {
            "origin_utm": [plan.origin_x, plan.origin_y],
            "leaf_spacing_m": plan.spacing_m,
            "samples": [plan.samples_x, plan.samples_y],
            "row_order": "north_to_south",
            "sampling": "corner_centered_pixel_is_a_point",
        },
        "parameters": params,
    }
