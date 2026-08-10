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
constructed, and this module constructs it exactly the way the raster already
does, because that is the behaviour that visibly works:

    z(v)     = leaf DTM sampled beneath vertex v          <- the BED
    level(v) = z(v) + raise_for_order(order)              <- the water SURFACE

Two values per vertex, deliberately, because they have different consumers:

  * `z` is the BED, and it is what the burn runs its running-minimum over. Burning
    against the surface instead would cut ~1-11 m too shallow and leave the DEM
    dams in place.
  * `level` is the authored water surface, and it is what a river pins to when it
    is not disturbed. It equals the raster's surface value at that pixel by
    construction, so the polyline and the raster cannot drift apart.

This is the same per-pixel rule as `watersurface.river_surface_moh`, applied at
vertices. It is NOT a fitted profile. An earlier revision rasterised the centreline,
fitted a monotone-descending profile (PAVA) and widened it back out; wherever the
chord between two sparse Elvenett vertices cut across a bank, the fit pooled the
surface metres above the bed and the widen smeared that inflated value onto lower
neighbours, so the runtime read depth >> 0 and the river spiked.

The defence here is DENSIFICATION, not fitting. Vertices are resampled to the leaf
spacing before Z is sampled, so the polyline follows the real channel instead of
cutting chords across it. Non-monotonic descent that survives is REPORTED, never
corrected: the runtime's running-minimum is what enforces descent during the burn.

FLOW DIRECTION
--------------
Source vertex order is trusted. ELVIS is designed to encode flow direction, but
that is design intent rather than a per-feature guarantee, and nothing verifies
it upstream of us. So a Z-derived direction is computed independently and the
DISAGREEMENT RATE is reported as a diagnostic — broken out by whether the segment
touches a lake, because clustering there indicates a real problem while a diffuse
scatter is just DTM noise. Z never flips a segment: on flat reaches, through
weirs, across lakes and over DTM artefacts it is the noisier signal of the two.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from . import core
from . import water as kvwater
from . import waterid as kvid
from .watersurface import DEFAULT_DEPTH_BY_ORDER, depth_for_order, _fill_nan_1d, _bresenham_path

RIVERS_BIN_FILE = "rivers.bin"
RIVERS_GEOJSON_FILE = "rivers.geojson"
LAKES_FILE = "lakes.json"
JUNCTIONS_FILE = "junctions.json"

BIN_MAGIC = b"KVRIVER1"
BIN_VERSION = 1

JUNCTION_INFLOW = "Inflow"
JUNCTION_OUTFLOW = "Outflow"

# A vertex is "the same point" as another within this fraction of leaf spacing,
# for endpoint matching during connectivity building.
ENDPOINT_TOL_CELLS = 1.0
# Descent tolerance: a rise smaller than this is DTM noise, not a real reversal.
DESCENT_TOL_M = 0.05

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
    upstream: list = field(default_factory=list)     # seg_ids
    downstream: Optional[int] = None                 # seg_id
    lake_spans: list = field(default_factory=list)   # (start_idx, end_idx, local_lake_id)

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
) -> RiverNetwork:
    """
    Turn the elvenett polylines into the serialisable network: densified, Z-sampled,
    connected, clipped at lake boundaries, validated.

    `height_leaf` must be the LEAF DTM the surface raster was built against, so the
    polyline `z` (the bed) matches the bed the raster used.

    `water_surface` is the COMBINED raster surface field. When supplied, each
    vertex's `level` is read straight out of it, which makes polyline/raster
    agreement exact rather than merely close. Recomputing `bed + raise`
    independently is not good enough: the raster applies rules the recomputation
    would miss — most importantly it pins river pixels ADJACENT to a lake to that
    lake's surface, so a vertex near an inlet would otherwise carry `bed + raise`
    while the raster carried the lake's `hoyde`. On steep inlet banks that is a
    100 m+ disagreement on the handful of vertices that matter most, since they
    are precisely where the burn hands off to the lake basin.

    Without `water_surface` the function falls back to `bed + raise` plus lake-span
    pinning, which is correct in the interior and wrong only at those boundaries.
    """
    depth_by_order = depth_by_order or DEFAULT_DEPTH_BY_ORDER
    stride = float(vertex_stride_m) if vertex_stride_m else float(plan.spacing_m)
    weight_raw = wg.weight_raw if wg.weight_raw is not None else wg.weight

    SY, SX = plan.samples_y, plan.samples_x
    if height_leaf.shape != (SY, SX):
        raise ValueError(f"height_leaf {height_leaf.shape} != grid {(SY, SX)}")

    # Resolved level, not the raw NVE field: a polyline crossing a lake whose level
    # was estimated must be pinned to that same level, or the polyline and the
    # surface raster disagree at exactly the inlet/outlet the burn hands off at.
    lake_hoyde = {lid: wg.lake_level(info)
                  for lid, info in (wg.lake_table or {}).items()}

    segments: list = []
    clip = {"features": 0, "clipped": 0, "split": 0,
            "vertices_in": 0, "vertices_dropped": 0, "features_dropped": 0}
    span_stats = {"raw": 0, "merged": 0, "dropped": 0, "kept": 0}

    for seg in feats.rivers:
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
            z = _bilinear(height_leaf, row, col)
            # Nodata bed samples along the channel are interpolated from their finite
            # neighbours rather than dropped — a hole in the middle of a polyline would
            # otherwise break the burn's running minimum.
            z = _fill_nan_1d(z)
            if not np.isfinite(z).any():
                continue

            raise_m = depth_for_order(order, depth_by_order, depth_scale)
            level = z + raise_m

            ri, ci = _nearest_idx(plan, xy)
            lake_at = wg.lake_id[ri, ci].astype(np.int64)
            spans, st = _runs_of_lake(lake_at)
            for k in span_stats:
                span_stats[k] += st[k]

            # Inside a lake, the authored lake surface wins over bed+raise: the lake's
            # NVE `hoyde` is real data and the raise is a synthetic constant, so a
            # through-line must not disagree with the pool it crosses.
            for a, b, lid in spans:
                h = lake_hoyde.get(int(lid))
                if h is not None and np.isfinite(h):
                    level[a:b + 1] = float(h)

            # Then, wherever the raster actually carries water at this vertex's pixel,
            # take ITS value verbatim. The raster is the display authority; this makes
            # the two products identical instead of independently derived, and it picks
            # up the adjacent-to-lake pinning that bed+raise cannot know about.
            if water_surface is not None:
                ras = np.asarray(water_surface, dtype=np.float64)[ri, ci]
                take = np.isfinite(ras)
                level[take] = ras[take]

            segments.append(RiverSegment(
                seg_id=len(segments),
                # Hash the CLIPPED run, not the source feature: after a split the
                # source geometry no longer identifies one emitted segment, and two
                # pieces sharing one hash would collide as keys.
                geom_hash=geometry_hash(xy),
                order=int(order),
                xy=xy,
                z=z.astype(np.float32),
                level=level.astype(np.float32),
                strekn_lnr=seg.strekn_lnr,
                elvid=seg.elvid,
                vassdragsnr=seg.vassdragsnr,
                vatnlnr=seg.vatnlnr,
                lake_spans=spans,
            ))

    _link_segments(segments, tol_m=ENDPOINT_TOL_CELLS * plan.spacing_m)
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
    Count vertices whose bed rises going downstream. Reported, NOT fixed: the tool
    must not smooth or correct river Z. The runtime's running minimum during the
    burn is what enforces descent, and it does so without inventing elevations.
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
        "descent_note": "reported only; not corrected. The runtime burn's "
                        "running-minimum enforces descent.",
    }


def validate_flow_direction(segments: list, tol_m: float = 0.5) -> dict:
    """
    Compare the SOURCE vertex order against a Z-derived direction, and report how
    often they disagree — split by whether the segment touches a lake.

    Source order wins in every case. This is a diagnostic, not a correction: a
    diffuse low disagreement rate is ordinary DTM noise on flat reaches, whereas a
    high rate, or one concentrated in lake-touching segments, points at something
    real (a genuinely mis-ordered layer, or lake-through-lines whose flat Z carries
    no directional signal at all).
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
        "flowdir_note": "source vertex order is authoritative; Z is an independent "
                        "QA signal only and never flips a segment.",
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
#   8s  magic "KVRIVER1"
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
                "<IqqHi", s.seg_id,
                -1 if s.strekn_lnr is None else int(s.strekn_lnr),
                -1 if s.vatnlnr is None else int(s.vatnlnr),
                int(s.order),
                -1 if s.downstream is None else int(s.downstream)))
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
        "format": "kvterrain-rivers-bin/1",
        "magic": BIN_MAGIC.decode("ascii"),
        "byte_order": "little_endian",
        "bytes": os.path.getsize(path),
        "segments": len(net.segments),
        "vertices": net.total_vertices(),
        "vertex_fields": ["x_rel_origin_m", "y_rel_origin_m", "z_bed_m", "level_surface_m"],
        "vertex_order": "upstream_to_downstream",
        "note": ("z is the sampled DTM BED — burn against this. level is the water "
                 "SURFACE (bed + raise by stream order, or the lake's authored "
                 "hoyde inside a lake span) — pin to this. Loaded once, resident; "
                 "not streamed."),
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
        "level_sources": {
            "nve_hoyde": "the lake's authored NVE `hoyde`",
            "dtm_interior_median": "read off the LiDAR water surface inside the "
                                   "polygon, because NVE publishes no hoyde for "
                                   "this lake. Pin it exactly like an NVE one — "
                                   "it is the level the surface raster carries.",
        },
        "note": "no polygon geometry is emitted; the per-pixel mask in "
                "water_id.atlas is the geometry reference. `hoyde_moh` is the raw "
                "NVE field and may be null; `authored_level_m` is always the level "
                "this export actually used.",
        "lakes": net.lakes,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return {"file": os.path.basename(path), "count": len(net.lakes),
            "with_authored_level": sum(
                1 for l in net.lakes if l.get("authored_level_m") is not None),
            "with_estimated_level": sum(
                1 for l in net.lakes
                if l.get("level_source") == "dtm_interior_median")}


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