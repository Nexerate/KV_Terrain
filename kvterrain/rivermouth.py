"""
kvterrain.rivermouth
====================

Carry each river's channel on into the lake it meets.

`lakebed.build_lake_beds` treats every sample around a lake that is not lake as
shore, and a river is not lake. So each river mouth got the treatment of a
shoreline: the 8-connected ring on the waterline, then a bed only as deep as a
beach a few metres out. On Lierne (2026-09-29) all 2 732 mouths had a two-sample
line at depth zero right across a channel about 2 m deep, and the lake stayed
shallower than the river for the next 10 m or so. The water shader draws the
zero-depth line as dry ground and the shallows as nearly clear water, so every
river looked cut off a few metres short of its lake.

A real channel does not stop at the shoreline. An inflow keeps its trench into
the lake and fans out into a delta until the lake floor falls away below it; an
outlet is fed by a funnel that narrows onto the channel. This module cuts both,
along the river polyline's own path through the lake:

    depth(s, r) = D(s) * P(r / w(s))

  * `s` is distance into the lake along the polyline from the mouth vertex, and
    `r` the distance from the polyline.
  * `D(s) = (d0 + grade * s) * fade(s)`: the river's own depth at the mouth,
    deepening slightly, held for the first half of the reach `L` and smoothly
    shallowed to nothing at `L`. `L` scales with the channel width.
  * `w(s) = w0 + flare * s`: the river's half-width at the mouth, widening into
    the lake (a wider flare for inflows, whose deltas spread, than for outlets).
  * `P` is flat over the inner half of the width and smoothstepped to zero at its
    edge, so the channel blends into the lake bed rather than stepping.

The lake bed is only ever LOWERED to this (`min`), so where the lake is already
deeper nothing changes and the channel simply runs out into it.

What it does not touch
----------------------

Every lake keeps its shore ring on the waterline: the 8-connected ring of samples
touching anything that is not the lake (land, islands, rivers, the map edge).
That ring is what holds a lake at its level in the depression hierarchy, and it is
the shoreline, so the channel profile is cut behind it and not through it. Only
samples of the SAME lake the polyline crosses are cut, so a channel never reaches
into a neighbouring lake, and land, rivers and islands are never touched.

One exception, the INLET THROAT: the ring samples within the river's width of its
line, a cell or two past an inflow's mouth, are opened so the channel runs into
the lake without a dry line across it. The lake then backs up its inflow's trench
until the river bed rises above the level, and fills any hollow beside it, as a
real lake would; that cannot lower its spill. It could only drain if the flood
below the level ESCAPES: flooding from the throat through samples below the level,
not crossing the lake (`_escapes`), reaches the map edge or runs beyond the check
window, i.e. finds lower ground. Then the throat goes back on the waterline. On
Lierne (2026-09-29) 1 488 of 1 573 inlets open (17 shut at the map edge, 68
unbounded).

OUTLETS stay shut for now (`open_outlets`). An outflow head is read only where a
node's spill is a RIVER sample (exporter and runtime alike), and the node matched
to a lake is flooded only to its spill, not spill + head. Cutting an outlet funnel
to the river's depth breaks both on Lierne: wherever the lake's own bed has a
ridge shallower than the river at the outlet, that ridge (a lake sample, no head)
becomes the spill of the lake's main basin (285 lakes `level_above_spill`), and a
lake shallower than its outlet river lies wholly above its spill and matches no
node (284). With the ring shut the funnel is cut behind the sill.

Measured on Lierne, median lake depth along the river's line at 0/5/10/15/20 m
past the mouth: inflows 0/0/1.0/1.4/1.7 m before, 1.5/1.6/1.7/2.4/2.4 m after,
with the river 1.9 m deep just outside; 71% of inflows are wet all the way from
the river into the lake. Outlets 0/0/0.5/0.8/1.1 m before, 0/0/1.6/1.7/1.7 m after
(the zeros are the sill). The water audit: 2 163 ok + 1 held by outflow (was
2 171 ok), 10 without a node (was 7), 6 above their spill (was 2); the river pits
it lists drop from 578 to 223 because rivers now run into their lakes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial import cKDTree

INFLOW = "inflow"
OUTFLOW = "outflow"


@dataclass
class MouthParams:
    """Every tunable of the mouth carve. Recorded verbatim in the manifest."""
    enabled: bool = True
    # Reach L into the lake, as channel widths, clamped to [min, max] metres.
    length_widths: float = 12.0
    min_length_m: float = 40.0
    max_length_m: float = 250.0
    # Extra depth gained per metre into the lake (the channel keeps descending).
    grade: float = 0.01
    # Half-width gained per metre into the lake: a delta spreads, a funnel less so.
    inflow_flare: float = 0.35
    outflow_flare: float = 0.25
    # Never cut a channel shallower than this at the mouth (m), however shallow
    # the river's own trench is there: a mouth must read as water.
    min_depth_m: float = 0.5
    # Open the lake's shore ring across an inlet / an outlet, in the channel's
    # throat only (`throat_cells` past the mouth).
    open_inlets: bool = True
    # Off until the runtime can carry a lake over an open outlet (the module
    # docstring has the measurements).
    open_outlets: bool = False
    throat_cells: float = 1.5
    # An outlet's funnel is cut this much below the river's first samples, so the
    # river sample, not the funnel, is the lake's spill.
    outlet_margin_m: float = 0.1
    # How far the escape check behind an opened inlet looks (metres).
    check_radius_m: float = 1500.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Mouth:
    """One place a river polyline crosses a lake outline."""
    kind: str                  # INFLOW | OUTFLOW
    lake_id: int               # local lake id (WaterGrid.lake_id)
    level_m: float             # the lake's authored level
    path_xy: np.ndarray        # (N, 2) world metres, from the mouth INTO the lake
    depth_m: float             # the river channel's depth at the mouth
    half_width_m: float        # the river channel's half-width at the mouth
    segment_id: int = -1
    # An outlet whose river is trenched deeper than twice its order's modelled
    # depth: the funnel is cut to that limit, so the lake's spill stays inside it.
    capped: bool = False
    # The channel's modelled depth for its order.
    order_depth_m: float = 0.0


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _densify(xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` metres along `xy`, and their arc length from xy[0]."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return xy[:1].copy(), np.zeros(1)
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], 1), t


def find_mouths(plan, segments, bed: np.ndarray, lake_surface: np.ndarray, *,
                spacing_m: float, depth_by_order: Optional[dict] = None,
                depth_scale: float = 1.0) -> list:
    """
    Every inflow and outflow of the traced network, with the river channel's depth
    and half-width at the mouth.

    `segments` are `rivernet.RiverSegment`s with their `lake_spans` found; `bed` is
    the river-carved, lake-UNCARVED terrain, so the depth read off the last few
    river vertices is the trench the river carve actually cut. That depth is
    capped at the channel's modelled depth for its order: the downhill river bed
    occasionally trenches a brook many metres into a plateau, and a mouth must
    not carry that on across a lake.
    """
    from . import bathymetry, rivernet

    table = depth_by_order or bathymetry.DEFAULT_RIVER_DEPTH_BY_ORDER

    mouths = []
    for s in segments:
        if not s.lake_spans:
            continue
        xy = s.xy
        n = xy.shape[0]
        ri, ci = rivernet._nearest_idx(plan, xy)
        for a, b, lid in s.lake_spans:
            lvl = lake_surface[ri[a:b + 1], ci[a:b + 1]]
            lvl = lvl[np.isfinite(lvl)]
            if lvl.size == 0:
                continue
            level = float(np.median(lvl))
            half_w = 0.5 * float(s.width_m or spacing_m)
            for kind, m, step in ((INFLOW, a, 1), (OUTFLOW, b, -1)):
                if (kind == INFLOW and a == 0) or (kind == OUTFLOW and b == n - 1):
                    continue            # the polyline starts/ends inside the lake
                river = [m - step * k for k in (1, 2, 3) if 0 <= m - step * k < n]
                d = level - bed[ri[river], ci[river]]
                d = d[np.isfinite(d)]
                order_depth = bathymetry.depth_for_order(int(s.order), table,
                                                         depth_scale)
                capped = False
                if kind == INFLOW:
                    # The river's depth over the three vertices before the mouth.
                    depth = float(np.median(d)) if d.size else 0.0
                    depth = min(depth, order_depth)
                else:
                    # An outlet's funnel must reach the HIGHEST of the first river
                    # samples downstream, so that sample is the lake's spill.
                    depth = max(float(d.min()), 0.0) if d.size else 0.0
                    if depth > 2.0 * order_depth:
                        depth, capped = 2.0 * order_depth, True
                # The path into the lake: from the last river vertex (so the
                # throat is covered) to the far end of the span.
                j0 = m - step
                j1 = b if kind == INFLOW else a
                idx = np.arange(j0, j1 + step, step)
                if idx.size < 2:
                    continue
                mouths.append(Mouth(kind=kind, lake_id=int(lid), level_m=level,
                                    path_xy=xy[idx].astype(np.float64),
                                    depth_m=depth, half_width_m=half_w,
                                    segment_id=int(s.seg_id), capped=capped,
                                    order_depth_m=float(order_depth)))
    return mouths


def _escapes(bed, tr, tc, mo, water_type, lake_id, island, *, radius: int) -> str:
    """Whether opening this throat lets the lake escape: flood 8-connected from it
    through samples below the level, not crossing the lake itself. A flood that
    stays enclosed only widens the lake into its trench (and any hollow beside
    it) and cannot lower its spill. One that reaches the map edge ('map_edge') or
    runs beyond `radius` samples ('unbounded') has found lower ground to drain to.
    Returns '' when it is enclosed."""
    from . import water as kvwater

    H, W = bed.shape
    r0, r1 = max(int(tr.min()) - radius, 0), min(int(tr.max()) + radius + 1, H)
    c0, c1 = max(int(tc.min()) - radius, 0), min(int(tc.max()) + radius + 1, W)
    own = ((water_type[r0:r1, c0:c1] == kvwater.TYPE_LAKE)
           & (lake_id[r0:r1, c0:c1] == mo.lake_id))
    if island is not None:
        own &= ~island[r0:r1, c0:c1]
    wet = (bed[r0:r1, c0:c1] < mo.level_m) & ~own
    seed = np.zeros_like(wet)
    seed[tr - r0, tc - c0] = True
    seed = binary_dilation(seed, structure=np.ones((3, 3), bool)) & wet
    if not seed.any():
        return ""
    lbl, _ = label(wet, structure=np.ones((3, 3), bool))
    comp = np.isin(lbl, np.unique(lbl[seed]))
    sides = ((comp[0].any(), r0 == 0), (comp[-1].any(), r1 == H),
             (comp[:, 0].any(), c0 == 0), (comp[:, -1].any(), c1 == W))
    if any(hit and at_map for hit, at_map in sides):
        return "map_edge"
    if any(hit for hit, _ in sides):
        return "unbounded"
    return ""


def carve_river_mouths(
    bed: np.ndarray,
    mouths: list,
    *,
    plan,
    water_type: np.ndarray,
    lake_id: np.ndarray,
    island: Optional[np.ndarray] = None,
    params: Optional[MouthParams] = None,
) -> tuple[np.ndarray, dict]:
    """
    Lower lake samples along each mouth's path to the channel profile. `bed` is the
    finished lake bed and is modified IN PLACE (and returned). See the module
    docstring for the shape and the shoreline rule.
    """
    from . import rivernet, water as kvwater

    p = params or MouthParams()
    report = {"params": p.to_dict(), "mouths": len(mouths),
              "inflows": 0, "outflows": 0, "carved": 0, "samples_lowered": 0,
              "max_lowering_m": 0.0, "outlets_opened": 0, "outlets_capped": 0}
    if not p.enabled or not mouths:
        return bed, report

    sp = float(plan.spacing_m)
    H, W = bed.shape
    lowered = np.zeros(bed.shape, bool)
    max_low = 0.0
    throats = []

    for mo in mouths:
        report["inflows" if mo.kind == INFLOW else "outflows"] += 1
        w0 = max(float(mo.half_width_m), 0.5 * sp)
        L = float(np.clip(p.length_widths * 2.0 * w0, p.min_length_m, p.max_length_m))
        outlet = mo.kind == OUTFLOW
        flare = p.outflow_flare if outlet else p.inflow_flare
        d0 = max(float(mo.depth_m), p.min_depth_m)
        if outlet and p.open_outlets:
            d0 += p.outlet_margin_m
        elif outlet and mo.order_depth_m > 0:
            # Behind a shut sill the funnel is only for looks: no deeper than the
            # channel's modelled depth, like an inflow.
            d0 = max(min(d0, mo.order_depth_m), p.min_depth_m)

        pts, s = _densify(mo.path_xy, 0.5 * sp)
        # The first point is the last river vertex; the mouth is one stride on.
        s_mouth = float(np.hypot(*(mo.path_xy[1] - mo.path_xy[0])))
        s = s - s_mouth
        # Where the channel starts to fade. An inflow fades over the second half of
        # its reach. An outlet's funnel holds its full depth until the lake bed
        # along the line is deeper, so the lake reaches its outlet everywhere below
        # the river's bed and the spill is the river sample, not a lake sample.
        hold = 0.5 * L
        if outlet and p.open_outlets:
            pr, pc = rivernet._nearest_idx(plan, pts)
            lake_d = mo.level_m - bed[pr, pc]
            deep = np.nonzero((s > 0) & (lake_d >= d0 + p.grade * np.maximum(s, 0)
                                         + p.outlet_margin_m))[0]
            hold = float(s[deep[0]]) if deep.size else float(s[-1])
        keep = s <= hold + 0.5 * L
        pts, s = pts[keep], s[keep]
        if pts.shape[0] < 2:
            continue
        reach = w0 + flare * float(s[-1])

        rows, cols = rivernet.world_to_grid(plan, pts[:, 0], pts[:, 1])
        pad = int(np.ceil(reach / sp)) + 2
        r0 = max(int(np.floor(rows.min())) - pad, 0)
        r1 = min(int(np.ceil(rows.max())) + pad + 1, H)
        c0 = max(int(np.floor(cols.min())) - pad, 0)
        c1 = min(int(np.ceil(cols.max())) + pad + 1, W)
        if r0 >= r1 or c0 >= c1:
            continue
        win_t = water_type[r0:r1, c0:c1]
        win_l = lake_id[r0:r1, c0:c1]
        this = (win_t == kvwater.TYPE_LAKE) & (win_l == mo.lake_id)
        if island is not None:
            this &= ~island[r0:r1, c0:c1]
        if not this.any():
            continue
        # The shore ring: this lake's samples touching anything that is not this
        # lake (land, islands, rivers, the map edge). It stays on the waterline
        # except in an open throat.
        lake_here = (win_t == kvwater.TYPE_LAKE) & (win_l == mo.lake_id)
        ring = this & binary_dilation(~lake_here, structure=np.ones((3, 3), bool),
                                      border_value=1)
        edge = np.zeros_like(this)
        edge[0 if r0 == 0 else slice(0, 0), :] = True
        edge[-1 if r1 == H else slice(0, 0), :] = True
        edge[:, 0 if c0 == 0 else slice(0, 0)] = True
        edge[:, -1 if c1 == W else slice(0, 0)] = True
        this &= ~edge                    # map-edge samples are outlets
        open_ring = p.open_outlets if outlet else p.open_inlets

        rr, cc = np.nonzero(this)
        # Grid (row, col) of a sample -> world metres, the inverse of world_to_grid.
        gy = rr + r0
        gx = cc + c0
        wx = plan.origin_x + gx * sp
        wy = plan.origin_y + (plan.samples_y - 1 - gy) * sp
        dist, k = cKDTree(pts).query(np.stack([wx, wy], 1),
                                     distance_upper_bound=reach + sp)
        hit = np.isfinite(dist)
        if not hit.any():
            continue
        rr, cc, dist, k = rr[hit], cc[hit], dist[hit], k[hit]
        sk = s[k]
        w = w0 + flare * np.maximum(sk, 0.0)
        core = dist <= max(w0, 0.75 * sp)
        # Full depth over the core (the river's own width, and at least the
        # sample nearest the line), smoothstepped out to the flared edge.
        across = np.where(core, 1.0, 1.0 - _smoothstep(
            (dist - 0.5 * w) / np.maximum(0.5 * w, 1e-6)))
        fade = 1.0 - _smoothstep((sk - hold) / max(0.5 * L, 1e-6))
        depth = (d0 + p.grade * np.maximum(sk, 0.0)) * fade * across
        # A ring sample opens only in this mouth's own throat: within the river's
        # width of its line, a cell or two past the mouth. So a channel never
        # cuts another river's sill further along, or the shore beside it.
        throat = open_ring & core & (sk <= p.throat_cells * sp)
        in_ring = ring[rr, cc]
        ok = (depth > 0) & (~in_ring | throat)
        if not ok.any():
            continue
        opened = in_ring[ok]
        rr, cc, depth = rr[ok] + r0, cc[ok] + c0, depth[ok]
        target = (mo.level_m - depth).astype(bed.dtype)
        cur = bed[rr, cc]
        lower = target < cur
        if not lower.any():
            continue
        report["carved"] += 1
        max_low = max(max_low, float((cur[lower] - target[lower]).max()))
        o = lower & opened
        if o.any() and not outlet:
            throats.append((mo, rr[o], cc[o], cur[o].copy()))
        if o.any() and outlet:
            report["outlets_opened"] += 1
            report["outlets_capped"] += int(mo.capped)
        bed[rr[lower], cc[lower]] = target[lower]
        lowered[rr[lower], cc[lower]] = True

    # An opened inlet lets the lake back up its river's trench until the bed rises
    # above the level, which is right, and fill any hollow beside it. Only a flood
    # that escapes (see _escapes) would let the lake drain below its level; that
    # inlet's sill goes back.
    report["inlets_opened"] = len(throats)
    shut = {}
    for mo, tr, tc, old in throats:
        why = _escapes(bed, tr, tc, mo, water_type, lake_id, island,
                       radius=int(p.check_radius_m / sp) + 1)
        if why:
            bed[tr, tc] = old
            lowered[tr, tc] = False
            shut[why] = shut.get(why, 0) + 1
    report["inlets_kept_shut"] = shut

    report["samples_lowered"] = int(lowered.sum())
    report["max_lowering_m"] = round(max_low, 2)
    return bed, report
