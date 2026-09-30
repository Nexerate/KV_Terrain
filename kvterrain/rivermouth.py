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

OUTLETS open onto a RIFFLE. A real lake stands above its outlet sill by the depth
of the water leaving over it, and the depression hierarchy models exactly that:
a node holds water to spill + outflow head, the head being the river surface at
the spill minus the spill, read only where the spill is a RIVER sample (exporter
and runtime alike). So an outlet is shaped for that reading:

  * the throat opens and the funnel behind it holds the river's depth until it
    meets deeper lake bed (a narrow channel on across a lake shallower than that);
  * the river samples beside the throat become the riffle: raised, never lowered,
    to a crest 10 cm above the river's own shallowest first samples, so the riffle
    is the one highest point on the way out and the lake's spill is a river
    sample;
  * the crest is raised further, to the rim of any hollow in the lake bed that a
    real share of the lake drains into (`hollow_min_*`). Below that rim the hollow
    would be a basin of its own and the lake would split in the hierarchy;
  * the river's surface over the riffle is the lake's level (`raised_out`, which
    process.py writes into the surface raster and the polylines), so the head is
    exactly level - crest.

An outlet with no room for a riffle (crest within `min_riffle_m` of the level)
keeps its sill. The lake matching in the water audit (and the runtime's) floods a
lake's candidate node to spill + head, not the bare spill: a lake shallower than
its outlet river stands wholly above its riffle and would otherwise match nothing.

Measured on Lierne (2026-09-30), median lake depth along the river's line at
0/5/10/15/20 m past the mouth: inflows 0/0/1.0/1.4/1.7 m before, 1.5/1.6/2.0/2.4/
2.5 m after, with the river 1.9 m deep just outside; outlets 0/0/0.5/0.8/1.1 m
before, 1.8/1.9/1.7/2.1/2.4 m after. 72% of inflows and 52% of outlets are wet
the whole way from river to lake (0% before). 827 of 1 159 outlets carry a riffle
(median 1.4 m deep, p10 0.6, p90 2.9); 332 keep their sill. The water audit:
1 371 ok + 789 held by their outflow (was 2 171 ok), 11 without a node (was 7),
9 above their spill (was 2); the river pits it lists drop from 578 to 223.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import heapq
from collections import defaultdict

import numba as nb
import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label, maximum
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
    open_outlets: bool = True
    throat_cells: float = 1.5
    # An outlet's funnel is cut this much below the river's first samples, so the
    # river sample, not the funnel, is the lake's spill.
    outlet_margin_m: float = 0.1
    # How far the escape check behind an opened inlet looks (metres).
    check_radius_m: float = 1500.0
    # A hollow in the lake bed counts toward an outlet's riffle when at least this
    # many samples, and this share of the lake, drain into it; smaller ones (shore
    # roughness) are left as basins of their own.
    hollow_min_samples: int = 25
    hollow_min_fraction: float = 0.01
    # A riffle shallower than this is no outlet at all: the sill stays shut.
    min_riffle_m: float = 0.1

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
            vw = getattr(s, "vertex_width_m", None)
            for kind, m, step in ((INFLOW, a, 1), (OUTFLOW, b, -1)):
                if (kind == INFLOW and a == 0) or (kind == OUTFLOW and b == n - 1):
                    continue            # the polyline starts/ends inside the lake
                # The channel's own (visual) width just outside the mouth.
                half_w = 0.5 * float(vw[m - step] if vw is not None and len(vw) == n
                                     else (s.width_m or spacing_m))
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


@nb.njit(cache=True)
def _connect_heights(bed, allowed, seed_r, seed_c):
    """For every allowed sample, the lowest height water must reach to join it to
    one of the seeds through allowed samples (8-connected): the highest bed on the
    best path. Equal to the bed wherever the sample drains to a seed; higher
    inside a hollow, by exactly the hollow's rim."""
    H, W = bed.shape
    out = np.full((H, W), np.inf)
    done = np.zeros((H, W), np.bool_)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for k in range(seed_r.shape[0]):
        r, c = seed_r[k], seed_c[k]
        out[r, c] = bed[r, c]
        heapq.heappush(heap, (float(bed[r, c]), np.int64(r * W + c)))
    while len(heap) > 0:
        h, i = heapq.heappop(heap)
        r = i // W
        c = i - r * W
        if done[r, c]:
            continue
        done[r, c] = True
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= H:
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if cc < 0 or cc >= W or done[rr, cc] or not allowed[rr, cc]:
                    continue
                v = max(float(bed[rr, cc]), h)
                if v < out[rr, cc]:
                    out[rr, cc] = v
                    heapq.heappush(heap, (v, np.int64(rr * W + cc)))
    return out


@nb.njit(cache=True)
def _drainage_counts(bed, allowed):
    """How many allowed samples drain, by steepest descent through allowed samples
    (8-connected), into each sample: nonzero only at the pits where descent ends."""
    H, W = bed.shape
    recv = np.full(H * W, -1, np.int64)
    for r in range(H):
        for c in range(W):
            if not allowed[r, c]:
                continue
            best = bed[r, c]
            k = -1
            for dr in (-1, 0, 1):
                rr = r + dr
                if rr < 0 or rr >= H:
                    continue
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if cc < 0 or cc >= W or not allowed[rr, cc]:
                        continue
                    if bed[rr, cc] < best:
                        best = bed[rr, cc]
                        k = rr * W + cc
            recv[r * W + c] = k
    term = np.full(H * W, -1, np.int64)
    counts = np.zeros(H * W, np.int64)
    stack = np.empty(H * W, np.int64)
    for i in range(H * W):
        if not allowed[i // W, i % W]:
            continue
        n = 0
        j = i
        while term[j] < 0 and recv[j] >= 0:
            stack[n] = j
            n += 1
            j = recv[j]
        t = term[j] if term[j] >= 0 else j
        term[j] = t
        for q in range(n):
            term[stack[q]] = t
        counts[t] += 1
    return counts.reshape(H, W)


def _outlet_riffles(bed, outlets, *, water_type, lake_id, island, p, report,
                    raised_out) -> None:
    """Hold every opened outlet with a riffle: raise the river's first samples to
    the lake's crest, so the lake's spill is a RIVER sample (which carries an
    outflow head) and the lake's whole bed joins up before it spills. See the
    module docstring."""
    from . import water as kvwater

    if not outlets:
        return
    H, W = bed.shape
    lakes_only = np.where(water_type == kvwater.TYPE_LAKE, lake_id, 0)
    slices = find_objects(lakes_only)
    del lakes_only
    by_lake = defaultdict(list)
    for o in outlets:
        by_lake[o[0].lake_id].append(o)
    depths = []
    shut = 0
    for lid, outs in by_lake.items():
        sl = slices[lid - 1] if lid - 1 < len(slices) else None
        if sl is None:
            continue
        r0, r1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)
        c0, c1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)
        b = bed[r0:r1, c0:c1]
        own = ((water_type[r0:r1, c0:c1] == kvwater.TYPE_LAKE)
               & (lake_id[r0:r1, c0:c1] == lid))
        if island is not None:
            own &= ~island[r0:r1, c0:c1]
        tr = np.concatenate([o[1] for o in outs]) - r0
        tc = np.concatenate([o[2] for o in outs]) - c0
        level = float(outs[0][0].level_m)
        conn = _connect_heights(b.astype(np.float64), own, tr.astype(np.int64),
                                tc.astype(np.int64))
        # The lake's crest: the highest rim of any hollow in its bed that a real
        # share of the lake drains into. Below it, that hollow would be a basin of
        # its own that joins the outlet only above the riffle, and the lake would
        # split between it and the outlet. Hollows judged by what drains INTO
        # them: a bowl whose low centre sits a hand's breadth below its rim holds
        # few samples under the rim, yet is where the whole lake drains.
        hollow = own & np.isfinite(conn) & (conn > b + 1e-3)
        crest = -np.inf
        lbl, n = label(hollow, structure=np.ones((3, 3), bool))
        if n:
            idx = np.arange(1, n + 1)
            drains = _drainage_counts(b.astype(np.float64), own)
            catch = np.bincount(lbl.ravel(), weights=drains.ravel(),
                                minlength=n + 1)[1:]
            rims = np.asarray(maximum(conn, lbl, idx))
            big = catch >= max(p.hollow_min_samples,
                               p.hollow_min_fraction * float(own.sum()))
            if big.any():
                crest = float(rims[big].max()) + p.outlet_margin_m
        for mo, orr, occ, old, d_funnel in outs:
            # A margin above the river's own bed too, so the riffle is the one
            # highest sample on the way out and not tied with the channel below.
            top = max(level - d_funnel + 2.0 * p.outlet_margin_m, crest)
            if top > level - p.min_riffle_m:
                bed[orr, occ] = old            # no room for a riffle: stay shut
                shut += 1
                continue
            # The river samples beside this outlet's throat carry the riffle.
            wr0, wr1 = max(int(orr.min()) - 2, 0), min(int(orr.max()) + 3, H)
            wc0, wc1 = max(int(occ.min()) - 2, 0), min(int(occ.max()) + 3, W)
            m = np.zeros((wr1 - wr0, wc1 - wc0), bool)
            m[orr - wr0, occ - wc0] = True
            near = (binary_dilation(m, structure=np.ones((3, 3), bool))
                    & (water_type[wr0:wr1, wc0:wc1] == kvwater.TYPE_RIVER))
            rr, cc = np.nonzero(near)
            rr, cc = rr + wr0, cc + wc0
            low = bed[rr, cc] < top
            bed[rr[low], cc[low]] = np.asarray(top, dtype=bed.dtype)
            if raised_out is not None and rr.size:
                raised_out.append((rr, cc, level))
            depths.append(level - top)
            report["riffle_samples"] += int(low.sum())
    report["outlets_kept_shut"] = shut
    if depths:
        d = np.asarray(depths)
        report["riffle_depth_m"] = {
            "p10": round(float(np.percentile(d, 10)), 2),
            "p50": round(float(np.median(d)), 2),
            "p90": round(float(np.percentile(d, 90)), 2)}


def carve_river_mouths(
    bed: np.ndarray,
    mouths: list,
    *,
    plan,
    water_type: np.ndarray,
    lake_id: np.ndarray,
    island: Optional[np.ndarray] = None,
    params: Optional[MouthParams] = None,
    raised_out: Optional[list] = None,
) -> tuple[np.ndarray, dict]:
    """
    Lower lake samples along each mouth's path to the channel profile. `bed` is the
    finished lake bed and is modified IN PLACE (and returned). See the module
    docstring for the shape and the shoreline rule.

    The one place this RAISES terrain is an outlet's riffle, on the river samples
    beside the outlet's throat. For each outlet, (rows, cols, lake level) of those
    samples is appended to `raised_out` when given, so the caller can carry the
    riffle into any other copy of the river bed (the polylines' `z`) and stand the
    river's surface there at the lake's level: water leaving over the riffle is
    the lake's own, and the outflow head is read from that surface.
    """
    from . import rivernet, water as kvwater

    p = params or MouthParams()
    report = {"params": p.to_dict(), "mouths": len(mouths),
              "inflows": 0, "outflows": 0, "carved": 0, "samples_lowered": 0,
              "max_lowering_m": 0.0, "outlets_opened": 0, "outlets_capped": 0,
              "riffle_samples": 0}
    if not p.enabled or not mouths:
        return bed, report

    sp = float(plan.spacing_m)
    H, W = bed.shape
    lowered = np.zeros(bed.shape, bool)
    max_low = 0.0
    throats = []
    outlets = []

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
        # The mouth widens over the first half of the reach and runs on at that
        # width: an outlet's channel held across a shallow lake is a channel, not
        # a funnel growing to swallow the lake.
        flare_s = 0.5 * L
        reach = w0 + flare * min(float(s[-1]), flare_s)

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
        w = w0 + flare * np.clip(sk, 0.0, flare_s)
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
            outlets.append((mo, rr[o], cc[o], cur[o].copy(), d0))
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

    _outlet_riffles(bed, outlets, water_type=water_type, lake_id=lake_id,
                    island=island, p=p, report=report, raised_out=raised_out)

    report["samples_lowered"] = int(lowered.sum())
    report["max_lowering_m"] = round(max_low, 2)
    return bed, report
