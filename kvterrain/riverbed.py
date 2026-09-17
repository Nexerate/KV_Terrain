"""
kvterrain.riverbed
==================

The DOWNHILL-ONLY river bed: enforce, before the river carve, that neither a
river's bed nor its water level ever rises going downstream.

WHY THIS MOVED HERE
-------------------
Until the water redesign the exporter reported non-monotonic descent and left the
running minimum to the runtime, which burned the polylines into its routing grid
(`rivernet.validate_descent`: 10.5% of open-channel vertex steps rose on Lierne,
worst 4.8 m). Under the redesign there is no runtime burn: the runtime builds a
depression hierarchy on the terrain as shipped, and every rise along a mapped
channel would be a pit it has to reason about. So the enforcement happens here,
on the terrain itself (WATER_REDESIGN.md §4.1).

WHAT IT DOES, AND WHAT IT DOES NOT
----------------------------------
It is a RUNNING MINIMUM over DENSIFIED profiles, never a fit. The earlier PAVA fit
failed because sparse Elvenett chords cut across banks and pooled surface levels
metres above the bed (see `rivernet`'s module docstring); densifying to leaf
spacing is the defence against that, and a running minimum only ever lowers, so
it cannot lift anything onto a bank.

Per vertex, on the traced network (`rivernet.trace_river_network`):

    level(v) = uncarved ground under v            # today's channel level
    bed(v)   = level(v) - depth_for_order(order)  # today's trench floor

Each is replaced by its running minimum along the segment, SEPARATELY. A shared
minimum would let a bigger stream order downstream raise the surface; two
minimums keep `level - bed >= depth` of whichever vertex set the level. At a
junction a segment starts no higher than the ends of all its upstream segments.
The minimum is carried straight through lake spans (owner's decision, 2026-09-17):
an outlet channel lower than the inflow may cut the lake rim, and the lake then
extends into that channel, which is acceptable.

Back on the grid, the per-vertex minimums are spread sideways over the channel
exactly as `watersurface.channel_level` spreads the level: nearest source sample,
guarded by the modelled half-width. Then

    river level  <- min(river level, downhill level)
    bed target   <- downhill bed, where it is below today's trench floor

and `bathymetry.carve_river_beds` lowers each sample TOWARD that target, weighted
by the usual cross-section and bank taper, never below it. (Cutting an extra
depth from each sample's own ground instead overshoots on every sample lower
than its channel level and leaves a pit beside the channel; on Lierne that was
9 172 pits of 0.5 m or more.) The lake tie-in (raise-only) is re-applied after
lowering.

That handles the channel's WIDTH, but not its continuity. At 5 m spacing a
2-3 m stream rasterises to a gappy mask or none at all, so on Lierne most of the
samples a centreline actually crosses are not river samples and the carve never
touches them (with the cross-section alone, 9.4% of open-channel steps along the
polylines' own samples still rose). So after the carve, `burn_downhill_paths`
walks every segment's 8-connected sample path in upstream-first order and lowers
each sample to the running minimum of the carved bed along it, carried across
links and through lake spans (lake samples carry the minimum but are not written;
the lake carve owns them). Where the carve already descends this changes
nothing; elsewhere it shaves the bumps off the path — including, on a mapped
channel, a road embankment crossing it. `pixel_descent` measures the result on
the vertices' own samples, which is the path the depression hierarchy sees;
`rivernet`'s descent numbers sample `z` bilinearly and also read the banks.
"""

from __future__ import annotations

from typing import Optional

import numba as nb
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import bathymetry as kvbathy
from . import core
from . import rivernet
from . import water as kvwater


def _topological_order(segments: list) -> tuple[list, int]:
    """Upstream-first order over downstream links. Segments caught in a cycle are
    appended in id order afterwards; the count is returned so it gets reported."""
    by_id = {s.seg_id: s for s in segments}
    indeg = {s.seg_id: 0 for s in segments}
    for s in segments:
        if s.downstream is not None and s.downstream in indeg:
            indeg[s.downstream] += 1
    ready = [sid for sid in sorted(indeg) if indeg[sid] == 0]
    order: list = []
    head = 0
    while head < len(ready):
        sid = ready[head]
        head += 1
        order.append(sid)
        d = by_id[sid].downstream
        if d is not None and d in indeg:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    placed = set(order)
    cyclic = [s.seg_id for s in segments if s.seg_id not in placed]
    return order + cyclic, len(cyclic)


def downhill_profiles(
    plan: core.GridPlan,
    traced: "rivernet.TracedNetwork",
    ground: np.ndarray,
    *,
    depth_by_order: Optional[dict] = None,
    depth_scale: float = 1.0,
) -> dict:
    """
    Per-segment (level, bed, level_downhill, bed_downhill), float64, keyed by
    seg_id, plus the count of segments that sat in a link cycle.
    """
    depth_by_order = depth_by_order or kvbathy.DEFAULT_RIVER_DEPTH_BY_ORDER
    segs = traced.segments
    by_id = {s.seg_id: s for s in segs}
    order, n_cyclic = _topological_order(segs)

    out: dict = {}
    for sid in order:
        s = by_id[sid]
        row, col = rivernet.world_to_grid(plan, s.xy[:, 0], s.xy[:, 1])
        g = rivernet._fill_nan_1d(rivernet._bilinear(ground, row, col))
        lvl = g
        bed = g - kvbathy.depth_for_order(s.order, depth_by_order, depth_scale)

        seed_l, seed_b = np.inf, np.inf
        for u in s.upstream:
            if u in out:          # an upstream still unplaced sits in a cycle
                seed_l = min(seed_l, out[u][2][-1])
                seed_b = min(seed_b, out[u][3][-1])
        # NaN (an all-nodata run) must not poison the minimum: fmin skips it.
        lm = np.fmin.accumulate(np.concatenate([[seed_l], lvl]))[1:]
        bm = np.fmin.accumulate(np.concatenate([[seed_b], bed]))[1:]
        lm[~np.isfinite(lm)] = np.nan
        bm[~np.isfinite(bm)] = np.nan
        out[sid] = (lvl, bed, lm, bm)
    return {"profiles": out, "cyclic_segments": n_cyclic}


def downhill_river_bed(
    plan: core.GridPlan,
    traced: "rivernet.TracedNetwork",
    wg: kvwater.WaterGrid,
    ground: np.ndarray,
    river_level: np.ndarray,
    *,
    depth_by_order: Optional[dict] = None,
    depth_scale: float = 1.0,
    width_by_order: Optional[dict] = None,
    width_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (lowered river level, bed target, report). The target is NaN except on
    river samples the downhill bed lowers.

    `ground` is the uncarved, void-repaired leaf; `river_level` is
    `watersurface.river_surface_moh` BEFORE the lake tie-in (the caller re-applies
    it). Pixels no downhill vertex reaches keep their level and get no target.
    """
    depth_by_order = depth_by_order or kvbathy.DEFAULT_RIVER_DEPTH_BY_ORDER
    prof = downhill_profiles(plan, traced, ground, depth_by_order=depth_by_order,
                             depth_scale=depth_scale)
    profiles = prof["profiles"]

    SY, SX = plan.samples_y, plan.samples_x
    is_river = wg.type == kvwater.TYPE_RIVER
    level_out = np.array(river_level, dtype=np.float32, copy=True)
    target = np.full((SY, SX), np.nan, dtype=np.float32)

    n_vert = n_level_low = n_bed_low = 0
    level_drops, bed_drops = [], []
    rr, cc, vl, vb = [], [], [], []
    for s in traced.segments:
        lvl, bed, lm, bm = profiles[s.seg_id]
        n_vert += lvl.size
        dl = lvl - lm
        db = bed - bm
        low_l = np.isfinite(dl) & (dl > 0)
        low_b = np.isfinite(db) & (db > 0)
        n_level_low += int(low_l.sum())
        n_bed_low += int(low_b.sum())
        if low_l.any():
            level_drops.append(dl[low_l])
        if low_b.any():
            bed_drops.append(db[low_b])
        ri, ci = rivernet._nearest_idx(plan, s.xy)
        rr.append(ri); cc.append(ci); vl.append(lm); vb.append(bm)

    report = {
        "enabled": True,
        "segments": len(traced.segments),
        "segments_in_link_cycles": prof["cyclic_segments"],
        "vertices": int(n_vert),
        "vertices_level_lowered": int(n_level_low),
        "vertices_bed_lowered": int(n_bed_low),
    }
    for key, drops in (("level", level_drops), ("bed", bed_drops)):
        d = np.concatenate(drops) if drops else np.zeros(0)
        report[f"{key}_lowered_max_m"] = float(d.max()) if d.size else 0.0
        report[f"{key}_lowered_median_m"] = float(np.median(d)) if d.size else 0.0

    if not rr or not is_river.any():
        report["pixels_level_lowered"] = report["pixels_deepened"] = 0
        return level_out, target, report

    rr = np.concatenate(rr); cc = np.concatenate(cc)
    vl = np.concatenate(vl).astype(np.float32); vb = np.concatenate(vb).astype(np.float32)
    ok = is_river[rr, cc] & np.isfinite(vl) & np.isfinite(vb)
    rr, cc, vl, vb = rr[ok], cc[ok], vl[ok], vb[ok]

    # Several vertices (a confluence, a stride finer than the grid) can land on one
    # pixel; the lowest wins, which is the downhill-safe choice.
    src_l = np.full((SY, SX), np.inf, dtype=np.float32)
    src_b = np.full((SY, SX), np.inf, dtype=np.float32)
    np.minimum.at(src_l, (rr, cc), vl)
    np.minimum.at(src_b, (rr, cc), vb)
    src = np.isfinite(src_l)
    if not src.any():
        report["pixels_level_lowered"] = report["pixels_deepened"] = 0
        return level_out, target, report

    dist, ind = distance_transform_edt(
        ~src, sampling=(plan.spacing_m, plan.spacing_m),
        return_distances=True, return_indices=True)
    half_w = kvbathy.channel_half_width_m(
        wg.weight, is_river, plan.spacing_m, width_by_order, width_scale)
    near = is_river & (dist <= half_w + plan.spacing_m)
    spread_l = src_l[ind[0], ind[1]]
    spread_b = src_b[ind[0], ind[1]]

    orders = np.asarray(wg.weight, dtype=np.int32)
    depth_px = np.zeros((SY, SX), dtype=np.float32)
    for o in np.unique(orders[is_river]):
        depth_px[is_river & (orders == o)] = kvbathy.depth_for_order(
            int(o), depth_by_order, depth_scale)

    have = near & np.isfinite(level_out)
    lowered = have & (spread_l < level_out)
    level_out[lowered] = spread_l[lowered]
    with np.errstate(invalid="ignore"):
        want = np.asarray(river_level, dtype=np.float32) - depth_px - spread_b
    deepen = have & (want > 0)
    target[deepen] = spread_b[deepen]

    report["pixels_level_lowered"] = int(lowered.sum())
    report["pixels_deepened"] = int(deepen.sum())
    report["bed_target_below_trench_max_m"] = (
        float(want[deepen].max()) if deepen.any() else 0.0)
    return level_out, target, report


@nb.njit(cache=True)
def _burn_paths(bed, is_river, order, starts, ends, rows, cols, in_lake,
                up_ptr, up_ids):
    """Running minimum along each segment's 8-connected sample path, in place."""
    SY, SX = bed.shape
    nseg = starts.shape[0]
    end_val = np.full(nseg, np.inf)
    done = np.zeros(nseg, dtype=np.uint8)
    lowered_river = 0
    lowered_land = 0
    worst_land = 0.0
    for k in range(order.shape[0]):
        sid = order[k]
        run = np.inf
        for u in range(up_ptr[sid], up_ptr[sid + 1]):
            j = up_ids[u]
            if done[j] and end_val[j] < run:
                run = end_val[j]
        pr = -1
        pc = -1
        for v in range(starts[sid], ends[sid]):
            r1 = rows[v]
            c1 = cols[v]
            lake = in_lake[v]
            if pr < 0:
                steps = 0
            else:
                steps = max(abs(r1 - pr), abs(c1 - pc))
            # Every sample between the previous vertex's and this one, then this
            # one; with stride == spacing there is at most one step.
            for t in range(1 if pr >= 0 else 0, steps + 1):
                if steps == 0:
                    r, c = r1, c1
                else:
                    r = pr + int(round((r1 - pr) * t / steps))
                    c = pc + int(round((c1 - pc) * t / steps))
                if r < 0 or r >= SY or c < 0 or c >= SX:
                    continue
                h = bed[r, c]
                if not np.isfinite(h):
                    continue
                if h <= run:
                    run = h
                elif not lake:
                    drop = h - run
                    bed[r, c] = run
                    if is_river[r, c]:
                        lowered_river += 1
                    else:
                        lowered_land += 1
                        if drop > worst_land:
                            worst_land = drop
            pr = r1
            pc = c1
        end_val[sid] = run
        done[sid] = 1
    return lowered_river, lowered_land, worst_land


def burn_downhill_paths(plan: core.GridPlan, traced: "rivernet.TracedNetwork",
                        wg: kvwater.WaterGrid, bed: np.ndarray) -> dict:
    """
    Lower every sample along every segment's path to the running minimum of `bed`
    along it (in place), upstream segments first. See the module docstring.
    """
    segs = traced.segments
    if not segs:
        return {"path_samples_lowered_river": 0, "path_samples_lowered_land": 0,
                "path_land_lowered_max_m": 0.0}
    order_ids, n_cyclic = _topological_order(segs)
    index = {s.seg_id: i for i, s in enumerate(segs)}
    starts = np.zeros(len(segs), dtype=np.int64)
    ends = np.zeros(len(segs), dtype=np.int64)
    rows, cols, lake = [], [], []
    acc = 0
    up_ptr = np.zeros(len(segs) + 1, dtype=np.int64)
    up_ids = []
    for i, s in enumerate(segs):
        ri, ci = rivernet._nearest_idx(plan, s.xy)
        m = np.zeros(s.n, dtype=np.uint8)
        for (a, b, _lid) in s.lake_spans:
            m[a:b + 1] = 1
        rows.append(ri); cols.append(ci); lake.append(m)
        starts[i] = acc
        acc += s.n
        ends[i] = acc
        ups = [index[u] for u in s.upstream if u in index]
        up_ids.extend(ups)
        up_ptr[i + 1] = up_ptr[i] + len(ups)
    lr, ll, worst = _burn_paths(
        bed, wg.type == kvwater.TYPE_RIVER,
        np.asarray([index[sid] for sid in order_ids], dtype=np.int64),
        starts, ends, np.concatenate(rows).astype(np.int64),
        np.concatenate(cols).astype(np.int64), np.concatenate(lake),
        up_ptr, np.asarray(up_ids, dtype=np.int64))
    return {"path_samples_lowered_river": int(lr),
            "path_samples_lowered_land": int(ll),
            "path_land_lowered_max_m": float(worst)}


def pixel_descent(plan: core.GridPlan, traced: "rivernet.TracedNetwork",
                  bed: np.ndarray, tol_m: float = 1e-3) -> dict:
    """
    Rises along each segment's vertex SAMPLES on a carved bed, open channel only
    (steps touching a lake span are skipped), with repeated samples collapsed.
    """
    ids, rows, cols, lake = [], [], [], []
    for s in traced.segments:
        ri, ci = rivernet._nearest_idx(plan, s.xy)
        m = np.zeros(s.n, dtype=bool)
        for (a, b, _lid) in s.lake_spans:
            m[a:b + 1] = True
        ids.append(np.full(s.n, s.seg_id, dtype=np.int64))
        rows.append(ri); cols.append(ci); lake.append(m)
    if not ids:
        return {"pixel_steps_checked": 0, "pixel_steps_rising": 0,
                "pixel_rising_pct": 0.0, "pixel_worst_rise_m": 0.0}
    ids = np.concatenate(ids); rows = np.concatenate(rows)
    cols = np.concatenate(cols); lake = np.concatenate(lake)
    same_seg = ids[1:] == ids[:-1]
    moved = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    open_step = ~(lake[1:] | lake[:-1])
    z = np.asarray(bed, dtype=np.float64)[rows, cols]
    d = z[1:] - z[:-1]
    step = same_seg & moved & open_step & np.isfinite(d)
    up = step & (d > tol_m)
    n = int(step.sum())
    return {"pixel_steps_checked": n,
            "pixel_steps_rising": int(up.sum()),
            "pixel_rising_pct": (100.0 * int(up.sum()) / n) if n else 0.0,
            "pixel_worst_rise_m": float(d[up].max()) if up.any() else 0.0}
