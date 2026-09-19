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

    raw(v)     = uncarved ground under v
    ceiling(v) = lowest dry bank at/outside the carved-strip edge at v
    cap(v)     = min(raw(v), ceiling(v))
    run(v)     = running minimum of cap(v)
    level(v)   = running minimum of min(boxmean(run, +/-R), cap(v))
    bed(v)     = level(v) - depth_for_order(order)

The symmetric box mean removes the staircase made by a running minimum without
fitting or raising anything. The cap and a second running minimum restore the two
invariants a mean can break: below both banks, and never rising downstream. At a
junction a segment starts no higher than the ends of all its upstream segments.
The minimum is carried straight through lake spans (owner's decision, 2026-09-17):
an outlet channel lower than the inflow may cut the lake rim.

A downstream-first pass then makes nearby confluences accordant and grades river
tails onto lake levels. A grade may lower a profile by at most that river order's
carve depth; beyond that it tapers back to the ungraded profile, preserving real
hanging valleys instead of driving a shallow artificial ramp into a hillside.
The one exception is a river arriving below a lake. The lake surface expands as
a flat backwater through the contiguous low tail, across upstream segment links,
until it meets a river point already at or above lake level. This may put the
surface above a local bank sample, deliberately: at an inlet the lake is the
authority, and a deeper flooded mouth is preferable to a hidden lake edge.

Back on the grid, the per-vertex minimums are spread sideways over the channel
exactly as `watersurface.channel_level` spreads the level: nearest source sample,
guarded by the modelled half-width. Then

    river level  <- min(river level, downhill level)
    bed target   <- downhill bed, where it is below today's trench floor

and `bathymetry.carve_river_beds` lowers each sample TOWARD that target, weighted
by the usual cross-section and bank taper, never below it. (Cutting an extra
depth from each sample's own ground instead overshoots on every sample lower
than its channel level and leaves a pit beside the channel; on Lierne that was
9 172 pits of 0.5 m or more.) The old raise-only lake tie-in remains as a raster
guard after the vertex-level inlet grade.

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


CONFLUENCE_MAX_GRADE = 0.30
GRADE_TAPER_VERTICES = 4
MASK_EDGE_DIAGONAL_CELLS = 0.70711


def _segment_half_width(order: int, spacing_m: float, width_by_order: Optional[dict],
                        width_scale: float) -> float:
    """The scalar form of `bathymetry.channel_half_width_m`."""
    table = width_by_order or kvwater.DEFAULT_WIDTH_BY_ORDER
    return max(0.5 * kvwater.order_to_width_m(order, table, width_scale),
               0.5 * float(spacing_m))


def _vertex_normals(xy: np.ndarray) -> np.ndarray:
    """Unit normals from the chord spanning each vertex's two neighbours."""
    p = np.asarray(xy, dtype=np.float64)
    n = p.shape[0]
    out = np.empty_like(p)
    if n == 0:
        return out
    if n == 1:
        out[0] = (0.0, 1.0)
        return out
    a = np.maximum(np.arange(n) - 1, 0)
    b = np.minimum(np.arange(n) + 1, n - 1)
    t = p[b] - p[a]
    length = np.hypot(t[:, 0], t[:, 1])
    good = length > 1e-9
    t[good] /= length[good, None]
    t[~good] = (1.0, 0.0)
    out[:, 0] = -t[:, 1]
    out[:, 1] = t[:, 0]
    return out


def _dry_bilinear(plan: core.GridPlan, ground: np.ndarray, water_type: np.ndarray,
                  xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear ground where the categorical sample nearest the point is dry.

    Height is continuous (bilinear); water ownership in this raster exporter is
    categorical. Requiring
    all four interpolation corners to be dry rejects almost every bank beside a
    one-texel or diagonal channel: one corner is the channel even though the
    sampled point itself is beyond the rim. Use the nearest categorical sample
    to decide ownership (the same convention as the acceptance measurement) and
    still interpolate the ground at the exact point.
    """
    row, col = rivernet.world_to_grid(plan, xy[:, 0], xy[:, 1])
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    inside = ((r0 >= 0) & (c0 >= 0) &
              (r0 + 1 < plan.samples_y) & (c0 + 1 < plan.samples_x))
    value = np.full(row.shape, np.inf, dtype=np.float64)
    if not inside.any():
        return value, inside

    ii = np.flatnonzero(inside)
    rr, cc = r0[ii], c0[ii]
    vals = np.stack([ground[rr, cc], ground[rr, cc + 1],
                     ground[rr + 1, cc], ground[rr + 1, cc + 1]]).astype(np.float64)
    rn = np.clip(np.rint(row[ii]).astype(np.int64), 0, plan.samples_y - 1)
    cn = np.clip(np.rint(col[ii]).astype(np.int64), 0, plan.samples_x - 1)
    ok = (water_type[rn, cn] == kvwater.TYPE_LAND) & np.isfinite(vals).all(axis=0)
    fr, fc = row[ii] - rr, col[ii] - cc
    w = np.stack([(1.0 - fr) * (1.0 - fc), (1.0 - fr) * fc,
                  fr * (1.0 - fc), fr * fc])
    value[ii[ok]] = np.sum(vals[:, ok] * w[:, ok], axis=0)
    valid = np.zeros(row.shape, dtype=bool)
    valid[ii[ok]] = True
    return value, valid


def _bank_ceiling(plan: core.GridPlan, xy: np.ndarray, ground: np.ndarray,
                  water_type: np.ndarray, half_width_m: float) -> np.ndarray:
    """Lowest dry ground at the carved-strip edge and one sample beyond.

    The raster mask extends past the modelled half-width because a centreline
    buffer is rasterised ``all_touched``.  The runtime stamps that whole strip:
    ``max(half_width, half a sample) + 0.70711 sample``.  Sampling the modelled
    rim can therefore read ground that this exporter itself carves and mistake
    it for a dry bank.
    """
    normal = _vertex_normals(xy)
    ceiling = np.full(xy.shape[0], np.inf, dtype=np.float64)
    spacing = float(plan.spacing_m)
    strip_edge = (max(float(half_width_m), 0.5 * spacing) +
                  MASK_EDGE_DIAGONAL_CELLS * spacing)
    for side in (-1.0, 1.0):
        for reach in (strip_edge, strip_edge + spacing):
            q = np.asarray(xy, dtype=np.float64) + normal * (side * reach)
            h, ok = _dry_bilinear(plan, ground, water_type, q)
            ceiling[ok] = np.minimum(ceiling[ok], h[ok])
    return ceiling


def _shape_profile(level: np.ndarray, ceiling: np.ndarray, radius: int,
                   seed: float = np.inf) -> np.ndarray:
    """Literal numpy port of DerivedRiverProfile.Shape."""
    level = np.asarray(level, dtype=np.float64)
    ceiling = np.asarray(ceiling, dtype=np.float64)
    n = level.size
    if n == 0:
        return level.copy()
    cap = np.minimum(level, ceiling)
    run = np.minimum.accumulate(np.concatenate([[float(seed)], cap]))[1:]
    prefix = np.concatenate([[0.0], np.cumsum(run, dtype=np.float64)])
    radius = max(int(radius), 0)
    surface = np.empty(n, dtype=np.float64)
    r = float(seed)
    for i in range(n):
        w = min(radius, i, n - 1 - i)  # shrink symmetrically at both ends
        mean = (prefix[i + w + 1] - prefix[i - w]) / (2 * w + 1)
        r = min(r, mean, cap[i])
        surface[i] = r
    return surface


def _lake_level(wg: kvwater.WaterGrid, lake_id: int) -> Optional[float]:
    info = (wg.lake_table or {}).get(int(lake_id))
    if info is None:
        return None
    level = wg.lake_level(info)
    return float(level) if level is not None and np.isfinite(level) else None


def _grade_tail(surface: np.ndarray, ceiling: np.ndarray, xy: np.ndarray,
                first: int, last: int, junction_level: float, touch_m: float,
                max_grade: float, max_lowering_m: float, *,
                backwater: bool = False,
                taper_vertices: int = GRADE_TAPER_VERTICES) -> bool:
    """Grade a tail locally, never lowering it by more than its carve depth.

    Starting at the junction, the grade is followed only until it would pull the
    authored profile down by more than ``max_lowering_m``.  The permitted
    lowering then tapers to zero over a few vertices farther upstream.  Without
    this envelope, every stream steeper than ``max_grade`` follows the synthetic
    grade all the way to its source and can be sunk hundreds of metres into its
    hillside.
    """
    if last < first or not np.isfinite(junction_level):
        return False
    p = np.asarray(xy[first:last + 1], dtype=np.float64)
    if p.shape[0] == 1:
        along = np.zeros(1, dtype=np.float64)
    else:
        step = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        along = np.concatenate([[0.0], np.cumsum(step)])
        along = along[-1] - along
    # Measure the grade from the OUTER edge of the touching zone. Measuring from
    # the junction itself and then replacing the inner zone with a flat level
    # would create a G*touch vertical step at that boundary — the waterfall this
    # tie-in exists to remove.
    bound = junction_level + float(max_grade) * np.maximum(along - float(touch_m), 0.0)
    original = surface[first:last + 1].copy()
    requested = np.maximum(original - bound, 0.0)
    limit = max(float(max_lowering_m), 0.0)

    # Walk upstream from the junction.  At the first over-limit vertex, stop
    # following the grade and blend its lowering back to zero farther upstream.
    # `envelope` is non-decreasing downstream, so max(bound, original-envelope)
    # remains downstream-only when both input profiles are downstream-only.
    over = np.flatnonzero(requested > limit + 1e-9)
    if over.size:
        stop = int(over[-1])
        envelope = np.zeros(original.size, dtype=np.float64)
        ramp_start = max(0, stop - max(int(taper_vertices), 0))
        if ramp_start < stop:
            envelope[ramp_start:stop + 1] = np.linspace(
                0.0, limit, stop - ramp_start + 1)
        else:
            envelope[stop] = limit
        envelope[stop + 1:] = limit
        bounded = np.maximum(bound, original - envelope)
        surface[first:last + 1] = np.minimum(original, bounded)
    else:
        surface[first:last + 1] = np.minimum(original, bound)

    if not backwater or surface[last] >= junction_level - 1e-9:
        return False
    # The lake expands into the contiguous below-lake tail.  Lake level is the
    # authority here, including where it stands above the local dry-bank sample:
    # the result is a deeper, flat backwater mouth rather than a river surface
    # hidden below the lake or an uphill step into it.  The unchanged upstream
    # point is already >= lake level, so replacing the tail cannot introduce a
    # downstream rise.
    k = last
    while k >= first and surface[k] < junction_level - 1e-9:
        k -= 1
    surface[k + 1:last + 1] = float(junction_level)
    return k < first


def downhill_profiles(
    plan: core.GridPlan,
    traced: "rivernet.TracedNetwork",
    ground: np.ndarray,
    *,
    wg: Optional[kvwater.WaterGrid] = None,
    depth_by_order: Optional[dict] = None,
    depth_scale: float = 1.0,
    width_by_order: Optional[dict] = None,
    width_scale: float = 1.0,
    confluence_grade: float = CONFLUENCE_MAX_GRADE,
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
    ceilings: dict = {}
    half_widths: dict = {}
    legacy: dict = {}
    for sid in order:
        s = by_id[sid]
        row, col = rivernet.world_to_grid(plan, s.xy[:, 0], s.xy[:, 1])
        g = rivernet._fill_nan_1d(rivernet._bilinear(ground, row, col))
        lvl = g
        depth = kvbathy.depth_for_order(s.order, depth_by_order, depth_scale)
        bed = g - depth
        half_w = _segment_half_width(s.order, plan.spacing_m, width_by_order, width_scale)
        half_widths[sid] = half_w
        ceiling = (_bank_ceiling(plan, s.xy, ground, wg.type, half_w)
                   if wg is not None else np.full(g.shape, np.inf, dtype=np.float64))
        ceilings[sid] = ceiling

        seed_l = np.inf
        seed_old = np.inf
        for u in s.upstream:
            if u in out:          # an upstream still unplaced sits in a cycle
                seed_l = min(seed_l, out[u][2][-1])
                seed_old = min(seed_old, legacy[u][-1])
        radius = min(6, max(2, int(np.ceil(half_w / max(plan.spacing_m, 1e-9)))))
        lm = _shape_profile(lvl, ceiling, radius, seed_l)
        # The old exporter profile, retained only to report how much this change
        # lowers levels relative to the previous running-minimum implementation.
        old = np.fmin.accumulate(np.concatenate([[seed_old], lvl]))[1:]
        legacy[sid] = old
        bm = lm - depth
        lm[~np.isfinite(lm)] = np.nan
        bm[~np.isfinite(bm)] = np.nan
        out[sid] = (lvl, bed, lm, bm)

    # The main stem is known only after its tributaries have been shaped. Walk
    # downstream-first so lowering a tail is visible when that segment becomes
    # the main stem for the confluence above it.
    if wg is not None:
        backwater_at_start: dict[int, float] = {}
        for sid in reversed(order):
            if sid not in out:
                continue
            s = by_id[sid]
            lm = out[sid][2]
            ceiling = ceilings[sid]
            spans = sorted(s.lake_spans)

            # Grade each open reach immediately before a lake onto its authored
            # level. Never carry an inlet grade backward through a previous lake.
            for si in range(len(spans) - 1, -1, -1):
                a, _b, lid = spans[si]
                if a <= 0:
                    continue
                lake_level = _lake_level(wg, lid)
                if lake_level is None:
                    continue
                first = spans[si - 1][1] + 1 if si > 0 else 0
                depth = kvbathy.depth_for_order(
                    s.order, depth_by_order, depth_scale)
                reached_first = _grade_tail(
                    lm, ceiling, s.xy, first, a - 1, lake_level,
                    half_widths[sid] + plan.spacing_m, confluence_grade, depth,
                    backwater=True)
                if reached_first and first == 0:
                    backwater_at_start[sid] = lake_level

            starts_in_lake = next((sp for sp in spans if sp[0] == 0), None)
            main_level = (_lake_level(wg, starts_in_lake[2])
                          if starts_in_lake is not None else float(lm[0]))
            if main_level is None or not np.isfinite(main_level):
                continue
            if starts_in_lake is not None:
                backwater_at_start[sid] = float(main_level)
            lake_backwater = sid in backwater_at_start
            for uid in s.upstream:
                if uid not in out:
                    continue
                u = by_id[uid]
                ulm = out[uid][2]
                # A confluence acts on the final open reach, never through a lake.
                first = 0
                for _a, b, _lid in u.lake_spans:
                    if b < u.n - 1:
                        first = max(first, b + 1)
                touch = half_widths[sid] + half_widths[uid] + plan.spacing_m
                depth = kvbathy.depth_for_order(
                    u.order, depth_by_order, depth_scale)
                reached_first = _grade_tail(
                    ulm, ceilings[uid], u.xy, first, u.n - 1,
                    main_level, touch, confluence_grade, depth,
                    backwater=lake_backwater)
                if reached_first and first == 0 and lake_backwater:
                    backwater_at_start[uid] = float(main_level)

        # Tie-ins changed surfaces after the first bed calculation.
        for sid, values in out.items():
            depth = kvbathy.depth_for_order(by_id[sid].order, depth_by_order, depth_scale)
            out[sid] = (values[0], values[1], values[2], values[2] - depth)

    no_bank = sum(int((~np.isfinite(ceilings[s.seg_id])).sum()) for s in segs)
    drops = []
    for s in segs:
        if s.seg_id not in out:
            continue
        open_vertex = np.ones(s.n, dtype=bool)
        for a, b, _lid in s.lake_spans:
            open_vertex[a:b + 1] = False
        d = legacy[s.seg_id] - out[s.seg_id][2]
        take = open_vertex & np.isfinite(d)
        if take.any():
            drops.append(np.maximum(d[take], 0.0))
    drop = np.concatenate(drops) if drops else np.zeros(0, dtype=np.float64)
    return {
        "profiles": out,
        "ceilings": ceilings,
        "cyclic_segments": n_cyclic,
        "vertices_no_dry_bank_sample": int(no_bank),
        "profile_lowering_median_m": float(np.median(drop)) if drop.size else 0.0,
        "profile_lowering_p90_m": float(np.percentile(drop, 90)) if drop.size else 0.0,
        "profile_lowering_max_m": float(drop.max()) if drop.size else 0.0,
        "confluence_max_grade": float(confluence_grade),
    }


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
    return_profiles: bool = False,
) -> tuple:
    """
    Returns (lowered river level, bed target, report). The target is NaN except on
    river samples the downhill bed lowers.

    `ground` is the uncarved, void-repaired leaf; `river_level` is
    `watersurface.river_surface_moh` BEFORE the lake tie-in (the caller re-applies
    it). Pixels no downhill vertex reaches keep their level and get no target.
    """
    depth_by_order = depth_by_order or kvbathy.DEFAULT_RIVER_DEPTH_BY_ORDER
    prof = downhill_profiles(
        plan, traced, ground, wg=wg, depth_by_order=depth_by_order,
        depth_scale=depth_scale, width_by_order=width_by_order,
        width_scale=width_scale)
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
        "vertices_no_dry_bank_sample": prof["vertices_no_dry_bank_sample"],
        "profile_lowering_median_m": prof["profile_lowering_median_m"],
        "profile_lowering_p90_m": prof["profile_lowering_p90_m"],
        "profile_lowering_max_m": prof["profile_lowering_max_m"],
        "confluence_max_grade": prof["confluence_max_grade"],
    }
    for key, drops in (("level", level_drops), ("bed", bed_drops)):
        d = np.concatenate(drops) if drops else np.zeros(0)
        report[f"{key}_lowered_max_m"] = float(d.max()) if d.size else 0.0
        report[f"{key}_lowered_median_m"] = float(np.median(d)) if d.size else 0.0

    if not rr or not is_river.any():
        report["pixels_level_lowered"] = report["pixels_level_changed"] = 0
        report["pixels_deepened"] = 0
        result = (level_out, target, report)
        return result + (profiles,) if return_profiles else result

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
        report["pixels_level_lowered"] = report["pixels_level_changed"] = 0
        report["pixels_deepened"] = 0
        result = (level_out, target, report)
        return result + (profiles,) if return_profiles else result

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
    changed = have & (np.abs(spread_l - level_out) > 1e-6)
    lowered = have & (spread_l < level_out)
    # The profile is now the authority. Spreading it into the raster (rather than
    # reading the raster back into the vertices later) removes the old sub-pixel
    # disagreement that made 7.3% of exported vertex steps rise.
    level_out[have] = spread_l[have]
    with np.errstate(invalid="ignore"):
        want = np.asarray(river_level, dtype=np.float32) - depth_px - spread_b
    deepen = have & (want > 0)
    target[deepen] = spread_b[deepen]

    report["pixels_level_lowered"] = int(lowered.sum())
    report["pixels_level_changed"] = int(changed.sum())
    report["pixels_deepened"] = int(deepen.sum())
    report["bed_target_below_trench_max_m"] = (
        float(want[deepen].max()) if deepen.any() else 0.0)
    result = (level_out, target, report)
    return result + (profiles,) if return_profiles else result


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
