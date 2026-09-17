"""
kvterrain.bathymetry
====================

Synthetic lakebed depth for lakes with no surveyed bathymetry, plus repair of
lakes whose interior arrives as a VOID in the source DTM.

Kartverket's DTM is LiDAR-derived and water returns essentially no pulse, so a
lake polygon comes back one of two ways:

  (a) HYDRO-FLATTENED — a flat finite surface at the lake's water level. This is
      the case the downstream water sim was built for: rim - bed ~= 0 and the
      carved bowl supplies the depth.

  (b) VOID — no ground return. CRUCIALLY, with this project's `exportImage`
      request (`noData=""`), the ImageServer does NOT return NaN there: it returns
      a finite fill (0, or a small sentinel) and the TIFF carries no nodata tag,
      so `core.export_image_fetch` passes it through as a real height. Left alone
      it becomes the export's `height_min_m`, every lake shares that same floor,
      and downstream `depth = rim - height_min` is a 100 m+ column in a vertical-
      walled pit -> the shallow-water solver spikes. (The Norli-north failure.)

Because (b) is FINITE, an `isfinite` test cannot find it. `fill_lake_surface`
therefore detects voids PHYSICALLY: a lake sample is a void if it is non-finite
OR sits more than `void_below_shore_m` below its own shoreline. It then fills
each lake's voids with that lake's (flat) shoreline elevation.

RIVERS ARE CARVED HERE TOO (`carve_river_beds`), for a different reason. A river's
water surface is the DTM's own terrain height beneath it — see
`watersurface.river_surface_moh` — so without a trench there is no water column at
all. The trench is what makes a river visible, and cutting DOWN for it is what
keeps the water surface welded to the ground it runs over instead of floating
above it.

CALL ORDER (this is the contract; the carves do NOT do it for you):

    leaf = fill_lake_surface(leaf, wg.type)          # repair voids
    uncarved = leaf.copy()                           # river SURFACE reads this
    leaf = carve_river_beds(leaf, wg.type, wg.weight, spacing, ...)
    leaf = carve_lake_beds(leaf, wg.type, spacing,   # then carve the bowl
                           surface_moh=lake_surf, ...)

The snapshot is load-bearing: `watersurface.river_surface_moh` must read the bed
as it was BEFORE the river carve, or the surface follows the trench down and the
water column closes to zero.

Earlier revisions of this docstring claimed `carve_lake_beds` invoked
`fill_lake_surface` automatically via `fill_voids=True`. IT NEVER DID — the
`fill_voids` kwarg is a legacy alias for `estimate_missing`, which gates
`estimate_lake_surface` (a different function that only picks a water LEVEL).
The carve then happened to hide interior voids as a side effect, because it
overwrites every lake pixel with `surface - carve_depth` regardless of what the
DTM returned there.

That side effect is not a substitute for the repair, for two reasons:

  * it misses lakes that have neither an NVE `hoyde` nor a usable shoreline, and
  * it cannot touch the thin ring of void land just OUTSIDE the polygon, where
    the source raster and the NVE geometry fail to align to the pixel. Those
    shore-ring voids stay in `bed` as a fake pit right at the waterline, which
    the runtime clip pass (`depth = max(0, L - bed)`) renders as a spike and the
    solver reads as a spurious basin.

`core.run_export` therefore calls `fill_lake_surface` explicitly, before the
carve. Any other caller must do the same.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (distance_transform_edt, label, binary_dilation,
                           maximum as _labeled_maximum)

TYPE_LAKE_VALUE = 2
TYPE_RIVER_VALUE = 1
DEFAULT_RAMP_RADIUS_M = 4.0   # legacy (pre-known-surface) ramp radius
DEFAULT_MAX_DEPTH_M = 4.0     # legacy (pre-known-surface) max depth
# Carve the bed toward a maximum REAL depth below the (known or estimated) water
# surface, over a long shore ramp. carve_depth is in real metres; the goal is
# opacity, so it must exceed the renderer's opaque threshold AFTER any vertical scale
# the consumer applies (e.g. a 1:5 height compression turns 20 m into 4 units).
DEFAULT_CARVE_DEPTH_M = 20.0
# The bed descends LINEARLY from the waterline at this slope, in metres of depth per
# metre of shore, until it reaches `carve_depth_m`. At the default 1.0 the ramp is
# exactly as long as the carve is deep: a 20 m lake bed reaches 20 m of depth 20 m
# from shore — four texels at 5 m spacing — and is flat from there inward.
#
# Linear, not smoothstepped, and short, not long. Both of those are corrections.
# A long smoothstep (this was 150 m of ramp) spends its first texels almost flat, so
# the near-shore depth contour is a smooth distance-from-shore curve rather than the
# shoreline: lakes rendered as rounded blobs sitting inside their own outlines, and a
# measured 55% of Kroktjønna's polygon fell below a 0.5 m visibility threshold. A
# straight ramp has a constant slope everywhere, so depth is proportional to how far
# inside the lake you are and nothing near the shore is ambiguous.
#
# (The stopgap for the smoothstep version was a second, shorter ramp underneath it —
# `edge_depth_m` / `edge_ramp_m`. It is DELETED: with a straight 1:1 ramp the first
# texel inside the waterline is already 5 m down at 5 m spacing, so there is nothing
# left for it to fix.)
DEFAULT_SHORE_SLOPE = 1.0
# ...with one floor: every connected water body reaches at least this depth at its
# deepest sample, by scaling that body's whole profile up (the shoreline stays at
# zero, so the shape is preserved). Without it a pond two texels across would carve
# essentially nothing and render as invisible water on flat ground — the failure the
# old per-body bevel shrink existed to prevent, kept, but at a couple of metres
# instead of the full 20.
DEFAULT_MIN_DEPTH_M = 2.0
DEFAULT_BEVEL_PX = 2          # legacy (pre-ramp_m) bevel width, in texels
DEFAULT_SHORE_BAND_PX = 3
# A flat lake surface is never this far below its own shore. Anything deeper is a
# void fill (0 / sentinel / NaN), not real bathymetry. Generous enough to preserve
# a few metres of genuine bank, small enough to catch a 0-fill in high terrain.
DEFAULT_VOID_BELOW_SHORE_M = 12.0

# ── River channel carve ──────────────────────────────────────────────────────
# Maximum depth of the trench cut UNDER a river channel, in real metres, by
# Strahler order. The river's water surface is the DTM's own terrain height, so
# this table is the only thing that gives a river visible depth, exactly as
# `carve_depth_m` is for lakes.
#
# This table used to live in `watersurface` as DEFAULT_DEPTH_BY_ORDER, where it
# RAISED the water surface above the terrain instead. That model is what made
# rivers read as "water hoses" — a solid tube of water standing 1-11 m proud of
# the ground it crossed, most visible exactly where the channel was shallowest and
# the raise had nothing to sink into. The numbers are unchanged; only their sign
# and their target changed. `--river-depth-scale` still scales the whole table.
DEFAULT_RIVER_DEPTH_BY_ORDER = {
    1: 1.5, 2: 2.25, 3: 3.0, 4: 4.0, 5: 5.5, 6: 7.0, 7: 9.0, 8: 11.0,
}
# How far above its channel's water level a sample may still be treated as part of
# the channel, in metres. It bounds the artefact from both sides:
#
#   * the water level is capped at this much above any sample's own ground, so
#     water never stands higher than this over the terrain anywhere, and
#   * the trench is cut in full up to this height above the level and tapers to
#     nothing by twice it, so a sample standing further above the water than this
#     keeps its terrain untouched and renders dry.
#
# So water lives in a band of a few metres around the channel level, whatever the
# river's modelled depth. Deliberately NOT scaled by stream order: an order-8
# trunk is carved 11 m deep, and letting its water climb 11 m up the walls of the
# gorge it runs through is exactly the artefact this bounds. 2 m is about the
# largest step a real bank makes within one 5 m texel of a channel.
DEFAULT_BANK_TOLERANCE_M = 2.0


def depth_for_order(order: int, table: dict, scale: float = 1.0) -> float:
    """Carve depth for a stream order, scaled; clamps to the table ends."""
    if not table:
        return 0.5 * scale
    keys = sorted(table)
    o = min(max(int(order), keys[0]), keys[-1])
    return float(table[o]) * float(scale)


def channel_half_width_m(order: np.ndarray, mask: np.ndarray, spacing_m: float,
                         width_by_order: dict | None = None,
                         width_scale: float = 1.0) -> np.ndarray:
    """
    Per-pixel modelled channel HALF-width in metres, floored at half a texel.

    Shared by the carve (which uses it to shape the cross-section) and by
    `watersurface.river_surface_moh` (which uses it to decide how far across the
    channel a water level is levelled out). They must agree: one is the trench,
    the other is what fills it.
    """
    from . import water as kvwater

    width_by_order = width_by_order or kvwater.DEFAULT_WIDTH_BY_ORDER
    orders = np.asarray(order, dtype=np.int32)
    half = np.full(mask.shape, 0.5 * spacing_m, dtype=np.float32)
    for o in np.unique(orders[mask]):
        half[mask & (orders == o)] = max(
            0.5 * kvwater.order_to_width_m(int(o), width_by_order, width_scale),
            0.5 * spacing_m)
    return half


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance_to_shore_m(water_type: np.ndarray, spacing_m: float,
                        *, island: np.ndarray | None = None,
                        edge_is_shore: bool = False) -> np.ndarray:
    """
    Distance from each lake sample to the nearest NON-lake sample, in metres.

    Note what this means at the edge: the outermost wet sample is one full cell
    from dry land, so its distance is `spacing_m` — NOT zero. `carve_lake_beds`
    subtracts that offset before ramping; see SHORE_ANCHOR_M.

    `island` (lake pixels that are really a filled polygon hole — see
    `water.rasterize_water`) counts as shore, not as water. The mask says "lake"
    there so the water SURFACE runs through the island, but the island is dry land
    that must be ramped away from like any other bank; without this a lake would
    reach full depth immediately against its own islands.

    `edge_is_shore` treats the ground beyond the MAP EDGE as shore. The EDT only
    measures to non-lake samples inside the array, so without it a lake the export
    boundary cuts through is at full depth right at the edge (Lierne: 26 lakes,
    3 796 edge samples, median 20.0 m deep). The depression hierarchy makes every
    edge sample an outlet, and such a lake would drain off the map through its own
    floor. Padding with shore instead ramps the bed back up to the waterline at the
    edge, so the cropped remainder is a bowl that holds its level.
    """
    is_lake = water_type == TYPE_LAKE_VALUE
    if island is not None:
        is_lake = is_lake & ~np.asarray(island, dtype=bool)
    if not is_lake.any():
        return np.zeros(water_type.shape, dtype=np.float32)
    if edge_is_shore:
        padded = np.pad(is_lake, 1, mode="constant", constant_values=False)
        dist = distance_transform_edt(padded, sampling=(spacing_m, spacing_m))
        return dist[1:-1, 1:-1].astype(np.float32)
    dist = distance_transform_edt(is_lake, sampling=(spacing_m, spacing_m))
    return dist.astype(np.float32)


# Where the bevel's zero point sits, in units of `spacing_m`, measured from the
# outermost wet sample outward. 1.0 puts it on the first DRY sample, so the last
# WET sample lands at exactly zero carve — the water's edge sits on the terrain.
#
# The true waterline is really about half a cell out (rasterisation is
# centre-in-polygon), which would leave the last wet sample ~3 m deep at the
# default bevel. 1.0 is the deliberate choice over 0.5: a mesh renderer that
# feathers between a wet vertex and a dry one wants that wet vertex ON the
# terrain, and zero is the only value that guarantees no step. Drop this to 0.5
# if a consumer culls samples at depth <= 0 and the outermost ring disappears.
#
# It was effectively 0.0 before, and that was the bug behind the "trench around
# every lake": distance_to_shore_m hands the outermost wet sample a distance of
# one whole cell, so with the default 2-texel bevel the ramp opened at
# smoothstep(5/10) = 0.5 and that sample was carved to HALF the full depth
# immediately. Measured over a 41x41 km export: 219744 shoreline samples, median
# depth at the outermost ring 10.01 m, and not one lake sample anywhere in the
# export shallower than 9 m. Every lake was a flat-bottomed pit with a 10 m
# vertical rim, and the water plane consequently sat metres below the ground it
# should have met. There was no shallow margin to render a shoreline with.
SHORE_ANCHOR_M = 1.0


def _carve_depth_m(is_lake: np.ndarray, dist: np.ndarray, spacing_m: float,
                   carve_depth_m: float, ramp_m: float,
                   min_depth_m: float, *, bodies_8_connected: bool = False) -> np.ndarray:
    """
    The carve profile in METRES: a straight ramp from 0 at the waterline to
    `carve_depth_m` at `ramp_m` inward, flat from there.

        depth(d) = carve_depth_m * clip((d - waterline) / ramp_m, 0, 1)

    Two things happen beyond that line.

    1. The ramp is anchored on the WATERLINE (`SHORE_ANCHOR_M`), because `dist`
       bottoms out at one whole cell inside a lake rather than at zero.

    2. Each connected body is scaled by a GAIN so its deepest sample reaches at
       least `min_depth_m`. Since the gain multiplies a profile that is zero at the
       shore, the shoreline stays exactly on the waterline and only the interior
       lifts; and because the gain is never applied above 1, a body big enough to
       reach `carve_depth_m` on its own is left untouched. A body with literally no
       room to ramp (one sample across, every sample on the waterline) takes
       `min_depth_m` as a step, since there is nowhere to ramp at all.

    This REPLACES the old per-body bevel SHRINK, which pulled the ramp width in
    until every body bottomed out at the full carve depth. That guaranteed depth at
    the cost of the thing depth is supposed to encode: a 40 m pond and a 4 km lake
    both hit 20 m, the pond doing it over two texels. Scaling a fixed-slope profile
    up to a floor keeps small lakes shallow, which is the whole point, while still
    refusing to render a pond as a dry patch.
    """
    anchor = SHORE_ANCHOR_M * spacing_m
    ramp = max(float(ramp_m), 1e-6)
    if not is_lake.any():
        return np.zeros(is_lake.shape, dtype=np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        depth = (float(carve_depth_m)
                 * np.clip((dist - anchor) / ramp, 0.0, 1.0)).astype(np.float32)
    depth[~is_lake] = 0.0

    # Bodies for the min-depth floor. 4-connected by default (the original rule); a
    # sliver touching its lake only diagonally is then a body of its own, too thin
    # to ramp, and is STEPPED to min_depth_m right at the shoreline — a notch in the
    # rim of an 8-connected depression hierarchy. See carve_lake_beds.
    lbl, n = label(is_lake, structure=np.ones((3, 3), dtype=bool)
                   if bodies_8_connected else None)
    if n == 0:
        return depth

    peak = np.atleast_1d(
        np.asarray(_labeled_maximum(depth, lbl, np.arange(1, n + 1)),
                   dtype=np.float32))
    floor = min(float(min_depth_m), float(carve_depth_m))

    gain = np.ones(n + 1, dtype=np.float32)
    deep_enough = peak > 1e-6
    gain[1:] = np.where(deep_enough,
                        np.maximum(1.0, floor / np.maximum(peak, 1e-6)), 1.0)
    # Bodies with no room to ramp at all: a step to the floor depth, flagged
    # explicitly rather than left to 0 * inf.
    stepped = np.zeros(n + 1, dtype=bool)
    stepped[1:] = ~deep_enough

    out = depth * gain[lbl]
    step_px = stepped[lbl] & is_lake
    out[step_px] = np.float32(floor)
    return out.astype(np.float32)


def _shore_surface(shore_vals: np.ndarray, shore_percentile: float,
                   void_below_shore_m: float) -> float:
    """
    Estimate a lake's flat water level from a band of shoreline heights, robust to
    the band itself containing some void fill. Anchor on the ring's UPPER cluster (a
    high percentile): real land sits high and any void sentinels sit low, so this
    lands on real land even when much of the ring is void. Drop anything a full
    void-depth below that anchor (the sentinels), then take a LOW percentile of what
    remains — low, because the sim sets the lake level to its LOWEST shoreline (the
    spill point), so we want the bed to end up below that, not above it.
    """
    land_ref = float(np.percentile(shore_vals, 75.0))
    clean = shore_vals[shore_vals > land_ref - void_below_shore_m]
    base = clean if clean.size else shore_vals
    return float(np.percentile(base, shore_percentile))


def estimate_lake_surface(
    height_m: np.ndarray,
    water_type: np.ndarray,
    *,
    shore_band_px: int = DEFAULT_SHORE_BAND_PX,
    shore_percentile: float = 20.0,
    void_below_shore_m: float = DEFAULT_VOID_BELOW_SHORE_M,
) -> np.ndarray:
    """
    Per-pixel estimate of each lake's flat water-surface elevation from the DTM,
    for lakes that have no authoritative `hoyde` from NVE. Returns float32, NaN
    outside lakes (and for any lake with no usable shoreline). This is the FALLBACK
    surface source; when NVE `hoyde` is available the carve uses that instead.

    All morphology (label + one EDT) runs ONCE over the whole grid; the only
    per-lake work is a percentile over that lake's shoreline band. Robust to void
    fill in the shoreline via `_shore_surface`'s upper-cluster anchor.
    """
    out = height_m
    is_lake = water_type == TYPE_LAKE_VALUE
    surface_px = np.full(out.shape, np.nan, dtype=np.float32)
    if not is_lake.any():
        return surface_px

    finite = np.isfinite(out)
    lbl, n = label(is_lake)
    if n == 0:
        return surface_px

    dist_lake, inds = distance_transform_edt(
        ~is_lake, return_distances=True, return_indices=True)
    nearest_lake_lbl = lbl[inds[0], inds[1]]

    shore = (~is_lake) & finite & (dist_lake <= float(shore_band_px))
    shore_lbl = nearest_lake_lbl[shore].astype(np.intp)
    shore_h = out[shore].astype(np.float64)

    surface_by_label = np.full(n + 1, np.nan, dtype=np.float64)
    if shore_h.size:
        order = np.argsort(shore_lbl, kind="stable")
        sl = shore_lbl[order]
        sh = shore_h[order]
        starts = np.searchsorted(sl, np.arange(1, n + 1), side="left")
        ends = np.searchsorted(sl, np.arange(1, n + 1), side="right")
        for k in range(n):
            a, b = starts[k], ends[k]
            if b > a:
                surface_by_label[k + 1] = _shore_surface(
                    sh[a:b], shore_percentile, void_below_shore_m)

    surf = np.where(is_lake, surface_by_label[lbl], np.nan)
    # Lakes whose shoreline was entirely void/absent: nearest real land height.
    if np.isnan(surface_by_label[1:]).any():
        real = finite & (~is_lake)
        if real.any():
            _, rinds = distance_transform_edt(
                ~real, return_distances=True, return_indices=True)
            nearest_real_h = out[rinds[0], rinds[1]]
            need = is_lake & np.isnan(surf)
            surf = np.where(need, nearest_real_h, surf)

    return surf.astype(np.float32)


# Estimated levels are pulled DOWN by this much before use. The estimator already
# reads the low band of the interior (see DEFAULT_LEVEL_BAND_PCT), so the one-sided
# hedge this used to provide is built into the statistic; the default is 0 and the
# knob is kept only for a caller that wants to sink every lake a little further.
#
# Note this NEVER applies to a lake that has an NVE `hoyde`. That level is used
# exactly as published — see `water.apply_estimated_levels`.
DEFAULT_LEVEL_MARGIN_M = 0.0
# The level is the MEAN OF THE LOW BAND of the interior: every sample between the
# 1st and 25th percentile, averaged. A weighted average toward the bottom, with
# both tails cut, and each cut is there for a specific failure:
#
#   * the 25% ceiling discards LAND inside the polygon — an NVE outline drawn wider
#     than the real water body, an unmapped shoal, a shoreline that has moved. Land
#     is higher than water, so a plain median walks up with it and the lake ends up
#     standing above the ground around it.
#   * the 1% floor discards a stray LOW cell — a void the repair pass did not
#     catch, an outlet channel clipped by the polygon, a DTM artefact. A minimum,
#     or a percentile close to it, follows those all the way down, and this
#     function has no way to tell one from a real water surface.
#
# Averaging the band rather than taking a single percentile of it just makes the
# result stop moving around: on a flat lake interior every choice between the min
# and the median lands within ~0.1 m of the others (measured over 56 Lierne lakes
# with a known hoyde: bias +0.06 to +0.11 m, |err| p50 0.35-0.37 m whichever is
# used), so the statistic is chosen for what it does on the PATHOLOGICAL lake, not
# for accuracy on the ordinary one.
DEFAULT_LEVEL_TRIM_PCT = 1.0
DEFAULT_LEVEL_BAND_PCT = 25.0
# Fewer interior samples than this and the statistic is not worth trusting.
MIN_INTERIOR_SAMPLES = 5
# The perimeter cap (see `perimeter_levels`) reads this percentile of the ring of
# land just outside a lake. Not the minimum: a lake's outlet is genuinely below its
# surface, and so is any channel the ring clips, so the low end of a perimeter is
# not evidence that the lake is too high. A quarter of the perimeter has to sit
# below the estimate before this binds.
DEFAULT_PERIMETER_PERCENTILE = 25.0


def perimeter_levels(
    height_m: np.ndarray,
    lake_id: np.ndarray,
    water_type: np.ndarray,
    *,
    band_px: int = DEFAULT_SHORE_BAND_PX,
    percentile: float = DEFAULT_PERIMETER_PERCENTILE,
    min_samples: int = MIN_INTERIOR_SAMPLES,
) -> dict:
    """
    Per-lake height of the LAND RING just outside the polygon — the perimeter scan.

    {local lake id -> a low percentile of the finite non-lake samples within
    `band_px` of that lake}. Used as a CAP on an estimated level: whatever the
    interior says, a lake should not be given a surface standing above the ground
    that encircles it, because that is the artefact you see through at the
    shoreline.

    Grouping is by nearest lake SAMPLE (so by lake id, not by connected body),
    which matters where two lakes at different levels sit within a band of each
    other. One EDT for the whole grid, then one percentile per lake.

    Only ever consulted for lakes with no NVE `hoyde`. A lake that has one is not
    second-guessed — see `water.apply_estimated_levels`.
    """
    is_lake = water_type == TYPE_LAKE_VALUE
    if not is_lake.any():
        return {}

    dist, ind = distance_transform_edt(
        ~is_lake, return_distances=True, return_indices=True)
    near_id = np.asarray(lake_id)[ind[0], ind[1]]
    ring = ((~is_lake) & np.isfinite(height_m)
            & (dist <= float(band_px)) & (near_id > 0))
    if not ring.any():
        return {}

    rid = near_id[ring]
    rh = np.asarray(height_m)[ring].astype(np.float64)
    order = np.argsort(rid, kind="stable")
    rid, rh = rid[order], rh[order]
    uniq, starts = np.unique(rid, return_index=True)
    ends = np.append(starts[1:], rid.size)

    p = float(np.clip(percentile, 0.0, 100.0))
    return {int(u): float(np.percentile(rh[a:b], p))
            for u, a, b in zip(uniq, starts, ends) if b - a >= min_samples}


def estimate_levels_from_interior(
    height_m: np.ndarray,
    lake_id: np.ndarray,
    *,
    margin_m: float = DEFAULT_LEVEL_MARGIN_M,
    trim_pct: float = DEFAULT_LEVEL_TRIM_PCT,
    band_pct: float = DEFAULT_LEVEL_BAND_PCT,
    min_samples: int = MIN_INTERIOR_SAMPLES,
    exclude: np.ndarray | None = None,
) -> dict:
    """
    Per-lake water level read straight off the DTM INSIDE each lake polygon.

    ONLY FOR LAKES NVE PUBLISHES NO `hoyde` FOR. A lake that has one uses it as
    published; this function is not consulted, and its answer is not allowed to
    second-guess the published one. See `water.apply_estimated_levels`.

    LiDAR does get a return from water — it reports the water SURFACE — so the
    samples inside a lake polygon are a direct measurement of the thing we want,
    not an inference from the surrounding bank. That makes this strictly better
    than `estimate_lake_surface`, which reads a percentile of the shoreline band
    outside the polygon. Measured against the 89 lakes in a Lierne block that do
    carry an NVE `hoyde`:

        interior median   bias +0.00 m, |err| p50 0.63 m, 28% high by >0.5 m
        shore band (old)  bias +0.35 m, |err| p50 0.75 m, 39% high by >0.5 m

    and the truth itself is quantised to 1 m, so 0.63 m is at the noise floor.

    The level is the MEAN OF THE LOW BAND (`trim_pct`..`band_pct`) of the interior,
    not the median and not the minimum — see DEFAULT_LEVEL_BAND_PCT for what each
    end of that band is protecting against.

    `exclude` marks lake pixels that are not water (islands from filled polygon
    holes — see `water.rasterize_water`). Their heights are real terrain and must
    not enter the statistic; reading the low band would rarely let them change the
    answer, but "the estimator reads island tops" is not a property worth relying
    on being harmless.

    `height_m` must be the VOID-REPAIRED, UNCARVED bed (i.e. run this after
    `fill_lake_surface` and before `carve_lake_beds`); sampling a carved array
    would just return `surface - carve_depth`.

    Returns {local lake id -> level in metres}, omitting lakes with too few
    usable samples.
    """
    ids = np.asarray(lake_id).ravel()
    h = np.asarray(height_m).ravel()
    sel = (ids > 0) & np.isfinite(h)
    if exclude is not None:
        sel &= ~np.asarray(exclude, dtype=bool).ravel()
    if not sel.any():
        return {}

    gid = ids[sel]
    gh = h[sel].astype(np.float64)
    order = np.argsort(gid, kind="stable")
    gid = gid[order]
    gh = gh[order]

    uniq, starts = np.unique(gid, return_index=True)
    ends = np.append(starts[1:], gid.size)

    lo = float(np.clip(trim_pct, 0.0, 100.0))
    hi = float(np.clip(band_pct, lo, 100.0))
    out: dict = {}
    for u, a, b in zip(uniq, starts, ends):
        if b - a < min_samples:
            continue
        v = gh[a:b]
        floor, ceil = np.percentile(v, lo), np.percentile(v, hi)
        band = v[(v >= floor) & (v <= ceil)]
        # A lake flat enough that the whole band collapses to one value still has
        # an answer: that value.
        level = float(band.mean()) if band.size else float(ceil)
        out[int(u)] = level - float(margin_m)
    return out


def fill_lake_surface(
    height_m: np.ndarray,
    water_type: np.ndarray,
    *,
    shore_band_px: int = DEFAULT_SHORE_BAND_PX,
    shore_percentile: float = 20.0,
    dilate_px: int = 1,
    void_below_shore_m: float = DEFAULT_VOID_BELOW_SHORE_M,
) -> np.ndarray:
    """
    Replace VOID samples inside (and just outside) lake polygons with the lake's
    flat shoreline elevation, so a void lake behaves like a hydro-flattened one.

    A lake sample is a VOID if it is non-finite OR more than `void_below_shore_m`
    below its lake's shoreline estimate. This is the key change over an isfinite-
    only test: this project's fetch returns voids as a finite 0/sentinel, which an
    isfinite test silently misses (and which then becomes the export's height_min).

    Per connected lake body: `surface` is a low percentile of the finite shoreline
    band, with sentinels scrubbed out (see `_shore_surface`). All voids in the body
    -- and, when `dilate_px > 0`, voids in a thin ring just outside it -- are set to
    `surface`. The outside ring matters because the void region and the NVE polygon
    come from different sources and rarely align to the pixel; a void land cell at
    the shore would otherwise drag the sim's rim scan (an InterlockedMin over
    shoreline land) down to the floor and seed the lake dry.

    Returns a copy; input is not modified. Non-lake samples are never written.

    PERFORMANCE: all morphology (label, one EDT, one dilation) runs ONCE over the
    whole grid, independent of lake count. The only per-lake work is a percentile
    over that lake's shoreline samples -- O(total shoreline pixels), not
    O(lakes * grid). Earlier revisions dilated the full array once per lake, which
    hung on lake-dense regions.
    """
    if height_m.shape != water_type.shape:
        raise ValueError(
            f"height_m {height_m.shape} and water_type {water_type.shape} must match"
        )

    out = height_m.copy()
    is_lake = water_type == TYPE_LAKE_VALUE
    if not is_lake.any():
        return out

    finite = np.isfinite(out)
    lbl, n = label(is_lake)
    if n == 0:
        return out

    # ONE global EDT: for every pixel, the distance to and label of its nearest
    # lake. `dist_lake` bounds the shoreline band; `nearest_lake_lbl` groups shore
    # pixels (and outside-ring pixels) by which lake they belong to.
    dist_lake, inds = distance_transform_edt(
        ~is_lake, return_distances=True, return_indices=True)
    nearest_lake_lbl = lbl[inds[0], inds[1]]

    # Shoreline = finite non-lake land within shore_band_px of its nearest lake.
    shore = (~is_lake) & finite & (dist_lake <= float(shore_band_px))
    shore_lbl = nearest_lake_lbl[shore].astype(np.intp)
    shore_h = out[shore].astype(np.float64)

    # Per-lake flat surface, computed by grouping shore samples by label via a
    # single sort (no full-array ops). Labels with no shoreline stay NaN.
    surface_by_label = np.full(n + 1, np.nan, dtype=np.float64)
    if shore_h.size:
        order = np.argsort(shore_lbl, kind="stable")
        sl = shore_lbl[order]
        sh = shore_h[order]
        starts = np.searchsorted(sl, np.arange(1, n + 1), side="left")
        ends = np.searchsorted(sl, np.arange(1, n + 1), side="right")
        for k in range(n):
            a, b = starts[k], ends[k]
            if b > a:
                surface_by_label[k + 1] = _shore_surface(
                    sh[a:b], shore_percentile, void_below_shore_m)

    # Fallback for any lake whose shoreline was entirely void/absent: nearest real
    # (finite, non-lake) land height. Only pays for a second EDT if actually needed.
    surface_px = np.where(is_lake, surface_by_label[lbl], np.nan)
    if np.isnan(surface_by_label[1:]).any():
        real = finite & (~is_lake)
        if real.any():
            _, rinds = distance_transform_edt(
                ~real, return_distances=True, return_indices=True)
            nearest_real_h = out[rinds[0], rinds[1]]
            need = is_lake & np.isnan(surface_px)
            surface_px = np.where(need, nearest_real_h, surface_px)

    with np.errstate(invalid="ignore"):  # NaN comparisons are intentional
        # Interior voids: non-finite OR implausibly far below the lake surface.
        void = is_lake & np.isfinite(surface_px) & (
            (~finite) | (out < surface_px - void_below_shore_m))
        out[void] = surface_px[void]

        # One dilation for the whole mask: repair void land in a thin ring just
        # outside the polygons (source-vs-polygon misalignment), using each ring
        # pixel's nearest-lake surface.
        if dilate_px > 0:
            near = binary_dilation(is_lake, iterations=dilate_px) & (~is_lake)
            ring_surface = np.where(near, surface_by_label[nearest_lake_lbl], np.nan)
            ring_void = near & np.isfinite(ring_surface) & (
                (~finite) | (out < ring_surface - void_below_shore_m))
            out[ring_void] = ring_surface[ring_void]

    return out


def carve_lake_beds(
    height_m: np.ndarray,
    water_type: np.ndarray,
    spacing_m: float,
    *,
    surface_moh: np.ndarray | None = None,
    carve_depth_m: float = DEFAULT_CARVE_DEPTH_M,
    shore_slope: float = DEFAULT_SHORE_SLOPE,
    ramp_m: float | None = None,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    island: np.ndarray | None = None,
    estimate_missing: bool = True,
    edge_is_shore: bool = False,
    shore_8_connected: bool = False,
    shore_band_px: int = DEFAULT_SHORE_BAND_PX,
    void_below_shore_m: float = DEFAULT_VOID_BELOW_SHORE_M,
    # ── legacy aliases (pre-known-surface / pre-ramp_m); mapped if provided ──
    bevel_px: int | None = None,
    ramp_radius_m: float | None = None,
    max_depth_m: float | None = None,
    fill_voids: bool | None = None,
) -> np.ndarray:
    """
    Carve each lake's bed below its water SURFACE on a STRAIGHT ramp: down at
    `shore_slope` metres of depth per metre of shore until it reaches
    `carve_depth_m`, flat from there inward. At the default slope of 1.0 the ramp is
    as long as the carve is deep — a 20 m bed is 20 m of depth over 20 m of shore,
    four texels at 5 m spacing. Pass `ramp_m` to set that distance directly instead.

    The bed value does NOT depend on what the DTM returned inside the lake (flat
    surface, void 0, sentinel) — it is `surface - depth(distance from shore)` — so
    the LiDAR water-flattening and the whole void saga are irrelevant here.

    The profile starts at ZERO, and that is the point: the outermost wet sample
    sits exactly at the water level, so the water has somewhere to be shallow and a
    mesh renderer has a vertex to feather against. See SHORE_ANCHOR_M for what this
    looked like when it was wrong, and `_carve_depth_m` for the ramp and the floor
    under small bodies.

    A lake wider than twice the ramp reaches `carve_depth_m` in its middle, which at
    the default slope is most lakes. That is a deliberate reversal: an earlier
    revision stretched the ramp to 150 m so that lake SIZE decided depth and small
    lakes stayed shallow, and the cost was that no lake's near-shore water was deep
    enough to render its own outline. Depth now follows distance from the shore, and
    nothing else. Lower `carve_depth_m` if lakes should be shallower; lower
    `shore_slope` if their sides should be gentler.

    `shore_8_connected` makes the shoreline watertight for the 8-connected depression
    hierarchy. The EDT puts the zero-depth waterline on samples with a non-lake
    4-neighbour; a sample whose only dry neighbour is DIAGONAL is sqrt(2) cells from
    shore and carved 2.07 m deep at the default slope and 5 m spacing. Wherever that
    dry diagonal ground is below the level, the hierarchy's lake drains through the
    corner: on Lierne about 390 lakes "spilled over land" up to exactly 2.08 m below
    their authored level that way. With it on, every lake sample with a non-lake
    8-neighbour sits on the waterline, and bodies for the min-depth floor are
    8-connected too, so a diagonal sliver is not stepped 2 m deep at the shore.

    `island` marks lake pixels that are really filled polygon holes. They keep their
    DTM height (they are land) and count as shore for the ramp, so a lake shelves
    away from its islands the same way it shelves away from its outer bank.

    Surface source, per lake pixel:
      1. `surface_moh` (authoritative NVE `hoyde`) wherever finite, else
      2. a DTM shore estimate (`estimate_lake_surface`) when `estimate_missing`.
    Pixels with no surface from either source are left untouched.

    The only purpose of the carve is to sink the bed far enough below the water
    surface that the consumer renders opaque water; it is not real bathymetry.

    WHY THIS IS SAFE FOR THE SOLVER (it looks like it should not be).
    The carve deliberately writes a depression into the visual terrain, which
    reads as a violation of "do not change the visual terrain". It is not, and
    the reason is worth recording because it is easy to talk yourself out of:

      * The source leaves us no choice. LiDAR gets no return from water, so the
        DTM carries a lake's SURFACE as its terrain height. Without a carve the
        water plane and the terrain are coincident and the surfaces z-fight.
        There is no true bed in the input to preserve.
      * It is the depression-hierarchy's Case 1: a hole beneath an existing lake
        that does not break the rim. The bevel guarantees the bed descends
        monotonically inward from the shoreline, so no rim cell is ever lowered
        and the drainage topology is untouched.
      * The basin it creates is never discovered as a free basin, because the
        lake carries an authored level (NVE `hoyde`) and is pinned
        `AuthoredWins`. The solver reads the pinned level; the carve only makes
        `max(0, L - bed)` deeper, which is the intended visual result.

    The pin is load-bearing. If a lake ever loses its authored level and falls
    through to the computed path, the solver will find this fabricated bowl and
    fill it to a spill elevation that means nothing. Keep lake records emitted
    and keep the pin flag wired.
    """
    if height_m.shape != water_type.shape:
        raise ValueError(
            f"height_m {height_m.shape} and water_type {water_type.shape} must match"
        )
    # Legacy mapping so existing callers keep working.
    if max_depth_m is not None:
        carve_depth_m = max_depth_m
    if ramp_radius_m is not None:
        ramp_m = float(ramp_radius_m)
    if bevel_px is not None:
        ramp_m = max(1.0, float(bevel_px)) * max(spacing_m, 1e-6)
    # No explicit ramp distance: derive it from the slope, so "20 m deep" means
    # "20 m of shore" at slope 1.0.
    if ramp_m is None or float(ramp_m) <= 0.0:
        ramp_m = float(carve_depth_m) / max(float(shore_slope), 1e-6)
    if fill_voids is not None:
        estimate_missing = fill_voids

    out = height_m.copy()
    is_lake = water_type == TYPE_LAKE_VALUE
    if island is not None:
        island = np.asarray(island, dtype=bool) & is_lake
        is_lake = is_lake & ~island
    if not is_lake.any():
        return out

    # 1. Known surface from NVE hoyde.
    surf = np.full(out.shape, np.nan, dtype=np.float32)
    if surface_moh is not None:
        known = is_lake & np.isfinite(surface_moh)
        surf[known] = np.asarray(surface_moh, dtype=np.float32)[known]

    # 2. Estimate surface only for lake pixels still missing one.
    if estimate_missing:
        need = is_lake & ~np.isfinite(surf)
        if need.any():
            est = estimate_lake_surface(
                height_m, water_type,
                shore_band_px=shore_band_px, void_below_shore_m=void_below_shore_m)
            take = need & np.isfinite(est)
            surf[take] = est[take]

    have = is_lake & np.isfinite(surf)
    if not have.any():
        return out

    # Bed at surface - depth(distance from shore), a straight ramp over `ramp_m`.
    #
    # The ramp is anchored on the WATERLINE, not on the outermost wet sample:
    # `dist` never reads below one cell inside a lake, so ramping straight off it
    # opens the ramp already part-open and leaves a step at the shore. See
    # SHORE_ANCHOR_M for the measurement that motivated this. The subtraction can
    # only make the carve SHALLOWER, so it cannot deepen a bowl, widen one, or
    # lower a rim sample that was previously left alone.
    dist = distance_to_shore_m(water_type, spacing_m, island=island,
                               edge_is_shore=edge_is_shore)
    depth = _carve_depth_m(is_lake, dist, spacing_m, carve_depth_m, ramp_m,
                           min_depth_m, bodies_8_connected=shore_8_connected)
    if shore_8_connected:
        # Outside the array counts as lake here (border_value 0), so the map edge
        # is left to edge_is_shore.
        ring = is_lake & binary_dilation(~is_lake, structure=np.ones((3, 3), bool))
        depth[ring] = 0.0
    if edge_is_shore:
        # The ramp already puts the edge ring on the waterline, except for a body
        # too thin to ramp at all, which `_carve_depth_m` steps to min depth. On the
        # map edge that step would be the outlet the lake drains through, so the
        # ring is pinned to zero depth outright.
        depth[0, :] = depth[-1, :] = depth[:, 0] = depth[:, -1] = 0.0
    bed = surf - depth
    out[have] = bed[have]
    return out


def carve_river_beds(
    height_m: np.ndarray,
    water_type: np.ndarray,
    order: np.ndarray,
    spacing_m: float,
    *,
    level: np.ndarray | None = None,
    bank_tolerance_m: float = DEFAULT_BANK_TOLERANCE_M,
    depth_by_order: dict | None = None,
    depth_scale: float = 1.0,
    width_by_order: dict | None = None,
    width_scale: float = 1.0,
    bed_target: np.ndarray | None = None,
) -> np.ndarray:
    """
    Cut a trench under every river pixel: deepest along the channel's middle,
    feathering to nearly nothing at the banks, with the maximum depth set by that
    pixel's Strahler order (`DEFAULT_RIVER_DEPTH_BY_ORDER`, scaled).

    WHY THIS EXISTS. The river water surface is the terrain height under the
    channel (`watersurface.river_surface_moh`), and the runtime derives
    `depth = surface - terrain`. Carve nothing and that depth is zero everywhere:
    the rivers vanish. The predecessor solved it from the other side — leave the
    terrain alone and RAISE the surface 1.5-11 m above it — which gave every river
    a guaranteed water column at the cost of standing it on top of the landscape.
    On flat ground, and anywhere the DTM's channel was too shallow to swallow the
    raise, the result read as a hose of water draped over the terrain, crossing
    contours it should have followed. Cutting down instead puts the water where
    water goes and cannot produce that artefact at any depth, because the surface
    is pinned to the ground by construction.

    The cross-section is scaled by the MODELLED channel width for the order (the
    same `water.DEFAULT_WIDTH_BY_ORDER` table that decided how many pixels the
    river seeded), not by a fixed number of texels, so a trunk river gets a wide
    bowl and a stream gets a narrow slot. A channel only one sample wide has no
    room to feather and takes its full depth as a step — which is correct: it is
    one texel of water, and half-carving it would render it as nothing at all.

    `level` (the water level from `watersurface.river_surface_moh`) makes the carve
    STOP AT THE WATERLINE. A sample on the level takes the full trench; one standing
    more than `bank_tolerance_m` above it takes less; one standing more than twice
    that above it — a bank, or a cliff face the rasterised buffer lapped onto — is
    not touched at all and renders dry. Without this the trench was cut wherever the
    mask went, several metres up cliffs included, notching terrain that will never
    hold water and painting a film of water up it.

    Rivers are NOT flattened, only lowered: the trench follows the DTM's own
    longitudinal profile, so the channel keeps descending exactly as the terrain
    does and no bed elevation is ever invented. Lake pixels are untouched (lake
    beats river in the mask, and lakes have their own carve).

    SAFE FOR THE SOLVER, for the same reason the lake carve is and one more: this
    only ever LOWERS, and it lowers along a drainage line, so it can deepen a
    channel the burn already wants open but cannot dam one. It could in principle
    breach a divide, if ELVIS ever routed a channel across one — that would be a
    source-data error, and a 1-11 m trench is not what would make it visible.

    NOT re-applied per pyramid level. A coarse level averages the trench with its
    banks, so `surface - coarse_terrain` shrinks and distant thin rivers fade (the
    readme quantifies it). Carving each level instead would hold the depth, at the
    cost of widening the trench to a whole coarse cell and popping as the viewer
    approaches — a display trade, not a correctness one, and not taken here.

    `bed_target` (m.o.h. per sample, NaN where unset; from
    `riverbed.downhill_river_bed`) is the downhill-only bed. A sample standing above
    it is lowered TOWARD it by the same weight the trench uses — cross-section times
    bank taper — so the channel middle reaches it, a bank above the waterline is
    still left alone, and no sample is ever cut below it. The taper keeps measuring
    against `level` as passed, which must be the ORIGINAL channel level, not the
    lowered one, or lowering the level would switch the carve off.

    `order` is the per-pixel stream order (`WaterGrid.weight`). Returns a copy.
    """
    from . import water as kvwater

    if height_m.shape != water_type.shape:
        raise ValueError(
            f"height_m {height_m.shape} and water_type {water_type.shape} must match"
        )
    depth_by_order = depth_by_order or DEFAULT_RIVER_DEPTH_BY_ORDER
    width_by_order = width_by_order or kvwater.DEFAULT_WIDTH_BY_ORDER

    out = height_m.copy()
    is_river = (water_type == TYPE_RIVER_VALUE) & np.isfinite(out)
    if not is_river.any():
        return out

    # Distance from each channel sample to the nearest non-channel sample. As in
    # the lake carve this bottoms out at one whole cell, so half a cell is taken
    # off to put the notional bank half a texel outside the outermost wet sample.
    dist = distance_transform_edt(
        water_type == TYPE_RIVER_VALUE, sampling=(spacing_m, spacing_m))
    dist = (dist - 0.5 * spacing_m).astype(np.float32)

    # Per-pixel max depth and channel half-width, filled by unique order so the
    # table lookups run a handful of times rather than once per pixel.
    orders = np.asarray(order, dtype=np.int32)
    depth_max = np.zeros(out.shape, dtype=np.float32)
    for o in np.unique(orders[is_river]):
        depth_max[is_river & (orders == o)] = depth_for_order(
            int(o), depth_by_order, depth_scale)
    half_w = channel_half_width_m(orders, is_river, spacing_m,
                                  width_by_order, width_scale)

    # The MODELLED half-width can exceed the width the channel actually got on the
    # grid — an order-4 river is modelled 8 m wide but rasterises to two texels at
    # 5 m spacing. Normalising by the model alone would then leave that channel at
    # ~two thirds of its depth everywhere, deepest-point included. So the reference
    # is capped by how far this channel really reaches from its own banks, taken as
    # a local maximum of `dist` over a window wide enough for the widest modelled
    # channel. Wide rivers are unaffected (their local reach IS the modelled
    # half-width); narrow ones now bottom out properly.
    from scipy.ndimage import maximum_filter

    widest = float(np.max(half_w[is_river])) if is_river.any() else spacing_m
    k = int(min(2 * int(np.ceil(widest / max(spacing_m, 1e-6))) + 1, 33))
    reach = maximum_filter(dist, size=max(k, 3))
    ref = np.minimum(half_w, np.maximum(reach, 0.5 * spacing_m))

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = _smoothstep(dist / np.maximum(ref, 1e-6))
    weight = frac
    cut = depth_max * frac

    # Stop at the waterline. `e` is how far this sample stands above its channel's
    # water level; the trench is cut in FULL while that is within the bank
    # tolerance, then tapers linearly to nothing at twice it. Cutting in full up to
    # the tolerance is what keeps the channel wet and continuous — the centreline
    # itself has e = 0 and always takes the whole trench — while the taper is what
    # stops the trench being gouged several metres up a cliff face the rasterised
    # buffer happened to lap onto, and stops it ending in a step.
    if level is not None:
        b = max(float(bank_tolerance_m), 1e-3)
        e = height_m.astype(np.float32) - np.asarray(level, dtype=np.float32)
        e = np.where(np.isfinite(e), e, 0.0)     # no level here -> no taper
        taper = np.clip((2.0 * b - e) / b, 0.0, 1.0)
        cut = cut * taper
        weight = frac * taper

    out[is_river] = (out[is_river] - cut[is_river]).astype(out.dtype)
    if bed_target is not None:
        tgt = np.asarray(bed_target, dtype=np.float32)
        h0 = height_m.astype(np.float32)
        sel = is_river & np.isfinite(tgt) & (h0 > tgt)
        pulled = h0[sel] - (h0[sel] - tgt[sel]) * weight[sel]
        out[sel] = np.minimum(out[sel], pulled).astype(out.dtype)
    return out