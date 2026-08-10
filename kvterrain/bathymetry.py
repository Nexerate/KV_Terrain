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

CALL ORDER (this is the contract; `carve_lake_beds` does NOT do it for you):

    leaf = fill_lake_surface(leaf, wg.type)          # repair voids
    leaf = carve_lake_beds(leaf, wg.type, spacing,   # then carve the bowl
                           surface_moh=lake_surf, ...)

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
DEFAULT_RAMP_RADIUS_M = 4.0   # legacy (pre-known-surface) ramp radius
DEFAULT_MAX_DEPTH_M = 4.0     # legacy (pre-known-surface) max depth
# New model: carve the bed a fixed REAL depth below the (known or estimated) water
# surface, flat, with a short shore bevel. carve_depth is in real metres; the goal is
# opacity, so it must exceed the renderer's opaque threshold AFTER any vertical scale
# the consumer applies (e.g. a 1:5 height compression turns 20 m into 4 units).
DEFAULT_CARVE_DEPTH_M = 20.0
DEFAULT_BEVEL_PX = 2
DEFAULT_SHORE_BAND_PX = 3
# A flat lake surface is never this far below its own shore. Anything deeper is a
# void fill (0 / sentinel / NaN), not real bathymetry. Generous enough to preserve
# a few metres of genuine bank, small enough to catch a 0-fill in high terrain.
DEFAULT_VOID_BELOW_SHORE_M = 12.0


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance_to_shore_m(water_type: np.ndarray, spacing_m: float) -> np.ndarray:
    """
    Distance from each lake sample to the nearest NON-lake sample, in metres.

    Note what this means at the edge: the outermost wet sample is one full cell
    from dry land, so its distance is `spacing_m` — NOT zero. `carve_lake_beds`
    subtracts that offset before ramping; see SHORE_ANCHOR_M.
    """
    is_lake = water_type == TYPE_LAKE_VALUE
    if not is_lake.any():
        return np.zeros(water_type.shape, dtype=np.float32)
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


def _shore_ramp(is_lake: np.ndarray, dist: np.ndarray,
                bevel_px: int, spacing_m: float) -> np.ndarray:
    """
    The bevel profile: 0 at the waterline rising to 1 at full carve depth.

    Two things happen here beyond a plain `smoothstep(dist / bevel)`.

    1. The ramp is anchored on the WATERLINE (`SHORE_ANCHOR_M`), because `dist`
       bottoms out at one whole cell inside a lake rather than at zero.

    2. The bevel is shrunk PER LAKE BODY so the ramp always reaches 1 somewhere
       in that body. Anchoring costs one cell of reach, and a body narrower than
       the nominal bevel would otherwise never bottom out — it would hold a few
       metres of water at its deepest point and render near-transparent. In the
       reference export that is 197 of 1484 bodies: tiny, 586 samples between
       them (0.02% of lake area), but "the pond vanished" is a worse failure than
       "the pond is a little too deep", and the guard is one labelled maximum.

    A body with NO room at all — one sample across, every sample sitting on the
    waterline — takes the full carve as a step. There is nowhere to ramp, so a
    vertical pit is the only available answer, and it is what this code did
    everywhere before the anchor was fixed.
    """
    nominal = max(float(bevel_px) * spacing_m, 1e-6)
    anchor = SHORE_ANCHOR_M * spacing_m
    lbl, n = label(is_lake)
    if n == 0:
        return np.zeros(is_lake.shape, dtype=np.float32)

    # Deepest reach of each body == how much ramp room it actually has.
    reach = np.atleast_1d(
        np.asarray(_labeled_maximum(dist, lbl, np.arange(1, n + 1)),
                   dtype=np.float32))
    room = reach - anchor

    width = np.empty(n + 1, dtype=np.float32)
    width[0] = nominal
    width[1:] = np.clip(room, 1e-6, nominal)
    # Bodies with no room take the full carve outright. Flagged explicitly rather
    # than left to `room/width`, which is 0/0 there and would silently yield a
    # ramp of zero — i.e. no carve at all, the pond rendered dry.
    stepped = np.zeros(n + 1, dtype=bool)
    stepped[1:] = room <= 1e-6

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = _smoothstep((dist - anchor) / width[lbl])
    return np.where(stepped[lbl], np.float32(1.0), frac).astype(np.float32)


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


# Estimated levels are pulled DOWN by this much before use. The estimator itself
# is unbiased at the median, so this is a deliberate one-sided hedge: water
# standing proud of its own banks reads as broken, whereas water sitting a
# fraction low is the mild version of an artifact the shore bevel already
# feathers. Calibrated on 89 Lierne lakes that DO carry an NVE hoyde — 0.25 m
# cuts the "more than 2 m too high" tail from 3.4% to 1.1% and offsets about half
# the +0.5 m bias seen on small lakes, at the cost of a 0.25 m median undershoot,
# which is a quarter of `hoyde`'s own 1 m quantisation.
DEFAULT_LEVEL_MARGIN_M = 0.25
# Fewer interior samples than this and the median is not worth trusting.
MIN_INTERIOR_SAMPLES = 5


def estimate_levels_from_interior(
    height_m: np.ndarray,
    lake_id: np.ndarray,
    *,
    margin_m: float = DEFAULT_LEVEL_MARGIN_M,
    min_samples: int = MIN_INTERIOR_SAMPLES,
) -> dict:
    """
    Per-lake water level read straight off the DTM INSIDE each lake polygon.

    LiDAR does get a return from water — it reports the water SURFACE — so the
    samples inside a lake polygon are a direct measurement of the thing we want,
    not an inference from the surrounding bank. That makes this strictly better
    than `estimate_lake_surface`, which reads a percentile of the shoreline band
    outside the polygon. Measured against the 89 lakes in a Lierne block that do
    carry an NVE `hoyde`:

        interior median   bias +0.00 m, |err| p50 0.63 m, 28% high by >0.5 m
        shore band (old)  bias +0.35 m, |err| p50 0.75 m, 39% high by >0.5 m

    and the truth itself is quantised to 1 m, so 0.63 m is at the noise floor.

    Takes the plain MEDIAN. Eroding the boundary ring first, or using a low
    percentile instead, were both measured and both changed the result by under
    0.02 m — lake interiors are flat, so there is nothing for a robust statistic
    to be robust against. Simplicity wins.

    `height_m` must be the VOID-REPAIRED, UNCARVED bed (i.e. run this after
    `fill_lake_surface` and before `carve_lake_beds`); sampling a carved array
    would just return `surface - carve_depth`.

    Returns {local lake id -> level in metres}, omitting lakes with too few
    usable samples.
    """
    ids = np.asarray(lake_id).ravel()
    h = np.asarray(height_m).ravel()
    sel = (ids > 0) & np.isfinite(h)
    if not sel.any():
        return {}

    gid = ids[sel]
    gh = h[sel].astype(np.float64)
    order = np.argsort(gid, kind="stable")
    gid = gid[order]
    gh = gh[order]

    uniq, starts = np.unique(gid, return_index=True)
    ends = np.append(starts[1:], gid.size)

    out: dict = {}
    for u, a, b in zip(uniq, starts, ends):
        if b - a < min_samples:
            continue
        out[int(u)] = float(np.median(gh[a:b])) - float(margin_m)
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
    bevel_px: int = DEFAULT_BEVEL_PX,
    estimate_missing: bool = True,
    shore_band_px: int = DEFAULT_SHORE_BAND_PX,
    void_below_shore_m: float = DEFAULT_VOID_BELOW_SHORE_M,
    # ── legacy aliases (pre-known-surface); mapped if provided ──
    ramp_radius_m: float | None = None,
    max_depth_m: float | None = None,
    fill_voids: bool | None = None,
) -> np.ndarray:
    """
    Set each lake's bed to a fixed real depth below its water SURFACE, flat, with a
    short shore bevel. The bed value does NOT depend on what the DTM returned inside
    the lake (flat surface, void 0, sentinel) — it is `surface - carve_depth_m` — so
    the LiDAR water-flattening and the whole void saga are irrelevant here.

    The resulting bed profile, going inward from the shore at the default 2-texel
    bevel and 5 m spacing, is `surface - [0, 10, 20, 20, ...]`. The leading ZERO is
    the point: the outermost wet sample sits exactly at the water level, so the
    water has somewhere to be shallow and a mesh renderer has a vertex to feather
    against. See SHORE_ANCHOR_M for what this looked like when it was wrong, and
    `_shore_ramp` for the per-body handling of lakes too narrow to ramp.

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
        bevel_px = max(1, int(round(ramp_radius_m / max(spacing_m, 1e-6))))
    if fill_voids is not None:
        estimate_missing = fill_voids

    out = height_m.copy()
    is_lake = water_type == TYPE_LAKE_VALUE
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

    # Flat bed at surface - carve_depth, with a smooth bevel over the first
    # `bevel_px` texels from shore so the shoreline isn't a vertical wall.
    #
    # The ramp is anchored on the WATERLINE, not on the outermost wet sample:
    # `dist` never reads below one cell inside a lake, so ramping straight off it
    # opens the bevel already half-open and leaves a vertical step at the shore.
    # See SHORE_ANCHOR_M for the measurement that motivated this. The subtraction
    # can only make the carve SHALLOWER, so it cannot deepen a bowl, widen one, or
    # lower a rim sample that was previously left alone.
    dist = distance_to_shore_m(water_type, spacing_m)      # metres from shore
    frac = _shore_ramp(is_lake, dist, bevel_px, spacing_m)  # 0 at shore -> 1 inside
    bed = surf - carve_depth_m * frac
    out[have] = bed[have]
    return out