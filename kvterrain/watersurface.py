"""
kvterrain.watersurface
======================

Per-pixel WATER SURFACE elevation (metres above sea level) for the whole grid,
unifying lakes and rivers into one field the runtime can read as:

    depth(x) = max(0, water_surface_moh(x) - terrain_height(x))

so no rim discovery, no rain-to-fill, and no reseed-from-neighbours is needed.
Terrain edits stay valid automatically: the surface is a property of the water,
the depth is re-derived from whatever terrain is current.

Lakes get their authoritative NVE `hoyde` (see water.WaterGrid.lake_surface_moh).
Rivers have NO elevation in Elvenett (2D polylines), so their surface is derived
here from the leaf DTM under each channel's CENTRELINE:

    surface(x) = ground under the nearest centreline sample   # see channel_level

i.e. a river's water surface is level across its channel and descends along it,
sitting on the ground rather than above it, and the water column comes from the
trench `bathymetry.carve_river_beds` cuts beneath. A river pixel adjacent to a lake
is RAISED to that lake's surface where the lake stands higher, so the two agree at
a flooded inlet/outlet — but never lowered to it, which used to clip the last
texels of every channel to zero depth and leave a dry gap between each river and
the lake it runs into.

Up to 0.5.0 this added a raise by Strahler order instead — `bed + 1.5..11 m` — and
carved nothing. That guaranteed a water column, but it built it upward: the water
sat proud of the terrain everywhere, which read as a hose of water lying over the
landscape. The depth model is now the same one lakes use — a surface on the ground,
a bed carved below it — so a river cannot stand above its banks however deep it is
asked to be, and because the surface is a level rather than a copy of each sample's
ground, it cannot be painted up the cliffs a channel runs past either.

This is deliberately per-pixel and deliberately NOT a fitted profile. A previous
revision rasterised each centreline, fitted a monotone-descending profile to the
sampled bed (isotonic regression / PAVA) and widened it to the buffer; that method
spiked wherever the chord between sparse vertices crossed a bank. `river_surface_moh`
documents the failure in full. Do not reintroduce it here.

The polyline products in `rivernet` apply this SAME rule at densified vertices, so
the raster and the polylines agree by construction rather than by coincidence. The
raster remains authoritative for display; the polylines exist for the runtime burn,
which needs the connectivity and ordering that rasterisation destroys.

Coarse levels use a WATER-ONLY downsample (a parent cell is water if ANY child is;
its surface is the min over water children -- the conservative spill level), so the
narrow-channel-averaged-with-banks failure of "subtract coarse terrain" is avoided
in the surface field itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import bathymetry as kvbathy
from . import core
from . import water as kvwater

TYPE_RIVER = kvwater.TYPE_RIVER   # 1
TYPE_LAKE = kvwater.TYPE_LAKE     # 2

# NOTE: `DEFAULT_DEPTH_BY_ORDER` and `depth_for_order` lived here and now live in
# `bathymetry` as `DEFAULT_RIVER_DEPTH_BY_ORDER` / `depth_for_order`. The numbers
# are the same; they no longer describe how far a river's surface is RAISED above
# its bed, they describe how deep the bed is CARVED below its surface. The table
# belongs beside the carve that applies it, and leaving a copy of it here would
# invite the two to drift.


# --------------------------------------------------------------------------- #
# Small numerical helpers                                                      #
# --------------------------------------------------------------------------- #

# NOTE (0.5.0): `_pava_nondecreasing` / `isotonic_decreasing` lived here and are
# now DELETED. They were the last remnants of the removed centreline approach
# (rasterise a polyline, fit a monotone-descending profile, widen to the buffer)
# whose widen step caused the river-spike failure described in
# `river_surface_moh` below. Nothing called them.
#
# We do NOT reintroduce a profile fit. Descent is enforced by a running MINIMUM
# over densified profiles (`riverbed`, the downhill-only bed), which only ever
# lowers and so cannot pool a level onto a bank the way the fit did. The level
# computed here stays the measured ground; `riverbed` lowers it afterwards.
#
# The remaining helpers below (`_fill_nan_1d`, `_line_rc`, `_bresenham_path`) are
# NOT dead: `rivernet` uses them to walk a centreline's pixel path and to repair
# nodata bed samples along it.


def _fill_nan_1d(y: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs along a 1-D profile (nodata bed samples)."""
    y = np.asarray(y, dtype=np.float64).copy()
    bad = ~np.isfinite(y)
    if not bad.any():
        return y
    if bad.all():
        return y  # nothing to interpolate from; caller drops these
    good = ~bad
    idx = np.arange(y.size)
    y[bad] = np.interp(idx[bad], idx[good], y[good])
    return y


def _line_rc(r0: int, c0: int, r1: int, c1: int):
    """Integer Bresenham line between two (row,col) pixels, 8-connected, ordered
    from (r0,c0) to (r1,c1). Pure Python — no scikit-image dependency."""
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    r, c = r0, c0
    rr: list[int] = []
    cc: list[int] = []
    while True:
        rr.append(r); cc.append(c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr; c += sc
        if e2 < dc:
            err += dc; r += sr
    return rr, cc


def _bresenham_path(rc):
    """Ordered, de-duplicated pixel path through vertex pixels (r,c)."""
    path: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(rc[:-1], rc[1:]):
        rr, cc = _line_rc(int(r0), int(c0), int(r1), int(c1))
        for k in range(len(rr)):
            p = (rr[k], cc[k])
            if not path or path[-1] != p:
                path.append(p)
    if not path and rc:
        path.append((int(rc[0][0]), int(rc[0][1])))
    return path


# --------------------------------------------------------------------------- #
# River surface                                                                #
# --------------------------------------------------------------------------- #

def channel_level(plan: core.GridPlan, feats: kvwater.WaterFeatures,
                  is_river: np.ndarray, ground: np.ndarray,
                  half_width_m: np.ndarray,
                  bank_tolerance_m: float = kvbathy.DEFAULT_BANK_TOLERANCE_M
                  ) -> np.ndarray:
    """
    The water LEVEL across each channel: the ground under the channel's own
    CENTRELINE, spread sideways to the samples that centreline seeded.

    A water surface is level across its channel and descends along it. Giving every
    sample its own ground instead — which is what this used to do — paints water
    onto whatever the rasterised buffer happens to cover, so wherever a channel
    runs past the foot of a cliff, the samples that lapped onto the rock carried
    water up it: a thin film climbing the cliff. Over a Lierne block that was 16.5%
    of all river samples standing above their own channel's level, a median of
    2.4 m up the bank and as much as 12.3 m of it.

    WHY THE CENTRELINE AND NOT A LOCAL MINIMUM. A minimum over each channel's own
    width is the obvious fix and it was measured first. It removes the film, but it
    also takes the minimum ALONG the channel, and a channel is supposed to descend:
    on any reach falling faster than the trench is deep, every sample sits above the
    level taken from its downstream neighbour and the whole reach clips to dry. The
    wet channel came apart into thousands of fragments (204 mask components ->
    1487-3259 wet runs, with 12-17% of centreline samples dry). The centreline has
    no such failure by construction: the level at a centreline sample is that
    sample's own ground, so every centreline sample keeps the full trench depth and
    the channel stays continuous however steep it gets. Only the sideways spread is
    flattened, which is the direction the artefact actually lives in.

    Two guards:
      * a sample further from any centreline than its own channel is wide keeps its
        own ground (a buffer with no centreline of its own — which happens where a
        centreline lies under a lake — must not inherit a level from some unrelated
        channel across the map), and
      * the level is capped at `bank_tolerance_m` above the sample's own ground, so
        water can never stand more than that above the terrain anywhere. Without it
        a channel running along a CLIFF TOP would hang a wall of water off the edge
        — the same artefact upside down.
    """
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import LineString

    out = np.where(is_river, ground, np.nan).astype(np.float32)
    if not is_river.any():
        return out

    shapes = [(LineString(s.xy), 1)
              for s in list(feats.rivers) + list(feats.main_rivers)
              if s.xy.shape[0] >= 2]
    if not shapes:
        return out
    centre = np.zeros(is_river.shape, dtype=np.uint8)
    rasterize(shapes, out=centre, transform=kvwater.sample_grid_transform(plan, 0),
              all_touched=True)
    # Only centrelines that landed on RIVER samples count. A lake-through-line's
    # ground is the pool's flat surface, and letting that be the level for a
    # channel climbing away from the lake would drain the channel; the lake tie-in
    # below is what joins those two, and it only ever raises.
    src = (centre > 0) & is_river & np.isfinite(ground)
    if not src.any():
        return out

    dist, ind = distance_transform_edt(
        ~src, sampling=(plan.spacing_m, plan.spacing_m),
        return_distances=True, return_indices=True)
    lvl = ground[ind[0], ind[1]].astype(np.float32)

    far = dist > (np.asarray(half_width_m, dtype=np.float32) + plan.spacing_m)
    lvl = np.where(far, ground, lvl)
    lvl = np.minimum(lvl, ground.astype(np.float32) + float(bank_tolerance_m))

    out = np.where(is_river & np.isfinite(ground), lvl, np.nan).astype(np.float32)
    return out


def river_surface_moh(
    plan: core.GridPlan,
    feats: kvwater.WaterFeatures,
    wg: kvwater.WaterGrid,
    height_uncarved: np.ndarray,
    *,
    lake_surface: Optional[np.ndarray] = None,
    width_by_order: Optional[dict] = None,
    width_scale: float = 1.0,
    bank_tolerance_m: float = kvbathy.DEFAULT_BANK_TOLERANCE_M,
) -> np.ndarray:
    """
    Per-pixel river water-surface elevation (m.o.h.), float32, NaN off-river.

    Each river pixel's surface is its CHANNEL's level — the leaf-DTM ground under that
    channel's centreline, spread flat across the buffer that centreline seeded, read from the
    bed as it stood BEFORE `bathymetry.carve_river_beds` cut the trench:

        surface(x) = channel_level(x)        # see channel_level

    The runtime derives depth as (surface - terrain) against the CARVED terrain it ships with,
    so the water column is the full trench depth along the channel, tapers off as the ground
    rises away from it, and reaches zero on the bank — the waterline lands where the ground
    crosses the level, at the resolution of the height data rather than at the resolution of
    the rasterised buffer.

    THREE RULES HAVE BEEN TRIED HERE, and the shape of the failures is the argument for this
    one. 0.5.0 rasterised each Elvenett polyline, fitted a monotone-descending isotonic
    profile (PAVA) to the sampled bed and widened it to the buffer; wherever the chord between
    sparse vertices cut across a ridge the fit pooled the surface metres ABOVE the bed and the
    widen smeared it onto lower neighbours — rivers spiked. 0.5.0's replacement gave every
    sample its own bed plus a raise by stream order, which could not spike but stood the whole
    river on top of the landscape — the water-hose. Dropping the raise and carving instead
    fixed the floating but kept the underlying mistake: a surface derived per SAMPLE follows
    the terrain, so water climbed every cliff the buffer lapped onto.

    A water surface is a LEVEL. It is flat across its channel and descends along it, and that
    is a property of the channel, not of the sample. Deriving it from the centreline is what
    makes it one, and it is why neither of the two failure modes can return: the level cannot
    spike (it is measured ground, never fitted) and cannot float (it is capped at
    `bank_tolerance_m` above any sample's own ground).

    `height_uncarved` is the LEAF (finest) DTM (SY,SX), void-repaired but not yet river-carved.
    `feats` supplies the centrelines. `lake_surface` (from WaterGrid.lake_surface_moh), when
    given, RAISES river pixels adjacent to a lake to the lake's surface — never lowers them —
    so the two agree at a flooded inlet/outlet without leaving a dry seam at every other one.
    """
    SX, SY = kvwater._level_samples(plan, 0)
    if height_uncarved.shape != (SY, SX):
        raise ValueError(f"height_uncarved {height_uncarved.shape} != grid {(SY, SX)}")

    is_river = wg.type == TYPE_RIVER
    out = np.full((SY, SX), np.nan, dtype=np.float32)
    if not is_river.any():
        return out

    # surface = the level across this sample's own channel. A NaN (nodata) bed stays NaN ->
    # that pixel reads no water. The half-width is the SAME one the carve shapes its trench
    # with, so the level is flattened over exactly the width that gets carved.
    half_w = kvbathy.channel_half_width_m(
        wg.weight, is_river, plan.spacing_m, width_by_order, width_scale)
    out = channel_level(plan, feats, is_river, height_uncarved, half_w,
                        bank_tolerance_m=bank_tolerance_m)

    # Lake tie-in: a river pixel adjacent to a lake is RAISED to that lake's surface where
    # the lake stands higher, so the two meet exactly at a flooded inlet/outlet.
    #
    # The tie-in only ever lifts. Pinning outright — taking the lake's level whether it is
    # above or below the channel — is what put a dry gap between every river and the lake it
    # runs into. A lake sits BELOW the ground beside it at 81% of these pixels (measured over
    # a Lierne block: 545 of 673), because a channel arrives over a bank; forcing the water
    # surface down to the pool's level there drops it below the carved bed, `depth =
    # max(0, surface - terrain)` clips to zero, and the last texel or two of river renders as
    # dry ground. That was 7.4% of all junction pixels. Taking the max instead leaves those
    # pixels at their own ground level — the channel keeps its full carved depth right up to
    # the shoreline — and still floods the ones the lake genuinely covers. Zero dry junction
    # pixels, same measurement.
    if lake_surface is not None:
        out = apply_lake_tie_in(out, lake_tie_in(wg, lake_surface))

    return out


def lake_tie_in(wg: kvwater.WaterGrid, lake_surface: np.ndarray):
    """
    Which river pixels touch a lake, and that lake's surface there: the inputs to
    the raise-only tie-in described in `river_surface_moh`. Returned rather than
    applied so a caller that adjusts the river level in between (the downhill
    river bed) can re-apply the SAME raise afterwards without paying for the
    distance transform twice. None when there is nothing to tie.
    """
    from scipy.ndimage import binary_dilation
    is_lake = wg.type == TYPE_LAKE
    if not is_lake.any():
        return None
    _, linds = distance_transform_edt(
        ~is_lake, return_distances=True, return_indices=True)
    nearest_lake_surf = lake_surface[linds[0], linds[1]]
    touch = (wg.type == TYPE_RIVER) & binary_dilation(is_lake, iterations=1)
    good = touch & np.isfinite(nearest_lake_surf)
    return good, nearest_lake_surf[good].astype(np.float32)


def apply_lake_tie_in(river_level: np.ndarray, tie) -> np.ndarray:
    """Raise (never lower) the river pixels in `tie` to their lake's surface."""
    if tie is None:
        return river_level
    good, vals = tie
    out = np.array(river_level, dtype=np.float32, copy=True)
    # fmax, not maximum: a nodata bed leaves NaN in `out`, and at a lake
    # edge the lake's own surface is a better answer than "no water".
    out[good] = np.fmax(out[good], vals)
    return out


# --------------------------------------------------------------------------- #
# Combine + downsample                                                         #
# --------------------------------------------------------------------------- #

def combine_water_surface(lake_surface: np.ndarray,
                          river_surface: np.ndarray) -> np.ndarray:
    """Merge lake + river surfaces into one field (NaN = no water). Lake wins
    overlaps, matching the mask's lake>river priority."""
    out = np.array(river_surface, dtype=np.float32, copy=True)
    lmask = np.isfinite(lake_surface)
    out[lmask] = np.asarray(lake_surface, dtype=np.float32)[lmask]
    return out


def downsample_surface_water_only(surf: np.ndarray) -> np.ndarray:
    """
    One pyramid step for the surface field: a parent sample is water if ANY of its
    up-to-9 corner-anchored child samples is water, and its surface is the MIN over
    the water children (the conservative spill level -- never floods a cell higher
    than its lowest water child would). NaN stays NaN (no water).

    The corner-anchored 3x3 gather matches `core.decimate_corner`'s sample lattice
    (parent sample i sits on child sample 2i), so a surface sample and its height
    twin stay on the same world point at every level. `waterid.downsample_water_id`
    uses the identical gather with a categorical rule.

    (This docstring used to cite `water.decimate_water_corner` as the scheme being
    matched. That function was deleted along with the categorical `.water` tile;
    the scheme it described is now defined here and in `waterid`.)
    """
    SY, SX = surf.shape
    oy, ox = (SY - 1) // 2 + 1, (SX - 1) // 2 + 1
    out = np.full((oy, ox), np.nan, dtype=np.float32)
    # gather the 3x3 corner neighbourhood of each parent's anchor child (2*i, 2*j)
    acc = np.full((oy, ox), np.inf, dtype=np.float32)
    any_water = np.zeros((oy, ox), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = np.clip(np.arange(oy) * 2 + dr, 0, SY - 1)
            c = np.clip(np.arange(ox) * 2 + dc, 0, SX - 1)
            child = surf[np.ix_(r, c)]
            w = np.isfinite(child)
            any_water |= w
            acc = np.where(w, np.minimum(acc, child), acc)
    out[any_water] = acc[any_water]
    return out


def build_surface_pyramid(leaf_surface: np.ndarray, num_levels: int) -> list:
    """Surface field for every pyramid level, water-only downsampled."""
    levels = [leaf_surface.astype(np.float32)]
    for _ in range(1, num_levels):
        levels.append(downsample_surface_water_only(levels[-1]))
    return levels


# --------------------------------------------------------------------------- #
# On-disk .wsurf tile: u16, packed on the SAME [hmin,hmax] as the .r16 height  #
# tile so the runtime unpacks identically; 65535 == "no water" sentinel.       #
# --------------------------------------------------------------------------- #

import os

SURFACE_NODATA_U16 = np.uint16(65535)
_SURF_MAX_CODE = 65534.0


def pack_surface_u16(surf: np.ndarray, hmin: float, hmax: float) -> np.ndarray:
    """Pack a surface field to <u2 over [hmin,hmax]->[0,65534]; NaN -> 65535."""
    rng = max(float(hmax) - float(hmin), 1e-6)
    out = np.full(surf.shape, SURFACE_NODATA_U16, dtype="<u2")
    w = np.isfinite(surf)
    if w.any():
        norm = np.clip((surf[w].astype(np.float64) - hmin) / rng, 0.0, 1.0)
        out[w] = np.rint(norm * _SURF_MAX_CODE).astype("<u2")
    return out


def unpack_surface_u16(packed: np.ndarray, hmin: float, hmax: float) -> np.ndarray:
    """Inverse of pack_surface_u16 (for tests/validation). 65535 -> NaN."""
    rng = max(float(hmax) - float(hmin), 1e-6)
    out = np.full(packed.shape, np.nan, dtype=np.float32)
    w = packed != SURFACE_NODATA_U16
    out[w] = (hmin + (packed[w].astype(np.float64) / _SURF_MAX_CODE) * rng).astype(np.float32)
    return out


def export_surface_tiles(plan: core.GridPlan, surface_levels: list, out_dir: str,
                         hmin: float, hmax: float, *,
                         atlas_name: str = core.ATLAS_SURFACE_FILE) -> dict:
    """
    Slice every surface pyramid level into (tile_cells+1)² u16 tiles and concatenate them
    into one dense `surface.atlas` blob (same §4 layout as the height atlas — identical tile
    geometry, so the runtime reads a surface tile at the SAME byte offset as its height
    twin). Returns a manifest fragment describing the encoding and naming the atlas file.
    """
    TC = plan.tile_cells
    TS = TC + 1

    os.makedirs(out_dir, exist_ok=True)
    tiles_written = 0
    with open(os.path.join(out_dir, atlas_name), "wb") as atlas_fh:
        for lvl, surf in enumerate(surface_levels):
            tiles_x, tiles_y = core.tiles_at_level(
                plan.leaf_tiles_x, plan.leaf_tiles_y, lvl)
            SY = surf.shape[0]
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    r0, c0, _ = core.north_up_tile_slice(SY, TC, tx, ty)
                    tile = surf[r0:r0 + TS, c0:c0 + TS]
                    pack_surface_u16(tile, hmin, hmax).tofile(atlas_fh)
                    tiles_written += 1

    expect = core.atlas_total_bytes(
        plan.leaf_tiles_x, plan.leaf_tiles_y, plan.num_levels, TS)
    actual = os.path.getsize(os.path.join(out_dir, atlas_name))
    if actual != expect:
        raise RuntimeError(
            f"dense surface atlas size mismatch: {atlas_name} is {actual} "
            f"bytes, expected {expect}. The surface tile grid was not dense.")

    manifest = {
        "atlas_file": atlas_name,
        "atlas_format": core.ATLAS_FORMAT,
        "dtype": "u16le",
        "packing": "same [height_min_m, height_max_m] as the height atlas, mapped to [0,65534]",
        "nodata_code": int(SURFACE_NODATA_U16),
        "units": "metres_above_sea_level",
        "row_order": "north_to_south",
        "runtime": "surface = (code==65535) ? NO_WATER : height_min_m + code/65534*(height_max_m-height_min_m); "
                   "depth = max(0, surface - terrain_height). Unifies lakes and rivers; "
                   "no rim scan, no rain-fill, valid across terrain edits.",
        "lake_surface": "authoritative NVE hoyde, or a level read off the LiDAR water "
                        "surface inside the polygon (metres above sea level)",
        "river_surface": "the UNCARVED leaf-DTM ground height under the channel — the river "
                         "surface sits ON the terrain, and the water column comes from the "
                         "trench carved beneath it (see river_carve_depth_by_order_m); "
                         "raised to the lake surface, never lowered to it, where a river "
                         "meets a lake.",
        "river_carve_depth_by_order_m": {
            str(k): v for k, v in kvbathy.DEFAULT_RIVER_DEPTH_BY_ORDER.items()},
        "tiles_written": tiles_written,
    }
    # Provenance for the NVE-derived water.
    manifest.update(kvwater.water_source_manifest())
    return manifest