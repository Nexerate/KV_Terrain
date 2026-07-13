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
Rivers have NO elevation in Elvenett (2D polylines), so their surface is estimated
here from the leaf DTM along each channel:

  1. Rasterise each centerline in DOWNSTREAM order (Elvenett vertices are ordered
     downstream) and sample the leaf bed height along it.
  2. Fit a MONOTONE-NON-INCREASING profile to that bed (isotonic regression / PAVA).
     Real river surfaces never flow uphill; DTM noise (bridges, vegetation, a coarse
     cell catching a bank) does. PAVA is the least-squares monotone fit, so it carves
     spurious bumps and fills spurious dips minimally, without a hand-tuned smoother.
  3. Add a small nominal depth by Strahler order -> channel surface.
  4. Tie to lakes: a river pixel adjacent to a lake is pinned to that lake's surface,
     so the river and lake agree at the outlet/inlet by construction.
  5. Widen: every pixel of the buffered channel takes its nearest centerline surface.

Coarse levels use a WATER-ONLY downsample (a parent cell is water if ANY child is;
its surface is the min over water children -- the conservative spill level), so the
narrow-channel-averaged-with-banks failure of "subtract coarse terrain" is avoided
in the surface field itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import core
from . import water as kvwater

TYPE_RIVER = kvwater.TYPE_RIVER   # 1
TYPE_LAKE = kvwater.TYPE_LAKE     # 2

# How far the river surface sits above its (already-incised) bed, in metres, by
# Strahler order. Rivers are NOT carved — the DTM already contains the channel and
# ravine — so this raise is the ONLY thing that gives a river visible depth
# (depth = surface - terrain), and it must clear the renderer's opaque threshold.
#
# Sizing rationale (mirrors the lake carve, which uses ~20 m real so it survives a
# 1:5 vertical compression -> ~4 units and reads opaque): a 1-3 m raise was far
# below that threshold, so rivers rendered essentially transparent — the symptom
# that motivated this table. The raise now ramps with stream order so big rivers,
# which sit in deep incised channels, fill those channels and read solidly opaque,
# while small streams (shallow/no channel) get just a few metres and don't balloon
# a wide sheet of water over flat ground. depth_for_order clamps any order beyond
# the table ends, so orders past 8 also get the order-8 value. The whole table is
# multiplied by `depth_scale` (UI "River surface raise ×" / CLI --river-depth-scale)
# for per-renderer tuning without editing code.
DEFAULT_DEPTH_BY_ORDER = {
    1: 3.0, 2: 4.5, 3: 6.0, 4: 8.0, 5: 11.0, 6: 14.0, 7: 18.0, 8: 22.0,
}


# --------------------------------------------------------------------------- #
# Small numerical helpers                                                      #
# --------------------------------------------------------------------------- #

def depth_for_order(order: int, table: dict, scale: float = 1.0) -> float:
    if not table:
        return 0.5 * scale
    keys = sorted(table)
    o = min(max(int(order), keys[0]), keys[-1])
    return float(table[o]) * float(scale)


def _pava_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators: least-squares monotone NON-DECREASING fit."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return y.copy()
    vals: list[float] = []
    wts: list[float] = []
    cnts: list[int] = []
    for yi in y:
        vals.append(float(yi)); wts.append(1.0); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2 = vals.pop(); w2 = wts.pop(); c2 = cnts.pop()
            v1 = vals.pop(); w1 = wts.pop(); c1 = cnts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2); cnts.append(c1 + c2)
    out = np.empty(n, dtype=np.float64)
    i = 0
    for v, c in zip(vals, cnts):
        out[i:i + c] = v
        i += c
    return out


def isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    """Least-squares monotone NON-INCREASING fit (upstream -> downstream)."""
    return -_pava_nondecreasing(-np.asarray(y, dtype=np.float64))


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

def river_surface_moh(
    plan: core.GridPlan,
    feats: kvwater.WaterFeatures,
    wg: kvwater.WaterGrid,
    height_leaf: np.ndarray,
    *,
    depth_by_order: Optional[dict] = None,
    depth_scale: float = 1.0,
    lake_surface: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-pixel river water-surface elevation (m.o.h.), float32, NaN off-river.

    Each river pixel's surface is simply its OWN leaf-DTM bed height plus a small nominal
    depth chosen by that pixel's stream order (wg.weight):

        surface(x) = height_leaf(x) + depth_for_order(order(x))

    The runtime derives depth as (surface - terrain) against the SAME leaf DTM, so this yields
    a UNIFORM ~depth_for_order water column along every channel. It cannot spike, because the
    surface tracks the local bed by construction (surface - terrain == bump, everywhere).

    This replaces the earlier centreline approach (rasterise each Elvenett polyline, fit a
    monotone-descending isotonic profile to the sampled bed, then widen to the buffer). That
    fit deviated from the local bed: wherever the straight Bresenham line between sparse
    vertices cut across a ridge/bank, the sampled bed was non-monotone, PAVA pooled the surface
    metres ABOVE the bed, and the widen smeared that inflated value onto lower neighbours. The
    runtime then read depth = surface - terrain >> 0 there — the river "spikes". Only rivers
    synthesise their surface (lakes use authoritative NVE hoyde), so only rivers spiked. Tying
    the surface to each pixel's own bed removes the failure mode entirely, at every LOD (the
    coarse min-over-water surface minus the averaged terrain is still <= bump, never a spike).

    `height_leaf` is the LEAF (finest) DTM (SY,SX). `lake_surface` (from
    WaterGrid.lake_surface_moh), when given, pins river pixels adjacent to a lake to the lake
    surface so the two agree at the inlet/outlet. `feats` is no longer needed (kept for a
    stable signature); per-pixel stream order comes from the rasterised wg.weight.
    """
    depth_by_order = depth_by_order or DEFAULT_DEPTH_BY_ORDER
    SX, SY = kvwater._level_samples(plan, 0)
    if height_leaf.shape != (SY, SX):
        raise ValueError(f"height_leaf {height_leaf.shape} != grid {(SY, SX)}")

    is_river = wg.type == TYPE_RIVER
    out = np.full((SY, SX), np.nan, dtype=np.float32)
    if not is_river.any():
        return out

    # Per-pixel nominal depth from the per-pixel stream order (wg.weight; the rasteriser stores
    # the order clamped to 1..255). Fill by unique order so depth_for_order runs a handful of
    # times rather than per pixel; the table lookup itself clamps orders beyond the table ends.
    orders = wg.weight.astype(np.int32)
    bump = np.zeros((SY, SX), dtype=np.float32)
    for o in np.unique(orders[is_river]):
        bump[is_river & (orders == o)] = depth_for_order(int(o), depth_by_order, depth_scale)

    # surface = own bed + own bump. A NaN (nodata) bed stays NaN -> that pixel reads no water.
    out[is_river] = height_leaf[is_river].astype(np.float32) + bump[is_river]

    # Lake tie-in: a river pixel adjacent to a lake is pinned to that lake's surface, so the
    # river and lake meet exactly at the inlet/outlet (overrides bed+bump on those pixels).
    if lake_surface is not None:
        from scipy.ndimage import binary_dilation
        is_lake = wg.type == TYPE_LAKE
        if is_lake.any():
            _, linds = distance_transform_edt(
                ~is_lake, return_distances=True, return_indices=True)
            nearest_lake_surf = lake_surface[linds[0], linds[1]]
            touch = is_river & binary_dilation(is_lake, iterations=1)
            good = touch & np.isfinite(nearest_lake_surf)
            out[good] = nearest_lake_surf[good].astype(np.float32)

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
    One pyramid step for the surface field, matching water.decimate_water_corner's
    corner-anchored 3x3 scheme: a parent sample is water if ANY of its up-to-9 child
    samples is water, and its surface is the MIN over the water children (the
    conservative spill level -- never floods a cell higher than its lowest water
    child would). NaN stays NaN (no water).
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
                         hmin: float, hmax: float, *, writer=None,
                         write_atlas: bool = True, write_per_tile: bool = True,
                         atlas_name: str = core.ATLAS_SURFACE_FILE) -> dict:
    """
    Slice every surface pyramid level into (tile_cells+1)² tiles and write each as a
    raw little-endian u16 `.wsurf` array beside the matching `.r16` tile, and/or
    concatenate them into one dense `surface.atlas` blob (same §4 layout as the
    height atlas). Returns a manifest fragment describing the encoding. Uses the
    same core.north_up_tile_slice tiling as the height export so files line up 1:1.
    """
    TC = plan.tile_cells
    TS = TC + 1

    def _default_writer(path, arr):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr.tofile(path)
    write = writer or _default_writer

    atlas_fh = None
    if write_atlas:
        os.makedirs(out_dir, exist_ok=True)
        atlas_fh = open(os.path.join(out_dir, atlas_name), "wb")

    tiles_written = 0
    try:
      for lvl, surf in enumerate(surface_levels):
        tiles_x, tiles_y = core.tiles_at_level(
            plan.leaf_tiles_x, plan.leaf_tiles_y, lvl)
        SY = surf.shape[0]
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                r0, c0, _ = core.north_up_tile_slice(SY, TC, tx, ty)
                tile = surf[r0:r0 + TS, c0:c0 + TS]
                packed = pack_surface_u16(tile, hmin, hmax)
                rel = f"L{lvl}/{tx}_{ty}.wsurf"
                if write_per_tile:
                    write(os.path.join(out_dir, rel), packed)
                if atlas_fh is not None:
                    packed.tofile(atlas_fh)
                tiles_written += 1
    finally:
        if atlas_fh is not None:
            atlas_fh.close()

    atlas_file = None
    if write_atlas:
        expect = core.atlas_total_bytes(
            plan.leaf_tiles_x, plan.leaf_tiles_y, plan.num_levels, TS)
        actual = os.path.getsize(os.path.join(out_dir, atlas_name))
        if actual != expect:
            raise RuntimeError(
                f"dense surface atlas size mismatch: {atlas_name} is {actual} "
                f"bytes, expected {expect}. The surface tile grid was not dense.")
        atlas_file = atlas_name

    manifest = {
        "tile_suffix": ".wsurf",
        "dtype": "u16le",
        "packing": "same [height_min_m, height_max_m] as .r16, mapped to [0,65534]",
        "nodata_code": int(SURFACE_NODATA_U16),
        "units": "metres_above_sea_level",
        "row_order": "north_to_south",
        "runtime": "surface = (code==65535) ? NO_WATER : height_min_m + code/65534*(height_max_m-height_min_m); "
                   "depth = max(0, surface - terrain_height). Unifies lakes and rivers; "
                   "no rim scan, no rain-fill, valid across terrain edits.",
        "lake_surface": "authoritative NVE hoyde (metres above sea level)",
        "river_surface": "leaf-DTM bed + a raise by stream order (see river_raise_by_order_m), "
                         "so depth = raise everywhere along a channel; pinned to the lake "
                         "surface where a river meets a lake.",
        "river_raise_by_order_m": {str(k): v for k, v in DEFAULT_DEPTH_BY_ORDER.items()},
        "tiles_written": tiles_written,
    }
    if atlas_file is not None:
        # Consumed by run_export/app to fill manifest["atlas"]["surface_file"];
        # also handy standalone for anyone reading only the water_surface block.
        manifest["atlas_file"] = atlas_file
        manifest["atlas_format"] = core.ATLAS_FORMAT
    # Provenance for the NVE-derived water (no .water tile carries it any more).
    manifest.update(kvwater.water_source_manifest())
    return manifest