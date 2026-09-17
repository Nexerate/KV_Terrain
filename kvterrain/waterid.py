"""
kvterrain.waterid
=================

The per-pixel WATER ID raster: one u16 channel that carries both the water class
(Dry / River / Ocean / Lake) and, for lakes, the AUTHORED lake identity.

    0        Dry
    1        River
    2        Ocean
    3..15    reserved
    16 + n   Lake, where n is the local lake id in WaterGrid.lake_table

Shader side:  isLake = v >= 16 ;  isRiver = v == 1 ;  isOcean = v == 2
Lake record:  lake_id = v - 16  ->  lakes.json entry -> authored `hoyde`

WHY ONE CHANNEL AND NOT TWO
---------------------------
The architecture calls for a `WaterClass : uint8 { Dry, Lake, River, Ocean }`
raster for the clip shader and the invention guard. Separately, the runtime needs
to know WHICH authored lake a pixel belongs to, in order to pin that lake's level
to the basin the solver discovers.

That second requirement is the binding one, and it is not optional. Basin ids are
derived fresh by the GPU on every solve; they are not authored and this tool
cannot emit them. The pin test (`pinned = hasAuthoredLevel AND NOT disturbed`)
has to answer "is this basin an authored lake, and which one" — and the mapping
is not 1:1, since at 16 m a single NVE lake can straddle several basins and a
basin can swallow several small lakes. The only way to resolve it is a per-basin
histogram over the authored lake ids carried by that basin's cells, which is a
per-pixel lookup by construction. A bounding box cannot do it: bboxes overlap
badly for lake clusters and are a poor proxy for the long, narrow, fjord-shaped
lakes that dominate this terrain.

Given that a u16 lake-id channel is required anyway, a separate u8 class channel
is redundant — the id already encodes "is lake" in its zero/non-zero split. So
class and identity are merged into one u16:

    class-only + lake-id  =  1 + 2 = 3 bytes/sample
    u16 class + u16 id    =      2 + 2 = 4 bytes/sample
    THIS                  =              2 bytes/sample

and, more valuably, 2 bytes/sample is exactly what the height and surface atlases
use, so a water-id tile sits at the IDENTICAL byte offset as its height and
surface twins: one offset computation for all three. (`core.atlas_*` later gained
a bytes-per-sample parameter for the 4-byte `labels.atlas`; its default of 2 is
this atlas's.)

This is a deliberate deviation from the letter of the spec'd uint8 enum. It
satisfies every stated USE of that enum and adds the identity the pin requires.

DOWNSAMPLE
----------
Categorical, never averaged. Precedence Lake > River > Ocean > Dry, matching the
rasteriser's lake > river priority; among lakes the id held by the most children
wins. The gather is the same corner-anchored 3x3 as
`watersurface.downsample_surface_water_only`, so a water-id sample and its height
twin stay on the same world point at every level.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import core
from . import water as kvwater

CODE_DRY = 0
CODE_RIVER = 1
CODE_OCEAN = 2
LAKE_ID_BASE = 16          # first code used by lakes; 3..15 reserved

WATER_ID_FILE = "water_id.atlas"
WATER_ID_NODATA = CODE_DRY

# Kartverket heights are metres above sea level, so real sea level is 0. (The
# consuming world renders its ocean plane at a different height in ITS units;
# that is a runtime concern and must not leak into the source data.)
DEFAULT_OCEAN_LEVEL_M = 0.0

assert LAKE_ID_BASE + kvwater.MAX_LOCAL_LAKE_ID <= 65535, \
    "lake id range overflows u16 — water.MAX_LOCAL_LAKE_ID and LAKE_ID_BASE disagree"


def ocean_mask_from_edges(
    height_m: np.ndarray,
    ocean_level_m: float = DEFAULT_OCEAN_LEVEL_M,
    *,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Ocean = the below-sea-level region CONNECTED TO A MAP EDGE, not simply every
    cell below sea level.

    The distinction matters in Norway specifically. A plain threshold would tag
    every inland below-sea-level depression as ocean — closed kettle holes, quarry
    floors, and the many sub-sea-level lake beds this very pipeline just carved
    (a 20 m carve under a lake sitting at 8 m.o.h. lands at -12 m). Those are not
    the sea. Flood-filling inward from the border keeps the connectivity the
    architecture asks for: the ocean is a fixed sink that connects to the map
    edges.

    `exclude` (typically the lake/river mask) is treated as a barrier AND removed
    from the result, so authored freshwater is never relabelled ocean.
    """
    from scipy.ndimage import label

    h = np.asarray(height_m)
    below = np.isfinite(h) & (h <= float(ocean_level_m))
    if exclude is not None:
        below &= ~np.asarray(exclude, dtype=bool)
    if not below.any():
        return np.zeros(h.shape, dtype=bool)

    lbl, n = label(below)          # 4-connected; diagonal leaks are not ocean
    if n == 0:
        return np.zeros(h.shape, dtype=bool)

    edge = np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])
    keep = np.unique(edge[edge > 0])
    if keep.size == 0:
        return np.zeros(h.shape, dtype=bool)

    return np.isin(lbl, keep)


def build_water_id(
    wg: kvwater.WaterGrid,
    ocean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fold the water grid into the u16 code raster. Precedence, lowest first:
    Dry -> Ocean -> River -> Lake, so a lake polygon always wins its pixels and a
    river always beats the sea (a tidal reach stays a river).
    """
    out = np.full(wg.type.shape, CODE_DRY, dtype=np.uint16)

    if ocean is not None:
        out[np.asarray(ocean, dtype=bool)] = CODE_OCEAN

    out[wg.type == kvwater.TYPE_RIVER] = CODE_RIVER

    is_lake = wg.type == kvwater.TYPE_LAKE
    if is_lake.any():
        lid = wg.lake_id[is_lake].astype(np.uint32)
        if lid.size and int(lid.max()) > kvwater.MAX_LOCAL_LAKE_ID:
            raise ValueError(
                f"local lake id {int(lid.max())} exceeds "
                f"{kvwater.MAX_LOCAL_LAKE_ID}; cannot encode in u16 water_id")
        out[is_lake] = (lid + LAKE_ID_BASE).astype(np.uint16)

    return out


def decode_water_id(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(class_array, lake_id_array) for tests/validation. class uses the spec'd
    enum ordering Dry=0, Lake=1, River=2, Ocean=3."""
    code = np.asarray(code)
    cls = np.zeros(code.shape, dtype=np.uint8)
    cls[code == CODE_RIVER] = 2
    cls[code == CODE_OCEAN] = 3
    is_lake = code >= LAKE_ID_BASE
    cls[is_lake] = 1
    lake_id = np.zeros(code.shape, dtype=np.uint16)
    lake_id[is_lake] = (code[is_lake].astype(np.uint32) - LAKE_ID_BASE).astype(np.uint16)
    return cls, lake_id


def downsample_water_id(w: np.ndarray) -> np.ndarray:
    """
    One categorical pyramid step. Corner-anchored 3x3 gather (parent sample i sits
    on child sample 2i), precedence Lake > River > Ocean > Dry, majority id among
    lakes. Never averages — averaging two lake ids yields a third, nonexistent
    lake.
    """
    SY, SX = w.shape
    oy, ox = (SY - 1) // 2 + 1, (SX - 1) // 2 + 1

    kids = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = np.clip(np.arange(oy) * 2 + dr, 0, SY - 1)
            c = np.clip(np.arange(ox) * 2 + dc, 0, SX - 1)
            kids.append(w[np.ix_(r, c)])
    stack = np.stack(kids)                      # (9, oy, ox)

    out = np.full((oy, ox), CODE_DRY, dtype=np.uint16)
    out[np.any(stack == CODE_OCEAN, axis=0)] = CODE_OCEAN
    out[np.any(stack == CODE_RIVER, axis=0)] = CODE_RIVER

    is_lake = stack >= LAKE_ID_BASE
    any_lake = np.any(is_lake, axis=0)
    if any_lake.any():
        best_cnt = np.zeros((oy, ox), dtype=np.int16)
        best_id = np.zeros((oy, ox), dtype=np.uint16)
        for k in range(9):
            cand = stack[k]
            cnt = np.zeros((oy, ox), dtype=np.int16)
            for j in range(9):
                cnt += ((stack[j] == cand) & is_lake[j]).astype(np.int16)
            take = is_lake[k] & (cnt > best_cnt)
            best_cnt[take] = cnt[take]
            best_id[take] = cand[take]
        out[any_lake] = best_id[any_lake]

    return out


def build_water_id_pyramid(leaf: np.ndarray, num_levels: int) -> list:
    levels = [np.asarray(leaf, dtype=np.uint16)]
    for _ in range(1, num_levels):
        levels.append(downsample_water_id(levels[-1]))
    return levels


def export_water_id_tiles(
    plan: core.GridPlan, id_levels: list, out_dir: str, *,
    atlas_name: str = WATER_ID_FILE,
) -> dict:
    """
    Slice every water-id level into (tile_cells+1)^2 u16 tiles, concatenated into
    one dense blob with the SAME layout as the height atlas — so the runtime reads
    a water-id tile at the identical byte offset as its height twin.
    """
    TC = plan.tile_cells
    TS = TC + 1

    os.makedirs(out_dir, exist_ok=True)
    tiles_written = 0
    with open(os.path.join(out_dir, atlas_name), "wb") as fh:
        for lvl, arr in enumerate(id_levels):
            tiles_x, tiles_y = core.tiles_at_level(
                plan.leaf_tiles_x, plan.leaf_tiles_y, lvl)
            SY = arr.shape[0]
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    r0, c0, _ = core.north_up_tile_slice(SY, TC, tx, ty)
                    tile = np.ascontiguousarray(
                        arr[r0:r0 + TS, c0:c0 + TS], dtype="<u2")
                    assert tile.shape == (TS, TS), \
                        f"bad water_id slice {tile.shape} L{lvl} {tx},{ty}"
                    tile.tofile(fh)
                    tiles_written += 1

    expect = core.atlas_total_bytes(
        plan.leaf_tiles_x, plan.leaf_tiles_y, plan.num_levels, TS)
    actual = os.path.getsize(os.path.join(out_dir, atlas_name))
    if actual != expect:
        raise RuntimeError(
            f"dense water_id atlas size mismatch: {atlas_name} is {actual} "
            f"bytes, expected {expect}. The water_id tile grid was not dense.")

    return {
        "atlas_file": atlas_name,
        "atlas_format": core.ATLAS_FORMAT,
        "dtype": "u16le",
        "row_order": "north_to_south",
        "encoding": {
            "dry": CODE_DRY,
            "river": CODE_RIVER,
            "ocean": CODE_OCEAN,
            "reserved": "3..15",
            "lake_id_base": LAKE_ID_BASE,
            "lake": f"code >= {LAKE_ID_BASE}; local lake id = code - {LAKE_ID_BASE}",
        },
        "runtime": (
            "isLake = v >= 16; isRiver = v == 1; isOcean = v == 2; "
            "lake_id = v - 16 keys into lakes.json for the authored level. "
            "Tile byte offsets are identical to the height atlas."),
        "downsample": "categorical; precedence lake > river > ocean > dry, "
                      "majority lake id; never averaged",
        "tiles_written": tiles_written,
    }