"""
kvterrain.core
==============

Headless engine that turns a geographic rectangle into a quad-tree pyramid of
R16 height tiles for a Unity terrain system.

Conventions (these matter, read them):

* CORNER-CENTERED / "pixel-is-a-point" sampling.
  A tile of N cells carries N+1 samples. Sample (i, j) IS the height at a fixed
  world point. Adjacent tiles SHARE their boundary samples (the last column of
  one tile equals the first column of the next), so meshed terrain has no cracks.

* SINGLE assembled array, then slice.
  The whole region is fetched into one continuous (Ny+1) x (Nx+1) float32 array.
  Tiles are sliced out of that array, so shared edges are bit-identical by
  construction rather than by trusting the server to return the same value twice.

* CENTERED ODD-KERNEL DECIMATION for parents ([1 2 1] / 4 separable).
  This is the correct downsample for corner-centered data: the kernel center sits
  on the retained even vertex, so a parent sample lands EXACTLY on world point of
  child sample 2i. A 2x2 box average (correct only for cell-centered data) would
  drift half a texel per level and break LOD edge registration -- do not use it.

* Row order is NORTH-UP (row 0 = northmost), matching the GeoTIFF the service
  returns, so the pipeline never flips. Unity's RAW import may need its "flip
  vertically" toggle depending on your setup; the manifest records the order.

* Heights packed to R16 (uint16, little-endian) against a single GLOBAL
  [height_min, height_max] range for the whole export, so all tiles at all
  levels share one vertical scale and blend cleanly.

Coordinates are handled in a single projected metre-based CRS (EUREF89 / UTM),
so "5 m spacing" is exactly 5 m and the tile grid sits on a clean integer lattice.

Data © Kartverket (CC BY 4.0).
"""

from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

IMAGESERVER = {
    "DTM": "https://hoydedata.no/arcgis/rest/services/DTM/ImageServer",
    "DOM": "https://hoydedata.no/arcgis/rest/services/DOM/ImageServer",
}
IMAGESERVER_NHM = {
    "DTM": "https://hoydedata.no/arcgis/rest/services/NHM_DTM_25833/ImageServer",
    "DOM": "https://hoydedata.no/arcgis/rest/services/NHM_DOM_25833/ImageServer",
}
POINT_API = "https://ws.geonorge.no/hoydedata/v1/punkt"
DEFAULT_MAX_FETCH_PX = 4096
ATTRIBUTION = "© Kartverket (CC BY 4.0)"


def utm_epsg_for_lon(lon_deg: float) -> int:
    if lon_deg < 12.0:
        return 25832
    if lon_deg < 30.0:
        return 25833
    return 25835


@dataclass
class GridPlan:
    epsg: int
    spacing_m: float
    origin_x: float
    origin_y: float
    tile_cells: int
    leaf_tiles_x: int
    leaf_tiles_y: int
    num_levels: int

    @property
    def leaf_cells_x(self) -> int:
        return self.leaf_tiles_x * self.tile_cells

    @property
    def leaf_cells_y(self) -> int:
        return self.leaf_tiles_y * self.tile_cells

    @property
    def samples_x(self) -> int:
        return self.leaf_cells_x + 1

    @property
    def samples_y(self) -> int:
        return self.leaf_cells_y + 1

    @property
    def width_m(self) -> float:
        return self.leaf_cells_x * self.spacing_m

    @property
    def height_m(self) -> float:
        return self.leaf_cells_y * self.spacing_m

    @property
    def bbox_utm(self) -> tuple[float, float, float, float]:
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.width_m,
            self.origin_y + self.height_m,
        )

    def fetch_pixels(self) -> int:
        return self.samples_x * self.samples_y

    def total_tiles(self) -> int:
        n = 0
        tx, ty = self.leaf_tiles_x, self.leaf_tiles_y
        for _ in range(self.num_levels):
            n += tx * ty
            tx = max(1, tx // 2)
            ty = max(1, ty // 2)
        return n


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def plan_grid(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float,
    spacing_m: float,
    tile_cells: int = 128,
    epsg: Optional[int] = None,
    snap_origin: bool = True,
) -> GridPlan:
    from pyproj import Transformer

    if epsg is None:
        epsg = utm_epsg_for_lon(0.5 * (lon_min + lon_max))

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max],
    )
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    if snap_origin:
        x0 = math.floor(x0 / spacing_m) * spacing_m
        y0 = math.floor(y0 / spacing_m) * spacing_m

    req_cells_x = max(1, math.ceil((x1 - x0) / spacing_m))
    req_cells_y = max(1, math.ceil((y1 - y0) / spacing_m))

    leaf_tiles_x = _next_pow2(math.ceil(req_cells_x / tile_cells))
    leaf_tiles_y = _next_pow2(math.ceil(req_cells_y / tile_cells))
    num_levels = int(math.log2(min(leaf_tiles_x, leaf_tiles_y))) + 1

    return GridPlan(
        epsg=epsg,
        spacing_m=float(spacing_m),
        origin_x=float(x0),
        origin_y=float(y0),
        tile_cells=int(tile_cells),
        leaf_tiles_x=int(leaf_tiles_x),
        leaf_tiles_y=int(leaf_tiles_y),
        num_levels=int(num_levels),
    )


Fetcher = Callable[[str, int, float, float, int, int, float], np.ndarray]


def export_image_fetch(
    server_url: str, epsg: int,
    sx0: float, sy0: float, nx: int, ny: int, spacing: float,
    *, source_kind: str = "DTM", session=None, timeout: int = 120,
    retries: int = 4, pause: float = 1.5,
) -> np.ndarray:
    import requests
    import rasterio

    h = spacing / 2.0
    xmin = sx0 - h
    xmax = sx0 + (nx - 1) * spacing + h
    ymin = sy0 - h
    ymax = sy0 + (ny - 1) * spacing + h

    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": str(epsg),
        "imageSR": str(epsg),
        "size": f"{nx},{ny}",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "noData": "",
        "noDataInterpretation": "esriNoDataMatchAny",
        "adjustAspectRatio": "false",
        "f": "image",
    }
    url = server_url.rstrip("/") + "/exportImage"
    sess = session or requests.Session()

    last_err = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "json" in ctype:
                raise RuntimeError(f"server error: {r.text[:400]}")
            with rasterio.open(io.BytesIO(r.content)) as ds:
                arr = ds.read(1).astype(np.float32)
                nodata = ds.nodata
            if nodata is not None and not math.isnan(nodata):
                arr[arr == nodata] = np.nan
            arr[(arr < -1000.0) | (arr > 9000.0)] = np.nan
            return arr
        except Exception as e:
            last_err = e
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"exportImage failed after {retries} tries: {last_err}")


def assemble_region(
    plan: GridPlan,
    fetcher: Fetcher,
    server_url: str,
    *,
    max_fetch_px: int = DEFAULT_MAX_FETCH_PX,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> np.ndarray:
    SX, SY = plan.samples_x, plan.samples_y
    out = np.full((SY, SX), np.nan, dtype=np.float32)

    x_starts = list(range(0, SX, max_fetch_px))
    y_starts = list(range(0, SY, max_fetch_px))
    total = len(x_starts) * len(y_starts)
    done = 0

    for jy in y_starts:
        ny = min(max_fetch_px, SY - jy)
        for ix in x_starts:
            nx = min(max_fetch_px, SX - ix)
            sx0 = plan.origin_x + ix * plan.spacing_m
            sy0 = plan.origin_y + jy * plan.spacing_m

            block = fetcher(server_url, plan.epsg, sx0, sy0, nx, ny, plan.spacing_m)
            top = SY - (jy + ny)
            out[top:top + ny, ix:ix + nx] = block

            done += 1
            if progress:
                progress(done, total, f"tile {done}/{total}")

    return out


def decimate_corner(arr: np.ndarray) -> np.ndarray:
    def _axis(a: np.ndarray, axis: int) -> np.ndarray:
        a = np.moveaxis(a, axis, -1)
        n = a.shape[-1] - 1
        assert n % 2 == 0, "axis length must be odd (cells even)"
        pad = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(1, 1)], mode="edge")
        w = np.isfinite(pad).astype(np.float32)
        v = np.where(np.isfinite(pad), pad, 0.0).astype(np.float32)
        ker = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        num = v[..., 0:-2] * ker[0] + v[..., 1:-1] * ker[1] + v[..., 2:] * ker[2]
        den = w[..., 0:-2] * ker[0] + w[..., 1:-1] * ker[1] + w[..., 2:] * ker[2]
        with np.errstate(invalid="ignore", divide="ignore"):
            full = np.where(den > 0, num / den, np.nan)
        out = full[..., 0::2]
        return np.moveaxis(out, -1, axis)

    return _axis(_axis(arr, -1), -2)


def build_pyramid(leaf: np.ndarray, num_levels: int) -> list[np.ndarray]:
    levels = [leaf]
    cur = leaf
    for _ in range(1, num_levels):
        cur = decimate_corner(cur)
        levels.append(cur)
    return levels


@dataclass
class PackResult:
    height_min: float
    height_max: float
    manifest: dict
    tiles_written: int
    # Water products, returned so a UI can preview them without re-running the
    # pipeline. `app.py` used to keep its own copy of the whole build sequence in
    # order to hold on to these; it now calls run_export and reads them here.
    water_grid: object = None       # water.WaterGrid | None
    river_net: object = None        # rivernet.RiverNetwork | None
    water_id_leaf: object = None    # np.ndarray | None
    coarse_level: object = None     # np.ndarray | None — root of the height pyramid
    # The finished leaf bed (all carves applied) and the unified water surface,
    # so a UI can show `depth = surface - terrain` — the quantity the runtime
    # actually renders, and the only honest way to see what a carve setting did.
    terrain_leaf: object = None     # np.ndarray | None
    water_surface: object = None    # np.ndarray | None — m.o.h., NaN off-water
    stage_seconds: dict = None      # {stage key -> seconds}, filled by process


def north_up_tile_slice(SY: int, tile_cells: int, tx: int, ty: int) -> tuple[int, int, int]:
    TS = tile_cells + 1
    col0 = tx * tile_cells
    j0 = ty * tile_cells
    row0 = SY - 1 - (j0 + tile_cells)
    return row0, col0, TS


# --------------------------------------------------------------------------- #
# Dense tile-atlas layout                                                      #
#                                                                              #
# A whole export can be concatenated into ONE dense blob per data type instead #
# of thousands of per-tile files, read at runtime by a pure-arithmetic byte    #
# offset. These helpers ARE the format contract: they must agree, exactly, with#
# the runtime's TilesAtLevel / offset arithmetic. They take primitives (not a  #
# GridPlan) so a consumer can recompute every offset from the manifest header  #
# alone — no per-tile offset table is written or needed.                        #
#                                                                              #
#   tileBytes           = tile_samples * tile_samples * 2                       #
#   tilesCount(l)       = tilesX(l) * tilesY(l)                                 #
#   levelByteBase(L)    = tileBytes * Σ_{l=0}^{L-1} tilesCount(l)               #
#   tileOffset(L, x, y) = levelByteBase(L) + (y * tilesX(L) + x) * tileBytes    #
#                                                                              #
# Levels are concatenated ascending (L0 finest first). Within a level, tiles   #
# are row-major: index = y * tilesX(L) + x. Intra-tile bytes are unchanged from #
# the per-tile files (LE u16, tile_samples², north->south rows).               #
# --------------------------------------------------------------------------- #

ATLAS_FORMAT = "dense_v1"
ATLAS_HEIGHT_FILE = "heights.atlas"
ATLAS_SURFACE_FILE = "surface.atlas"


def tiles_at_level(leaf_tiles_x: int, leaf_tiles_y: int, level: int) -> tuple[int, int]:
    """(tilesX, tilesY) at a pyramid level. MUST match the runtime's TilesAtLevel
    and the per-level grid used by export_tiles / export_surface_tiles."""
    return (max(1, int(leaf_tiles_x) // (2 ** level)),
            max(1, int(leaf_tiles_y) // (2 ** level)))


def atlas_tile_bytes(tile_samples: int) -> int:
    return int(tile_samples) * int(tile_samples) * 2


def atlas_level_byte_bases(
    leaf_tiles_x: int, leaf_tiles_y: int, num_levels: int, tile_samples: int,
) -> list[int]:
    """Byte offset at which each level begins (prefix sum over tile counts).
    Length == num_levels; every value is a Python int (unbounded, so no int32
    overflow — the runtime side must use a 64-bit type)."""
    tb = atlas_tile_bytes(tile_samples)
    bases: list[int] = []
    acc = 0
    for lvl in range(int(num_levels)):
        bases.append(acc)
        tx, ty = tiles_at_level(leaf_tiles_x, leaf_tiles_y, lvl)
        acc += tb * tx * ty
    return bases


def atlas_total_bytes(
    leaf_tiles_x: int, leaf_tiles_y: int, num_levels: int, tile_samples: int,
) -> int:
    """Exact size a dense atlas file must have. The export self-check asserts the
    written file matches this; a mismatch means the grid wasn't dense (ragged)."""
    tb = atlas_tile_bytes(tile_samples)
    total = 0
    for lvl in range(int(num_levels)):
        tx, ty = tiles_at_level(leaf_tiles_x, leaf_tiles_y, lvl)
        total += tb * tx * ty
    return total


def atlas_tile_offset(
    leaf_tiles_x: int, leaf_tiles_y: int, num_levels: int, tile_samples: int,
    level: int, x: int, y: int,
) -> int:
    """Byte offset of tile (x, y) at `level` inside a dense atlas."""
    tx, _ = tiles_at_level(leaf_tiles_x, leaf_tiles_y, level)
    base = atlas_level_byte_bases(
        leaf_tiles_x, leaf_tiles_y, num_levels, tile_samples)[level]
    return base + (int(y) * tx + int(x)) * atlas_tile_bytes(tile_samples)


def pack_r16(value_m: np.ndarray, hmin: float, hmax: float) -> np.ndarray:
    rng = max(hmax - hmin, 1e-6)
    norm = (value_m - hmin) / rng
    norm = np.clip(norm, 0.0, 1.0)
    return np.rint(norm * 65535.0).astype("<u2")


def export_tiles(
    plan: GridPlan,
    levels: list[np.ndarray],
    out_dir: str,
    *,
    nodata_fill_m: float = 0.0,
    height_min: Optional[float] = None,
    height_max: Optional[float] = None,
    source_kind: str = "DTM",
    atlas_name: str = ATLAS_HEIGHT_FILE,
) -> PackResult:
    import os

    TC = plan.tile_cells
    TS = TC + 1

    leaf = levels[0]
    finite = leaf[np.isfinite(leaf)]
    if height_min is None:
        height_min = float(finite.min()) if finite.size else 0.0
    if height_max is None:
        height_max = float(finite.max()) if finite.size else 1.0
    if height_max <= height_min:
        height_max = height_min + 1.0

    os.makedirs(out_dir, exist_ok=True)
    tiles_written = 0

    # One dense blob for the whole export. Tiles are written in level-then-row-major order
    # (L0 finest first; within a level, y outer, x inner), which is exactly the §4 offset
    # layout — so a plain sequential append reproduces it with no seeking. The runtime reads
    # any tile back by pure arithmetic (levelByteBase[L] + (y*tilesX + x)*tileBytes).
    with open(os.path.join(out_dir, atlas_name), "wb") as atlas_fh:
        for lvl, arr in enumerate(levels):
            tiles_x, tiles_y = tiles_at_level(plan.leaf_tiles_x, plan.leaf_tiles_y, lvl)
            arr_filled = np.where(np.isfinite(arr), arr, nodata_fill_m)
            SY = arr.shape[0]
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    r_lo, i0, _ = north_up_tile_slice(SY, TC, tx, ty)
                    sub = arr_filled[r_lo:r_lo + TS, i0:i0 + TS]
                    assert sub.shape == (TS, TS), f"bad slice {sub.shape} L{lvl} {tx},{ty}"
                    pack_r16(sub, height_min, height_max).tofile(atlas_fh)
                    tiles_written += 1

    # Fail LOUDLY if the grid wasn't dense: the runtime reads by pure arithmetic and a
    # short/long file would silently misalign every tile past the gap. Guardrail for the
    # ragged-edge (border) case — though this exporter is dense by construction.
    expect = atlas_total_bytes(plan.leaf_tiles_x, plan.leaf_tiles_y, plan.num_levels, TS)
    actual = os.path.getsize(os.path.join(out_dir, atlas_name))
    if actual != expect:
        raise RuntimeError(
            f"dense atlas size mismatch: {atlas_name} is {actual} bytes, expected "
            f"{expect} (tileBytes={atlas_tile_bytes(TS)} * "
            f"{expect // atlas_tile_bytes(TS)} tiles). The tile grid was not dense — "
            f"the arithmetic offset path would misalign.")

    manifest = {
        "format": "kvterrain-atlas-r16/1",
        "attribution": ATTRIBUTION,
        "source": source_kind,
        "crs": f"EPSG:{plan.epsg}",
        "row_order": "north_to_south",
        "byte_order": "little_endian",
        "bits": 16,
        "origin_utm": [plan.origin_x, plan.origin_y],
        "leaf_spacing_m": plan.spacing_m,
        "tile_cells": TC,
        "tile_samples": TS,
        "leaf_tiles": [plan.leaf_tiles_x, plan.leaf_tiles_y],
        "num_levels": plan.num_levels,
        "height_min_m": height_min,
        "height_max_m": height_max,
        "nodata_fill_m": nodata_fill_m,
        "region_bbox_utm": list(plan.bbox_utm),
        # The whole tile pyramid is implied by the header (leaf_tiles, tile_cells,
        # num_levels) via TilesAtLevel + the §4 offset formula; no per-tile table is stored.
        "atlas": {
            "format": ATLAS_FORMAT,
            "height_file": atlas_name,
            "tile_bytes": atlas_tile_bytes(TS),
            "height_bytes": expect,
        },
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return PackResult(height_min, height_max, manifest, tiles_written)


def run_export(
    plan: GridPlan,
    out_dir: str,
    *,
    source_kind: str = "DTM",
    fetcher: Optional[Fetcher] = None,
    server_url: Optional[str] = None,
    max_fetch_px: int = DEFAULT_MAX_FETCH_PX,
    nodata_fill_m: float = 0.0,
    height_min: Optional[float] = None,
    height_max: Optional[float] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    include_water: bool = False,
    water_opts: Optional[dict] = None,
) -> PackResult:
    """
    One-shot fetch + process: the original single-pass build, kept intact.

    The tool is now split in two — `kvterrain.fetch` writes a dataset,
    `kvterrain.process` turns one into an export — because refetching twenty
    square kilometres of LiDAR to retune a lake carve was the dominant cost of
    working on this pipeline. This function is the two of them back to back with
    nothing stored in between, for callers that genuinely want one pass (`cli
    build`, and anything that was calling it before the split).

    Prefer `fetch.run_fetch` + `process.process_dataset` when you expect to
    process the same region more than once.
    """
    from . import process as _process

    if fetcher is None:
        fetcher = lambda u, e, sx, sy, nx, ny, sp: export_image_fetch(
            u, e, sx, sy, nx, ny, sp, source_kind=source_kind)
    if server_url is None:
        server_url = IMAGESERVER[source_kind]

    leaf = assemble_region(plan, fetcher, server_url,
                           max_fetch_px=max_fetch_px, progress=progress)

    features = None
    if include_water:
        from . import water as _water
        wopts = dict(water_opts or {})
        # `features` (pre-built) and `fetcher` (a WaterFetcher) are the two ways a
        # caller can supply water without the network — the demo path uses the
        # second. Everything else in the dict is a process-stage setting and is
        # passed straight through; `include_main_rivers` is READ here (it decides
        # whether the hovedelv layer is queried) but left in place, because
        # `run_process` records it and drops it.
        features = wopts.pop("features", None)
        water_fetcher = wopts.pop("fetcher", None)
        if features is None:
            if water_fetcher is not None:
                features = water_fetcher(plan)
            else:
                features = _water.fetch_water_features(
                    plan, progress=progress,
                    include_main_rivers=bool(wopts.get("include_main_rivers", True)))
        water_opts = wopts

    # `run_process` reports a fraction; this function's callers were written
    # against the (done, total, msg) fetch callback, so keep that contract.
    def _proc_progress(frac: float, label: str) -> None:
        if progress:
            progress(int(round(frac * 1000)), 1000, label)

    return _process.run_process(
        plan, leaf, out_dir,
        features=features,
        source_kind=source_kind,
        nodata_fill_m=nodata_fill_m,
        height_min=height_min,
        height_max=height_max,
        include_water=include_water,
        water_opts=water_opts,
        progress=_proc_progress,
    )
