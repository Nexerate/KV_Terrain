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
    if fetcher is None:
        fetcher = lambda u, e, sx, sy, nx, ny, sp: export_image_fetch(
            u, e, sx, sy, nx, ny, sp, source_kind=source_kind)
    if server_url is None:
        server_url = IMAGESERVER[source_kind]

    leaf = assemble_region(plan, fetcher, server_url,
                           max_fetch_px=max_fetch_px, progress=progress)

    surface_levels = None
    water_id_levels = None
    water_leaf = None
    river_net = None
    water_params: dict = {}
    lake_count = 0
    carve_depth_m = 0.0
    lake_ramp_m = 0.0
    lake_min_depth_m = 0.0
    lake_slope = 0.0
    snap_report: dict = {}
    emit_geojson = False

    if include_water:
        from . import water, bathymetry, watersurface, waterid, rivernet
        opts = dict(water_opts or {})
        features = opts.pop("features", None)
        fetcher_water = opts.pop("fetcher", None)
        # Lake carve: deepen inward toward a maximum REAL depth over a long ramp,
        # so how deep a lake gets is decided by how big it is.
        carve_depth_m = float(opts.pop("lake_max_depth_m", bathymetry.DEFAULT_CARVE_DEPTH_M))
        lake_slope = float(opts.pop("lake_shore_slope", bathymetry.DEFAULT_SHORE_SLOPE))
        lake_ramp_m = float(opts.pop("lake_ramp_radius_m", 0.0))
        if "lake_bevel_px" in opts:        # legacy alias -> ramp width in metres
            lake_ramp_m = max(1.0, float(opts.pop("lake_bevel_px"))) * plan.spacing_m
        lake_min_depth_m = float(opts.pop("lake_min_depth_m", bathymetry.DEFAULT_MIN_DEPTH_M))
        lake_snap_px = int(opts.pop("lake_snap_px", water.DEFAULT_LAKE_SNAP_PX))
        depth_scale = float(opts.pop("river_depth_scale", 1.0))
        bank_tol_m = float(opts.pop("river_bank_tolerance_m",
                                    bathymetry.DEFAULT_BANK_TOLERANCE_M))
        ocean_level_m = float(opts.pop("ocean_level_m", waterid.DEFAULT_OCEAN_LEVEL_M))
        vertex_stride_m = float(opts.pop("river_vertex_stride_m", plan.spacing_m))
        emit_geojson = bool(opts.pop("emit_geojson", False))
        estimate_lake_levels = bool(opts.pop("estimate_lake_levels", True))
        perimeter_cap = bool(opts.pop("lake_perimeter_cap", True))
        fill_lake_holes = bool(opts.pop("fill_lake_holes", True))

        # Pop the rasterise-only keys so they never leak into fetch_water_features
        # (which doesn't accept them). What remains in `opts` is fetch kwargs only.
        raster_opts = {}
        for k in ("width_by_order", "width_scale", "all_touched_rivers"):
            if k in opts:
                raster_opts[k] = opts.pop(k)

        if features is None:
            if fetcher_water is not None:
                features = fetcher_water(plan)
            else:
                features = water.fetch_water_features(plan, progress=progress, **opts)

        water_leaf = water.rasterize_water(
            plan, features, fill_lake_holes=fill_lake_holes, **raster_opts)
        lake_count = len(water_leaf.lake_table)

        # 1. VOID REPAIR, before anything reads the bed. Kartverket returns a void
        #    lake interior as a finite 0/sentinel rather than NaN, so an isfinite
        #    test misses it entirely. Left in place it becomes the export's
        #    height_min and the shore ring keeps a fake pit right at the waterline.
        #    The carve below hides interior voids as a side effect but cannot reach
        #    the ring outside the polygon, and does nothing for lakes with neither
        #    an NVE hoyde nor a usable shoreline. See bathymetry's module docstring.
        leaf = bathymetry.fill_lake_surface(leaf, water_leaf.type)

        # 1a2. SNAP each lake to its own flat water surface in the DTM. The NVE
        #      outline and the LiDAR block are independent products, so a one- or
        #      two-texel ring of the lake's own water plane falls outside the
        #      polygon and would render as a raised rim tracing the true shoreline
        #      just outside the water. See snap_lakes_to_flat_water.
        snap_report = water.snap_lakes_to_flat_water(
            water_leaf, leaf, max_px=lake_snap_px)

        # 1b. SNAPSHOT the repaired bed before ANY carve. Two consumers need it:
        #     the lake level estimator (which must read the LiDAR water surface,
        #     not a bowl we dug) and the river water level, whose whole model is
        #     "the water sits at the lowest ground the channel has here".
        leaf_uncarved = leaf.copy()

        # 1c. RESOLVE EVERY LAKE'S LEVEL, once, before anything reads one. NVE
        #     `hoyde` VERBATIM where NVE publishes one — it is the authored level
        #     and the DTM does not get a vote on it. Only for a lake without one is
        #     a level read off the (now repaired, still uncarved) DTM inside the
        #     polygon, and that estimate is capped by the ground ringing the lake.
        #     Everything downstream — the carve, the surface raster, the river
        #     tie-in, lakes.json — goes through WaterGrid.lake_level, so they cannot
        #     drift apart.
        level_report = water.apply_estimated_levels(
            water_leaf, leaf_uncarved, enabled=estimate_lake_levels,
            island=water_leaf.lake_island, perimeter_cap=perimeter_cap)
        lake_surf = water_leaf.lake_surface_moh()

        # 2. THE RIVER WATER LEVEL, from the uncarved ground: the lowest ground
        #    across each channel, levelled over that channel's own width, then
        #    raised (never lowered) to a lake's surface where the two meet. This
        #    has to come BEFORE the carve, because the carve stops at the waterline
        #    this defines — otherwise the trench is cut wherever the rasterised
        #    buffer went, cliff faces included.
        river_surf = watersurface.river_surface_moh(
            plan, features, water_leaf, leaf_uncarved,
            lake_surface=lake_surf,
            width_by_order=raster_opts.get("width_by_order"),
            width_scale=raster_opts.get("width_scale", 1.0),
            bank_tolerance_m=bank_tol_m)

        # 2b. Carve the river channels. The water surface sits on the ground, so
        #     this trench is the entire water column — see carve_river_beds.
        leaf = bathymetry.carve_river_beds(
            leaf, water_leaf.type, water_leaf.weight, plan.spacing_m,
            level=river_surf,
            bank_tolerance_m=bank_tol_m,
            depth_scale=depth_scale,
            width_by_order=raster_opts.get("width_by_order"),
            width_scale=raster_opts.get("width_scale", 1.0),
        )

        # 2c. SNAPSHOT the river-carved but LAKE-uncarved bed, for the polylines.
        #     The lake carve writes a fabricated bowl (carve_depth_m under the
        #     authored surface) inside every lake polygon. That bowl is a display
        #     device — it exists so the water plane does not z-fight the
        #     LiDAR-flattened lake surface — and it is NOT terrain. The runtime
        #     burns polyline `z` straight into bedConditioned, so a `z` sampled
        #     from the carved array cuts the solver's routing grid to a fabricated
        #     depth at exactly the points where a channel hands off to a lake
        #     basin. The river trench, by contrast, IS the channel the burn is
        #     trying to open, so `z` is sampled after it. Sampling this array does
        #     not weaken the polyline/raster agreement, because `level` is read
        #     verbatim out of `water_surface` (see build_river_network) rather than
        #     recomputed from `z`.
        leaf_bed_uncarved = leaf.copy()

        # 3. Carve the lake bowls so the water plane does not z-fight the
        #    LiDAR-flattened lake surface. Safe for the solver only because the lake
        #    level is authored and pinned — see carve_lake_beds.
        #
        #    `estimate_missing` is tied to the SAME switch that governs the surface
        #    raster. It must never be true while the raster is empty: the carve's
        #    private shore-estimate fallback is exactly how lakes without an NVE
        #    hoyde ended up as 20 m dry pits. Either both know a level, or neither
        #    touches the lake.
        leaf = bathymetry.carve_lake_beds(
            leaf,
            water_leaf.type,
            plan.spacing_m,
            surface_moh=lake_surf,
            carve_depth_m=carve_depth_m,
            shore_slope=lake_slope,
            ramp_m=lake_ramp_m or None,
            min_depth_m=lake_min_depth_m,
            island=water_leaf.lake_island,
            estimate_missing=False,
        )

        # 4. Unified water surface (lakes = authored level, rivers = the channel
        #    level from step 2), then its water-only pyramid.
        water_surface = watersurface.combine_water_surface(lake_surf, river_surf)
        surface_levels = watersurface.build_surface_pyramid(water_surface, plan.num_levels)

        # 5. Class + authored lake identity. Ocean is flood-filled inward from the
        #    map edges, not thresholded, so inland sub-sea-level ground (including
        #    the bowls just carved) is never mislabelled sea.
        ocean = waterid.ocean_mask_from_edges(
            leaf, ocean_level_m,
            exclude=(water_leaf.type != water.TYPE_LAND))
        water_id_leaf = waterid.build_water_id(water_leaf, ocean)
        water_id_levels = waterid.build_water_id_pyramid(water_id_leaf, plan.num_levels)

        # 6. The polylines. `z` is sampled from the river-carved, lake-UNCARVED bed
        #    (step 2b) because the runtime burns it; `level` is read verbatim from
        #    `water_surface`, so polyline level still equals the raster surface
        #    pixel-for-pixel.
        river_net = rivernet.build_river_network(
            plan, features, water_leaf, leaf_bed_uncarved,
            water_surface=water_surface,
            vertex_stride_m=vertex_stride_m, depth_scale=depth_scale)

        water_params = {
            "river_width_scale": raster_opts.get("width_scale", 1.0),
            "river_depth_scale": depth_scale,
            "river_bank_tolerance_m": bank_tol_m,
            "river_vertex_stride_m": vertex_stride_m,
            "lake_carve_depth_m": carve_depth_m,
            "lake_shore_slope": lake_slope,
            "lake_shore_ramp_m": lake_ramp_m or (carve_depth_m / max(lake_slope, 1e-6)),
            "lake_min_depth_m": lake_min_depth_m,
            "lake_snap_px": lake_snap_px,
            "lake_holes_filled": fill_lake_holes,
            "ocean_level_m": ocean_level_m,
            "include_main_rivers": opts.get("include_main_rivers", True),
            "estimate_lake_levels": estimate_lake_levels,
            "lake_perimeter_cap": perimeter_cap,
        }

    levels = build_pyramid(leaf, plan.num_levels)
    res = export_tiles(plan, levels, out_dir,
                       nodata_fill_m=nodata_fill_m,
                       height_min=height_min, height_max=height_max,
                       source_kind=source_kind)
    res.water_grid = water_leaf
    res.river_net = river_net
    res.water_id_leaf = water_id_levels[0] if water_id_levels else None
    res.coarse_level = levels[-1]

    if surface_levels is not None:
        import os
        from . import watersurface, waterid, rivernet

        # The surface atlas packs on the SAME [height_min, height_max] as the height atlas,
        # so it can only be written now that export_tiles has finalised that range.
        wsurf = watersurface.export_surface_tiles(
            plan, surface_levels, out_dir, res.height_min, res.height_max)
        # Name the surface atlas in the header so the runtime opens the second handle.
        res.manifest["atlas"]["surface_file"] = wsurf["atlas_file"]
        wsurf["lake_count"] = lake_count
        wsurf["lake_levels"] = level_report
        wsurf["lake_bathymetry"] = {
            "enabled": True,
            "carve_depth_m": float(carve_depth_m),
            "shore_slope_m_per_m": float(lake_slope),
            "shore_ramp_m": float(lake_ramp_m or (carve_depth_m / max(lake_slope, 1e-6))),
            "min_depth_m": float(lake_min_depth_m),
            "method": "linear_ramp_below_known_surface",
            "ramp_anchor": "waterline",
            "ramp_profile_m": "a STRAIGHT ramp: depth = carve_depth_m * clip((metres "
                              "from the waterline) / shore_ramp_m, 0, 1), flat from "
                              "there inward. shore_ramp_m = carve_depth_m / "
                              "shore_slope_m_per_m, so at slope 1.0 a 20 m bed takes "
                              "20 m of shore. A body too small to reach carve_depth_m "
                              "has its profile scaled up just far enough to reach "
                              "min_depth_m at its deepest sample",
            "shoreline_snap": snap_report,
            "islands": ("lake polygon holes are rasterised as lake so the surface "
                        "runs through them, but their DTM height is left alone and "
                        "they act as shore for the ramp; depth = max(0, level - "
                        "terrain) hides them"),
            "rationale": "LiDAR returns the lake SURFACE as terrain height, so "
                         "without a carve the water plane and the terrain are "
                         "coincident. Safe for the solver: ramped, never breaks "
                         "the rim, and the lake level is authored and pinned.",
        }
        wsurf["river_bathymetry"] = {
            "enabled": True,
            "depth_scale": float(water_params.get("river_depth_scale", 1.0)),
            "method": "trench_under_the_channel_scaled_by_modelled_width",
            "bank_tolerance_m": float(water_params.get("river_bank_tolerance_m", 0.0)),
            "level": "the ground under the channel's own centreline, spread flat "
                     "across the samples that centreline seeded — so the water "
                     "surface is level ACROSS a channel and descends ALONG it",
            "profile": "depth = depth_by_order(order) * smoothstep(distance from "
                       "bank / modelled channel half-width), cut in full while the "
                       "sample is within bank_tolerance_m of the water level and "
                       "tapering to nothing at twice that; a one-sample channel "
                       "takes full depth as a step",
            "rationale": "the river water surface is the uncarved ground under the "
                         "channel, so the trench is the entire water column. Carving "
                         "down rather than raising the surface keeps a river in its "
                         "landscape instead of on top of it; levelling across the "
                         "channel and stopping the carve at the waterline keeps it "
                         "off the cliffs beside it.",
        }
        res.manifest["water_surface"] = wsurf

        wid = waterid.export_water_id_tiles(plan, water_id_levels, out_dir)
        res.manifest["atlas"]["water_id_file"] = wid["atlas_file"]
        res.manifest["water_id"] = wid

        res.manifest["water_vector"] = rivernet.export_river_network(
            river_net, plan, out_dir, emit_geojson=emit_geojson)

        res.manifest["generator"] = rivernet.generator_metadata(plan, {
            "source_kind": source_kind,
            "nodata_fill_m": nodata_fill_m,
            "height_min_m": res.height_min,
            "height_max_m": res.height_max,
            "tile_cells": plan.tile_cells,
            "water": water_params,
        })

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(res.manifest, f, indent=2)
    else:
        import os
        from . import rivernet
        res.manifest["generator"] = rivernet.generator_metadata(plan, {
            "source_kind": source_kind,
            "nodata_fill_m": nodata_fill_m,
            "height_min_m": res.height_min,
            "height_max_m": res.height_max,
            "tile_cells": plan.tile_cells,
            "water": None,
        })
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(res.manifest, f, indent=2)

    return res