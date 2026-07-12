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


def quadkey(level: int, x: int, y: int, num_levels: int) -> str:
    depth = (num_levels - 1) - level
    if depth <= 0:
        return ""
    key = []
    for b in range(depth - 1, -1, -1):
        digit = 0
        if (x >> b) & 1:
            digit += 1
        if (y >> b) & 1:
            digit += 2
        key.append(str(digit))
    return "".join(key)


@dataclass
class PackResult:
    height_min: float
    height_max: float
    manifest: dict
    tiles_written: int


def north_up_tile_slice(SY: int, tile_cells: int, tx: int, ty: int) -> tuple[int, int, int]:
    TS = tile_cells + 1
    col0 = tx * tile_cells
    j0 = ty * tile_cells
    row0 = SY - 1 - (j0 + tile_cells)
    return row0, col0, TS


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
    writer: Optional[Callable[[str, np.ndarray], None]] = None,
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

    def _default_writer(path: str, a16: np.ndarray) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        a16.tofile(path)

    write = writer or _default_writer
    manifest_levels = []
    tiles_written = 0

    for lvl, arr in enumerate(levels):
        spacing = plan.spacing_m * (2 ** lvl)
        tiles_x = max(1, plan.leaf_tiles_x // (2 ** lvl))
        tiles_y = max(1, plan.leaf_tiles_y // (2 ** lvl))
        arr_filled = np.where(np.isfinite(arr), arr, nodata_fill_m)

        level_entry = {
            "level": lvl,
            "spacing_m": spacing,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "tile_samples": TS,
            "tiles": [],
        }

        SY = arr.shape[0]
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                i0 = tx * TC
                j0 = ty * TC
                r_lo, i0, _ = north_up_tile_slice(SY, TC, tx, ty)
                sub = arr_filled[r_lo:r_lo + TS, i0:i0 + TS]
                assert sub.shape == (TS, TS), f"bad slice {sub.shape} L{lvl} {tx},{ty}"
                a16 = pack_r16(sub, height_min, height_max)
                rel = f"L{lvl}/{tx}_{ty}.r16"
                write(os.path.join(out_dir, rel), a16)
                tiles_written += 1

                bx0 = plan.origin_x + i0 * spacing
                by0 = plan.origin_y + j0 * spacing
                level_entry["tiles"].append({
                    "x": tx, "y": ty,
                    "key": quadkey(lvl, tx, ty, plan.num_levels),
                    "file": rel,
                    "bbox_utm": [bx0, by0, bx0 + TC * spacing, by0 + TC * spacing],
                })
        manifest_levels.append(level_entry)

    manifest = {
        "format": "kvterrain-quadtree-r16/1",
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
        "levels": manifest_levels,
    }

    if writer is None:
        os.makedirs(out_dir, exist_ok=True)
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

    water_section = None
    surface_levels = None
    if include_water:
        from . import water, bathymetry, watersurface
        opts = dict(water_opts or {})
        features = opts.pop("features", None)
        fetcher_water = opts.pop("fetcher", None)
        # New model: flat carve to a fixed REAL depth below the (known) lake surface.
        carve_depth_m = float(opts.pop("lake_max_depth_m", bathymetry.DEFAULT_CARVE_DEPTH_M))
        bevel_px = int(opts.pop("lake_bevel_px", bathymetry.DEFAULT_BEVEL_PX))
        if "lake_ramp_radius_m" in opts:   # legacy alias -> bevel width in texels
            bevel_px = max(1, int(round(float(opts.pop("lake_ramp_radius_m")) / plan.spacing_m)))
        depth_scale = float(opts.pop("river_depth_scale", 1.0))

        if features is None:
            if fetcher_water is not None:
                features = fetcher_water(plan)
            else:
                features = water.fetch_water_features(plan, progress=progress, **opts)

        raster_opts = {}
        for k in ("width_by_order", "width_scale", "all_touched_rivers"):
            if k in opts:
                raster_opts[k] = opts[k]

        water_leaf = water.rasterize_water(plan, features, **raster_opts)
        lake_surf = water_leaf.lake_surface_moh()          # authoritative NVE hoyde
        leaf = bathymetry.carve_lake_beds(
            leaf,
            water_leaf.type,
            plan.spacing_m,
            surface_moh=lake_surf,
            carve_depth_m=carve_depth_m,
            bevel_px=bevel_px,
        )
        # Unified water surface (lakes = hoyde, rivers = DTM-estimated along channel),
        # then its water-only pyramid. Rivers sample the *leaf* bed, so this must run
        # on `leaf` before the height pyramid is built.
        river_surf = watersurface.river_surface_moh(
            plan, features, water_leaf, leaf,
            depth_scale=depth_scale, lake_surface=lake_surf)
        water_surface = watersurface.combine_water_surface(lake_surf, river_surf)
        surface_levels = watersurface.build_surface_pyramid(water_surface, plan.num_levels)

        water_levels = water.build_water_pyramid(water_leaf, plan.num_levels)
        wres = water.export_water_tiles(
            plan,
            water_levels,
            out_dir,
            width_scale=float(opts.get("width_scale", 1.0)),
        )
        water_section = wres.water_manifest

    levels = build_pyramid(leaf, plan.num_levels)
    res = export_tiles(plan, levels, out_dir,
                       nodata_fill_m=nodata_fill_m,
                       height_min=height_min, height_max=height_max,
                       source_kind=source_kind)

    if water_section is not None:
        res.manifest["water"] = water_section
    if surface_levels is not None:
        # .wsurf packs on the SAME [height_min, height_max] as the .r16 tiles, so it
        # can only be written now that export_tiles has finalised that range.
        from . import watersurface
        res.manifest["water_surface"] = watersurface.export_surface_tiles(
            plan, surface_levels, out_dir, res.height_min, res.height_max)
    if water_section is not None or surface_levels is not None:
        import os
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(res.manifest, f, indent=2)

    return res
