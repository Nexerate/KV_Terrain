"""
kvterrain.preview
=================

Small raster previews, for the dataset picker and for looking at what a
post-processing run actually did.

Everything here is numpy + Pillow on purpose. The pipeline already depends on
Pillow through Streamlit, and a hypsometric ramp with a hillshade is ten lines of
array arithmetic — not worth pulling matplotlib in for. Nothing in this module is
part of the exported data format; it exists only to be looked at.

Two audiences:

* `dataset_thumbnail` runs once at FETCH time and is stored in the dataset. It
  draws the NVE vectors straight onto the shaded terrain with `ImageDraw` rather
  than rasterising them, because at thumbnail scale a polygon fill is both
  cheaper and truer than a 512-px burn of a 5 m mask.

* `terrain_rgb` / `water_class_rgb` / `depth_rgb` / `surface_rgb` run at PROCESS
  time on the real arrays. `depth_rgb` is the one to watch when tuning a carve:
  it shows `water_surface - terrain`, which is exactly what the runtime renders.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

# Hypsometric stops (fraction of the height range -> RGB). A conventional
# lowland-green / upland-brown / summit-white ramp, dark enough at the bottom
# that the hillshade still reads over it.
_TERRAIN_STOPS = np.array([
    [0.00, 0.18, 0.35, 0.22],
    [0.15, 0.30, 0.50, 0.25],
    [0.35, 0.60, 0.62, 0.30],
    [0.55, 0.65, 0.52, 0.33],
    [0.75, 0.60, 0.48, 0.42],
    [0.90, 0.78, 0.75, 0.72],
    [1.00, 0.98, 0.98, 0.98],
], dtype=np.float32)

# Water class colours, shared by every preview so blue always means the same thing.
COLOR_LAKE = (0.10, 0.35, 0.85)
COLOR_RIVER = (0.05, 0.62, 0.95)
COLOR_OCEAN = (0.04, 0.16, 0.42)
COLOR_NODATA = (0.85, 0.20, 0.55)


def stride_for(shape, max_px: int) -> int:
    """Integer stride that brings the longer axis of `shape` under `max_px`."""
    return max(1, int(max(shape)) // max(int(max_px), 1))


def downsample(arr: np.ndarray, max_px: int = 640) -> np.ndarray:
    s = stride_for(arr.shape[:2], max_px)
    return arr[::s, ::s]


def hillshade(h: np.ndarray, spacing_m: float, *,
              azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
              z_factor: float = 1.0) -> np.ndarray:
    """
    Classic Horn hillshade in [0, 1]. NaN heights shade as flat ground (0.5) so a
    void reads as a hole in the relief rather than as a black scar.
    """
    hh = np.where(np.isfinite(h), h, np.nan).astype(np.float32)
    filled = np.nan_to_num(hh, nan=float(np.nanmean(hh)) if np.isfinite(hh).any() else 0.0)
    dy, dx = np.gradient(filled * float(z_factor), float(spacing_m))
    slope = np.arctan(np.hypot(dx, dy))
    # np.gradient's first axis is rows, which run NORTH -> SOUTH here, so the
    # north-facing direction is -dy. Getting this backwards inverts the relief
    # and mountains read as valleys.
    aspect = np.arctan2(-dy, dx)
    az = np.deg2rad(360.0 - float(azimuth_deg) + 90.0)
    alt = np.deg2rad(float(altitude_deg))
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def _ramp(t: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Piecewise-linear RGB lookup; `t` in [0,1], returns (..., 3) float."""
    t = np.clip(t, 0.0, 1.0)
    out = np.empty((*t.shape, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, stops[:, 0], stops[:, c + 1])
    return out


def terrain_rgb(h: np.ndarray, spacing_m: float, *,
                hmin: Optional[float] = None, hmax: Optional[float] = None,
                shade: bool = True) -> np.ndarray:
    """Hypsometric colour × hillshade, float RGB in [0,1]. NaN -> magenta."""
    finite = np.isfinite(h)
    if hmin is None:
        hmin = float(h[finite].min()) if finite.any() else 0.0
    if hmax is None:
        hmax = float(h[finite].max()) if finite.any() else 1.0
    t = (np.nan_to_num(h, nan=hmin) - hmin) / max(hmax - hmin, 1e-6)
    rgb = _ramp(t.astype(np.float32), _TERRAIN_STOPS)
    if shade:
        # 0.55..1.25 rather than 0..1: a straight multiply by the shade crushes
        # the flats to black and loses the hypsometric colour entirely.
        rgb = np.clip(rgb * (0.55 + 0.7 * hillshade(h, spacing_m))[..., None], 0, 1)
    rgb[~finite] = COLOR_NODATA
    return rgb.astype(np.float32)


def water_class_rgb(water_type: np.ndarray, weight: Optional[np.ndarray] = None,
                    *, base: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Lake / river classes over an optional terrain base. River brightness tracks
    stream order, so a trunk river is visibly a trunk river.
    """
    from . import water as _water
    shape = water_type.shape
    rgb = (base.copy() if base is not None
           else np.full((*shape, 3), 0.08, dtype=np.float32))
    lake = water_type == _water.TYPE_LAKE
    river = water_type == _water.TYPE_RIVER
    rgb[lake] = COLOR_LAKE
    if weight is None:
        rgb[river] = COLOR_RIVER
    else:
        inten = (0.45 + 0.55 * np.clip(weight.astype(np.float32) / 8.0, 0, 1))[river]
        rgb[river] = np.stack([COLOR_RIVER[0] * inten,
                               COLOR_RIVER[1] * inten,
                               COLOR_RIVER[2] * inten], axis=-1)
    return np.clip(rgb, 0, 1).astype(np.float32)


_DEPTH_STOPS = np.array([
    [0.00, 0.80, 0.93, 0.98],
    [0.20, 0.42, 0.76, 0.93],
    [0.50, 0.13, 0.45, 0.80],
    [0.80, 0.06, 0.22, 0.55],
    [1.00, 0.02, 0.08, 0.27],
], dtype=np.float32)


def depth_rgb(depth_m: np.ndarray, *, max_depth_m: Optional[float] = None,
              base: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """
    `depth = max(0, water_surface - terrain)` — the quantity the runtime actually
    renders, and therefore the honest check on a carve. Returns (rgb, scale_max).

    Dry samples keep the terrain base, so you can see at a glance whether a lake
    came out as water or as a flat dry pan.
    """
    d = np.where(np.isfinite(depth_m), depth_m, 0.0).astype(np.float32)
    wet = d > 1e-3
    if max_depth_m is None:
        max_depth_m = float(np.percentile(d[wet], 99)) if wet.any() else 1.0
    max_depth_m = max(float(max_depth_m), 1e-3)
    rgb = (base.copy() if base is not None
           else np.full((*d.shape, 3), 0.10, dtype=np.float32))
    rgb[wet] = _ramp(d[wet] / max_depth_m, _DEPTH_STOPS)
    return np.clip(rgb, 0, 1).astype(np.float32), max_depth_m


# Diverging ramp for a signed change: warm = ground ADDED (a void filled or a
# shoreline raised), cool = ground REMOVED (a carve). Neutral in the middle so
# untouched terrain reads as untouched.
_DELTA_STOPS = np.array([
    [0.00, 0.05, 0.25, 0.55],
    [0.30, 0.30, 0.62, 0.88],
    [0.50, 0.92, 0.92, 0.90],
    [0.70, 0.93, 0.62, 0.25],
    [1.00, 0.62, 0.22, 0.04],
], dtype=np.float32)


def delta_rgb(delta_m: np.ndarray, *, scale_m: Optional[float] = None,
              base: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """
    A SIGNED change map: `fetched - processed`, so cuts and fills are told apart.

    Everything the pipeline did to the ground shows up here, not just the carve:
    the lake void repair RAISES ground to the shoreline, and the shoreline snap
    moves a ring of samples. A one-sided "how deep did we dig" ramp would render
    both of those as nothing, which is exactly the case where you most want to
    see them. Returns (rgb, scale) where the ramp runs -scale..+scale.
    """
    d = np.where(np.isfinite(delta_m), delta_m, 0.0).astype(np.float32)
    touched = np.abs(d) > 1e-3
    if scale_m is None:
        scale_m = float(np.percentile(np.abs(d[touched]), 99)) if touched.any() else 1.0
    scale_m = max(float(scale_m), 1e-3)
    rgb = (base.copy() if base is not None
           else np.full((*d.shape, 3), 0.10, dtype=np.float32))
    rgb[touched] = _ramp(0.5 + 0.5 * np.clip(d[touched] / scale_m, -1, 1), _DELTA_STOPS)
    return np.clip(rgb, 0, 1).astype(np.float32), scale_m


def surface_rgb(surface_moh: np.ndarray, *, base: Optional[np.ndarray] = None
                ) -> np.ndarray:
    """Water-surface elevation, ramped over the range of the water itself (not the
    terrain) so a 3 m fall along a river is actually visible."""
    s = surface_moh.astype(np.float32)
    wet = np.isfinite(s)
    rgb = (base.copy() if base is not None
           else np.full((*s.shape, 3), 0.10, dtype=np.float32))
    if wet.any():
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        rgb[wet] = _ramp((s[wet] - lo) / max(hi - lo, 1e-6), _DEPTH_STOPS[::-1].copy())
    return np.clip(rgb, 0, 1).astype(np.float32)


def to_png(rgb: np.ndarray) -> bytes:
    """Float RGB in [0,1] -> PNG bytes."""
    from PIL import Image
    arr = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Dataset thumbnail: shaded terrain with the NVE vectors drawn over it          #
# --------------------------------------------------------------------------- #

def dataset_thumbnail(
    heights: np.ndarray, lattice, *,
    rivers: Optional[list] = None, lakes: Optional[list] = None,
    max_px: int = 512,
) -> bytes:
    """
    The picture stored in a dataset, so the picker can show you what you fetched.

    `rivers` / `lakes` are RAW GeoJSON feature lists (what the fetch actually
    holds) in the lattice's CRS, drawn as vectors rather than rasterised: at 512
    px a 5 m raster burn of a stream is a dotted line, while a 1 px stroke is a
    stream.
    """
    from PIL import Image, ImageDraw

    s = stride_for(heights.shape, max_px)
    small = np.asarray(heights[::s, ::s], dtype=np.float32)
    rgb = terrain_rgb(small, lattice.spacing_m * s)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")

    H, W = small.shape
    x0, y0, x1, y1 = lattice.bbox_utm

    def to_px(xy):
        """World (x, y) -> thumbnail (col, row). Rows run north -> south."""
        xs = (np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        cols = (xs[:, 0] - x0) / max(x1 - x0, 1e-9) * (W - 1)
        rows = (y1 - xs[:, 1]) / max(y1 - y0, 1e-9) * (H - 1)
        return list(zip(cols.tolist(), rows.tolist()))

    draw = ImageDraw.Draw(img, "RGBA")
    lake_fill = tuple(int(c * 255) for c in COLOR_LAKE) + (215,)
    river_col = tuple(int(c * 255) for c in COLOR_RIVER) + (235,)

    for feat in (lakes or []):
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        polys = ([geom.get("coordinates") or []] if gtype == "Polygon"
                 else (geom.get("coordinates") or []) if gtype == "MultiPolygon" else [])
        for rings in polys:
            if not rings:
                continue
            pts = to_px(rings[0])
            if len(pts) >= 3:
                draw.polygon(pts, fill=lake_fill)

    for feat in (rivers or []):
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        lines = ([geom.get("coordinates") or []] if gtype == "LineString"
                 else (geom.get("coordinates") or []) if gtype == "MultiLineString" else [])
        for line in lines:
            pts = to_px(line) if line else []
            if len(pts) >= 2:
                draw.line(pts, fill=river_col, width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
