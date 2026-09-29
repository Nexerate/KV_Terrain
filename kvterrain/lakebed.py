"""
kvterrain.lakebed
=================

Lake beds that still look like ground when the water is gone.

`bathymetry.carve_lake_beds` was built for a static viewer: sink every lake 20 m
on a 1:1 ramp so the water renders opaque, and never look at the bed again. The
runtime is not a viewer. Terrain is edited and water moves, so a lake can drain,
and what that bed then shows — a flat floor at exactly 20 m behind a uniform
45-degree wall tracing the polygon — is obviously made up.

This module builds the bed from what a lake actually is: a continuation of the
terrain around it. Three tiers, best first, per lake:

  1. SURVEYED — NVE has depth contours and/or spot depths (`Innsjodatabase2`
     layers 1 and 2, about 500 lakes nationally). The synthetic shape below is
     bent to pass through them (see `_fit_ratio_field`).
  2. SUMMARY — NVE has only the survey's max and mean depth (Tunnsjøen: 222 m and
     87 m, no contours). The synthetic shape is scaled to that max, with its
     profile exponent chosen so the mean matches too.
  3. SYNTHETIC — no survey. The shape alone.

The synthetic shape
-------------------

    depth(p) = S(p) * dmax * f(q(p)),    f(q) = b * (1 - exp(-k * q / b))

  * `q` is distance from the shore, normalised to the body's own maximum `dmax`
    and slightly rounded at the centreline (see `_shape_distance`), so the
    deepest water is where the lake is furthest from any shore, narrows are
    shallow and islands are shored like the outer bank.
  * `S(p)` is the slope of the land around the lake — the height gained per metre
    walked away from the water, measured over a ring outside the shore and
    blended inward over the lake (`_shore_slope_field`). Cliffs make deep lakes,
    bogs make shallow ones, and the deepest point leans toward the steeper side,
    which is where it tends to be in a glacially carved basin.
  * `f` starts at slope `k` (`slope_ratio`: the submerged slope as a fraction of
    the land slope) and bends over with curvature `b`, so a lake keeps getting
    deeper toward its centre but more and more gently — sediment fills a basin's
    floor — and never goes exactly flat.

This is the relation Hollister, Milstead & Urrutia (2011, PLoS ONE 6(9) e25764)
used to predict maximum lake depth from surrounding slope and distance to shore,
extended from one number per lake to a whole surface.

Noise is MULTIPLICATIVE: `depth * (1 + a * n)` with `|a * n| < 1`. The shoreline
is where depth is zero, and a factor cannot move a zero, so the outline is exactly
the polygon's; and a positive depth stays positive, so no shoal ever reaches the
surface. Shallow water gets proportionally small bumps and deep water big ones.
On top of that, a fine DETAIL term in metres (`detail_m`) roughens the shallows
the way the shore around them is rough and fades out with depth, where a real
floor is sediment. It is bounded by 30% of the local depth, so the same two
guarantees hold. Both noises are anchored to WORLD coordinates, so a lake looks
the same whatever region or tiling it is processed in.

Invariants shared with the legacy carve (the depression hierarchy relies on them):
the 8-connected shore ring and the map edge (with `edge_is_shore`) sit exactly on
the waterline, island samples keep their DTM height, and nothing outside a lake is
touched — so no rim is ever lowered. Closed sub-basins inside a bed are allowed
(owner, 2026-09-29): a drained lake leaves water in its hollows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import (binary_dilation, distance_transform_edt,
                           find_objects, gaussian_filter, label)

TYPE_LAKE_VALUE = 2

MODEL_TERRAIN = "terrain"
MODEL_RAMP = "ramp"
MODELS = (MODEL_TERRAIN, MODEL_RAMP)
DEFAULT_MODEL = MODEL_TERRAIN

TIER_SURVEYED = "surveyed"
TIER_SUMMARY = "summary"
TIER_SYNTHETIC = "synthetic"


@dataclass
class LakeBedParams:
    """Every tunable of the terrain model. Recorded verbatim in the manifest.

    slope_ratio, curvature and ring_fraction are CALIBRATED (`lakecalib`,
    2026-09-29) against 130 NVE-surveyed lakes of up to 25 km², synthetic bed vs
    survey: median error x1.57 at the survey's contour and spot depths, x1.35 on
    max depth and x1.28 on mean depth, with biases within 12%. The same shape with
    the land slope held constant does markedly worse (x1.64 on max depth, x1.7+ on
    mean depth at its best constant), so the surrounding terrain does carry the
    signal. Re-run `kvterrain calibrate-lake-beds` before changing them.
    """
    # Submerged slope at the shore as a fraction of the land slope around it.
    slope_ratio: float = 0.9
    # How soon the profile bends over (f(q) above). Small = a bowl whose floor
    # flattens early; large = a straight cone. The floor never goes exactly flat.
    curvature: float = 0.6
    # The land ring the slope is read from: this fraction of the body's maximum
    # distance from shore, clamped to [ring_min_m, ring_max_m]. A pond is judged
    # by the bank beside it, a big lake by the valley sides above it.
    ring_fraction: float = 1.0
    ring_min_m: float = 25.0
    ring_max_m: float = 500.0
    # Land slope clamps (m per m). The floor keeps a lake in a bog from carving to
    # nothing; the ceiling stops a lone cliff pixel turning a pond into a well.
    min_slope: float = 0.02
    max_slope: float = 1.5
    # Every body reaches at least this deep at its deepest sample (tier 3 only):
    # a pond must still render as water.
    min_depth_m: float = 2.0
    # Optional hard ceiling on synthetic depth. None = no ceiling.
    max_depth_m: Optional[float] = None
    # Multiplicative noise: amplitude a in depth * (1 + a * n), n in [-1, 1].
    noise_amplitude: float = 0.35
    # Longest noise wavelength in metres; each further octave halves it.
    noise_wavelength_m: float = 400.0
    noise_octaves: int = 5
    noise_seed: int = 0
    # Fine relief in the shallows, in METRES rather than as a fraction of depth:
    # the bed a falling lake exposes first is rock and boulder with the texture of
    # the shore around it, while a deep floor is smooth sediment. Its amplitude is
    # detail_m * exp(-depth / detail_fade_m), and never more than 30% of the depth
    # there, so it cannot move the shoreline or lift a shoal to the surface either.
    detail_m: float = 1.0
    detail_wavelength_m: float = 60.0
    detail_fade_m: float = 8.0
    # Surveyed lakes already have a real shape; they get this share of the noise.
    survey_noise_scale: float = 0.35
    use_surveys: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# NVE surveys                                                                  #
# --------------------------------------------------------------------------- #

SURVEY_FIELD_CANDIDATES = {
    "vatnlnr": ("vatnlnr", "vatnLnr"),
    "name": ("innsjonavn", "navn"),
    "max_depth": ("maksdyp_m", "maksdyp"),
    "mean_depth": ("middeldyp_m", "middeldyp"),
    "level": ("oppmaltvannstand_m", "oppmaltvannstand"),
    "year": ("oppmaltaar",),
}
SAMPLE_FIELD_CANDIDATES = {
    "vatnlnr": ("vatnlnr", "vatnLnr"),
    "depth": ("dybde_m", "dybde"),
}


@dataclass
class LakeSurvey:
    """One NVE survey, joined to its lake by `vatnlnr`. Depths are positive down,
    measured from `level_m` (the water level at survey time) when it is known."""
    vatnlnr: int
    name: Optional[str] = None
    max_depth_m: Optional[float] = None
    mean_depth_m: Optional[float] = None
    level_m: Optional[float] = None
    year: Optional[int] = None
    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    depth_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def has_samples(self) -> bool:
        return self.depth_m.size > 0


def _positive(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _densify(coords: np.ndarray, step: float) -> np.ndarray:
    """Vertices of a polyline resampled to at most `step` apart (ends kept)."""
    if coords.shape[0] < 2:
        return coords
    out = [coords[:1]]
    for a, b in zip(coords[:-1], coords[1:]):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / step)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def parse_surveys(raw: Optional[dict], *, step_m: float) -> dict:
    """
    Raw bathymetry layers (as `dataset.bathymetry_geojson` returns them) ->
    `{vatnlnr: LakeSurvey}`. Contours are densified to `step_m` so a long straight
    contour weighs in by its length rather than by how few vertices it was drawn
    with.
    """
    from . import water

    if not raw:
        return {}
    surveys: dict = {}
    feats = raw.get("surveys") or []
    if feats:
        names = set().union(*[set((f.get("properties") or {}).keys()) for f in feats])
        fm = water.resolve_fields(names, SURVEY_FIELD_CANDIDATES)
        for f in feats:
            p = f.get("properties") or {}
            try:
                vid = int(p.get(fm["vatnlnr"]))
            except (TypeError, ValueError):
                continue
            s = surveys.setdefault(vid, LakeSurvey(vatnlnr=vid))
            s.name = s.name or (p.get(fm["name"]) if fm["name"] else None)
            s.max_depth_m = s.max_depth_m or _positive(p.get(fm["max_depth"]))
            s.mean_depth_m = s.mean_depth_m or _positive(p.get(fm["mean_depth"]))
            s.level_m = s.level_m or _positive(p.get(fm["level"]))
            yr = _positive(p.get(fm["year"])) if fm["year"] else None
            s.year = s.year or (int(yr) if yr else None)

    xy_by: dict = {}
    d_by: dict = {}
    for key in ("contours", "points"):
        feats = raw.get(key) or []
        if not feats:
            continue
        names = set().union(*[set((f.get("properties") or {}).keys()) for f in feats])
        fm = water.resolve_fields(names, SAMPLE_FIELD_CANDIDATES)
        for f in feats:
            p = f.get("properties") or {}
            try:
                vid = int(p.get(fm["vatnlnr"]))
                depth = float(p.get(fm["depth"]))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(depth):
                continue
            g = f.get("geometry") or {}
            t, c = g.get("type"), g.get("coordinates")
            parts = []
            if t == "LineString":
                parts = [_densify(np.asarray(c, float)[:, :2], step_m)]
            elif t == "MultiLineString":
                parts = [_densify(np.asarray(pc, float)[:, :2], step_m) for pc in c]
            elif t == "Point":
                parts = [np.asarray([c], float)[:, :2]]
            elif t == "MultiPoint":
                parts = [np.asarray(c, float)[:, :2]]
            for xy in parts:
                if xy.size:
                    xy_by.setdefault(vid, []).append(xy)
                    d_by.setdefault(vid, []).append(np.full(xy.shape[0], depth))
    for vid, parts in xy_by.items():
        s = surveys.setdefault(vid, LakeSurvey(vatnlnr=vid))
        s.xy = np.concatenate(parts)
        s.depth_m = np.concatenate(d_by[vid])
    return surveys


# --------------------------------------------------------------------------- #
# Small raster helpers                                                         #
# --------------------------------------------------------------------------- #

def _upsample_axis(c: np.ndarray, n: int, f: int, axis: int) -> np.ndarray:
    """Linear upsample along one axis from block centres back to `n` samples."""
    pos = (np.arange(n) - (f - 1) / 2.0) / f
    pos = np.clip(pos, 0.0, c.shape[axis] - 1)
    i0 = np.floor(pos).astype(np.intp)
    i1 = np.minimum(i0 + 1, c.shape[axis] - 1)
    w = (pos - i0).astype(np.float32)
    shape = [1, 1]
    shape[axis] = n
    w = w.reshape(shape)
    return np.take(c, i0, axis=axis) * (1 - w) + np.take(c, i1, axis=axis) * w


def _blur(a: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Gaussian blur that stays cheap at large sigma: block-average down, blur, and
    interpolate back up. A lake the size of Tunnsjøen asks for sigmas of hundreds
    of pixels, where a direct separable filter would dominate a whole process run.
    """
    a = np.asarray(a, dtype=np.float32)
    if sigma_px <= 0:
        return a.copy()
    if sigma_px <= 4.0 or min(a.shape) < 16:
        return gaussian_filter(a, sigma_px, mode="nearest", truncate=3.0)
    f = max(1, 2 ** int(math.floor(math.log2(sigma_px / 2.0))))
    H, W = a.shape
    Hp, Wp = -(-H // f) * f, -(-W // f) * f
    pad = np.pad(a, ((0, Hp - H), (0, Wp - W)), mode="edge")
    c = pad.reshape(Hp // f, f, Wp // f, f).mean(axis=(1, 3))
    c = gaussian_filter(c, sigma_px / f, mode="nearest", truncate=3.0)
    return _upsample_axis(_upsample_axis(c, H, f, 0), W, f, 1).astype(np.float32)


def _normalized_blur(values: np.ndarray, weights: np.ndarray, sigma_px: float,
                     *, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Blur of scattered samples: sum(w*v)/sum(w) under a Gaussian. Returns the
    estimate and the blurred weight, so a caller can tell data from extrapolation."""
    w = _blur(weights, sigma_px)
    v = _blur(values * weights, sigma_px)
    return v / np.maximum(w, floor), w


# --------------------------------------------------------------------------- #
# World-anchored noise                                                         #
# --------------------------------------------------------------------------- #

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xC2B2AE3D27D4EB4F)
_M3 = np.uint64(0xBF58476D1CE4E5B9)
_M4 = np.uint64(0x94D049BB133111EB)


def _hash01(i: np.ndarray, j: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic lattice hash -> [-1, 1). splitmix64 over (i, j, salt)."""
    with np.errstate(over="ignore"):
        h = (i.astype(np.int64).view(np.uint64) * _M1) ^ \
            (j.astype(np.int64).view(np.uint64) * _M2) ^ \
            np.uint64((salt * 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF)
        h ^= h >> np.uint64(30)
        h *= _M3
        h ^= h >> np.uint64(27)
        h *= _M4
        h ^= h >> np.uint64(31)
    return ((h >> np.uint64(11)).astype(np.float64) * (2.0 ** -52) - 1.0)


def _value_noise(x: np.ndarray, y: np.ndarray, cell: float, salt: int) -> np.ndarray:
    gx, gy = x / cell, y / cell
    i0, j0 = np.floor(gx), np.floor(gy)
    tx, ty = gx - i0, gy - j0
    tx = tx * tx * tx * (tx * (tx * 6 - 15) + 10)       # quintic fade
    ty = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
    i0 = i0.astype(np.int64)
    j0 = j0.astype(np.int64)
    a = _hash01(i0, j0, salt)
    b = _hash01(i0 + 1, j0, salt)
    c = _hash01(i0, j0 + 1, salt)
    d = _hash01(i0 + 1, j0 + 1, salt)
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty


def world_noise(x: np.ndarray, y: np.ndarray, *, wavelength_m: float,
                octaves: int, seed: int = 0) -> np.ndarray:
    """
    Domain-warped fractal value noise at world coordinates, in [-1, 1].

    Normalised by its octave weights so the amplitude parameter means the same
    whatever the octave count. The warp bends the lattice's axis-aligned grain
    into something that reads as sediment and rock rather than as a texture.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    warp = 0.35 * wavelength_m
    wx = x + warp * _value_noise(x, y, wavelength_m, seed * 131 + 7)
    wy = y + warp * _value_noise(x, y, wavelength_m, seed * 131 + 11)
    total = np.zeros(x.shape, dtype=np.float64)
    norm = 0.0
    amp, cell = 1.0, float(wavelength_m)
    for o in range(max(1, int(octaves))):
        total += amp * _value_noise(wx, wy, cell, seed * 131 + 17 + o)
        norm += amp
        amp *= 0.5
        cell *= 0.5
    # Value noise sums rarely approach +-1; stretch toward the range, then clip.
    return np.clip(1.6 * total / norm, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# The synthetic shape                                                          #
# --------------------------------------------------------------------------- #

def _profile(q: np.ndarray, k: float, b: float) -> np.ndarray:
    b = max(float(b), 1e-3)
    return b * (1.0 - np.exp(-float(k) * q / b))


def _shape_distance(d_px: np.ndarray, water: np.ndarray) -> np.ndarray:
    """
    Distance from shore, rounded at the centreline. A raw distance transform has a
    sharp ridge along the medial axis that a drained bed shows as an upturned roof
    keel. A single Gaussian of ~15% of the body's width takes the keel off, and it
    is blended in only once the distance is a few sigmas from the shore, where
    the raw transform is straight and a blur cannot change it. (A sigma that grew
    with distance would sink the very centre below its surroundings.)
    """
    dmax = float(d_px[water].max()) if water.any() else 0.0
    if dmax < 3.0:
        return d_px
    sigma = min(max(0.15 * dmax, 1.0), 60.0)
    blurred = _blur(np.where(water, d_px, 0.0), sigma)
    w = np.clip(d_px / (2.5 * sigma), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    return np.where(water, d_px * (1.0 - w) + blurred * w, 0.0).astype(np.float32)


@dataclass
class _Body:
    """One connected water body, cut out with a margin wide enough for its ring."""
    rows: slice
    cols: slice
    water: np.ndarray        # body water samples (islands excluded)
    lake: np.ndarray         # body samples incl. islands
    island: np.ndarray
    d_m: np.ndarray          # distance to shore (metres, waterline-anchored)
    shape_m: np.ndarray      # centreline-rounded distance (metres)
    dmax_m: float
    on_map_edge: bool


def _shore_slope_field(body: _Body, ground: np.ndarray, level: np.ndarray,
                       other_water: np.ndarray, spacing: float,
                       p: LakeBedParams) -> tuple[np.ndarray, float, float]:
    """
    Land slope around the body, blended over its water. Every land sample within
    the ring contributes its SECANT slope, (height above the lake) / (distance to
    the waterline): robust to a noisy DTM, honest about cliffs, and the quantity
    that actually extrapolates — how fast the ground falls toward the water.
    Returns (slope field, ring width m, median ring slope).
    """
    ring_m = float(np.clip(p.ring_fraction * body.dmax_m, p.ring_min_m, p.ring_max_m))
    water = body.water
    dist_px, (ir, ic) = distance_transform_edt(~water, return_indices=True)
    dist_m = np.maximum(dist_px * spacing - 0.5 * spacing, 0.5 * spacing)
    lvl = level[ir, ic]
    ring = (~water) & (dist_px * spacing <= ring_m) & ~other_water \
        & np.isfinite(ground) & np.isfinite(lvl)
    slope = np.zeros(water.shape, dtype=np.float32)
    rise = np.where(ring, ground - np.where(np.isfinite(lvl), lvl, 0.0), 0.0)
    slope[ring] = np.clip(rise[ring] / dist_m[ring], 0.0, p.max_slope)
    wts = ring.astype(np.float32)
    if not ring.any():
        return (np.full(water.shape, p.min_slope, np.float32), ring_m, p.min_slope)
    sigma_px = max(0.6 * body.dmax_m, ring_m) / spacing
    s, _ = _normalized_blur(slope, wts, sigma_px)
    s = np.clip(s, p.min_slope, p.max_slope)
    return s.astype(np.float32), ring_m, float(np.median(slope[ring]))


def _cut_body(lake_all: np.ndarray, body_lbl: np.ndarray, idx: int, sl: tuple,
              island: np.ndarray, spacing: float, edge_is_shore: bool,
              pad_px: int) -> _Body:
    H, W = lake_all.shape
    r0 = max(sl[0].start - pad_px, 0)
    r1 = min(sl[0].stop + pad_px, H)
    c0 = max(sl[1].start - pad_px, 0)
    c1 = min(sl[1].stop + pad_px, W)
    rows, cols = slice(r0, r1), slice(c0, c1)
    lake = body_lbl[rows, cols] == idx
    isl = island[rows, cols] & lake
    water = lake & ~isl
    # Outside the map: shore when edge_is_shore (a cropped lake ramps back up to
    # its level at the crop), otherwise more water (the EDT's own behaviour).
    edge_val = not edge_is_shore
    touches = (r0 == 0, r1 == H, c0 == 0, c1 == W)
    on_edge = bool((touches[0] and water[0].any()) or (touches[1] and water[-1].any())
                   or (touches[2] and water[:, 0].any())
                   or (touches[3] and water[:, -1].any()))
    padded = np.pad(water, 1, mode="constant", constant_values=False)
    if touches[0]:
        padded[0, 1:-1] = water[0] & edge_val
    if touches[1]:
        padded[-1, 1:-1] = water[-1] & edge_val
    if touches[2]:
        padded[1:-1, 0] = water[:, 0] & edge_val
    if touches[3]:
        padded[1:-1, -1] = water[:, -1] & edge_val
    d_px = distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)
    # Anchor on the waterline: the outermost wet sample is one cell from dry land
    # (bathymetry.SHORE_ANCHOR_M).
    d_px = np.where(water, np.maximum(d_px - 1.0, 0.0), 0.0).astype(np.float32)
    shape_px = _shape_distance(d_px, water)
    dmax = float(shape_px[water].max()) * spacing if water.any() else 0.0
    return _Body(rows, cols, water, lake, isl, d_px * spacing, shape_px * spacing,
                 dmax, on_edge)


# --------------------------------------------------------------------------- #
# Surveys -> depth                                                             #
# --------------------------------------------------------------------------- #

def _fit_ratio_field(base: np.ndarray, mask: np.ndarray, rr: np.ndarray,
                     cc: np.ndarray, target: np.ndarray, spacing: float,
                     dmax_m: float) -> tuple[np.ndarray, dict]:
    """
    A smooth multiplier field M with base * M passing (nearly) through the
    surveyed depths. Fitted in log space, coarse to fine: each pass blurs the
    remaining misfit at the samples and applies it, fading to no correction away
    from any data. Working on a RATIO of the synthetic shape rather than on the
    depths themselves is what keeps the shoreline at zero (base is zero there),
    and what lets the bed keep deepening past the innermost contour instead of
    going flat at its value, which is what interpolating contours directly does.
    """
    b = base[rr, cc]
    ok = (b > 0.25) & (target > 0.25) & mask[rr, cc]
    rr, cc, lr = rr[ok], cc[ok], np.log(target[ok] / b[ok])
    if lr.size == 0:
        return np.ones(base.shape, np.float32), {"samples_used": 0}
    F = np.full(base.shape, float(np.median(lr)), dtype=np.float32)
    sigma_m = max(dmax_m, 4.0 * spacing)
    passes = 0
    while True:
        sigma_px = sigma_m / spacing
        e = lr - F[rr, cc]
        vals = np.zeros(base.shape, np.float32)
        wts = np.zeros(base.shape, np.float32)
        np.add.at(vals, (rr, cc), e)
        np.add.at(wts, (rr, cc), 1.0)
        v = _blur(vals, sigma_px)
        w = _blur(wts, sigma_px)
        tau = 0.2 / (2.0 * math.pi * sigma_px * sigma_px)
        F += v / (w + tau)
        passes += 1
        if sigma_px <= 3.0:
            break
        sigma_m *= 0.5
    resid = lr - F[rr, cc]
    M = np.exp(np.clip(F, math.log(0.05), math.log(20.0))).astype(np.float32)
    return M, {"samples_used": int(lr.size), "passes": passes,
               "log_rms_before": float(np.sqrt(np.mean((lr - np.median(lr)) ** 2))),
               "log_rms_after": float(np.sqrt(np.mean(resid ** 2)))}


def _gamma_for_mean(q: np.ndarray, max_d: float, mean_d: float) -> float:
    """Exponent g with mean(max_d * q**g) == mean_d (q in [0, 1]); bisection."""
    lo, hi = 0.1, 8.0
    target = mean_d / max_d
    if target >= float(np.mean(q ** lo)):
        return lo
    if target <= float(np.mean(q ** hi)):
        return hi
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if float(np.mean(q ** mid)) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# --------------------------------------------------------------------------- #
# The whole pass                                                               #
# --------------------------------------------------------------------------- #

def build_lake_beds(
    height_m: np.ndarray,
    water_type: np.ndarray,
    spacing_m: float,
    *,
    surface_moh: np.ndarray,
    ground_m: np.ndarray,
    lake_id: Optional[np.ndarray] = None,
    lake_table: Optional[dict] = None,
    island: Optional[np.ndarray] = None,
    surveys: Optional[dict] = None,
    world_origin: tuple = (0.0, 0.0),
    params: Optional[LakeBedParams] = None,
    edge_is_shore: bool = False,
    shore_8_connected: bool = False,
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[np.ndarray, dict]:
    """
    Replace every lake sample that has a level with `level - depth` from the model
    in the module docstring, and return (new heights, report).

    `ground_m` is the UNCARVED bed (void-repaired, before any carve): the land the
    slope is read from. `surface_moh` is the per-sample lake level. `world_origin`
    is the world (x, y) of sample (row = last, col = 0) — the lattice origin — and
    anchors the noise. `lake_id`/`lake_table` join samples to NVE lake numbers
    for the surveys.

    Two passes. The first builds the synthetic bed body by body and writes every
    body that holds no surveyed lake. The second applies each survey to its lake
    AS A WHOLE: a lake can fall into several bodies (a causeway, a strait narrower
    than its own shore ring), and scaling each fragment to the survey's max depth
    would put that max in every one of them.
    """
    p = params or LakeBedParams()
    out = np.array(height_m, dtype=np.float32, copy=True)
    H, W = water_type.shape
    is_lake = water_type == TYPE_LAKE_VALUE
    island = (np.zeros_like(is_lake) if island is None
              else np.asarray(island, bool) & is_lake)
    report: dict = {"model": MODEL_TERRAIN, "params": p.to_dict(),
                    "bodies": 0, "tiers": {TIER_SURVEYED: 0, TIER_SUMMARY: 0,
                                           TIER_SYNTHETIC: 0},
                    "surveys_in_region": len(surveys or {}), "surveyed_lakes": []}
    if not is_lake.any():
        return out, report

    structure = np.ones((3, 3), bool) if shore_8_connected else None
    body_lbl, n = label(is_lake, structure=structure)
    slices = find_objects(body_lbl)
    report["bodies"] = int(n)

    # vatnlnr -> local lake ids, for the survey join.
    surveys = (surveys or {}) if p.use_surveys else {}
    local_to_survey: dict = {}
    for lid, info in (lake_table or {}).items():
        v = info.get("lopenr")
        if v is not None and int(v) in surveys:
            local_to_survey[int(lid)] = int(v)

    ox, oy = world_origin
    a = float(np.clip(p.noise_amplitude, 0.0, 0.95))
    depths_max: list = []
    stash: list = []                     # bodies holding a surveyed lake
    shape_pad = int(math.ceil(p.ring_max_m / spacing_m)) + 2

    def finish(b: dict) -> None:
        """Floor, waterline ring, map edge; then write the body."""
        depth, have, body = b["depth"], b["have"], b["body"]
        rows, cols = body.rows, body.cols
        # Floor for small bodies, on the synthetic part only: scale up (the
        # shoreline stays at zero) so the deepest sample reaches min_depth_m.
        syn = have & ~b["tier_mask"]
        if syn.any():
            peak = float(depth[syn].max())
            floor = float(p.min_depth_m)
            if peak <= 1e-6:
                depth = np.where(syn, floor, depth)
            elif peak < floor:
                depth = np.where(syn, depth * (floor / peak), depth)
        depth = np.maximum(depth, 0.0)
        if p.detail_m > 0:
            amp = np.minimum(0.3 * depth, float(p.detail_m)
                             * np.exp(-depth / max(float(p.detail_fade_m), 1e-3)))
            depth = depth + b["detail"] * amp
        if shore_8_connected:
            ring = body.water & binary_dilation(~body.water,
                                                structure=np.ones((3, 3), bool))
            depth[ring] = 0.0
        if edge_is_shore:
            if rows.start == 0:
                depth[0, :] = 0.0
            if rows.stop == H:
                depth[-1, :] = 0.0
            if cols.start == 0:
                depth[:, 0] = 0.0
            if cols.stop == W:
                depth[:, -1] = 0.0
        win = out[rows, cols]
        win[have] = (b["surf"] - depth)[have]
        depths_max.append(float(depth[have].max()))

    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        if progress and (idx % 50 == 0 or idx == n):
            progress(idx / max(n, 1), f"{idx}/{n} bodies")
        # Cheap first cut for the size; the ring needs a margin that depends on it.
        small = _cut_body(is_lake, body_lbl, idx, sl, island, spacing_m,
                          edge_is_shore, pad_px=2)
        ring_m = float(np.clip(p.ring_fraction * small.dmax_m, p.ring_min_m,
                               p.ring_max_m))
        pad = min(shape_pad, int(math.ceil(ring_m / spacing_m)) + 2)
        body = small if pad <= 2 else _cut_body(
            is_lake, body_lbl, idx, sl, island, spacing_m, edge_is_shore, pad_px=pad)
        rows, cols = body.rows, body.cols
        surf = surface_moh[rows, cols]
        have = body.lake & ~body.island & np.isfinite(surf)
        if not have.any():
            continue
        other_water = (water_type[rows, cols] == TYPE_LAKE_VALUE) & ~body.lake
        slope, _, _ = _shore_slope_field(
            body, ground_m[rows, cols], np.where(body.water, surf, np.nan),
            other_water, spacing_m, p)

        # The smooth synthetic depth, before noise.
        dmax = max(body.dmax_m, 1e-6)
        q = np.clip(body.shape_m / dmax, 0.0, 1.0)
        base = np.where(body.water, slope * dmax * _profile(q, p.slope_ratio,
                                                            p.curvature), 0.0)
        base = base.astype(np.float32)
        if p.max_depth_m is not None:
            base = np.minimum(base, float(p.max_depth_m))

        yy, xx = np.mgrid[rows, cols]
        wx = ox + xx.astype(np.float64) * spacing_m
        wy = oy + (H - 1 - yy).astype(np.float64) * spacing_m
        noise = world_noise(wx, wy, wavelength_m=p.noise_wavelength_m,
                            octaves=p.noise_octaves, seed=p.noise_seed)
        detail = (world_noise(wx, wy, wavelength_m=p.detail_wavelength_m, octaves=3,
                              seed=p.noise_seed + 1009)
                  if p.detail_m > 0 else None)
        b = {"body": body, "have": have, "surf": surf, "base": base, "noise": noise,
             "detail": detail,
             "depth": base * (1.0 + a * noise),
             "tier_mask": np.zeros(base.shape, dtype=bool)}
        lids = None
        if lake_id is not None and local_to_survey:
            lid_win = lake_id[rows, cols]
            lids = {int(v) for v in np.unique(lid_win[have])} & set(local_to_survey)
            b["lid"] = lid_win
        if lids:
            stash.append(b)
        else:
            finish(b)
            report["tiers"][TIER_SYNTHETIC] += 1

    # ---- pass 2: each surveyed lake, across all of its bodies -------------- #
    by_survey: dict = {}
    for b in stash:
        for lid in np.unique(b["lid"][b["have"]]):
            v = local_to_survey.get(int(lid))
            if v is not None:
                by_survey.setdefault(v, []).append((b, b["have"] & (b["lid"] == lid)))
    for vid, parts in by_survey.items():
        sv = surveys[vid]
        lvl = float(np.median(np.concatenate([b["surf"][m] for b, m in parts])))
        offset = (lvl - sv.level_m) if sv.level_m else 0.0
        px = sum(int(m.sum()) for _, m in parts)
        area = sum(float((lake_table or {}).get(lid, {}).get("area_m2") or 0.0)
                   for lid, v in local_to_survey.items() if v == vid)
        coverage = (px * spacing_m * spacing_m / area) if area > 0 else 1.0
        cropped = any(b["body"].on_map_edge for b, _ in parts)
        entry = {"vatnlnr": vid, "name": sv.name, "survey_year": sv.year,
                 "bodies": len(parts), "level_m": round(lvl, 2),
                 "survey_level_m": sv.level_m, "level_offset_m": round(offset, 2),
                 "survey_max_depth_m": sv.max_depth_m,
                 "survey_mean_depth_m": sv.mean_depth_m,
                 "cropped_by_map_edge": cropped,
                 "in_region_share": round(min(coverage, 1.0), 3)}

        tier = TIER_SYNTHETIC
        if sv.has_samples:
            fits, fitted = [], 0
            for b, m in parts:
                body, rows, cols = b["body"], b["body"].rows, b["body"].cols
                c = np.rint((sv.xy[:, 0] - ox) / spacing_m).astype(np.int64) - cols.start
                r = (H - 1 - np.rint((sv.xy[:, 1] - oy) / spacing_m).astype(np.int64)) \
                    - rows.start
                inside = (r >= 0) & (r < m.shape[0]) & (c >= 0) & (c < m.shape[1])
                r, c, tgt = r[inside], c[inside], sv.depth_m[inside] + offset
                keep = m[r, c]
                if not keep.any():
                    continue
                M, fit = _fit_ratio_field(b["base"], m, r[keep], c[keep], tgt[keep],
                                          spacing_m, max(body.dmax_m, 1e-6))
                if fit.get("samples_used", 0) == 0:
                    continue
                an = a * float(np.clip(p.survey_noise_scale, 0.0, 1.0))
                b["depth"] = np.where(m, b["base"] * M * (1.0 + an * b["noise"]),
                                      b["depth"])
                b["tier_mask"] |= m
                fits.append(fit)
                fitted += 1
            if fitted:
                tier = TIER_SURVEYED
                entry["fit"] = {
                    "bodies_fitted": fitted,
                    "samples_used": sum(f["samples_used"] for f in fits),
                    "log_rms_before": round(max(f["log_rms_before"] for f in fits), 3),
                    "log_rms_after": round(max(f["log_rms_after"] for f in fits), 3)}
        if tier == TIER_SYNTHETIC and sv.max_depth_m:
            peak = max(float(b["depth"][m].max()) for b, m in parts)
            max_d = sv.max_depth_m + offset
            if peak > 0 and coverage >= 0.5:
                # Scale to the survey's max over the WHOLE lake, and pick the
                # profile exponent that reproduces its mean too.
                qs = [np.clip(b["depth"][m] / peak, 0.0, 1.0) for b, m in parts]
                g = 1.0
                if sv.mean_depth_m:
                    g = float(np.clip(_gamma_for_mean(np.concatenate(qs), max_d,
                                                      sv.mean_depth_m + offset),
                                      0.5, 3.0))
                for (b, m), qn in zip(parts, qs):
                    dm = b["depth"].copy()
                    dm[m] = max_d * qn ** g
                    b["depth"] = dm
                    b["tier_mask"] |= m
                tier = TIER_SUMMARY
                entry["profile_exponent"] = round(g, 3)
            elif peak > max_d:
                # Only part of the lake is here, so its deepest point may not be:
                # keep the synthetic shape, just never deeper than the survey says.
                for b, m in parts:
                    b["depth"] = np.where(m, b["depth"] * (max_d / peak), b["depth"])
                entry["note"] = ("mostly outside the region: synthetic, scaled "
                                 "down to the survey's max depth")
            else:
                entry["note"] = "mostly outside the region: synthetic"
        elif tier == TIER_SYNTHETIC:
            entry["note"] = "survey has no depths inside the lake"
        entry["tier"] = tier
        report["tiers"][tier] += 1
        dm = np.concatenate([b["depth"][m] for b, m in parts])
        entry["model_max_depth_m"] = round(float(dm.max()), 2)
        entry["model_mean_depth_m"] = round(float(dm.mean()), 2)
        report["surveyed_lakes"].append(entry)
    for b in stash:
        finish(b)

    if depths_max:
        dm = np.asarray(depths_max)
        report["body_max_depth_m"] = {
            "max": round(float(dm.max()), 2),
            "p50": round(float(np.median(dm)), 2),
            "p90": round(float(np.percentile(dm, 90)), 2)}
    return out, report
