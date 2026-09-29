"""
kvterrain.lakecalib
===================

Calibrate the synthetic lake bed (`lakebed`) against lakes NVE has surveyed.

About 690 Norwegian lakes carry an NVE depth survey, and they are the only ground
truth there is for what the synthetic model should produce. This module picks a
sample of them, fetches for each a small DTM window and its polygon, builds the
SYNTHETIC bed (surveys switched off, noise off), and scores it against the survey:

  * sample depths — every contour vertex and spot depth, re-referenced from the
    survey water level to the lake's `hoyde`; scored as |log(model / survey)|
    so a 2 m pond and a 200 m lake weigh the same;
  * max and mean depth from the survey summary.

Everything fetched is cached per lake, so sweeping parameters costs no network
after the first run. `calibrate` sweeps `slope_ratio` x `curvature` x
`ring_fraction` and reports the best combination and how far each lake sits from
it. The model is deliberately simple; the spread of per-lake errors is the honest
statement of how much a synthetic bed can know.

    kvterrain calibrate-lake-beds --n 80 --cache data/lakecalib
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
import os
import random
from typing import Optional

import numpy as np

from . import core, lakebed, water

CALIB_EPSG = 25833
DEFAULT_SPACING_M = 10.0
MAX_WINDOW_PX = 1600
WINDOW_PAD_M = 600.0
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "lakecalib")


def _query(layer: int, where: str, *, geometry: bool, session, out_fields="*") -> list:
    if geometry:
        return water._arcgis_query_geojson(water.LAKE_SERVICE, layer, CALIB_EPSG, None,
                                           where=where, session=session,
                                           out_fields=out_fields)
    import requests
    sess = session or requests.Session()
    r = sess.get(f"{water.LAKE_SERVICE}/{layer}/query",
                 params={"where": where, "outFields": out_fields,
                         "returnGeometry": "false", "f": "json"}, timeout=120)
    r.raise_for_status()
    return [f["attributes"] for f in r.json().get("features", [])]


def candidate_lakes(session=None) -> list[dict]:
    """Every survey with contours or points and a max depth, as attribute dicts."""
    surveys = _query(water.BATHY_SURVEY_LAYER, "maksdyp_m > 0", geometry=False,
                     session=session,
                     out_fields="vatnlnr,innsjonavn,maksdyp_m,middeldyp_m,"
                                "oppmaltvannstand_m,arealdybdekart_km2")
    with_lines = set()
    for layer in (water.BATHY_CONTOUR_LAYER, water.BATHY_POINT_LAYER):
        import requests
        sess = session or requests.Session()
        r = sess.get(f"{water.LAKE_SERVICE}/{layer}/query",
                     params={"where": "dybde_m > 0", "outFields": "vatnlnr",
                             "returnDistinctValues": "true",
                             "returnGeometry": "false", "f": "json"}, timeout=120)
        r.raise_for_status()
        with_lines |= {f["attributes"]["vatnlnr"] for f in r.json().get("features", [])}
    seen, out = set(), []
    for s in surveys:
        v = s.get("vatnlnr")
        if v in with_lines and v not in seen:
            seen.add(v)
            out.append(s)
    return out


def fetch_lake(vatnlnr: int, cache: str, *, spacing_m: float = DEFAULT_SPACING_M,
               session=None) -> Optional[dict]:
    """Polygon, survey layers and a DTM window for one lake; cached on disk."""
    path = os.path.join(cache, f"{vatnlnr}.npz")
    meta_path = os.path.join(cache, f"{vatnlnr}.json.gz")
    if os.path.exists(path) and os.path.exists(meta_path):
        with gzip.open(meta_path, "rt") as f:
            meta = json.load(f)
        if meta.get("skip"):
            return None
        arr = np.load(path)
        meta["dtm"] = arr["dtm"]
        return meta

    os.makedirs(cache, exist_ok=True)
    where = f"vatnlnr = {int(vatnlnr)}"
    lake = _query(water.LAKE_LAYER, where, geometry=True, session=session)
    raw = {"surveys": _query(water.BATHY_SURVEY_LAYER, where, geometry=False,
                             session=session),
           "contours": _query(water.BATHY_CONTOUR_LAYER, where, geometry=True,
                              session=session),
           "points": _query(water.BATHY_POINT_LAYER, where, geometry=True,
                            session=session)}
    raw["surveys"] = [{"properties": s} for s in raw["surveys"]]

    def skip(reason):
        with gzip.open(meta_path, "wt") as f:
            json.dump({"skip": reason}, f)
        return None

    if not lake:
        return skip("no lake polygon")
    xs, ys = [], []
    for feat in lake:
        for rings in water._iter_polygon_rings(feat.get("geometry")):
            ext = np.asarray(rings[0], float)
            xs += [ext[:, 0].min(), ext[:, 0].max()]
            ys += [ext[:, 1].min(), ext[:, 1].max()]
    x0, x1 = min(xs) - WINDOW_PAD_M, max(xs) + WINDOW_PAD_M
    y0, y1 = min(ys) - WINDOW_PAD_M, max(ys) + WINDOW_PAD_M
    sp = spacing_m
    while max(x1 - x0, y1 - y0) / sp > MAX_WINDOW_PX:
        sp *= 2
    x0, y0 = math.floor(x0 / sp) * sp, math.floor(y0 / sp) * sp
    nx, ny = int(math.ceil((x1 - x0) / sp)) + 1, int(math.ceil((y1 - y0) / sp)) + 1
    # The ImageServer is erratic on small windows (0.4 s to a 30 s cold start),
    # so give up on a request sooner than a region fetch would and retry.
    dtm = core.export_image_fetch(core.IMAGESERVER["DTM"], CALIB_EPSG, x0, y0, nx, ny,
                                  sp, session=session, timeout=45, retries=4)
    hoyde = None
    for feat in lake:
        h = (feat.get("properties") or {}).get("hoyde")
        if h is not None:
            hoyde = float(h)
    meta = {"vatnlnr": int(vatnlnr), "lake": lake, "bathymetry": raw,
            "origin": [x0, y0], "spacing_m": sp, "hoyde": hoyde}
    np.savez_compressed(path, dtm=np.asarray(dtm, np.float32))
    with gzip.open(meta_path, "wt") as f:
        json.dump(meta, f)
    meta["dtm"] = np.asarray(dtm, np.float32)
    return meta


def _rasterise(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lake mask incl. islands, island mask) on the cached window."""
    from affine import Affine
    from rasterio.features import rasterize
    from shapely.geometry import Polygon

    H, W = meta["dtm"].shape
    sp = meta["spacing_m"]
    x0, y0 = meta["origin"]
    tr = Affine(sp, 0.0, x0 - 0.5 * sp, 0.0, -sp, y0 + (H - 0.5) * sp)
    shells, holes = [], []
    for feat in meta["lake"]:
        for rings in water._iter_polygon_rings(feat.get("geometry")):
            shells.append((Polygon(rings[0]), 1))
            holes += [(Polygon(h), 1) for h in rings[1:]]
    lake = rasterize(shells, out_shape=(H, W), transform=tr, fill=0) > 0
    isl = (rasterize(holes, out_shape=(H, W), transform=tr, fill=0) > 0) & lake \
        if holes else np.zeros((H, W), bool)
    return lake, isl


class Prepared:
    """One lake, rasterised and ready for repeated model runs."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.lake, self.island = _rasterise(meta)
        self.sp = float(meta["spacing_m"])
        self.origin = tuple(meta["origin"])
        surveys = lakebed.parse_surveys(meta["bathymetry"], step_m=self.sp)
        self.survey = surveys.get(meta["vatnlnr"])
        sv = self.survey
        self.level = meta["hoyde"] or (sv.level_m if sv else None)
        self.offset = (self.level - sv.level_m) if (sv and sv.level_m and self.level) else 0.0
        H, W = self.lake.shape
        water_px = self.lake & ~self.island
        self.ok = bool(sv is not None and sv.has_samples and self.level is not None
                       and water_px.sum() >= 20)
        if not self.ok:
            return
        c = np.rint((sv.xy[:, 0] - self.origin[0]) / self.sp).astype(int)
        r = H - 1 - np.rint((sv.xy[:, 1] - self.origin[1]) / self.sp).astype(int)
        inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        r, c, d = r[inside], c[inside], sv.depth_m[inside] + self.offset
        keep = water_px[r, c] & (d > 0.5)
        self.rr, self.cc, self.depth = r[keep], c[keep], d[keep]
        self.ok = self.depth.size >= 10
        self.type = np.where(self.lake, lakebed.TYPE_LAKE_VALUE, 0).astype(np.uint8)
        self.surf = np.where(self.lake, self.level, np.nan).astype(np.float32)
        dtm = self.meta["dtm"].astype(np.float32)
        # The DTM inside a lake is its flattened (or void) water surface; the model
        # only reads land, but keep voids (a finite 0 or sentinel, well below the
        # water — see bathymetry.fill_lake_surface) out of the ring.
        self.ground = np.where(np.isfinite(dtm) & (dtm > self.level - 12.0), dtm, np.nan)

    def model_depth(self, params: lakebed.LakeBedParams) -> np.ndarray:
        out, _ = lakebed.build_lake_beds(
            self.ground.copy(), self.type, self.sp, surface_moh=self.surf,
            ground_m=self.ground, island=self.island, params=params,
            world_origin=self.origin, edge_is_shore=True, shore_8_connected=True)
        return np.where(self.lake & ~self.island, self.level - out, np.nan)

    def score(self, params: lakebed.LakeBedParams) -> dict:
        d = self.model_depth(params)
        m = d[self.rr, self.cc]
        good = np.isfinite(m) & (m > 0.05)
        err = np.log(np.maximum(m, 0.05) / self.depth)
        water_px = self.lake & ~self.island
        sv = self.survey
        out = {"vatnlnr": self.meta["vatnlnr"], "name": sv.name,
               "samples": int(self.depth.size),
               "sample_abs_log_err": float(np.median(np.abs(err))),
               "sample_log_bias": float(np.median(err)),
               "sample_coverage": float(good.mean())}
        mx = float(np.nanmax(d[water_px]))
        if sv.max_depth_m:
            out["max_log_err"] = math.log(max(mx, 0.05) / (sv.max_depth_m + self.offset))
        if sv.mean_depth_m:
            mean = float(np.nanmean(d[water_px]))
            out["mean_log_err"] = math.log(max(mean, 0.05) /
                                           max(sv.mean_depth_m + self.offset, 0.1))
        out["model_max_m"] = round(mx, 1)
        out["survey_max_m"] = sv.max_depth_m
        return out


def _summary(scores: list[dict]) -> dict:
    def med(key, f=lambda v: v):
        vals = [f(s[key]) for s in scores if key in s]
        return float(np.median(vals)) if vals else float("nan")
    return {
        "lakes": len(scores),
        "sample_abs_log_err": med("sample_abs_log_err"),
        "sample_log_bias": med("sample_log_bias"),
        "max_abs_log_err": med("max_log_err", abs),
        "max_log_bias": med("max_log_err"),
        "mean_abs_log_err": med("mean_log_err", abs),
        "mean_log_bias": med("mean_log_err"),
        "objective": med("sample_abs_log_err") + 0.5 * med("max_log_err", abs)
                     + 0.5 * med("mean_log_err", abs),
    }


def calibrate(n: int = 80, *, cache: str = DEFAULT_CACHE, seed: int = 1,
              max_area_km2: float = 25.0, grid: Optional[dict] = None,
              workers: int = 8, cache_only: bool = False, session=None,
              log=print) -> dict:
    import requests
    sess = session or requests.Session()
    cands = candidate_lakes(sess)
    cands = [c for c in cands if (c.get("arealdybdekart_km2") or 0) <= max_area_km2]
    random.Random(seed).shuffle(cands)
    lakes: list[Prepared] = []

    if cache_only:
        cands = [c for c in cands
                 if os.path.exists(os.path.join(cache, f"{int(c['vatnlnr'])}.npz"))]

    def one(c):
        try:
            # One session per thread: requests.Session is not thread-safe.
            return c, fetch_lake(int(c["vatnlnr"]), cache)
        except Exception as e:      # noqa: BLE001 — one bad lake must not stop a sweep
            return c, e

    from concurrent.futures import ThreadPoolExecutor, as_completed
    pos = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(lakes) < n and pos < len(cands):
            chunk = cands[pos:pos + 2 * workers]
            pos += len(chunk)
            futures = [pool.submit(one, c) for c in chunk]
            for fut in as_completed(futures):
                c, meta = fut.result()
                if isinstance(meta, Exception):
                    log(f"  {c['vatnlnr']}: fetch failed ({meta})")
                    continue
                if meta is None or len(lakes) >= n:
                    continue
                p = Prepared(meta)
                if p.ok:
                    lakes.append(p)
                    log(f"  [{len(lakes)}/{n}] {c.get('innsjonavn')} ({c['vatnlnr']}): "
                        f"{p.depth.size} samples, max {c.get('maksdyp_m')} m")

    # Centred on the 2026-09-29 optimum (0.9 / 0.6 / 1.0); a first, wider sweep
    # (0.4-1.3 / 0.15-1.0 / 0.25-1.0) had it at the edge only in ring_fraction.
    grid = grid or {"slope_ratio": [0.7, 0.9, 1.1],
                    "curvature": [0.4, 0.6, 0.8],
                    "ring_fraction": [0.75, 1.0, 1.5]}
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        params = lakebed.LakeBedParams(noise_amplitude=0.0, detail_m=0.0, min_depth_m=0.0,
                                       use_surveys=False, **kw)
        scores = [lk.score(params) for lk in lakes]
        summ = _summary(scores)
        results.append({"params": kw, **summ, "per_lake": scores})
        log(f"  {kw}: objective {summ['objective']:.3f}, sample |log| "
            f"{summ['sample_abs_log_err']:.3f} (bias {summ['sample_log_bias']:+.2f}), "
            f"max bias {summ['max_log_bias']:+.2f}, mean bias {summ['mean_log_bias']:+.2f}")
    results.sort(key=lambda r: r["objective"])
    return {"lakes": len(lakes), "grid": grid, "best": results[0],
            "ranking": [{k: v for k, v in r.items() if k != "per_lake"}
                        for r in results]}
