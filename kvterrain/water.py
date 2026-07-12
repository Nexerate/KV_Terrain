"""
kvterrain.water
===============

River + lake water masks aligned pixel-for-pixel with the R16 height tiles.

This module fetches NVE's national hydrology vectors and rasterises them onto the
SAME corner-centered sample lattice the height pipeline uses (`core.GridPlan`).
The result is an in-memory `WaterGrid` (per-pixel `type`, river `weight`, and
`lake_id` + `lake_table`) that downstream passes consume:

  * `bathymetry.carve_lake_beds`  reads `type` to carve lake bowls, and
  * `watersurface`                reads `type` + `weight` (stream order) and
                                  `lake_surface_moh()` to build the per-pixel
                                  `.wsurf` water-surface field that is actually
                                  written to disk.

There is NO on-disk `.water` tile any more. The runtime consumes water purely as
the `.wsurf` surface-elevation field (depth = max(0, surface - terrain)); the old
categorical `.water` record (type/weight/flow/lake_id) and its quad-tree pyramid
have been removed. `weight` (stream order) is still carried in memory because the
river surface raise is sized by it; the old `flow` bearing byte is gone with the
tile it served.

Sources (open data, ArcGIS REST MapServers — same service family as the height
ImageServer, so we query them the same way and let the server reproject to the
plan's UTM zone via `outSR`):

* Rivers — NVE **Elvenett (ELVIS)**, the national river-network database. Every
  watercourse (river/stream/lake-through-line) is a polyline in a connected
  network *with defined flow direction*. We use two layers:
    - `elvenett` (all segments)   -> every stream, base ("minor") size
    - `hovedelv` (main rivers)    -> upgrades size where a main river runs
* Lakes — NVE **Innsjødatabase**, ~243k lake polygons; lakes > 2500 m² carry a
  unique national number (løpenummer).

Conventions match core.py exactly: corner-centered ("pixel is a point"), one
assembled array sliced into shared-edge tiles, north-up rows.

Field/attribute names on the live services are not contractually stable; every
attribute read here is *optional* and the pipeline degrades gracefully (size
falls back to a main/minor split). Endpoints, layer indices and candidate field
names are constants below — adjust if NVE renames things.

Data © NVE (Elvenett / Innsjødatabase).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import core


# --------------------------------------------------------------------------- #
# Endpoints / layers / attribute candidates (NVE ArcGIS Enterprise, cloud-hosted)
# --------------------------------------------------------------------------- #
#
# NVE migrated ALL map services to a new cloud host in Dec 2025 (announced
# 17.12.2025: https://www.nve.no/kart/nytt-om-gis-api/nye-url-er-for-alle-nve-s-karttjenester/).
# The old host `nve.geodataonline.no` no longer resolves at all (NXDOMAIN) --
# it wasn't a network/firewall issue, the domain was retired. Service names and
# layer indices carried over unchanged; only the host + path prefix changed:
#   old: https://nve.geodataonline.no/arcgis/rest/services
#   new: https://kart.nve.no/enterprise/rest/services
# If these ever move again, browse https://kart.nve.no/enterprise/rest/services
# (or whatever NVE's "Nytt om GIS API" page points to) to find the new base.

RIVER_SERVICE = "https://kart.nve.no/enterprise/rest/services/Elvenett1/MapServer"
RIVER_LAYER_ALL = 2        # 'elvenett'  — all watercourse segments (polyline)
RIVER_LAYER_MAIN = 1       # 'hovedelv'  — main rivers only (polyline)

LAKE_SERVICE = "https://kart.nve.no/enterprise/rest/services/Innsjodatabase2/MapServer"
LAKE_LAYER = 5             # 'Innsjodatabase' — lake polygons

# Optional attribute names, tried in order. None are required.
RIVER_ORDER_FIELDS = ("STRAHLER", "strahler", "elveOrden", "elveorden", "orden", "ORDEN")
LAKE_ID_FIELDS = ("vatnLnr", "vatnlnr", "lopenr", "LOPENR", "vassdragLnr", "objektNr")
LAKE_NAME_FIELDS = ("navn", "NAVN", "sjonavn", "objektNavn", "vatnNavn")
LAKE_AREA_FIELDS = ("areal_km2", "arealKm2", "areal", "AREAL")   # km² if *_km2, else m²
# Lake surface elevation (metres above sea level). Innsjodatabase layer 5 exposes
# `hoyde`; the others are defensive aliases. This is the authoritative water level —
# when present, the runtime reads depth = hoyde - terrain directly, and the carve
# places the bed relative to it, so no rim discovery or shore estimation is needed.
LAKE_HOYDE_FIELDS = ("hoyde", "hoyde_moh", "hoydeMoh", "vannstand_moh",
                     "vannstandMoh", "HOYDE", "z_moh")

WATER_ATTRIBUTION = "© NVE — Elvenett (ELVIS) & Innsjødatabase"
WATER_LICENSE = "NLOD / CC BY 4.0 (confirm current terms at geonorge.no)"

# ArcGIS MapServer paging cap for these layers (MaxRecordCount=2000 on the new
# cloud host, up from 1000 on the old geodataonline.no host).
DEFAULT_PAGE = 2000


# --------------------------------------------------------------------------- #
# On-disk encoding                                                             #
# --------------------------------------------------------------------------- #

TYPE_LAND = 0
TYPE_RIVER = 1
TYPE_LAKE = 2

# Relative river size -> full channel width in metres. `weight` carried per sample
# IS the stream order (clamped 1..255); this table only drives how wide the
# centerline is buffered before rasterising, so bigger rivers seed more pixels.
DEFAULT_WIDTH_BY_ORDER = {
    1: 2.0, 2: 3.0, 3: 5.0, 4: 8.0, 5: 14.0, 6: 25.0, 7: 45.0, 8: 80.0,
}
DEFAULT_ORDER_MAIN = 5      # assumed order for a 'hovedelv' segment lacking an order attr
DEFAULT_ORDER_MINOR = 2     # assumed order for an 'elvenett' segment lacking an order attr


def order_to_width_m(order: int, width_by_order: dict, scale: float) -> float:
    """Full channel width for a stream order, scaled; clamps to the table ends."""
    if not width_by_order:
        return max(0.5, scale)
    keys = sorted(width_by_order)
    o = min(max(order, keys[0]), keys[-1])
    return float(width_by_order[o]) * float(scale)


# --------------------------------------------------------------------------- #
# Feature containers                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class RiverSeg:
    """One polyline in the plan's CRS (metres), vertices ordered downstream."""
    xy: np.ndarray             # (N,2) float64
    order: int                 # stream size class (>=1)


@dataclass
class LakePoly:
    """A lake as exterior + optional holes, in the plan's CRS (metres)."""
    rings: list                # [exterior(M,2), hole0, hole1, ...]
    lopenr: Optional[int]      # NVE national lake number, if present
    navn: Optional[str]
    area_m2: Optional[float]
    hoyde_moh: Optional[float] = None   # lake surface elevation (m.o.h.), if present


@dataclass
class WaterFeatures:
    rivers: list = field(default_factory=list)   # list[RiverSeg]
    lakes: list = field(default_factory=list)    # list[LakePoly]


@dataclass
class WaterGrid:
    """North-up assembled masks, shape (SY, SX) each; lake_table maps id->info.

    `weight` is the per-pixel river stream order (clamped 1..255; 0 off-river),
    which `watersurface` reads to size the river-surface raise. The old `flow`
    bearing byte was dropped together with the `.water` tile it served.
    """
    type: np.ndarray           # u1
    weight: np.ndarray         # u1
    lake_id: np.ndarray        # u2
    lake_table: dict           # {local_id:int -> {lopenr,navn,area_m2,hoyde_moh}}

    def lake_surface_moh(self) -> np.ndarray:
        """
        Per-pixel lake water-surface elevation (m.o.h.), float32, NaN everywhere
        except lake pixels whose lake has a known `hoyde_moh`. This is the array the
        carve reads to place the bed and the runtime reads for depth = surface -
        terrain. Lakes missing `hoyde_moh` stay NaN so callers can fall back to the
        shore-estimate path for just those.
        """
        surf = np.full(self.type.shape, np.nan, dtype=np.float32)
        if not self.lake_table:
            return surf
        # Build an id->hoyde lookup array indexed by local lake id (0..maxid).
        maxid = int(self.lake_id.max()) if self.lake_id.size else 0
        if maxid == 0:
            return surf
        lut = np.full(maxid + 1, np.nan, dtype=np.float32)
        for lid, info in self.lake_table.items():
            h = info.get("hoyde_moh")
            if lid <= maxid and h is not None:
                lut[lid] = np.float32(h)
        is_lake = self.type == TYPE_LAKE
        surf[is_lake] = lut[self.lake_id[is_lake]]
        return surf


# --------------------------------------------------------------------------- #
# Grid transform: pixel CENTERS coincide with corner-centered sample points    #
# --------------------------------------------------------------------------- #

def sample_grid_transform(plan: core.GridPlan, level: int = 0):
    """
    Affine mapping raster (col,row) -> world (UTM) for a north-up raster whose
    pixel CENTERS land exactly on the sample lattice. This is the raster-space
    equivalent of the half-texel bbox offset `export_image_fetch` uses, so a
    rasterised water sample sits on the identical world point as the height
    sample with the same (row,col).
    """
    from affine import Affine

    sp = plan.spacing_m * (2 ** level)
    SY = _level_samples(plan, level)[1]
    return Affine(sp, 0.0, plan.origin_x - 0.5 * sp,
                  0.0, -sp, plan.origin_y + (SY - 0.5) * sp)


def _level_samples(plan: core.GridPlan, level: int) -> tuple[int, int]:
    """(SX, SY) sample counts at a pyramid level (matches build_pyramid shapes)."""
    SX, SY = plan.samples_x, plan.samples_y
    for _ in range(level):
        SX = (SX - 1) // 2 + 1
        SY = (SY - 1) // 2 + 1
    return SX, SY


# --------------------------------------------------------------------------- #
# Fetch: ArcGIS REST /query -> GeoJSON, paged, reprojected to the plan CRS     #
# --------------------------------------------------------------------------- #

# A WaterFetcher takes a GridPlan and returns WaterFeatures. Swappable so the
# rasterise/pyramid/pack pipeline is testable offline with no network.
WaterFetcher = Callable[[core.GridPlan], WaterFeatures]


def _arcgis_query_geojson(
    server: str, layer: int, epsg: int, bbox: tuple[float, float, float, float],
    *, session=None, out_fields: str = "*", page: int = DEFAULT_PAGE,
    timeout: int = 120, retries: int = 4, pause: float = 1.5,
    progress: Optional[Callable[[int, int, str], None]] = None, label: str = "",
) -> list:
    """
    Page through an ArcGIS MapServer feature layer's /query endpoint, asking the
    server for GeoJSON already projected into `epsg` (outSR). Returns the raw list
    of GeoJSON feature dicts intersecting `bbox`.
    """
    import requests

    url = f"{server.rstrip('/')}/{layer}/query"
    xmin, ymin, xmax, ymax = bbox
    base = {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": str(epsg),
        "outSR": str(epsg),
        "spatialRel": "esriSpatialRelIntersects",
        "where": "1=1",
        "outFields": out_fields,
        "returnGeometry": "true",
        "geometryPrecision": "2",
        "f": "geojson",
        "resultRecordCount": str(page),
    }
    sess = session or requests.Session()
    feats: list = []
    offset = 0
    while True:
        params = dict(base, resultOffset=str(offset))
        last_err = None
        for attempt in range(retries):
            try:
                r = sess.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                js = r.json()
                if isinstance(js, dict) and js.get("error"):
                    raise RuntimeError(f"ArcGIS error: {js['error']}")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(pause * (attempt + 1))
        else:
            raise RuntimeError(f"{label} query failed after {retries} tries: {last_err}")

        batch = js.get("features", []) or []
        feats.extend(batch)
        if progress:
            progress(len(feats), len(feats) + len(batch), f"{label}: {len(feats)} feats")
        # ArcGIS signals more pages via exceededTransferLimit or a full batch.
        more = js.get("exceededTransferLimit") or (len(batch) == page)
        if not batch or not more:
            break
        offset += page
    return feats


def _first_attr(props: dict, names) -> Optional[object]:
    for n in names:
        if n in props and props[n] is not None:
            return props[n]
    return None


def _iter_line_coords(geom: dict):
    """Yield each list-of-[x,y] linestring from a (Multi)LineString GeoJSON geom."""
    if not geom:
        return
    t = geom.get("type")
    c = geom.get("coordinates")
    if t == "LineString":
        yield c
    elif t == "MultiLineString":
        for part in c:
            yield part


def _iter_polygon_rings(geom: dict):
    """Yield [exterior, *holes] ring-lists for each polygon of a (Multi)Polygon."""
    if not geom:
        return
    t = geom.get("type")
    c = geom.get("coordinates")
    if t == "Polygon":
        yield c
    elif t == "MultiPolygon":
        for poly in c:
            yield poly


def features_from_geojson(
    river_all: list, river_main: list, lakes: list,
    *, order_fields=RIVER_ORDER_FIELDS,
    default_order_minor: int = DEFAULT_ORDER_MINOR,
    default_order_main: int = DEFAULT_ORDER_MAIN,
) -> WaterFeatures:
    """Turn raw GeoJSON feature lists into typed, CRS-metre WaterFeatures."""
    rivers: list = []

    def _add_rivers(feats, default_order):
        for f in feats or []:
            props = f.get("properties", {}) or {}
            ov = _first_attr(props, order_fields)
            try:
                order = int(round(float(ov))) if ov is not None else int(default_order)
            except (TypeError, ValueError):
                order = int(default_order)
            order = max(1, min(255, order))
            for line in _iter_line_coords(f.get("geometry") or {}):
                if line and len(line) >= 2:
                    rivers.append(RiverSeg(np.asarray(line, dtype=np.float64)[:, :2], order))

    _add_rivers(river_all, default_order_minor)
    _add_rivers(river_main, default_order_main)   # main rivers overwrite by higher weight

    lake_polys: list = []
    for f in lakes or []:
        props = f.get("properties", {}) or {}
        lopenr = _first_attr(props, LAKE_ID_FIELDS)
        try:
            lopenr = int(lopenr) if lopenr is not None else None
        except (TypeError, ValueError):
            lopenr = None
        navn = _first_attr(props, LAKE_NAME_FIELDS)
        area = _first_attr(props, LAKE_AREA_FIELDS)
        area_m2 = None
        if area is not None:
            try:
                area_m2 = float(area)
                # Heuristic: fields ending _km2 are km²; convert.
                for nm in LAKE_AREA_FIELDS:
                    if nm in props and props[nm] is area and nm.lower().endswith("km2"):
                        area_m2 *= 1e6
                        break
            except (TypeError, ValueError):
                area_m2 = None
        hv = _first_attr(props, LAKE_HOYDE_FIELDS)
        hoyde_moh = None
        if hv is not None:
            try:
                hoyde_moh = float(hv)
            except (TypeError, ValueError):
                hoyde_moh = None
        for rings in _iter_polygon_rings(f.get("geometry") or {}):
            arr_rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings if len(r) >= 4]
            if arr_rings:
                lake_polys.append(LakePoly(arr_rings, lopenr, navn, area_m2, hoyde_moh))

    return WaterFeatures(rivers=rivers, lakes=lake_polys)


def fetch_water_features(
    plan: core.GridPlan, *, include_main_rivers: bool = True,
    session=None, progress: Optional[Callable[[int, int, str], None]] = None,
) -> WaterFeatures:
    """Live fetch of rivers + lakes intersecting the plan bbox, in the plan CRS."""
    import requests
    sess = session or requests.Session()
    bbox = plan.bbox_utm

    river_all = _arcgis_query_geojson(RIVER_SERVICE, RIVER_LAYER_ALL, plan.epsg, bbox,
                                      session=sess, progress=progress, label="rivers")
    river_main = []
    if include_main_rivers:
        river_main = _arcgis_query_geojson(RIVER_SERVICE, RIVER_LAYER_MAIN, plan.epsg,
                                           bbox, session=sess, progress=progress,
                                           label="main-rivers")
    lakes = _arcgis_query_geojson(LAKE_SERVICE, LAKE_LAYER, plan.epsg, bbox,
                                  session=sess, progress=progress, label="lakes")
    return features_from_geojson(river_all, river_main, lakes)


# --------------------------------------------------------------------------- #
# Rasterise: burn features onto the leaf sample grid                           #
# --------------------------------------------------------------------------- #

def rasterize_water(
    plan: core.GridPlan, feats: WaterFeatures, *,
    width_by_order: Optional[dict] = None, width_scale: float = 1.0,
    all_touched_rivers: bool = True,
) -> WaterGrid:
    """
    Rasterise rivers (buffered by size) and lakes onto the leaf (SY,SX) grid.
    Priority at overlaps is lake > river; among rivers the larger (higher order)
    wins, and its weight (stream order) is the one kept.
    """
    from shapely.geometry import LineString, Polygon
    from rasterio.features import rasterize

    width_by_order = width_by_order or DEFAULT_WIDTH_BY_ORDER
    SX, SY = plan.samples_x, plan.samples_y
    transform = sample_grid_transform(plan, level=0)
    shape = (SY, SX)

    # ---- rivers -----------------------------------------------------------
    # One shapely polygon per sub-segment (each straight piece between two
    # vertices) buffered to the channel width, so a meandering river is covered
    # piece-by-piece. Only `weight` (stream order) is burned now — the surface
    # raise downstream is sized by it; flow bearing is no longer produced.
    weight_shapes = []   # (polygon, order)
    for seg in feats.rivers:
        xy = seg.xy
        if xy.shape[0] < 2:
            continue
        half = 0.5 * order_to_width_m(seg.order, width_by_order, width_scale)
        half = max(half, 0.51 * plan.spacing_m)  # ensure at least ~1 sample wide
        for k in range(xy.shape[0] - 1):
            (x0, y0), (x1, y1) = xy[k], xy[k + 1]
            if x0 == x1 and y0 == y1:
                continue
            poly = LineString([(x0, y0), (x1, y1)]).buffer(
                half, cap_style=2, join_style=2)  # flat caps, mitre joins
            if poly.is_empty:
                continue
            weight_shapes.append((poly, int(seg.order)))

    # Burn larger rivers LAST so they win (rasterize is last-write per pixel).
    order_sort = np.argsort([w for _, w in weight_shapes], kind="stable") \
        if weight_shapes else np.array([], dtype=int)
    weight_arr = np.zeros(shape, dtype=np.uint8)
    if len(order_sort):
        w_sorted = [weight_shapes[i] for i in order_sort]
        rasterize(w_sorted, out=weight_arr, transform=transform,
                  all_touched=all_touched_rivers, merge_alg=_replace())

    type_arr = np.where(weight_arr > 0, TYPE_RIVER, TYPE_LAND).astype(np.uint8)

    # ---- lakes ------------------------------------------------------------
    # Map each lake's national number (or a running counter) to a compact local
    # uint16 id so the sim can group a lake's samples without a flood-fill.
    lake_id_arr = np.zeros(shape, dtype=np.uint16)
    lake_table: dict = {}
    lake_shapes = []
    next_id = 1
    lopenr_to_local: dict = {}
    for lk in feats.lakes:
        key = lk.lopenr if lk.lopenr is not None else ("_anon", next_id)
        if key in lopenr_to_local:
            local = lopenr_to_local[key]
        else:
            local = next_id
            next_id += 1
            if local > 65535:
                # Extremely unlikely for any drawn region; stop assigning ids.
                break
            lopenr_to_local[key] = local
            lake_table[local] = {
                "lopenr": lk.lopenr, "navn": lk.navn, "area_m2": lk.area_m2,
                "hoyde_moh": lk.hoyde_moh,
            }
        shell = lk.rings[0]
        holes = lk.rings[1:] if len(lk.rings) > 1 else None
        poly = Polygon(shell, holes)
        if not poly.is_empty:
            lake_shapes.append((poly, int(local)))

    if lake_shapes:
        rasterize(lake_shapes, out=lake_id_arr, transform=transform,
                  all_touched=False, merge_alg=_replace())

    # lake > river priority: where a lake covers a sample, it's a lake, and the
    # river weight there is cleared (weight is read only when type==RIVER).
    lake_mask = lake_id_arr > 0
    type_arr[lake_mask] = TYPE_LAKE
    weight_arr[lake_mask] = 0

    return WaterGrid(type_arr, weight_arr, lake_id_arr, lake_table)


def _replace():
    from rasterio.enums import MergeAlg
    return MergeAlg.replace


# --------------------------------------------------------------------------- #
# Water source description (for the .wsurf manifest; no .water tiles are written)
# --------------------------------------------------------------------------- #

def water_source_manifest(width_scale: float = 1.0) -> dict:
    """
    Metadata describing where the water masks came from and how rivers were
    buffered. The runtime no longer reads a `.water` tile — this just travels in
    the manifest beside the `.wsurf` (water_surface) section for provenance.
    """
    return {
        "attribution": WATER_ATTRIBUTION,
        "license": WATER_LICENSE,
        "sources": {
            "rivers": f"{RIVER_SERVICE} (layers {RIVER_LAYER_ALL} elvenett, "
                      f"{RIVER_LAYER_MAIN} hovedelv)",
            "lakes": f"{LAKE_SERVICE} (layer {LAKE_LAYER} Innsjodatabase)",
        },
        "width_model": {
            "note": "centerline buffered to this FULL width (m) by stream order, "
                    "then scaled; drives pixel coverage only.",
            "scale": width_scale,
            "width_by_order_m": DEFAULT_WIDTH_BY_ORDER,
            "default_order_main": DEFAULT_ORDER_MAIN,
            "default_order_minor": DEFAULT_ORDER_MINOR,
        },
    }


# --------------------------------------------------------------------------- #
# Synthetic features for offline demo/tests (no network)                       #
# --------------------------------------------------------------------------- #

def synthetic_water_features(plan: core.GridPlan) -> WaterFeatures:
    """
    A deterministic river + lake placed in the plan's UTM extent, for demo mode
    and offline tests. A diagonal 'main' river crosses the region; a rectangular
    lake sits near the centre.
    """
    x0, y0, x1, y1 = plan.bbox_utm
    # Diagonal river SW->NE, a few vertices so it meanders across the region.
    river = RiverSeg(
        xy=np.array([
            [x0 + 0.10 * (x1 - x0), y0 + 0.10 * (y1 - y0)],
            [x0 + 0.40 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
            [x0 + 0.55 * (x1 - x0), y0 + 0.60 * (y1 - y0)],
            [x0 + 0.90 * (x1 - x0), y0 + 0.90 * (y1 - y0)],
        ], dtype=np.float64),
        order=5,
    )
    # A lake rectangle near the centre.
    cx0, cy0 = x0 + 0.60 * (x1 - x0), y0 + 0.25 * (y1 - y0)
    cx1, cy1 = x0 + 0.80 * (x1 - x0), y0 + 0.45 * (y1 - y0)
    lake = LakePoly(
        rings=[np.array([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1], [cx0, cy0]],
                        dtype=np.float64)],
        lopenr=999001, navn="Synthetic Lake", area_m2=(cx1 - cx0) * (cy1 - cy0),
        hoyde_moh=320.0,
    )
    return WaterFeatures(rivers=[river], lakes=[lake])