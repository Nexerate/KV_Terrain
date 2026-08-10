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

On-disk products derived from this grid (0.5.0):

  * `surface.atlas`   — per-pixel water-surface elevation (see `watersurface`).
  * `water_id.atlas`  — per-pixel class + authored lake identity (see `waterid`).

The old categorical `.water` record (type/weight/flow/lake_id, four channels) and
its quad-tree pyramid are still gone. `water_id` is its narrower replacement: a
single u16 channel carrying Dry/River/Ocean plus the authored lake id, which is
what the runtime actually needs to pin an authored lake level to a solver basin.
`weight` (stream order) stays in memory only — the river surface raise is sized by
it and `rivernet` samples it — and the old `flow` bearing byte remains gone with
the tile it served, since flow direction now lives on the polylines where it
belongs.

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

# Attribute names, tried in order; the FIRST entry of each tuple is the name the
# live service actually uses, verified against the layer's own `?f=json` metadata
# (see `describe_layer` / `kvterrain describe-services`). The rest are historical
# or defensive spellings kept only so an older/mirrored service still resolves.
#
# Read the metadata before editing these. Two of these tuples were wrong for the
# whole of 0.5.0 because they were written from guesswork and from field ALIASES
# rather than field NAMES — ArcGIS returns names in GeoJSON `properties`, and the
# alias ("Strahler") is not the name ("elveordenstrahler").
RIVER_ORDER_FIELDS = ("elveordenstrahler", "STRAHLER", "strahler",
                      "elveOrden", "elveorden", "orden", "ORDEN")

# ── ELVIS stable identity ────────────────────────────────────────────────────
# NVE documents these as persistent across service updates; they are what makes a
# river segment addressable from one export to the next.
#
#   strekn_lnr    segment-level national serial number — THE primary key. NVE
#                 describes it as the national reference for all watercourse
#                 elements.
#   elvid         identifies the river/tributary as a whole (string, len 15).
#   vassdragsnr   REGINE drainage-basin number (string, len 15).
#   vatnlnr       lake serial number, present when the segment touches a lake —
#                 also the natural cross-check for a detected junction.
#
# `objectid` is deliberately NOT read: it is an Esri-internal OID, ephemeral
# across service updates, and persisting it would silently rot.
RIVER_STREKN_FIELDS = ("strekninglnr", "strekn_lnr", "streknLnr", "STREKN_LNR",
                       "streknlnr")
RIVER_ELVID_FIELDS = ("elvid", "elvId", "ELVID", "elv_id")
RIVER_VASSDRAG_FIELDS = ("vassdragsnr", "vassdragNr", "VASSDRAGSNR", "vassdragsnummer")
RIVER_VATNLNR_FIELDS = ("vatnlnr", "vatnLnr", "VATNLNR", "vatn_lnr")
RIVER_CATCHMENT_FIELDS = ("vnrnfelt", "vnrNfelt", "VNRNFELT")
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

# Local lake ids are re-encoded into the u16 water-id raster as
# `waterid.LAKE_ID_BASE + local_id`, so the usable range is shortened by the
# reserved low codes. Kept here (not in waterid) because this is where ids are
# handed out; waterid asserts the two agree.
MAX_LOCAL_LAKE_ID = 65535 - 16


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
    """
    One polyline in the plan's CRS (metres), vertices ordered downstream.

    Vertex order is taken on trust from ELVIS, which is DESIGNED to encode flow
    direction — but that is design intent, not a per-feature guarantee, and
    nothing upstream of us verifies it. `rivernet.validate_flow_direction`
    therefore derives a direction independently from the sampled Z and reports
    the disagreement rate as a diagnostic. It never flips a segment: Z-derived
    direction is noisy on flat reaches, weirs, lakes and DTM artefacts, so it is
    a worse authority than the source order, not a better one.
    """
    xy: np.ndarray             # (N,2) float64
    order: int                 # stream size class (>=1)
    # ── ELVIS stable identity (all optional; None when the service omits them) ──
    strekn_lnr: Optional[int] = None    # primary segment key
    elvid: Optional[str] = None         # river/tributary key
    vassdragsnr: Optional[str] = None   # REGINE catchment key
    vatnlnr: Optional[int] = None       # lake serial, when the segment touches one
    vnrnfelt: Optional[str] = None      # parent catchment's vassdragsnr
    part_index: int = 0                 # part number within a MultiLineString
    feature_index: int = -1             # index of the parent GeoJSON feature


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
    """
    `rivers` is the AUTHORITATIVE network: the `elvenett` layer, one RiverSeg per
    linestring part, carrying ELVIS identity. This is what `rivernet` serialises.

    `main_rivers` is the `hovedelv` layer, kept SEPARATE and used for one purpose
    only: it is rasterised alongside `rivers` so that the last-write-wins burn
    upgrades the per-pixel stream order where a main river runs.

    Before 0.5.0 the hovedelv segments were appended straight into `rivers`, so
    every main river existed twice as overlapping geometry with two different
    orders. Harmless while the only product was a raster; fatal for a connectivity
    graph, which would have emitted each trunk river twice and linked neither copy
    correctly. Keeping them apart means the polyline set is exactly the elvenett
    network, while the raster still gets its upgrade — and `rivernet` recovers each
    segment's upgraded order by sampling the rasterised weight grid along its own
    pixel path, so the polyline order and the raster order cannot disagree.
    """
    rivers: list = field(default_factory=list)        # list[RiverSeg] — elvenett
    main_rivers: list = field(default_factory=list)   # list[RiverSeg] — hovedelv, raster only
    lakes: list = field(default_factory=list)         # list[LakePoly]


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
    # Stream order BEFORE the lake mask zeroes river pixels under lakes. The
    # display path must not see these (a lake pixel is a lake), but `rivernet`
    # must: Elvenett includes lake-through-lines, and a through-line's order can
    # only be recovered from the pre-mask grid. Reading `weight` there would
    # return 0 and silently demote every trunk river crossing a lake.
    weight_raw: np.ndarray = None      # u1

    def lake_level(self, info: dict):
        """
        The level this lake should actually be rendered and carved at.

        `level_m` when `apply_estimated_levels` has filled one in, else the NVE
        `hoyde_moh`. Kept as one accessor so the carve, the surface raster, the
        river tie-in and lakes.json cannot disagree about a lake's level — they
        did disagree before, and the result was a 20 m dry pit wherever NVE had
        no `hoyde` (see `apply_estimated_levels`).
        """
        lvl = info.get("level_m")
        return info.get("hoyde_moh") if lvl is None else lvl

    def lake_surface_moh(self) -> np.ndarray:
        """
        Per-pixel lake water-surface elevation (m.o.h.), float32, NaN everywhere
        except lake pixels whose lake has a resolved level (see `lake_level`).
        This is the array the carve reads to place the bed and the runtime reads
        for depth = surface - terrain. Lakes with no level from either source stay
        NaN — and MUST then be left uncarved, or they become a dry hole.
        """
        surf = np.full(self.type.shape, np.nan, dtype=np.float32)
        if not self.lake_table:
            return surf
        # Build an id->level lookup array indexed by local lake id (0..maxid).
        maxid = int(self.lake_id.max()) if self.lake_id.size else 0
        if maxid == 0:
            return surf
        lut = np.full(maxid + 1, np.nan, dtype=np.float32)
        for lid, info in self.lake_table.items():
            h = self.lake_level(info)
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


def _norm_key(s) -> str:
    """Fold a property name to letters+digits only, lowercased."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# NOTE: `_first_attr` lived here and is now DELETED. It resolved one property per
# call, per feature, by rescanning the whole property dict — so a schema mismatch
# was invisible unless a caller happened to check the result and complain.
# `resolve_fields` replaces it: the schema is resolved ONCE per layer against the
# union of property names, and `report_field_map` prints the outcome whether or
# not it succeeded. That inversion is the actual fix here; the two corrected
# spellings above are just the symptom it was hiding.


# Logical field -> candidate spellings, for the two feature layers we parse.
# `resolve_fields` turns these into "logical name -> the name this service really
# uses", once per layer per run, and `report_field_map` prints the result.
RIVER_FIELD_CANDIDATES = {
    "order": RIVER_ORDER_FIELDS,
    "strekn_lnr": RIVER_STREKN_FIELDS,
    "elvid": RIVER_ELVID_FIELDS,
    "vassdragsnr": RIVER_VASSDRAG_FIELDS,
    "vatnlnr": RIVER_VATNLNR_FIELDS,
    "vnrnfelt": RIVER_CATCHMENT_FIELDS,
}
LAKE_FIELD_CANDIDATES = {
    "lopenr": LAKE_ID_FIELDS,
    "navn": LAKE_NAME_FIELDS,
    "area": LAKE_AREA_FIELDS,
    "hoyde": LAKE_HOYDE_FIELDS,
}

# Logical fields whose absence changes the OUTPUT rather than just dropping a
# label, and which must therefore be shouted about rather than mentioned. `order`
# is here because losing it is invisible downstream: every segment silently takes
# DEFAULT_ORDER_MINOR, every river gets the same width and the same surface
# raise, and nothing in the export says so. That is exactly what happened for the
# whole of 0.5.0.
#
# Criticality is PER LAYER, not global: hovedelv carries no order field at all
# and is not supposed to, so flagging it there would train the reader to ignore
# the warning — which is how the real one went unnoticed in the first place.
CRITICAL_FIELDS = frozenset({"order", "hoyde"})
NO_CRITICAL_FIELDS: frozenset = frozenset()


def resolve_fields(available, candidates: dict) -> dict:
    """
    Map each logical field to the name THIS service actually uses (or None).

    Three passes, narrowing in confidence: exact name, then case/punctuation-
    folded name, then a folded SUBSTRING match that is only accepted when exactly
    one field matches. The substring pass is what would have caught
    `elveordenstrahler` from the candidate `strahler` without anyone noticing the
    rename; the uniqueness requirement is what stops it pairing `navn` with
    `elvenavnhierarki`.
    """
    available = list(available)
    exact = set(available)
    folded: dict = {}
    for k in available:                      # first spelling wins ties
        folded.setdefault(_norm_key(k), k)

    out: dict = {}
    for logical, names in candidates.items():
        hit = None
        for n in names:
            if n in exact:
                hit = n
                break
        if hit is None:
            for n in names:
                hit = folded.get(_norm_key(n))
                if hit:
                    break
        if hit is None:
            for n in names:
                want = _norm_key(n)
                near = [k for fk, k in folded.items() if want in fk]
                if len(near) == 1:
                    hit = near[0]
                    break
        out[logical] = hit
    return out


_FIELD_MAP_REPORTED: set = set()


def report_field_map(layer_label: str, fmap: dict, available,
                     critical=CRITICAL_FIELDS, note: Optional[str] = None) -> None:
    """
    Print, once per layer per run, how every logical field resolved.

    Deliberately prints on success too. The predecessor only spoke up when the
    river primary key was missing, so a missing `order` — which matters more —
    went unreported for an entire release. Silence should mean "not run", never
    "nothing to say".
    """
    if layer_label in _FIELD_MAP_REPORTED:
        return
    _FIELD_MAP_REPORTED.add(layer_label)

    got = {k: v for k, v in fmap.items() if v}
    lost = [k for k, v in fmap.items() if not v]
    pairs = ", ".join(f"{k}->{v}" for k, v in sorted(got.items()))
    print(f"[kvterrain.water] {layer_label}: matched {len(got)}/{len(fmap)} fields ({pairs})")
    if not lost:
        return
    hot = sorted(set(lost) & set(critical))
    if not hot:
        # Expected absence (see WATER_LAYERS notes): say so and stay quiet.
        print(f"[kvterrain.water] {layer_label}: no {', '.join(sorted(lost))} field"
              f"{' — ' + note if note else ''}")
        return
    print(f"[kvterrain.water] {layer_label}: UNMATCHED {', '.join(sorted(lost))}\n"
          f"  *** {', '.join(hot)} affects the exported geometry — "
          f"fix before trusting this export ***\n"
          f"  service fields: {sorted(available)}\n"
          f"  add the right spelling to the *_FIELDS tuples in water.py, or run "
          f"`kvterrain describe-services` to see the live schema.")


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
    """
    Turn raw GeoJSON feature lists into typed, CRS-metre WaterFeatures.

    `river_all` (elvenett) becomes the authoritative network with full ELVIS
    identity. `river_main` (hovedelv) becomes `main_rivers`, geometry retained for
    the raster order upgrade only — see WaterFeatures for why the two are no
    longer merged.
    """
    def _to_int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _to_str(v):
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    def _parse_rivers(feats, default_order, keep_identity: bool, label: str) -> list:
        out: list = []
        # Resolve the schema ONCE per layer instead of re-scanning every property
        # of every feature for every field, then read straight through the map.
        cands = dict(RIVER_FIELD_CANDIDATES, order=tuple(order_fields))
        if not keep_identity:
            cands = {"order": cands["order"]}     # hovedelv carries no identity
        keys: set = set()
        for f in feats or []:
            keys |= set((f.get("properties") or {}).keys())
        fmap = resolve_fields(keys, cands)
        if keys:
            report_field_map(
                label, fmap, keys,
                critical=CRITICAL_FIELDS if keep_identity else NO_CRITICAL_FIELDS,
                note=None if keep_identity else HOVEDELV_NO_ORDER)
        f_order = fmap.get("order")

        for fi, f in enumerate(feats or []):
            props = f.get("properties", {}) or {}
            ov = props.get(f_order) if f_order else None
            try:
                order = int(round(float(ov))) if ov is not None else int(default_order)
            except (TypeError, ValueError):
                order = int(default_order)
            order = max(1, min(255, order))

            if keep_identity:
                strekn = _to_int(props.get(fmap["strekn_lnr"]) if fmap["strekn_lnr"] else None)
                elvid = _to_str(props.get(fmap["elvid"]) if fmap["elvid"] else None)
                vdrag = _to_str(props.get(fmap["vassdragsnr"]) if fmap["vassdragsnr"] else None)
                vatn = _to_int(props.get(fmap["vatnlnr"]) if fmap["vatnlnr"] else None)
                nfelt = _to_str(props.get(fmap["vnrnfelt"]) if fmap["vnrnfelt"] else None)
            else:
                strekn = elvid = vdrag = vatn = nfelt = None

            # A MultiLineString is several parts of ONE feature. Each part becomes
            # its own RiverSeg (they are geometrically disjoint) but all of them
            # keep the parent's identity plus a part_index, so the link back to the
            # source feature survives instead of being dissolved as it used to be.
            for pi, line in enumerate(_iter_line_coords(f.get("geometry") or {})):
                if line and len(line) >= 2:
                    out.append(RiverSeg(
                        xy=np.asarray(line, dtype=np.float64)[:, :2],
                        order=order,
                        strekn_lnr=strekn, elvid=elvid, vassdragsnr=vdrag,
                        vatnlnr=vatn, vnrnfelt=nfelt,
                        part_index=pi, feature_index=fi,
                    ))
        return out

    rivers = _parse_rivers(river_all, default_order_minor, keep_identity=True,
                           label="elvenett")
    main_rivers = _parse_rivers(river_main, default_order_main, keep_identity=False,
                                label="hovedelv")

    lake_keys: set = set()
    for f in lakes or []:
        lake_keys |= set((f.get("properties") or {}).keys())
    lmap = resolve_fields(lake_keys, LAKE_FIELD_CANDIDATES)
    if lake_keys:
        report_field_map("innsjodatabase", lmap, lake_keys)
    # The km² -> m² conversion keys off the resolved field's own NAME, so it stays
    # correct whichever spelling the service turned out to use.
    area_is_km2 = bool(lmap["area"]) and lmap["area"].lower().endswith("km2")

    lake_polys: list = []
    for f in lakes or []:
        props = f.get("properties", {}) or {}
        lopenr = props.get(lmap["lopenr"]) if lmap["lopenr"] else None
        try:
            lopenr = int(lopenr) if lopenr is not None else None
        except (TypeError, ValueError):
            lopenr = None
        navn = props.get(lmap["navn"]) if lmap["navn"] else None
        area = props.get(lmap["area"]) if lmap["area"] else None
        area_m2 = None
        if area is not None:
            try:
                area_m2 = float(area) * (1e6 if area_is_km2 else 1.0)
            except (TypeError, ValueError):
                area_m2 = None
        hv = props.get(lmap["hoyde"]) if lmap["hoyde"] else None
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

    return WaterFeatures(rivers=rivers, main_rivers=main_rivers, lakes=lake_polys)


LEVEL_SOURCE_NVE = "nve_hoyde"
LEVEL_SOURCE_ESTIMATED = "dtm_interior_median"


def apply_estimated_levels(wg: WaterGrid, height_repaired: np.ndarray, *,
                           enabled: bool = True, margin_m: Optional[float] = None) -> dict:
    """
    Stamp every lake in `wg.lake_table` with a resolved `level_m` + `level_source`,
    estimating a level from the DTM for the lakes NVE gives no `hoyde` for.

    WHY THIS EXISTS. `hoyde` is null for a lot of small lakes — 25% of the records
    in a Lierne export, 37% of the lakes in the source data there, against 11% in
    a Krøderen one. Those lakes still rasterise, still classify as Lake, and still
    got carved (the carve had its own private shore-estimate fallback), but the
    surface raster only ever read `hoyde`, so it left them empty. A 20 m bowl dug
    into the terrain with no water in it — strictly worse than not carving at all.

    Resolving the level ONCE, here, and having every consumer read it through
    `WaterGrid.lake_level`, is what stops that divergence recurring. An estimated
    level is treated as authoritative from this point on: it goes into the surface
    raster and into `authored_level_m`, so the runtime pins it exactly as it pins
    an NVE one and needs to know nothing about where it came from. `level_source`
    records the provenance for anyone who does care.

    With `enabled=False` no estimate is made, those lakes keep a null level, and
    the caller MUST also stop carving them (`estimate_missing=False`) — otherwise
    it recreates the dry-pit bug this function was written to remove.

    `height_repaired` must be the void-repaired, UNCARVED bed. Returns a small
    summary dict for the manifest.
    """
    from . import bathymetry

    table = wg.lake_table or {}
    for info in table.values():
        h = info.get("hoyde_moh")
        info["level_m"] = h
        info["level_source"] = LEVEL_SOURCE_NVE if h is not None else None

    missing = [lid for lid, info in table.items() if info.get("level_m") is None]
    estimated = 0
    if enabled and missing:
        kw = {} if margin_m is None else {"margin_m": float(margin_m)}
        est = bathymetry.estimate_levels_from_interior(
            height_repaired, wg.lake_id, **kw)
        for lid in missing:
            v = est.get(lid)
            if v is not None and np.isfinite(v):
                table[lid]["level_m"] = float(v)
                table[lid]["level_source"] = LEVEL_SOURCE_ESTIMATED
                estimated += 1

    unresolved = sum(1 for i in table.values() if i.get("level_m") is None)
    return {
        "lakes_total": len(table),
        "levels_from_nve": sum(1 for i in table.values()
                               if i.get("level_source") == LEVEL_SOURCE_NVE),
        "levels_estimated": estimated,
        "levels_unresolved": unresolved,
        "estimation_enabled": bool(enabled),
        "margin_m": (bathymetry.DEFAULT_LEVEL_MARGIN_M
                     if margin_m is None else float(margin_m)),
        "method": ("median of the void-repaired DTM inside each lake polygon "
                   "(LiDAR returns the water surface), minus margin_m"),
    }


def describe_layer(server: str, layer: int, *, session=None, timeout: int = 60) -> dict:
    """
    The layer's own schema, from ArcGIS's `?f=json` endpoint: name, geometry type,
    paging cap, and every field with its type and alias.

    This is the authoritative answer to "what is this service actually called
    now", and it is why the *_FIELDS tuples no longer need to be guesswork. Note
    that GeoJSON `properties` carry field NAMES, not aliases — reading the alias
    column and writing it into a candidate tuple is the mistake that cost 0.5.0
    its Strahler orders.
    """
    import requests

    sess = session or requests.Session()
    url = f"{server.rstrip('/')}/{layer}"
    r = sess.get(url, params={"f": "json"}, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(f"ArcGIS error describing {url}: {js['error']}")
    return {
        "url": url,
        "name": js.get("name"),
        "geometry_type": js.get("geometryType"),
        "max_record_count": js.get("maxRecordCount"),
        "fields": [{"name": f.get("name"),
                    "type": str(f.get("type", "")).replace("esriFieldType", ""),
                    "alias": f.get("alias")}
                   for f in js.get("fields", []) or []],
    }


HOVEDELV_NO_ORDER = ("this layer has no Strahler field by design; every segment "
                     "takes DEFAULT_ORDER_MAIN and only upgrades the raster")

WATER_LAYERS = (
    {"label": "elvenett", "server": RIVER_SERVICE, "layer": RIVER_LAYER_ALL,
     "candidates": RIVER_FIELD_CANDIDATES, "critical": CRITICAL_FIELDS, "note": None},
    {"label": "hovedelv", "server": RIVER_SERVICE, "layer": RIVER_LAYER_MAIN,
     "candidates": {"order": RIVER_ORDER_FIELDS}, "critical": NO_CRITICAL_FIELDS,
     "note": HOVEDELV_NO_ORDER},
    {"label": "innsjodatabase", "server": LAKE_SERVICE, "layer": LAKE_LAYER,
     "candidates": LAKE_FIELD_CANDIDATES, "critical": CRITICAL_FIELDS, "note": None},
)


def describe_water_services(*, session=None) -> list:
    """Schema + logical-field resolution for every layer this module reads."""
    out = []
    for spec in WATER_LAYERS:
        base = {"label": spec["label"], "critical": spec["critical"],
                "note": spec["note"]}
        try:
            info = describe_layer(spec["server"], spec["layer"], session=session)
        except Exception as e:  # noqa: BLE001 — reported, not raised
            out.append(dict(base, url=f"{spec['server']}/{spec['layer']}",
                            error=str(e)))
            continue
        names = [f["name"] for f in info["fields"]]
        info.update(base)
        info["resolved"] = resolve_fields(names, spec["candidates"])
        out.append(info)
    return out


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
    # BOTH layers are rasterised: the elvenett network plus the hovedelv overlay.
    # They deliberately overlap; the sort below burns higher orders last, so a main
    # river upgrades the order of the pixels it shares with its elvenett twin.
    # Only `feats.rivers` is serialised as polylines — see WaterFeatures.
    weight_shapes = []   # (polygon, order)
    for seg in list(feats.rivers) + list(feats.main_rivers):
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
            if local > MAX_LOCAL_LAKE_ID:
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

    # Keep the order grid as it stood BEFORE lakes claimed their pixels. Elvenett
    # runs lake-through-lines across every lake, and those pixels are about to be
    # zeroed; `rivernet` needs them to recover a through-line's order (and, more
    # importantly, the through-line is the inflow->outflow connection that makes
    # the river network continuous across a lake).
    weight_raw = weight_arr.copy()

    # lake > river priority: where a lake covers a sample, it's a lake, and the
    # river weight there is cleared (weight is read only when type==RIVER).
    lake_mask = lake_id_arr > 0
    type_arr[lake_mask] = TYPE_LAKE
    weight_arr[lake_mask] = 0

    return WaterGrid(type_arr, weight_arr, lake_id_arr, lake_table,
                     weight_raw=weight_raw)


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
            "rivers": f"{RIVER_SERVICE} (layer {RIVER_LAYER_ALL} elvenett — the "
                      f"authoritative network, serialised as polylines)",
            "rivers_order_overlay": f"{RIVER_SERVICE} (layer {RIVER_LAYER_MAIN} "
                                    f"hovedelv — rasterised for the stream-order "
                                    f"upgrade only, not serialised)",
            "lakes": f"{LAKE_SERVICE} (layer {LAKE_LAYER} Innsjodatabase)",
        },
        "identity_fields": {
            "segment_primary": "strekn_lnr",
            "river": "elvid",
            "catchment": "vassdragsnr",
            "lake": "vatnlnr",
            "note": "objectid is Esri-internal and NOT persisted — it is not "
                    "stable across NVE service updates.",
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

    def P(fx, fy):
        return [x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)]

    # Diagonal trunk river SW->NE. Split at the confluence, the way Elvenett splits
    # real segments, so the connectivity builder has an end-meets-start join to
    # find rather than a T-junction mid-polyline (which it correctly would not
    # link, and which would make this demo silently under-report connectivity).
    trunk_up = RiverSeg(
        xy=np.array([P(0.10, 0.10), P(0.25, 0.22), P(0.40, 0.35)], dtype=np.float64),
        order=5,
        strekn_lnr=100001, elvid="SYN-ELV-0000001", vassdragsnr="002.A1Z",
        feature_index=0,
    )
    trunk_dn = RiverSeg(
        xy=np.array([P(0.40, 0.35), P(0.55, 0.60), P(0.90, 0.90)], dtype=np.float64),
        order=5,
        strekn_lnr=100002, elvid="SYN-ELV-0000001", vassdragsnr="002.A1Z",
        feature_index=1,
    )
    # A tributary meeting the trunk exactly at that split, so the confluence has
    # two upstream segments feeding one downstream.
    trib = RiverSeg(
        xy=np.array([P(0.20, 0.55), P(0.30, 0.45), P(0.40, 0.35)], dtype=np.float64),
        order=3,
        strekn_lnr=100003, elvid="SYN-ELV-0000009", vassdragsnr="002.A1Z",
        feature_index=2,
    )

    # A lake rectangle, plus an inflow / through-line / outflow chain across it, so
    # junction detection and the lake-span logic are exercised offline.
    cx0, cy0 = x0 + 0.60 * (x1 - x0), y0 + 0.25 * (y1 - y0)
    cx1, cy1 = x0 + 0.80 * (x1 - x0), y0 + 0.45 * (y1 - y0)
    lake = LakePoly(
        rings=[np.array([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1], [cx0, cy0]],
                        dtype=np.float64)],
        lopenr=999001, navn="Synthetic Lake", area_m2=(cx1 - cx0) * (cy1 - cy0),
        hoyde_moh=320.0,
    )
    inflow = RiverSeg(
        xy=np.array([P(0.95, 0.60), P(0.85, 0.45), P(0.72, 0.40)], dtype=np.float64),
        order=4, strekn_lnr=100004, elvid="SYN-ELV-0000003",
        vassdragsnr="002.A2Z", vatnlnr=999001, feature_index=3,
    )
    through = RiverSeg(
        xy=np.array([P(0.72, 0.40), P(0.68, 0.32)], dtype=np.float64),
        order=4, strekn_lnr=100005, elvid="SYN-ELV-0000003",
        vassdragsnr="002.A2Z", vatnlnr=999001, feature_index=4,
    )
    outflow = RiverSeg(
        xy=np.array([P(0.68, 0.32), P(0.55, 0.22), P(0.40, 0.12)], dtype=np.float64),
        order=4, strekn_lnr=100006, elvid="SYN-ELV-0000003",
        vassdragsnr="002.A2Z", vatnlnr=999001, feature_index=5,
    )

    # hovedelv overlay: the trunk geometry again at a higher order, and NOT added
    # to `rivers`. Exercises the "upgrade by raster, not by duplicate polyline"
    # path — rivernet must recover order 6 for the trunk by sampling the weight
    # grid, without a sixth polyline ever existing.
    main = RiverSeg(xy=np.vstack([trunk_up.xy, trunk_dn.xy[1:]]), order=6)

    return WaterFeatures(
        rivers=[trunk_up, trunk_dn, trib, inflow, through, outflow],
        main_rivers=[main],
        lakes=[lake],
    )