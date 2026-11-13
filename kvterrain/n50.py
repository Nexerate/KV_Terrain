"""
kvterrain.n50
=============

River widths from Kartverket's N50 Kartdata (open data, CC BY 4.0).

The river network (NVE Elvenett) carries no width, so every river's width was
modelled from its discharge. That is right for streams, but the trunk rivers came
out a median 12 m wide where N50 maps them as 15-57 m wide areas (Lierne,
2026-09-30). N50 has what we need:

  * `ElvBekk` centrelines with `vannbredde`, a width class at 1:50 000:
    2 = narrower than 3 m, 3 = 3-15 m. NVE Elvenett was drawn from the same
    geometry, so the two coincide (median offset 0 m on Lierne).
  * `Elv` polygons for rivers wider than that: the water area itself, from which
    a width can be measured at every vertex of our centreline.

N50 is delivered per municipality (the Geonorge download API; its polygon
selection is not used). A fetch finds the municipalities under the region with
Kartverket's point lookup, downloads each one's ready-made GML, and keeps only the
river features from its land-cover theme (`N50Arealdekke`) that reach the region.
Raw, like every other fetch-stage layer: the widths are derived at process time
(`riverwidth`).
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import zipfile
import xml.etree.ElementTree as ET
from typing import Callable, Optional

import numpy as np

N50_UUID = "ea192681-d039-42ec-b1bc-f3ce04c189ac"
ORDER_URL = "https://nedlasting.geonorge.no/api/order"
KOMMUNE_URL = "https://ws.geonorge.no/kommuneinfo/v1/punkt"
ATTRIBUTION = "© Kartverket, N50 Kartdata (CC BY 4.0)"
LICENSE = "CC BY 4.0"
LOOKUP_STEP_M = 2000.0
REGION_MARGIN_M = 200.0
KEEP = ("ElvBekk", "Elv")
GML = "{http://www.opengis.net/gml/3.2}"

Progress = Callable[[float, str], None]


def _session(session=None):
    if session is not None:
        return session
    import requests
    return requests.Session()


def region_bbox(plan) -> tuple:
    x0, y0 = float(plan.origin_x), float(plan.origin_y)
    x1 = x0 + (plan.samples_x - 1) * float(plan.spacing_m)
    y1 = y0 + (plan.samples_y - 1) * float(plan.spacing_m)
    return x0, y0, x1, y1


def municipalities(plan, *, session=None, step_m: float = LOOKUP_STEP_M,
                   workers: int = 8) -> list:
    """(number, name) of every municipality under a grid of points `step_m` apart
    over the region (edges included). Points outside Norway find none."""
    s = _session(session)
    x0, y0, x1, y1 = region_bbox(plan)
    xs = np.linspace(x0, x1, max(int(np.ceil((x1 - x0) / step_m)) + 1, 2))
    ys = np.linspace(y0, y1, max(int(np.ceil((y1 - y0) / step_m)) + 1, 2))

    def look(p):
        x, y = p
        try:
            r = s.get(KOMMUNE_URL, params={"nord": f"{y:.0f}", "ost": f"{x:.0f}",
                                           "koordsys": str(plan.epsg)}, timeout=30)
            if r.status_code != 200:
                return None
            d = r.json()
            return str(d["kommunenummer"]), str(d.get("kommunenavn", ""))
        except Exception:
            return None

    with cf.ThreadPoolExecutor(workers) as ex:
        found = set(r for r in ex.map(look, [(x, y) for x in xs for y in ys]) if r)
    return sorted(found)


def _order(number: str, name: str, epsg: int, session) -> str:
    body = {"orderLines": [{
        "metadataUuid": N50_UUID,
        "areas": [{"code": number, "type": "kommune", "name": name}],
        "projections": [{"code": str(epsg)}],
        "formats": [{"name": "GML"}]}]}
    r = session.post(ORDER_URL, json=body, timeout=60)
    r.raise_for_status()
    files = r.json().get("files") or []
    ok = [f for f in files if f.get("status") == "ReadyForDownload"]
    if not ok:
        raise RuntimeError(f"N50 for municipality {number} {name} is not ready "
                           f"for download: {[f.get('status') for f in files]}")
    return ok[0]["downloadUrl"]


def _coords(el) -> list:
    return [[[float(a), float(b)] for a, b in
             zip(*[iter(p.text.split())] * 2)] for p in el.iter(f"{GML}posList")]


def parse_arealdekke(fh, bbox: tuple, *, source: str = "") -> list:
    """GeoJSON features for the N50 river objects in an `N50Arealdekke` GML stream
    whose bounds reach `bbox` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = bbox
    out = []
    ns = None
    for _, el in ET.iterparse(fh, events=("end",)):
        tag = el.tag
        if ns is None and tag.startswith("{") and "produktspesifikasjon/N50" in tag:
            ns = tag.split("}")[0] + "}"
        if ns is None or not tag.startswith(ns):
            continue
        name = tag[len(ns):]
        if name not in KEEP:
            if name[:1].isupper() and name not in ("Posisjonskvalitet", "Retning"):
                el.clear()
            continue
        rings = _coords(el)
        el_id = el.get(f"{GML}id")
        vb = el.find(f"{ns}vannbredde")
        el.clear()
        if not rings:
            continue
        pts = np.asarray([p for r in rings for p in r])
        if (pts[:, 0].max() < x0 or pts[:, 0].min() > x1 or
                pts[:, 1].max() < y0 or pts[:, 1].min() > y1):
            continue
        geom = ({"type": "Polygon", "coordinates": rings} if name == "Elv"
                else {"type": "LineString", "coordinates": rings[0]})
        props = {"objtype": name, "id": el_id, "source": source}
        if vb is not None and vb.text:
            props["vannbredde"] = int(vb.text)
        out.append({"type": "Feature", "properties": props, "geometry": geom})
    return out


def fetch_n50_rivers(plan, *, session=None, progress: Optional[Progress] = None,
                     workers: int = 4) -> dict:
    """The N50 river features reaching the region, from every municipality under
    it: {"features": [...], "municipalities": [[number, name], ...]}."""
    s = _session(session)
    if progress:
        progress(0.0, "N50: finding municipalities")
    munis = municipalities(plan, session=s)
    x0, y0, x1, y1 = region_bbox(plan)
    bbox = (x0 - REGION_MARGIN_M, y0 - REGION_MARGIN_M,
            x1 + REGION_MARGIN_M, y1 + REGION_MARGIN_M)

    def one(m):
        number, name = m
        url = _order(number, name, int(plan.epsg), s)
        r = s.get(url, timeout=600)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            member = next((n for n in z.namelist()
                           if "Arealdekke" in n and n.lower().endswith(".gml")), None)
            if member is None:
                raise RuntimeError(f"no N50Arealdekke GML for {number} {name}")
            with z.open(member) as fh:
                return parse_arealdekke(fh, bbox, source=f"{number} {name}")

    feats, seen = [], set()
    done = 0
    with cf.ThreadPoolExecutor(max(1, workers)) as ex:
        for part in ex.map(one, munis):
            done += 1
            if progress:
                progress(done / max(len(munis), 1),
                         f"N50: {done}/{len(munis)} municipalities")
            for f in part:
                key = f["properties"].get("id")
                if key in seen:
                    continue          # a feature on a municipality border
                seen.add(key)
                feats.append(f)
    return {"features": feats, "municipalities": [list(m) for m in munis]}
