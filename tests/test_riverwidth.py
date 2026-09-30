import io
import struct
from types import SimpleNamespace

import numpy as np

from kvterrain import n50, rivernet, riverwidth


def _seg(sid, xy, width, spans=()):
    return SimpleNamespace(seg_id=sid, xy=np.asarray(xy, float), width_m=width,
                           lake_spans=list(spans))


def _poly(x0, y0, x1, y1):
    return {"type": "Feature", "properties": {"objtype": "Elv"},
            "geometry": {"type": "Polygon", "coordinates": [
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


def _line(xy, cls):
    return {"type": "Feature", "properties": {"objtype": "ElvBekk", "vannbredde": cls},
            "geometry": {"type": "LineString", "coordinates": [list(p) for p in xy]}}


def test_a_river_in_an_n50_area_takes_its_width_even_off_centre():
    # A river area 40 m wide; our centreline runs 6 m off its middle.
    area = _poly(0, -20, 1000, 20)
    xs = np.arange(100, 900, 5.0)
    s = _seg(1, np.stack([xs, np.full_like(xs, 6.0)], 1), 12.0)
    w, rep = riverwidth.vertex_widths([s], {"features": [area]})
    mid = w[1][20:-20]
    assert np.allclose(mid, 40.0, atol=0.5), mid[:5]
    assert rep["from_polygon"] > 0


def test_the_width_eases_back_where_the_area_ends():
    area = _poly(0, -20, 500, 20)
    xs = np.arange(100, 900, 5.0)
    s = _seg(1, np.stack([xs, np.zeros_like(xs)], 1), 12.0)
    w = riverwidth.vertex_widths([s], {"features": [area]},
                                 params=riverwidth.WidthParams(taper_m=50))[0][1]
    x = xs
    assert np.allclose(w[x < 450], 40.0, atol=0.5)
    assert np.allclose(w[x > 560], 12.0)
    between = w[(x > 500) & (x < 550)]
    assert (np.diff(between) <= 1e-6).all() and between.max() < 40 and between.min() >= 12


def test_a_tributary_reaching_into_a_big_river_keeps_its_own_width():
    area = _poly(0, -40, 1000, 40)
    ys = np.arange(200, -1, -5.0)                    # runs south, ends mid-river
    s = _seg(2, np.stack([np.full_like(ys, 500.0), ys], 1), 4.0)
    w = riverwidth.vertex_widths([s], {"features": [area]})[0][2]
    assert np.allclose(w, 4.0)


def test_width_classes_hold_the_modelled_width():
    xy = np.stack([np.arange(0, 200, 5.0), np.zeros(40)], 1)
    small = _seg(1, xy, 5.0)
    mid = _seg(2, xy + [0, 100], 2.0)
    feats = [_line(xy, 2), _line(xy + [0, 100], 3)]
    w = riverwidth.vertex_widths([small, mid], {"features": feats})[0]
    assert np.allclose(w[1], 3.0) and np.allclose(w[2], 3.0)


def test_no_n50_means_the_modelled_width():
    s = _seg(1, [[0, 0], [10, 0], [20, 0]], 7.0)
    w, rep = riverwidth.vertex_widths([s], None)
    assert np.allclose(w[1], 7.0) and rep["modelled"] == 3


GML = b"""<?xml version="1.0" encoding="UTF-8"?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2"
  xmlns:app="https://skjema.geonorge.no/SOSI/produktspesifikasjon/N50/20230401">
 <gml:featureMember><app:ElvBekk gml:id="a1">
  <app:senterlinje><gml:LineString><gml:posList>100 100 200 200</gml:posList></gml:LineString></app:senterlinje>
  <app:vannbredde>3</app:vannbredde></app:ElvBekk></gml:featureMember>
 <gml:featureMember><app:Elv gml:id="a2"><app:omrade><gml:Surface><gml:patches><gml:PolygonPatch>
  <gml:exterior><gml:LinearRing><gml:posList>0 0 50 0 50 50 0 50 0 0</gml:posList></gml:LinearRing></gml:exterior>
  </gml:PolygonPatch></gml:patches></gml:Surface></app:omrade></app:Elv></gml:featureMember>
 <gml:featureMember><app:Myr gml:id="a3"><app:omrade><gml:Surface><gml:patches><gml:PolygonPatch>
  <gml:exterior><gml:LinearRing><gml:posList>0 0 5 0 5 5 0 0</gml:posList></gml:LinearRing></gml:exterior>
  </gml:PolygonPatch></gml:patches></gml:Surface></app:omrade></app:Myr></gml:featureMember>
 <gml:featureMember><app:ElvBekk gml:id="a4">
  <app:senterlinje><gml:LineString><gml:posList>9000 9000 9100 9100</gml:posList></gml:LineString></app:senterlinje>
  <app:vannbredde>2</app:vannbredde></app:ElvBekk></gml:featureMember>
</gml:FeatureCollection>"""


def test_n50_parser_keeps_river_features_that_reach_the_region():
    feats = n50.parse_arealdekke(io.BytesIO(GML), (-10, -10, 1000, 1000))
    kinds = sorted((f["properties"]["objtype"], f["properties"].get("vannbredde"))
                   for f in feats)
    assert kinds == [("Elv", None), ("ElvBekk", 3)]
    poly = next(f for f in feats if f["properties"]["objtype"] == "Elv")
    assert poly["geometry"]["coordinates"][0][2] == [50.0, 50.0]


def test_river_widths_bin_follows_rivers_bin_vertex_order(tmp_path):
    segs = [rivernet.RiverSegment(seg_id=0, geom_hash="00" * 16, order=1,
                                  xy=np.zeros((3, 2)), z=None, level=None, width_m=2.0,
                                  vertex_width_m=np.array([2, 5, 9], np.float32)),
            rivernet.RiverSegment(seg_id=1, geom_hash="00" * 16, order=1,
                                  xy=np.zeros((2, 2)), z=None, level=None, width_m=4.0)]
    net = rivernet.RiverNetwork(segments=segs)
    p = tmp_path / "w.bin"
    man = rivernet.write_river_widths_bin(net, str(p))
    b = p.read_bytes()
    magic, ver, ns, nv = struct.unpack_from("<8sIII", b, 0)
    assert (magic, ver, ns, nv) == (b"KVRWIDT1", 1, 2, 5) and man["vertices"] == 5
    off = 20
    sid, n = struct.unpack_from("<II", b, off)
    w0 = np.frombuffer(b, "<f4", n, off + 8)
    off += 8 + 4 * n
    sid1, n1 = struct.unpack_from("<II", b, off)
    w1 = np.frombuffer(b, "<f4", n1, off + 8)
    assert (sid, n, sid1, n1) == (0, 3, 1, 2)
    assert w0.tolist() == [2, 5, 9] and w1.tolist() == [4, 4]
