"""
kvterrain.riverwidth
====================

A width for every river vertex, from N50 where it knows and discharge where not.

`rivernet.assign_discharge_and_width` gives each segment ONE width from its modelled
discharge (`width = clamp(a * Q^b)`). The runtime shares that function, and it stays
the hydraulic width in `rivers.bin`. It is right for streams and wrong for the trunk
rivers: on Lierne (2026-09-30) the rivers N50 maps as water areas are 15-57 m wide
(median 28 m) where the discharge model made them 5-15 m (median 12 m), a uniform
ribbon inside a channel twice its width.

So the VISUAL width — what is rasterised, carved, levelled and stamped — is set per
vertex (`vertex_widths`):

  * Near an N50 river POLYGON (`Elv`): the polygon's width across the river at that
    vertex, measured along the centreline's normal, then median- and mean-smoothed
    along the river so a bay or an island does not make a spike. Where the polygon
    ends and the river goes on as a line, the width eases back to the modelled one
    over `taper_m`.
  * On an N50 CENTRELINE with a width class (`ElvBekk.vannbredde`): the modelled
    width held to the class — at most 3 m for class 2, 3-15 m for class 3 — so it
    still grows downstream the way discharge says.
  * Elsewhere (outside Norway, or no N50 in the dataset): the modelled width.

N50 is 1:50 000 data (position accuracy about 20 m), but NVE's Elvenett was drawn
from the same geometry, so our centrelines sit on its lines; the polygon widths are
measured across our own centreline. Where the polygon and the scanned channel still
disagree, the carve's bank tolerance keeps the water off anything standing above
the river's level, as it always has.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class WidthParams:
    enabled: bool = True
    # A vertex this close to an N50 river polygon is measured across it.
    polygon_reach_m: float = 20.0
    # A vertex this close to an N50 centreline takes its width class.
    line_reach_m: float = 15.0
    # How far across the river the normal is cast each way.
    max_half_width_m: float = 150.0
    # Smoothing along the river: a running median, then a running mean (vertices).
    median_window: int = 7
    mean_window: int = 7
    # From a polygon's end back to the modelled width, over this distance.
    taper_m: float = 50.0
    # Class limits (vannbredde 2 and 3).
    class2_max_m: float = 3.0
    class3_min_m: float = 3.0
    class3_max_m: float = 15.0

    def to_dict(self) -> dict:
        return asdict(self)


def _fill_gaps(a: np.ndarray) -> np.ndarray:
    """NaNs linearly interpolated from their finite neighbours (ends held)."""
    ok = np.isfinite(a)
    if ok.all() or not ok.any():
        return a
    i = np.arange(a.size)
    return np.interp(i, i[ok], a[ok])


def _running(fn, a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or a.size < 2:
        return a.copy()
    h = w // 2
    p = np.pad(a, h, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(p, w)
    return fn(win, axis=1)


def _normals(xy: np.ndarray) -> np.ndarray:
    t = np.gradient(xy, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def vertex_widths(segments: list, n50: Optional[dict], *,
                  params: Optional[WidthParams] = None,
                  centres: Optional[dict] = None) -> tuple:
    """({seg_id: f32 width per vertex}, report). Segments are traced
    `rivernet.RiverSegment`s with their modelled `width_m` set.

    `centres`, when a dict is passed, receives {seg_id: offset per vertex}: how
    far along the vertex's normal the middle of the N50 river area lies, smoothed
    like the widths and eased to zero over `taper_m` where the area ends; 0 where
    the width is not an area's. N50's lines run off-centre inside their own areas
    (Lierne, 2026-09-30: by a median 3 m and up to 75 m), and a width stamped
    around such a line spills past the near bank by the offset, which
    `thalweg.snap_segments` removes by moving the line."""
    from shapely import STRtree, LineString, Polygon
    from shapely import points as shp_points, distance as shp_distance

    p = params or WidthParams()
    out: dict = {}
    rep = {"params": p.to_dict(), "n50": n50 is not None, "vertices": 0,
           "from_polygon": 0, "from_class2": 0, "from_class3": 0, "modelled": 0}
    feats = (n50 or {}).get("features") or [] if p.enabled else []
    lines, classes, polys = [], [], []
    for f in feats:
        g = f.get("geometry") or {}
        props = f.get("properties") or {}
        if g.get("type") == "LineString" and props.get("vannbredde") in (2, 3):
            c = np.asarray(g["coordinates"], float)
            if c.shape[0] >= 2:
                lines.append(LineString(c))
                classes.append(int(props["vannbredde"]))
        elif g.get("type") == "Polygon":
            rings = [np.asarray(r, float) for r in g["coordinates"]]
            if rings and rings[0].shape[0] >= 4:
                poly = Polygon(rings[0], [r for r in rings[1:] if r.shape[0] >= 4])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    polys.append(poly)
    line_tree = STRtree(lines) if lines else None
    classes = np.asarray(classes, int)
    polys_arr = np.asarray(polys, dtype=object)
    poly_tree = STRtree(polys) if polys else None
    widths_measured = []

    for s in segments:
        xy = np.asarray(s.xy, float)
        n = xy.shape[0]
        base = float(s.width_m or 0.0)
        w = np.full(n, base, np.float64)
        src = np.zeros(n, np.int8)            # 0 modelled, 2/3 class, 9 polygon
        inlake = np.zeros(n, bool)
        for a, b, _ in (s.lake_spans or ()):
            inlake[a:b + 1] = True
        rep["vertices"] += n
        if n < 2 or (line_tree is None and poly_tree is None):
            out[s.seg_id] = w.astype(np.float32)
            rep["modelled"] += n
            continue
        pts = shp_points(xy)

        if line_tree is not None:
            idx, dist = line_tree.query_nearest(pts, max_distance=p.line_reach_m,
                                                return_distance=True, all_matches=False)
            if idx.size:
                v, li = idx[0], idx[1]
                cl = classes[li]
                c2 = v[cl == 2]
                c3 = v[cl == 3]
                w[c2] = np.minimum(w[c2], p.class2_max_m)
                w[c3] = np.clip(w[c3], p.class3_min_m, p.class3_max_m)
                src[c2], src[c3] = 2, 3

        if poly_tree is not None:
            pidx, pdist = poly_tree.query_nearest(pts, max_distance=p.polygon_reach_m,
                                                  return_distance=True, all_matches=False)
            keep = ~inlake[pidx[0]] if pidx.size else np.zeros(0, bool)
            near, which = pidx[0][keep], pidx[1][keep]
            if near.size:
                meas = np.full(n, np.nan)
                mid = np.full(n, np.nan)
                meas[near], mid[near] = _measure(xy, near, polys_arr[which], p)
                have = np.isfinite(meas)
                if have.any():
                    # Smooth over the measured run(s) only.
                    m = meas.copy()
                    for a, b in _runs(have):
                        seg = m[a:b]
                        seg = _running(np.median, seg, p.median_window)
                        seg = _running(np.mean, seg, p.mean_window)
                        m[a:b] = seg
                    # Ease back to the width outside the polygon over taper_m.
                    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
                    hv = np.flatnonzero(have)
                    j = np.searchsorted(arc[hv], arc)
                    lo = hv[np.clip(j - 1, 0, hv.size - 1)]
                    hi = hv[np.clip(j, 0, hv.size - 1)]
                    near_i = np.where(np.abs(arc - arc[lo]) <= np.abs(arc - arc[hi]), lo, hi)
                    d = np.abs(arc - arc[near_i])
                    t = np.clip(d / max(p.taper_m, 1e-6), 0.0, 1.0)
                    t = t * t * (3 - 2 * t)
                    blend = (1.0 - t) * m[near_i] + t * w
                    take = (~inlake) & (d <= p.taper_m)
                    w[take] = blend[take]
                    if centres is not None:
                        c = np.zeros(n)
                        for a, b in _runs(have):
                            seg = _fill_gaps(mid[a:b])
                            if not np.isfinite(seg).any():
                                continue
                            seg = _running(np.median, seg, p.median_window)
                            c[a:b] = _running(np.mean, seg, p.mean_window)
                        c = np.where(take, (1.0 - t) * c[near_i], 0.0)
                        centres[s.seg_id] = c
                    src[have] = 9
                    widths_measured.append(m[have])
        w = np.maximum(w, 0.5)
        out[s.seg_id] = w.astype(np.float32)
        rep["from_polygon"] += int((src == 9).sum())
        rep["from_class2"] += int((src == 2).sum())
        rep["from_class3"] += int((src == 3).sum())
        rep["modelled"] += int((src == 0).sum())
    if widths_measured:
        allm = np.concatenate(widths_measured)
        rep["polygon_width_m"] = {
            "p10": round(float(np.percentile(allm, 10)), 1),
            "p50": round(float(np.median(allm)), 1),
            "p90": round(float(np.percentile(allm, 90)), 1)}
    return out, rep


def water_areas(n50: Optional[dict], *, margin_m: float = 0.0):
    """The union of N50's river polygons grown by `margin_m`, or None. The river
    mask is clipped to it wherever a width came from those polygons, so a width
    measured at a confluence or a bend cannot spill past the mapped banks."""
    import shapely
    polys = []
    for f in (n50 or {}).get("features") or []:
        g = f.get("geometry") or {}
        if g.get("type") == "Polygon":
            rings = [np.asarray(r, float) for r in g["coordinates"]]
            if rings and rings[0].shape[0] >= 4:
                poly = shapely.Polygon(rings[0], [r for r in rings[1:] if r.shape[0] >= 4])
                polys.append(poly if poly.is_valid else poly.buffer(0))
    if not polys:
        return None
    u = shapely.union_all(polys)
    return u.buffer(margin_m) if margin_m > 0 else u


def _measure(xy: np.ndarray, idx: np.ndarray, polys: np.ndarray, p) -> tuple:
    """Width of `polys[k]` across the centreline at vertex `idx[k]`: the piece of the
    normal through the vertex that lies in the polygon, nearest the vertex. Capped
    at four times the vertex's distance to the bank, so a normal that runs along
    the river at a bend, or out into a confluence, cannot report the river's
    length as its width. Also returns the signed distance along the normal to that
    piece's middle, NaN where the width is NaN or was capped (the piece is then
    not the river's cross-section)."""
    import shapely
    nrm = _normals(xy)[idx]
    c = xy[idx]
    a = c - p.max_half_width_m * nrm
    b = c + p.max_half_width_m * nrm
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    inter = shapely.intersection(lines, polys)
    pts = shapely.points(c)
    out = np.full(idx.size, np.nan)
    mid = np.full(idx.size, np.nan)
    for k, g in enumerate(inter):
        if g is None or g.is_empty:
            continue
        parts = getattr(g, "geoms", [g])
        best = min(parts, key=lambda q: q.distance(pts[k]))
        if best.distance(pts[k]) <= p.polygon_reach_m and best.length > 0:
            out[k] = best.length
            ends = np.asarray(best.coords, dtype=np.float64)
            mid[k] = float(np.dot(0.5 * (ends[0] + ends[-1]) - c[k], nrm[k]))
    # A tributary poking into a big river's polygon is not that river: across its
    # own line the polygon is long (the main river's length), along it short (the
    # main river's width). Keep a measurement only where the polygon runs at least
    # nearly as far along the line as across it.
    tng = np.stack([nrm[:, 1], -nrm[:, 0]], axis=1)
    along = shapely.intersection(shapely.linestrings(np.stack(
        [c - p.max_half_width_m * tng, c + p.max_half_width_m * tng], axis=1)), polys)
    for k, g in enumerate(along):
        if not np.isfinite(out[k]):
            continue
        parts = [] if g is None or g.is_empty else getattr(g, "geoms", [g])
        near_parts = [q for q in parts if q.distance(pts[k]) <= 1e-6]
        if not near_parts or near_parts[0].length < 0.75 * out[k]:
            out[k] = np.nan
    to_bank = shapely.distance(pts, shapely.boundary(polys))
    # Only a vertex inside the polygon is measured; one outside it, where a polygon
    # ends or beside it, takes the taper from its measured neighbours.
    inside = shapely.contains(polys, pts)
    cap = np.maximum(4.0 * to_bank, 1.0)
    width = np.where(inside, np.minimum(out, cap), np.nan)
    mid = np.where(np.isfinite(width) & (out <= cap), mid, np.nan)
    return width, mid


def _runs(mask: np.ndarray) -> list:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fit_to_areas(segments: list, areas, *, margin_m: float = 0.0) -> dict:
    """Cap each vertex width inside an N50 river area at twice its distance to the
    area's nearest bank (plus `margin_m`), in place on `vertex_width_m`.

    The runtime stamps a river as circles of the vertex width around the line.
    A width measured straight across is right for the river, but where the area
    bends, narrows ahead or has an island, the circle reaches a nearer bank in
    another direction and spills past it. On Lierne (2026-09-30), even with the
    lines centred, 4% of the vertices in N50 areas spilled more than 10 m. Run
    after `thalweg.snap_segments`, on the lines as they will ship.
    `areas` is `water_areas(...)`, None to do nothing."""
    import shapely
    from shapely import STRtree

    rep = {"vertices_capped": 0, "reduction_median_m": 0.0}
    if areas is None:
        return rep
    parts = list(getattr(areas, "geoms", [areas]))
    tree = STRtree(parts)
    rings = [shapely.boundary(p) for p in parts]
    drops = []
    for s in segments:
        vw = getattr(s, "vertex_width_m", None)
        if vw is None or len(vw) != s.n:
            continue
        pts = shapely.points(np.asarray(s.xy, dtype=np.float64))
        pi, pj = tree.query(pts, predicate="within")
        if pi.size == 0:
            continue
        d = np.full(s.n, np.inf)
        for i, j in zip(pi, pj):
            d[i] = min(d[i], shapely.distance(pts[i], rings[j]))
        cap = 2.0 * d + float(margin_m)
        w = np.asarray(vw, dtype=np.float64)
        over = w > cap
        if not over.any():
            continue
        # Ease into the cap rather than notching one vertex.
        capped = np.minimum(w, _running(np.min, np.minimum(w, cap), 5))
        capped = np.minimum(_running(np.mean, capped, 5), np.minimum(w, cap))
        changed = capped < w - 1e-6
        drops.append((w - capped)[changed])
        rep["vertices_capped"] += int(changed.sum())
        s.vertex_width_m = capped.astype(np.float32)
    if drops:
        dd = np.concatenate(drops)
        rep["reduction_median_m"] = round(float(np.median(dd)), 2)
        rep["reduction_max_m"] = round(float(dd.max()), 2)
    return rep
