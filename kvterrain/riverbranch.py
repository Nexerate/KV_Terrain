"""
Side channels: the parts of N50's river areas that no mapped line runs through.

NVE's network has one line per river. Where a river splits around islands or fans
into a lake over several mouths, the line follows one channel and N50's river area
holds them all. The runtime draws water only along lines, so the other channels
were bare ground beside a river that narrowed to a third of its width (Ågardselva
into Skjeldbreivatnet, owner, 2026-10-01). On Lierne 18% of the N50 river area
inside the region was undrawn, 89% of that with no line through it at all.

This module draws a line down the middle of each undrawn piece that connects:

* a SIDE ARM leaves a drawn river and rejoins it further down;
* a DELTA CHANNEL leaves a drawn river and ends in the lake that river runs into.

Where the river's own line runs beside its N50 area rather than in it, the
undrawn piece is that river, not a second channel, and is left alone.

A piece with a single contact (a bulge, a backwater, a pool) is left for later.

The line is the cheapest path through the piece with a cost that falls with the
distance to its banks, so it keeps to the middle; its width at each vertex is the
piece's width there (twice the distance to the nearer bank), so the circle the
runtime stamps stays inside the mapped area.

Each side channel is a segment of its own with no graph links: the runtime's flow
pass and the exporter's accumulation both assume one downstream per segment, and a
side arm linked back into its river would count the same water twice. Instead it
carries `branch_of`, written to `river_branches.bin`: the segment and vertex it
leaves from, its share of that segment's flow, and where it rejoins. Its share is
w_b^2 / (w_t^2 + sum w^2), over the median widths of the river alongside and of
every channel beside the same reach of it (owner, 2026-10-01; `_share_groups`).
Its level runs from the river's level at the split to the river's level where it
rejoins, or to the lake's (`riverbed.tie_branches`).
"""

from __future__ import annotations

import heapq
import struct
import os
from dataclasses import dataclass
from typing import Optional

import numba as nb
import numpy as np

from . import core
from . import rivernet
from . import water as kvwater

# A piece smaller than this is a bank wobble, not a channel.
BRANCH_MIN_AREA_M2 = 1500.0
# Raster resolution of the per-piece work. N50 arms run down to ~5 m wide.
BRANCH_RES_M = 1.0
# Opening radius: a strip of N50 area narrower than twice this beside a drawn
# river is the river's own bank showing past its stamped width, not a channel.
BRANCH_OPEN_M = 3.0
# How far a piece may stand off a drawn river or a lake and still touch it: the
# opening plus the stamp's rounding.
BRANCH_CONTACT_M = 6.0
BRANCH_MIN_LENGTH_M = 40.0
BRANCH_MIN_WIDTH_M = 4.0
# A segment carries an N50 area (rather than crossing one) when this share of its
# vertices, or this length, lies inside it.
CARRIER_INSIDE_FRACTION = 0.5
CARRIER_INSIDE_M = 200.0
# Two contact vertices further apart than this along the river are two contacts.
CONTACT_SPLIT_M = 20.0
# A delta channel's lake must lie this close downstream along its river.
LAKE_SEARCH_M = 3000.0
# How far a delta channel runs on into its lake, so it has a lake span.
LAKE_OVERRUN_M = 15.0
# The scale the traced line, and its width, are smoothed over.
BRANCH_SMOOTH_M = 15.0
BRANCH_WIDTH_SMOOTH_M = 25.0
# A channel leaves its river this far (plus the river's half-width) upstream of
# where it begins, and rejoins as far downstream of where it ends.
BRANCH_LEAD_M = 10.0

BRANCHES_FILE = "river_branches.bin"
BRANCHES_MAGIC = b"KVRBRAN1"
BRANCHES_VERSION = 1
NO_SEGMENT = -1


@dataclass
class BranchOf:
    parent: int                 # seg_id of the river it leaves
    parent_vertex: int
    share: float                # of the parent's flow
    rejoin: int = NO_SEGMENT    # seg_id it rejoins, NO_SEGMENT for a delta channel
    rejoin_vertex: int = 0
    lake_id: int = 0            # local lake id a delta channel ends in


# --------------------------------------------------------------------------- #
# Network position                                                             #
# --------------------------------------------------------------------------- #

def _seg_lengths(s) -> np.ndarray:
    """Cumulative length from each vertex to the segment's end."""
    d = np.hypot(*np.diff(s.xy, axis=0).T)
    return np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])


def _downstream_positions(segments: list) -> tuple[dict, dict, dict]:
    """Per segment: metres from each vertex to its network's sink, the sink's id,
    and the vertex-to-end lengths."""
    by_id = {s.seg_id: s for s in segments}
    tail = {s.seg_id: _seg_lengths(s) for s in segments}
    below: dict = {}
    sink: dict = {}
    for s in segments:
        if s.seg_id in below:
            continue
        chain, seen = [], set()
        sid = s.seg_id
        while sid is not None and sid in by_id and sid not in seen and sid not in below:
            seen.add(sid)
            chain.append(sid)
            sid = by_id[sid].downstream
        if sid is not None and sid in below:
            acc, end = below[sid] + tail[sid][0], sink[sid]
        else:
            acc, end = 0.0, chain[-1]
        for cid in reversed(chain):
            below[cid] = acc
            sink[cid] = end
            acc += tail[cid][0]
    pos = {sid: tail[sid] + below[sid] for sid in by_id}
    return pos, sink, tail


def _chain_after(by_id: dict, sid: int, k: int, metres: float):
    """(seg_id, vertex) pairs from (sid, k) downstream for `metres`."""
    out, run, seen = [], 0.0, set()
    while sid is not None and sid in by_id and sid not in seen:
        seen.add(sid)
        s = by_id[sid]
        for v in range(k, s.n):
            out.append((sid, v))
            if v > k:
                run += float(np.hypot(*(s.xy[v] - s.xy[v - 1])))
            if run >= metres:
                return out
        sid, k = s.downstream, 0
    return out


# --------------------------------------------------------------------------- #
# Path                                                                         #
# --------------------------------------------------------------------------- #

@nb.njit(cache=True)
def _cheapest_path(cost, src_r, src_c, target):
    """8-connected Dijkstra from one pixel to the first `target` pixel reached;
    the path as flat indices, empty when none is reachable. `cost` is per metre,
    inf where blocked."""
    H, W = cost.shape
    dist = np.full((H, W), np.inf)
    prev = np.full((H, W), -1, dtype=np.int64)
    dist[src_r, src_c] = 0.0
    heap = [(0.0, np.int64(src_r * W + src_c))]
    end = np.int64(-1)
    while len(heap) > 0:
        d, i = heapq.heappop(heap)
        r = i // W
        c = i - r * W
        if d > dist[r, c]:
            continue
        if target[r, c]:
            end = i
            break
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                w = cost[rr, cc]
                if not np.isfinite(w):
                    continue
                step = 1.4142135623730951 if (dr != 0 and dc != 0) else 1.0
                nd = d + step * 0.5 * (cost[r, c] + w)
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    prev[rr, cc] = i
                    heapq.heappush(heap, (nd, np.int64(rr * W + cc)))
    if end < 0:
        return np.zeros(0, dtype=np.int64)
    n = 0
    i = end
    while i >= 0:
        n += 1
        i = prev[i // W, i - (i // W) * W]
    path = np.empty(n, dtype=np.int64)
    i = end
    for k in range(n - 1, -1, -1):
        path[k] = i
        i = prev[i // W, i - (i // W) * W]
    return path


def _even(xy: np.ndarray, step: float) -> np.ndarray:
    """The polyline resampled at `step` metres by arc length, ends kept."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    m = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], m)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], 1)


def _smooth(xy: np.ndarray, passes: int = 3, win: int = 5) -> np.ndarray:
    """Moving average with the ends held, to take the pixel staircase out."""
    out = np.asarray(xy, dtype=np.float64).copy()
    if out.shape[0] <= win:
        return out
    h = win // 2
    k = np.ones(win) / win
    for _ in range(passes):
        pad = np.concatenate([np.repeat(out[:1], h, 0), out, np.repeat(out[-1:], h, 0)])
        sm = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(2)], 1)
        sm[0], sm[-1] = out[0], out[-1]
        out = sm
    return out


def _disc(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius_px * radius_px + 1e-9


# --------------------------------------------------------------------------- #
# The pass                                                                     #
# --------------------------------------------------------------------------- #

class _Window:
    """A north-up local raster: pixel (i, j) has its centre at
    (x0 + (j + 0.5) res, y1 - (i + 0.5) res)."""

    def __init__(self, x0, y0, x1, y1, res):
        self.res = float(res)
        self.x0 = float(np.floor(x0))
        self.y1 = float(np.ceil(y1))
        self.W = int(np.ceil((x1 - self.x0) / res)) + 1
        self.H = int(np.ceil((self.y1 - y0) / res)) + 1
        from affine import Affine
        self.transform = Affine(res, 0, self.x0, 0, -res, self.y1)

    def ij(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        j = np.floor((xy[:, 0] - self.x0) / self.res).astype(np.int64)
        i = np.floor((self.y1 - xy[:, 1]) / self.res).astype(np.int64)
        return i, j

    def xy(self, i, j):
        return np.stack([self.x0 + (np.asarray(j) + 0.5) * self.res,
                         self.y1 - (np.asarray(i) + 0.5) * self.res], 1)

    def inside(self, i, j):
        return (i >= 0) & (i < self.H) & (j >= 0) & (j < self.W)


def _carriers(segments: list, areas) -> tuple[dict, list]:
    """seg_id -> bool mask of vertices inside an area, for the segments that carry
    one; and the carrier segments."""
    import shapely
    out, carriers = {}, []
    for s in segments:
        inside = shapely.contains_xy(areas, s.xy[:, 0], s.xy[:, 1])
        if not inside.any():
            continue
        step = np.hypot(*np.diff(s.xy, axis=0).T)
        inside_m = float(step[inside[1:] & inside[:-1]].sum())
        if inside.mean() >= CARRIER_INSIDE_FRACTION or inside_m >= CARRIER_INSIDE_M:
            out[s.seg_id] = inside
            carriers.append(s)
    return out, carriers


def _half_widths(s) -> np.ndarray:
    vw = getattr(s, "vertex_width_m", None)
    if vw is None or len(vw) != s.n:
        return np.full(s.n, 0.5 * float(s.width_m or 1.0))
    return 0.5 * np.asarray(vw, dtype=np.float64)


def _candidate_windows(plan: core.GridPlan, areas, carriers: list, wg) -> list:
    """Bounding boxes, in world metres, of the undrawn pieces of N50 area found
    on the leaf grid (coarse; each is redone at BRANCH_RES_M)."""
    import shapely
    from rasterio.features import rasterize
    from scipy.ndimage import label, find_objects
    SY, SX = plan.samples_y, plan.samples_x
    tr = kvwater.sample_grid_transform(plan, level=0)
    a = rasterize([(g, 1) for g in getattr(areas, "geoms", [areas])],
                  out_shape=(SY, SX), transform=tr, fill=0, dtype="uint8").astype(bool)
    shapes = []
    for s in carriers:
        hw = _half_widths(s) + 0.5 * plan.spacing_m
        shapes.extend(shapely.buffer(shapely.points(s.xy), hw).tolist())
    drawn = (rasterize([(g, 1) for g in shapes], out_shape=(SY, SX), transform=tr,
                       fill=0, dtype="uint8").astype(bool) if shapes
             else np.zeros((SY, SX), bool))
    miss = a & ~drawn & (wg.type != kvwater.TYPE_LAKE)
    lab, n = label(miss, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    cells = np.bincount(lab.ravel(), minlength=n + 1)
    big = np.flatnonzero(cells * plan.spacing_m ** 2 >= BRANCH_MIN_AREA_M2)
    big = big[big > 0]
    out = []
    sl = find_objects(lab)
    for k in big:
        rs, cs = sl[k - 1]
        x0 = plan.origin_x + cs.start * plan.spacing_m
        x1 = plan.origin_x + (cs.stop - 1) * plan.spacing_m
        y1 = plan.origin_y + (SY - 1 - rs.start) * plan.spacing_m
        y0 = plan.origin_y + (SY - rs.stop) * plan.spacing_m
        out.append((x0, y0, x1, y1, lab[rs, cs] == k, rs, cs))
    return out


def add_side_channels(
    plan: core.GridPlan,
    traced: "rivernet.TracedNetwork",
    areas,
    wg: kvwater.WaterGrid,
    *,
    width_scale: float = 1.0,
    depth_from_q: bool = True,
) -> dict:
    """Append a segment for each side arm and delta channel to `traced.segments`,
    each with `branch_of` set. Run after the visual widths are final for the mapped
    lines (`riverwidth.fit_to_areas`) and before the river raster is stamped."""
    import shapely
    from rasterio.features import rasterize
    from scipy.ndimage import (binary_dilation, binary_opening,
                               distance_transform_edt, label)

    rep = {"pieces_examined": 0, "side_arms": 0, "delta_channels": 0,
           "skipped_one_contact": 0, "skipped_lake_upstream": 0,
           "skipped_no_path": 0, "skipped_short_or_narrow": 0,
           "skipped_line_beside_area": 0,
           "branch_length_m": 0.0, "share_median": None}
    if areas is None or areas.is_empty:
        return rep
    segments = traced.segments
    by_id = {s.seg_id: s for s in segments}
    inside_by_seg, carriers = _carriers(segments, areas)
    if not carriers:
        return rep
    pos, sink, _tail = _downstream_positions(segments)
    windows = _candidate_windows(plan, areas, carriers, wg)

    # Every carrier vertex, for the per-window lookups.
    cv_xy = np.concatenate([s.xy for s in carriers])
    cv_hw = np.concatenate([_half_widths(s) for s in carriers])
    cv_seg = np.concatenate([np.full(s.n, s.seg_id) for s in carriers])
    cv_k = np.concatenate([np.arange(s.n) for s in carriers])
    cv_in_lake = np.concatenate([_span_mask(s) for s in carriers])
    tree = shapely.STRtree(shapely.points(cv_xy))
    area_parts = list(getattr(areas, "geoms", [areas]))
    area_tree = shapely.STRtree(area_parts)

    stride = float(traced.stride_m)
    res = BRANCH_RES_M
    open_px = BRANCH_OPEN_M / res
    contact_px = BRANCH_CONTACT_M / res
    margin = 40.0 + BRANCH_CONTACT_M
    new: list = []
    shares: list = []
    reach: list = []
    for (x0, y0, x1, y1, coarse, rs, cs) in windows:
        win = _Window(x0 - margin, y0 - margin, x1 + margin, y1 + margin, res)
        box = shapely.box(win.x0, win.y1 - win.H * res, win.x0 + win.W * res, win.y1)
        local = [area_parts[i] for i in area_tree.query(box)]
        if not local:
            continue
        A = rasterize([(g, 1) for g in local], out_shape=(win.H, win.W),
                      transform=win.transform, fill=0, dtype="uint8").astype(bool)
        # Lake and the coarse piece, from the leaf grid at each pixel's sample.
        ii, jj = np.mgrid[0:win.H, 0:win.W]
        cxy = win.xy(ii.ravel(), jj.ravel())
        lr, lc = rivernet._nearest_idx(plan, cxy)
        lake_id = np.where(wg.type[lr, lc] == kvwater.TYPE_LAKE,
                           wg.lake_id[lr, lc], 0).reshape(win.H, win.W).astype(np.int64)
        in_coarse = np.zeros(win.H * win.W, bool)
        cr, cc = lr - rs.start, lc - cs.start
        okc = (cr >= 0) & (cr < coarse.shape[0]) & (cc >= 0) & (cc < coarse.shape[1])
        in_coarse[okc] = coarse[cr[okc], cc[okc]]
        in_coarse = in_coarse.reshape(win.H, win.W)

        vi = tree.query(box)
        vi = vi[shapely.contains_xy(box, cv_xy[vi, 0], cv_xy[vi, 1])]
        drawn = np.zeros((win.H, win.W), bool)
        if vi.size:
            shapes = shapely.buffer(shapely.points(cv_xy[vi]), cv_hw[vi] + res).tolist()
            drawn = rasterize([(g, 1) for g in shapes], out_shape=(win.H, win.W),
                              transform=win.transform, fill=0,
                              dtype="uint8").astype(bool)
        miss = A & ~drawn & (lake_id == 0)
        miss = binary_opening(miss, structure=_disc(open_px))
        lab, n = label(miss, structure=np.ones((3, 3), bool))
        for k in range(1, n + 1):
            piece = lab == k
            if not (piece & in_coarse).any():
                continue          # another window's piece
            if piece.sum() * res * res < BRANCH_MIN_AREA_M2:
                continue
            rep["pieces_examined"] += 1
            made = _branch_for_piece(
                plan, win, piece, A, lake_id, vi, cv_xy, cv_hw, cv_seg, cv_k,
                cv_in_lake, by_id, pos, sink, contact_px, rep, inside_by_seg)
            if made is None:
                continue
            px_xy, widths, info = made
            # The pixel path to the network's vertex stride, by arc length.
            src_s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(px_xy, axis=0).T))])
            seg_len = float(src_s[-1])
            m = max(2, int(np.ceil(seg_len / stride)) + 1)
            dst_s = np.linspace(0.0, seg_len, m)
            xy = np.stack([np.interp(dst_s, src_s, px_xy[:, 0]),
                           np.interp(dst_s, src_s, px_xy[:, 1])], 1)
            vw = np.interp(dst_s, src_s, widths)
            if not rivernet.in_region(plan, xy).all():
                rep["skipped_no_path"] += 1
                continue
            if (seg_len < BRANCH_MIN_LENGTH_M
                    or float(np.median(vw)) < BRANCH_MIN_WIDTH_M):
                rep["skipped_short_or_narrow"] += 1
                continue
            parent = by_id[info.parent]
            # The river's width alongside: its vertices from the split on, as far
            # as the branch is long, short of a lake.
            along = _chain_after(by_id, info.parent, info.parent_vertex, seg_len)
            wt = [2.0 * _half_widths(by_id[sid])[v] for sid, v in along
                  if not _span_mask(by_id[sid])[v]]
            w_t = float(np.median(wt)) if wt else float(parent.width_m or 1.0)
            w_b = float(np.median(vw))
            # The reach of the river the channel runs beside, as distances to the
            # network's sink, for sharing the flow with channels beside the same.
            if info.rejoin != NO_SEGMENT:
                lo = float(pos[info.rejoin][info.rejoin_vertex])
            else:
                open_along = [(sid, v) for sid, v in along
                              if not _span_mask(by_id[sid])[v]] or along[:1]
                lo = float(pos[open_along[-1][0]][open_along[-1][1]])
            ri, ci = rivernet._nearest_idx(plan, xy)
            lake_at = wg.lake_id[ri, ci].astype(np.int64)
            lake_at[wg.type[ri, ci] != kvwater.TYPE_LAKE] = 0
            spans, _st = rivernet._runs_of_lake(lake_at)
            if info.rejoin == NO_SEGMENT:
                # A delta channel keeps only the lake span it ends in.
                spans = [sp for sp in spans if sp[1] == xy.shape[0] - 1]
                if not spans:
                    rep["skipped_no_path"] += 1
                    continue
                info.lake_id = int(spans[-1][2])
            else:
                spans = []
            seg = rivernet.RiverSegment(
                seg_id=len(segments) + len(new),
                geom_hash=rivernet.geometry_hash(xy),
                order=int(parent.order),
                xy=xy, z=None, level=None,
                elvid=parent.elvid, vassdragsnr=None,
                lake_spans=spans,
            )
            seg.vertex_width_m = np.maximum(vw, 1.0).astype(np.float32)
            seg.branch_of = info
            new.append(seg)
            reach.append((seg, float(pos[info.parent][info.parent_vertex]), lo,
                          w_b, w_t))
            rep["side_arms" if info.rejoin != NO_SEGMENT else "delta_channels"] += 1
            rep["branch_length_m"] += seg_len
    groups = _share_groups(by_id, sink, reach)
    for members in groups:
        w_t = float(np.median([m[4] for m in members]))
        total = w_t ** 2 + sum(m[3] ** 2 for m in members)
        for seg, _hi, _lo, w_b, _wt in members:
            info = seg.branch_of
            info.share = float(w_b ** 2 / total)
            shares.append(info.share)
            q = float(np.float32(info.share * float(by_id[info.parent].discharge_m3_s)))
            seg.discharge_m3_s = q
            seg.width_m = float(np.float32(rivernet.width_from_discharge(
                q, scale=width_scale)))
            seg.depth_m = (float(np.float32(rivernet.depth_from_discharge(q)))
                           if depth_from_q else None)
    rep["channels_sharing_a_reach"] = int(sum(len(g) for g in groups if len(g) > 1))
    segments.extend(new)
    rep["branch_length_m"] = round(rep["branch_length_m"], 1)
    if shares:
        rep["share_median"] = round(float(np.median(shares)), 3)
    rep["rule"] = ("a line down the middle of each undrawn piece of N50 river area "
                   f"over {BRANCH_MIN_AREA_M2:g} m2 that leaves a drawn river and "
                   "rejoins it (side arm) or ends in the lake it runs into (delta "
                   "channel); share of the river's flow w_b^2/(w_t^2+sum w^2) over "
                   "the channels beside the same reach of it")
    return rep


def _share_groups(by_id: dict, sink: dict, reach: list) -> list:
    """
    Channels that take water from the same reach of one river share it: each gets
    w_b^2 / (w_t^2 + sum of every member's w^2), not w_b^2 / (w_t^2 + w_b^2) on
    its own, or two mouths leaving one vertex took 1.4 times the river between
    them (runtime, 2026-10-01). A reach is the stretch of river from where a
    channel leaves to where it rejoins or the river meets its lake; channels whose
    reaches overlap on the same river form one group. `reach` holds
    (segment, hi, lo, w_b, w_t), hi and lo in metres to the network's sink.
    """
    rows = sorted(reach, key=lambda m: (sink[m[0].branch_of.parent], -m[1]))
    groups: list = []
    rivers: list = []
    for m in rows:
        b = m[0].branch_of
        for g, river in zip(reversed(groups), reversed(rivers)):
            if (sink[g[0][0].branch_of.parent] == sink[b.parent]
                    and b.parent in river and m[1] > min(x[2] for x in g)):
                g.append(m)
                break
        else:
            groups.append([m])
            rivers.append({sid for sid, _v in _chain_after(
                by_id, b.parent, b.parent_vertex, 1e12)})
    return groups


def _span_mask(s) -> np.ndarray:
    m = np.zeros(s.n, dtype=bool)
    for a, b, _lid in s.lake_spans:
        m[max(int(a), 0):int(b) + 1] = True
    return m


def _lakes_downstream(by_id: dict, sid: int, k: int) -> set:
    out = set()
    for psid, v in _chain_after(by_id, sid, k, LAKE_SEARCH_M):
        for a, b, lid in by_id[psid].lake_spans:
            if a <= v <= b:
                out.add(int(lid))
    return out


def _branch_for_piece(plan, win, piece, A, lake_id, vi, cv_xy, cv_hw, cv_seg,
                      cv_k, cv_in_lake, by_id, pos, sink, contact_px, rep,
                      inside_by_seg):
    """(polyline, widths, BranchOf) for one piece, or None."""
    from scipy.ndimage import binary_dilation, distance_transform_edt, label

    res = win.res
    near_piece = binary_dilation(piece, structure=_disc(contact_px))
    # Distance from each pixel to the piece, for the river's contact vertices.
    to_piece = distance_transform_edt(~piece) * res
    att = []
    if vi.size:
        i, j = win.ij(cv_xy[vi])
        ok = win.inside(i, j)
        sel = vi[ok]
        d = to_piece[i[ok], j[ok]]
        touch = (d <= cv_hw[sel] + BRANCH_CONTACT_M + res) & ~cv_in_lake[sel]
        att = sel[touch]
    if len(att) == 0:
        rep["skipped_one_contact"] += 1
        return None
    att_pos = np.asarray([pos[int(cv_seg[a])][int(cv_k[a])] for a in att])
    att_sink = np.asarray([sink[int(cv_seg[a])] for a in att])
    root = att[int(np.argmax(att_pos))]
    same = att_sink == sink[int(cv_seg[root])]
    att, att_pos = att[same], att_pos[same]
    order = np.argsort(-att_pos)
    att, att_pos = att[order], att_pos[order]
    clusters = 1 + int((np.diff(-att_pos) > CONTACT_SPLIT_M).sum())
    root_seg, root_k = int(cv_seg[root]), int(cv_k[root])

    # A lake the piece touches, that its river runs into.
    lakes_here = lake_id[near_piece & (lake_id > 0)]
    target_lake = 0
    other_lake = False
    if lakes_here.size:
        down = _lakes_downstream(by_id, root_seg, root_k)
        ids, counts = np.unique(lakes_here, return_counts=True)
        for lid, cnt in sorted(zip(ids.tolist(), counts.tolist()), key=lambda t: -t[1]):
            if cnt * res * res < 25.0:
                continue
            if lid in down:
                target_lake = int(lid)
                break
            other_lake = True
    # A side arm rejoins its own river: the furthest contact down the root's chain,
    # far enough below the root to be a second contact.
    rejoin = None
    if not target_lake and clusters >= 2:
        chain = {}
        for sid, v in _chain_after(by_id, root_seg, root_k, 1e9):
            chain.setdefault(sid, v)
        root_pos = pos[root_seg][root_k]
        for a in att[::-1]:
            sid, v = int(cv_seg[a]), int(cv_k[a])
            if (sid in chain and v >= chain[sid]
                    and root_pos - pos[sid][v] >= BRANCH_MIN_LENGTH_M):
                rejoin = a
                break
    if not target_lake and rejoin is None:
        rep["skipped_lake_upstream" if other_lake else "skipped_one_contact"] += 1
        return None

    # The cost: cheap down the middle of the piece.
    dt = distance_transform_edt(piece) * res
    domain = piece | (near_piece & (A | (lake_id > 0)))
    cost = np.where(domain, 1.0 / (dt + 0.5) ** 2, np.inf)
    ri, rj = win.ij(cv_xy[root])
    ri, rj = int(ri[0]), int(rj[0])
    _connect(cost, piece, ri, rj, win)
    target = np.zeros_like(piece)
    if target_lake:
        target = near_piece & (lake_id == target_lake)
    else:
        ti, tj = win.ij(cv_xy[rejoin])
        ti, tj = int(ti[0]), int(tj[0])
        if not win.inside(np.asarray(ti), np.asarray(tj)):
            rep["skipped_no_path"] += 1
            return None
        _connect(cost, piece, ti, tj, win)
        target[ti, tj] = True
    if not win.inside(np.asarray(ri), np.asarray(rj)):
        rep["skipped_no_path"] += 1
        return None
    path = _cheapest_path(cost, ri, rj, target)
    if path.size < 2:
        rep["skipped_no_path"] += 1
        return None
    pi, pj = path // win.W, path % win.W
    # It must actually run through the piece, not round its edge.
    if piece[pi, pj].mean() < 0.5:
        rep["skipped_no_path"] += 1
        return None
    xy = win.xy(pi, pj)
    inside = piece[pi, pj]
    first_in = int(np.argmax(inside))
    last_in = int(inside.size - 1 - np.argmax(inside[::-1]))
    # Leave the river a little upstream of where the channel begins, and rejoin
    # it a little downstream of where it ends, so the joins are a Y and not a
    # hook: the river's vertices from the root on, as far as the channel goes.
    along = _chain_after(by_id, root_seg, root_k, 1e9)
    along = [(sid, v) for sid, v in along if not _span_mask(by_id[sid])[v]]
    a_xy = np.asarray([by_id[sid].xy[v] for sid, v in along])
    a_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(a_xy, axis=0).T))])
    a_hw = np.asarray([_half_widths(by_id[sid])[v] for sid, v in along])

    def lead(k, step):
        target_s = a_s[k] + step * (BRANCH_LEAD_M + a_hw[k])
        return int(np.clip(np.searchsorted(a_s, target_s), 0, len(along) - 1))

    k_in = int(np.argmin(np.hypot(*(a_xy - xy[first_in]).T)))
    # A river whose own line runs beside its N50 area, not in it, leaves the
    # whole area undrawn: that piece is the river itself, not a second channel.
    k_far = int(np.argmin(np.hypot(*(a_xy - xy[last_in]).T)))
    lo, hi = sorted((k_in, k_far))
    beside = [bool(inside_by_seg.get(sid, np.zeros(by_id[sid].n, bool))[v])
              for sid, v in along[lo:hi + 1]]
    if beside and float(np.mean(beside)) < CARRIER_INSIDE_FRACTION:
        rep["skipped_line_beside_area"] += 1
        return None
    k_root = lead(k_in, -1)
    root_seg, root_k = along[k_root]
    rejoin_at = None
    if rejoin is not None:
        rj = int(cv_seg[rejoin]), int(cv_k[rejoin])
        k_end = along.index(rj) if rj in along else len(along) - 1
        k_out = int(np.argmin(np.hypot(*(a_xy[:k_end + 1] - xy[last_in]).T)))
        k_rejoin = max(lead(k_out, +1), k_root + 1)
        if a_s[k_rejoin] - a_s[k_root] < BRANCH_MIN_LENGTH_M:
            rep["skipped_short_or_narrow"] += 1
            return None
        rejoin_at = along[k_rejoin]
        xy = np.vstack([a_xy[k_root], xy[first_in:last_in + 1], a_xy[k_rejoin]])
    else:
        xy = np.vstack([a_xy[k_root], xy[first_in:]])
        # On into the lake, so the channel has a span there.
        d = xy[-1] - xy[max(0, xy.shape[0] - 8)]
        nd = float(np.hypot(*d))
        if nd > 0:
            xy = np.vstack([xy, xy[-1] + d / nd * LAKE_OVERRUN_M])
    # Even spacing first, so the smoothing works in metres: it rounds the
    # pixel staircase and the corner where a connector meets the piece.
    xy = _even(xy, res)
    xy = _smooth(xy, passes=3, win=max(3, int(round(BRANCH_SMOOTH_M / res)) | 1))
    # Width: twice the distance to the piece's banks where the line runs; the
    # connectors outside the piece take the nearest value inside it. Never below
    # half the channel's median, so a neck in the mapped area does not pinch it.
    i, j = win.ij(xy)
    ok = win.inside(i, j)
    w = np.zeros(xy.shape[0])
    w[ok] = 2.0 * dt[i[ok], j[ok]]
    good = w > 0
    if not good.any():
        rep["skipped_short_or_narrow"] += 1
        return None
    idx = np.arange(w.size)
    w = np.interp(idx, idx[good], w[good])
    h = max(1, int(round(0.5 * BRANCH_WIDTH_SMOOTH_M / res)))
    k = np.ones(2 * h + 1) / (2 * h + 1)
    w = np.convolve(np.pad(w, h, mode="edge"), k, mode="valid")
    w = np.maximum(w, 0.5 * float(np.median(w)))
    info = BranchOf(parent=int(root_seg), parent_vertex=int(root_k), share=0.0)
    if rejoin_at is not None:
        info.rejoin = int(rejoin_at[0])
        info.rejoin_vertex = int(rejoin_at[1])
    return xy, w, info


def _connect(cost, piece, i, j, win):
    """Open a straight corridor from pixel (i, j) to the nearest piece pixel."""
    from scipy.ndimage import distance_transform_edt
    if not (0 <= i < cost.shape[0] and 0 <= j < cost.shape[1]):
        return
    _d, ind = distance_transform_edt(~piece, return_indices=True)
    ti, tj = int(ind[0][i, j]), int(ind[1][i, j])
    n = max(abs(ti - i), abs(tj - j)) + 1
    for t in np.linspace(0.0, 1.0, n):
        r = int(round(i + (ti - i) * t))
        c = int(round(j + (tj - j) * t))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < cost.shape[0] and 0 <= cc < cost.shape[1]:
                    if not np.isfinite(cost[rr, cc]):
                        cost[rr, cc] = 4.0


# --------------------------------------------------------------------------- #
# Serialisation                                                                #
# --------------------------------------------------------------------------- #

def write_branches_bin(segments: list, path: str) -> dict:
    """
    `river_branches.bin`: header `<8sII` (magic, version, count), then per branch
    `<IIIfiIHH` (seg_id, parent seg_id, parent vertex, share of the parent's flow,
    rejoin seg_id or -1, rejoin vertex, local lake id or 0, pad). Ids are rivers.bin
    segment ids, which are their indices. A parent is always a mapped segment, so
    one pass in file order settles every branch.
    """
    rows = [s for s in segments if getattr(s, "branch_of", None) is not None]
    with open(path, "wb") as fh:
        fh.write(struct.pack("<8sII", BRANCHES_MAGIC, BRANCHES_VERSION, len(rows)))
        for s in rows:
            b = s.branch_of
            fh.write(struct.pack("<IIIfiIHH", int(s.seg_id), int(b.parent),
                                 int(b.parent_vertex), float(b.share), int(b.rejoin),
                                 int(b.rejoin_vertex), int(b.lake_id), 0))
    return {
        "file": os.path.basename(path),
        "format": "kvterrain-river-branches/1",
        "magic": BRANCHES_MAGIC.decode("ascii"),
        "version": BRANCHES_VERSION,
        "byte_order": "little_endian",
        "header_layout": "<8sII (magic, version, branch_count)",
        "record_layout": "<IIIfiIHH (seg_id, parent_seg_id, parent_vertex, share, "
                         "rejoin_seg_id or -1, rejoin_vertex, lake_id or 0, pad)",
        "branches": len(rows),
        "bytes": os.path.getsize(path),
        "semantics": "a side channel's flow is share * its parent's flow; it has no "
                     "upstream or downstream links in rivers.bin. Its level runs "
                     "from the parent's at parent_vertex to the rejoin segment's at "
                     "rejoin_vertex, or to the lake's.",
    }
