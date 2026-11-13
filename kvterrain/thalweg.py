"""
kvterrain.thalweg
=================

Move each mapped stream onto the channel the scan shows.

NVE's Elvenett was drawn from N50 (1:50 000, position accuracy about 20 m), so a
stream's line often runs a few metres up the valley side rather than along its
bottom. The downhill river bed then does what it must: the water may not stand
above either bank, so where the line runs along a slope the level is dropped to
the lower bank and a trench is cut into the hillside to hold it. On Lierne
(2026-09-30) that bank cap was 96% of all river level lowering: 1 000 km of
channel sat more than 1 m below the ground under its own line, and at 78% of
those vertices a valley bottom lay a median 10 m to one side.

`snap_segments` shifts each vertex sideways, along the line's normal and by at
most `max_offset_m`, to where it needs the least cut. The cut at an offset is how
far the ground there stands above the lower of its two banks, read at the same
reaches `riverbed._bank_ceiling` uses. Zero at a valley bottom, the slope times
the channel width on a hillside. Choosing the offsets is a shortest path over
the whole segment (dynamic programming), with a small cost per metre of offset
so flat ground keeps its line, a cost per metre of sideways change so the line
stays smooth, and a hard limit on how fast it may drift sideways. A uniform
hillside therefore keeps its line: moving along it saves no cut.

What stays put:
  * vertices in a lake, and vertices whose visual half-width is over
    `max_half_width_m` (N50 draws those rivers as water areas; the area wins,
    and those vertices are moved to its middle instead, see `snap_segments`);
  * any offset that would land in a lake, off the map, or on nodata;
  * offsets beyond the inside of a bend's radius, so a line never folds.

Segments are snapped downstream-first. A segment's last vertex follows the first
vertex of the segment it flows into, so confluences stay joined. The vertex count
is unchanged, so lake spans, widths and profiles keep their indices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numba as nb
import numpy as np

from . import core
from . import rivernet
from . import water as kvwater
from .riverbed import MASK_EDGE_DIAGONAL_CELLS, _topological_order, _vertex_normals


@dataclass
class SnapParams:
    enabled: bool = True
    # How far a vertex may move off its mapped line, each way.
    max_offset_m: float = 15.0
    # Lateral resolution of the search.
    step_m: float = 1.0
    # Metres of cut one metre of offset must save to be worth it.
    offset_cost: float = 0.02
    # Cost per metre of sideways change between neighbouring vertices.
    turn_cost: float = 0.05
    # Most sideways drift per metre along the river.
    max_drift: float = 0.6
    # Wider rivers keep their line.
    max_half_width_m: float = 7.5
    # A residual at a confluence is eased in over this many vertices.
    join_taper_vertices: int = 4
    # Fraction of a bend's radius the inside may move by.
    bend_fraction: float = 0.8


@nb.njit(cache=True)
def _best_offsets(unary, max_step, turn_cost, step_m):
    """Shortest path through (vertex, offset) with a per-step sideways limit.
    Returns state indices, or -1 everywhere when no path exists."""
    n, K = unary.shape
    back = np.zeros((n, K), dtype=np.int32)
    cost = unary[0].copy()
    new = np.empty(K)
    for i in range(1, n):
        ms = max_step[i]
        for k in range(K):
            new[k] = np.inf
            u = unary[i, k]
            if not np.isfinite(u):
                continue
            best = np.inf
            bj = -1
            lo = max(0, k - ms)
            hi = min(K, k + ms + 1)
            for j in range(lo, hi):
                c = cost[j] + turn_cost * abs(k - j) * step_m
                if c < best:
                    best = c
                    bj = j
            if bj >= 0:
                new[k] = best + u
                back[i, k] = bj
        for k in range(K):
            cost[k] = new[k]
    out = np.full(n, -1, dtype=np.int32)
    k = -1
    best = np.inf
    for j in range(K):
        if cost[j] < best:
            best = cost[j]
            k = j
    if k < 0:
        return out
    for i in range(n - 1, -1, -1):
        out[i] = k
        k = back[i, k]
    return out


def _signed_curvature(xy: np.ndarray) -> np.ndarray:
    """Turn per metre at each vertex, positive when the line bends toward +normal."""
    n = xy.shape[0]
    kappa = np.zeros(n)
    if n < 3:
        return kappa
    a = xy[1:-1] - xy[:-2]
    b = xy[2:] - xy[1:-1]
    la = np.hypot(a[:, 0], a[:, 1])
    lb = np.hypot(b[:, 0], b[:, 1])
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(axis=1)
    turn = np.arctan2(cross, dot)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa[1:-1] = np.where(la + lb > 0, 2.0 * turn / (la + lb), 0.0)
    return kappa


def _snap_one(plan: core.GridPlan, xy: np.ndarray, half_w: np.ndarray,
              fixed: np.ndarray, end_offset: Optional[float],
              ground: np.ndarray, water_type: np.ndarray,
              params: SnapParams) -> tuple[np.ndarray, float, float, bool]:
    """Offsets (metres along the normal) for one segment, its cut before and
    after summed over vertices, and whether a path was found."""
    n = xy.shape[0]
    spacing = float(plan.spacing_m)
    step = float(params.step_m)
    K = int(round(params.max_offset_m / step))
    offsets = np.arange(-K, K + 1) * step
    normal = _vertex_normals(xy)

    edge = np.maximum(half_w, 0.5 * spacing) + MASK_EDGE_DIAGONAL_CELLS * spacing
    reach_max = float(edge.max()) + spacing
    R = int(np.ceil((params.max_offset_m + reach_max) / step))
    t = np.arange(-R, R + 1) * step
    pts = xy[:, None, :] + t[None, :, None] * normal[:, None, :]
    row, col = rivernet.world_to_grid(plan, pts[..., 0], pts[..., 1])
    prof = rivernet._bilinear(ground, row, col)
    inside = ((row >= 0) & (col >= 0) &
              (row <= plan.samples_y - 1) & (col <= plan.samples_x - 1))
    prof = np.where(inside, prof, np.nan)

    centre = R
    idx_off = centre + np.arange(-K, K + 1)
    h = prof[:, idx_off]
    bank = np.full(h.shape, np.inf)
    rows = np.arange(n)[:, None]
    for reach in (edge, edge + 0.5 * spacing, edge + spacing):
        r = np.rint(reach / step).astype(np.int64)[:, None]
        for side in (-1, 1):
            j = np.clip(idx_off[None, :] + side * r, 0, 2 * R)
            v = prof[rows, j]
            bank = np.fmin(bank, np.where(np.isfinite(v), v, np.inf))
    cut = np.maximum(h - bank, 0.0)
    cut = np.where(np.isfinite(cut), cut, 0.0)
    unary = cut + params.offset_cost * np.abs(offsets)[None, :]

    # Never onto nodata, off the map, or into a lake.
    rn = np.clip(np.rint(row[:, idx_off]).astype(np.int64), 0, plan.samples_y - 1)
    cn = np.clip(np.rint(col[:, idx_off]).astype(np.int64), 0, plan.samples_x - 1)
    bad = ~np.isfinite(h) | (water_type[rn, cn] == kvwater.TYPE_LAKE)
    # Never past the inside of a bend.
    kappa = _signed_curvature(xy)
    with np.errstate(divide="ignore"):
        limit = np.where(np.abs(kappa) > 1e-9,
                         params.bend_fraction / np.abs(kappa), np.inf)
    inner = np.sign(kappa)[:, None] * offsets[None, :] > limit[:, None]
    bad |= inner
    bad[:, K] = False
    unary[bad] = np.inf
    unary[fixed] = np.inf
    unary[fixed, K] = 0.0
    if end_offset is not None:
        k_end = int(np.clip(np.rint(end_offset / step), -K, K)) + K
        keep = unary[-1, k_end]
        unary[-1] = np.inf
        unary[-1, k_end] = keep if np.isfinite(keep) else 0.0

    seg = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
    max_step = np.maximum(np.floor(params.max_drift * seg / step), 0).astype(np.int64)
    max_step[0] = 2 * K
    state = _best_offsets(unary, max_step, float(params.turn_cost), step)
    before = float(cut[:, K].sum())
    if state[0] < 0:
        return np.zeros(n), before, before, False
    after = float(cut[np.arange(n), state].sum())
    return offsets[state], before, after, True


def snap_segments(plan: core.GridPlan, segments: list, ground: np.ndarray,
                  wg: kvwater.WaterGrid,
                  params: Optional[SnapParams] = None,
                  centres: Optional[dict] = None) -> dict:
    """Shift every segment's vertices onto the scanned channel, in place.
    `ground` is the uncarved, void-repaired leaf.

    `centres` ({seg_id: offset per vertex}, from `riverwidth.vertex_widths`) first
    moves each vertex that has one to the middle of its N50 river area, so the
    width stamped around the line fills the area instead of spilling past its
    near bank. That applies with the snap switched off too."""
    params = params or SnapParams()
    report = {"enabled": bool(params.enabled), "params": asdict(params)}
    centres = {k: v for k, v in (centres or {}).items() if np.any(v)}
    if not segments or (not params.enabled and not centres):
        return report

    by_id = {s.seg_id: s for s in segments}
    order, _ = _topological_order(segments)
    moved_start: dict[int, np.ndarray] = {}
    orig_start = {s.seg_id: np.asarray(s.xy[0], dtype=np.float64).copy()
                  for s in segments}
    shifts, lengths = [], []
    cut_before = cut_after = 0.0
    no_path = 0
    centred = []
    ground = np.asarray(ground)
    for sid in reversed(order):
        s = by_id[sid]
        xy0 = np.asarray(s.xy, dtype=np.float64)
        n = xy0.shape[0]
        if n < 2:
            moved_start[sid] = xy0[0].copy()
            continue
        xy = xy0
        c = centres.get(sid)
        if c is not None and len(c) == n:
            xy = xy0 + np.asarray(c, dtype=np.float64)[:, None] * _vertex_normals(xy0)
            centred.append(np.abs(np.asarray(c))[np.abs(np.asarray(c)) > 0.5])
        vw = getattr(s, "vertex_width_m", None)
        if vw is not None and len(vw) == n:
            half_w = 0.5 * np.asarray(vw, dtype=np.float64)
        else:
            half_w = np.full(n, 0.5 * float(s.width_m or 0.0))
        fixed = half_w > params.max_half_width_m
        for a, b, _lid in s.lake_spans:
            fixed[max(int(a), 0):min(int(b) + 1, n)] = True

        # The last vertex follows the start of the segment this one joins.
        target_end = None
        d = s.downstream
        if d is not None and d in moved_start:
            if np.hypot(*(xy[-1] - orig_start[d])) <= 2.0 * plan.spacing_m:
                target_end = moved_start[d]
        end_offset = None
        if target_end is not None and not fixed[-1]:
            end_offset = float(np.dot(target_end - xy[-1], _vertex_normals(xy)[-1]))

        if params.enabled:
            off, before, after, found = _snap_one(plan, xy, half_w, fixed,
                                                  end_offset, ground, wg.type, params)
            if not found and end_offset is not None:
                # Too short to drift to the join: snap freely, the tail eases in.
                off, before, after, found = _snap_one(plan, xy, half_w, fixed, None,
                                                      ground, wg.type, params)
            cut_before += before
            cut_after += after
            no_path += int(not found)
        else:
            off = np.zeros(n)
        new = xy + off[:, None] * _vertex_normals(xy)
        if target_end is not None:
            # Whatever the normal offset could not reach, ease in over the tail.
            residual = target_end - new[-1]
            k = min(max(int(params.join_taper_vertices), 1), n - 1)
            w = np.linspace(0.0, 1.0, k + 1)
            new[n - k - 1:] += w[:, None] * residual[None, :]
        moved_start[sid] = new[0].copy()
        seg_len = np.r_[0.0, np.hypot(*np.diff(xy0, axis=0).T)]
        shifts.append(np.hypot(*(new - xy0).T))
        lengths.append(seg_len)
        s.xy = new

    shift = np.concatenate(shifts) if shifts else np.zeros(0)
    length = np.concatenate(lengths) if lengths else np.zeros(0)
    total = float(length.sum()) or 1.0
    report.update({
        "segments": len(segments),
        "segments_without_path": int(no_path),
        "cut_sum_before_m": round(cut_before, 1),
        "cut_sum_after_m": round(cut_after, 1),
        "moved_pct_of_length": round(100.0 * float(length[shift > 0.5].sum()) / total, 1),
        "shift_median_moved_m": (round(float(np.median(shift[shift > 0.5])), 2)
                                 if (shift > 0.5).any() else 0.0),
        "shift_max_m": round(float(shift.max()), 2) if shift.size else 0.0,
    })
    cen = np.concatenate(centred) if centred else np.zeros(0)
    report["centred_in_n50_area"] = {
        "vertices": int(cen.size),
        "shift_median_m": round(float(np.median(cen)), 2) if cen.size else 0.0,
        "shift_max_m": round(float(cen.max()), 2) if cen.size else 0.0,
    }
    return report
