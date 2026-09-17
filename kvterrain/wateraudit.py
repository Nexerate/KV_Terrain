"""
kvterrain.wateraudit
====================

Does the terrain as shipped reproduce the authored water? WATER_REDESIGN.md §4.4.

Terrain is the source of truth under the redesign: the runtime renders a lake as
a depression node at a level. So the exporter has to say, lake by lake and river
by river, whether the depression hierarchy built on the packed heights agrees
with NVE — and where it does not. That report is `water_audit.json`.

LAKES
-----
For each authored lake with a level L:

1. Take the leaf holding most of the lake's mask (islands excluded).
2. Walk up from it. At each node N the pool is N's subtree below
   min(L, spill(N)) — connected by construction, because every merge inside N
   happened below N's spill. Measure its overlap with the mask (IoU), flooding
   out from the mask samples with a size cap.
3. The best-overlapping node is the lake's node. Climbing stops once the pool
   is far larger than the lake or the root is reached.

Findings:
  * `held_by_outflow`  spill < L <= spill + outflow head + tolerance. The spill is
    only where a lake is STABLE up to; while water leaves through the outflow
    river, the lake stands above it by the river's depth at the spill point (the
    exported river surface there minus the spill; `hierarchy.set_outflow_heads`). The authored level is taken as right, so the
    node's `outflow_head_m` is raised to L - spill and the node reproduces it.
    Not a finding.
  * `level_above_spill`  L > spill + outflow head + tolerance. The lake should be
    leaking: the level is wrong, the carve broke the rim, or an outlet (dam, weir)
    is not in the DEM. `spill_through` says what the saddle is made of —
    `river_channel`, `land`, `other_lake` or `map_edge` — because those are
    different problems with different fixes.
  * `below_spill_regulated`  L well below spill and a river flows in. Plausible
    for a regulated lake; reported, not flagged.
  * `no_node`  nothing overlaps the mask well (a lake on a slope in the DEM).
  * `ok`.

RIVERS
------
Steepest descent on the packed codes from each segment's first open-channel
vertex. The flow-direction atlas (§4.3) is not built yet, so this walks the
heights directly; flats are crossed by a small search for the nearest sample
with a lower neighbour. Findings: `leaves_corridor` (the walk steps off the
river/lake mask grown by one sample) and `pit` (it ends in a depression along
the channel). A walk that reaches the segment's end, a lake or the map edge, or
simply keeps going inside the corridor for long enough, is fine.

UNEXPLAINED DEPRESSIONS
-----------------------
Without catchment areas (§4.3) this is only the geometric half: nodes that are
deep and large, match no lake and contain none, keeping only the outermost of
nested candidates, ranked by area x depth.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numba as nb
import numpy as np

from . import core
from . import hierarchy as H
from . import rivernet
from . import water as kvwater

AUDIT_FILE = "water_audit.json"

DEFAULT_SPILL_TOL_M = 0.05          # plus one height step, see audit_lakes
DEFAULT_REGULATED_MARGIN_M = 1.0
DEFAULT_MIN_IOU = 0.3
POOL_CAP_FACTOR = 50                # stop climbing once a pool is this many x the mask
POOL_CAP_MIN = 20000
FLAT_SEARCH_CELLS = 512
UNEXPLAINED_MIN_DEPTH_M = 2.0
UNEXPLAINED_MIN_AREA_M2 = 5000.0
UNEXPLAINED_LIMIT = 500

RIVER_OK = 0
RIVER_LEAVES_CORRIDOR = 1
RIVER_PIT = 2
RIVER_FLAT_STALL = 3
RIVER_STATUS = {RIVER_OK: "ok", RIVER_LEAVES_CORRIDOR: "leaves_corridor",
                RIVER_PIT: "pit", RIVER_FLAT_STALL: "flat_stall"}
# A pit shallower than this along a channel is counted, not listed: at 5 m spacing
# the bed of a wide river is full of centimetre-deep noise pits.
RIVER_PIT_LIST_MIN_DEPTH_M = 0.5


def _spill_through(wg: kvwater.WaterGrid, cells: tuple, SY: int, SX: int,
                   own_lake: int) -> str:
    kinds = set()
    for cell in cells:
        r, c = divmod(int(cell), SX)
        if r in (0, SY - 1) or c in (0, SX - 1):
            return "map_edge"
        t = int(wg.type[r, c])
        if t == kvwater.TYPE_RIVER:
            kinds.add("river_channel")
        elif t == kvwater.TYPE_LAKE and int(wg.lake_id[r, c]) != own_lake:
            kinds.add("other_lake")
    for k in ("river_channel", "other_lake"):
        if k in kinds:
            return k
    return "land"


# --------------------------------------------------------------------------- #
# Kernels                                                                      #
# --------------------------------------------------------------------------- #

@nb.njit(cache=True)
def _pool_overlap(codes, labels, tin, tout, node, level_code, seeds, lake_id,
                  lake_code, island, stamp, stamp_val, cap, buf):
    """
    Flood the pool of `node` below `level_code` (fractional code, strict <) from
    the mask samples in `seeds`. Returns (pool_size, overlap, capped).
    """
    SY, SX = codes.shape
    lo = tin[node]
    hi = tout[node]
    n = 0
    inter = 0
    for s in range(seeds.shape[0]):
        i = seeds[s]
        r = i // SX
        c = i - r * SX
        if stamp[r, c] == stamp_val:
            continue
        if codes[r, c] >= level_code:
            continue
        t = tin[labels[r, c]]
        if t < lo or t >= hi:
            continue
        stamp[r, c] = stamp_val
        buf[n] = i
        n += 1
    read = 0
    while read < n:
        i = buf[read]
        read += 1
        r = i // SX
        c = i - r * SX
        if lake_id[r, c] == lake_code and not island[r, c]:
            inter += 1
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= SY:
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if cc < 0 or cc >= SX or (dr == 0 and dc == 0):
                    continue
                if stamp[rr, cc] == stamp_val or codes[rr, cc] >= level_code:
                    continue
                t = tin[labels[rr, cc]]
                if t < lo or t >= hi:
                    continue
                if n >= cap:
                    return n, inter, True
                stamp[rr, cc] = stamp_val
                buf[n] = rr * SX + cc
                n += 1
    return n, inter, False


@nb.njit(cache=True)
def _walk_rivers(codes, corridor, lake, labels, floor_code, spill_code, leaf_max,
                 starts, ends, max_steps):
    """Steepest descent per segment. Returns (status, stop_cell, steps)."""
    SY, SX = codes.shape
    m = starts.shape[0]
    status = np.zeros(m, dtype=np.int8)
    stop = np.full(m, -1, dtype=np.int64)
    steps_out = np.zeros(m, dtype=np.int64)
    flat = np.empty(512, dtype=np.int64)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for s in range(m):
        cur = starts[s]
        er = ends[s] // SX
        ec = ends[s] - er * SX
        steps = 0
        while True:
            r = cur // SX
            c = cur - r * SX
            if abs(r - er) <= 1 and abs(c - ec) <= 1:
                break
            if lake[r, c] or r == 0 or r == SY - 1 or c == 0 or c == SX - 1:
                break
            if steps >= max_steps[s]:
                break
            if not corridor[r, c]:
                status[s] = 1
                stop[s] = cur
                break
            h = codes[r, c]
            best = -1
            best_drop = 0.0
            for dr in (-1, 0, 1):
                rr = r + dr
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if dr == 0 and dc == 0:
                        continue
                    d = float(h) - float(codes[rr, cc])
                    if d <= 0:
                        continue
                    if dr != 0 and dc != 0:
                        d *= inv_sqrt2
                    if d > best_drop:
                        best_drop = d
                        best = rr * SX + cc
            if best >= 0:
                cur = best
                steps += 1
                continue
            # No lower neighbour: a pit bottom, or a flat to cross.
            l = labels[r, c]
            if l >= 1 and l <= leaf_max and floor_code[l] == h and spill_code[l] > h:
                status[s] = 2
                stop[s] = cur
                break
            flat[0] = cur
            n = 1
            read = 0
            exit_cell = -1
            while read < n and exit_cell < 0:
                j = flat[read]
                read += 1
                rj = j // SX
                cj = j - rj * SX
                for dr in (-1, 0, 1):
                    rr = rj + dr
                    if rr < 0 or rr >= SY:
                        continue
                    for dc in (-1, 0, 1):
                        cc = cj + dc
                        if cc < 0 or cc >= SX or (dr == 0 and dc == 0):
                            continue
                        k = rr * SX + cc
                        if codes[rr, cc] < h:
                            exit_cell = j
                            break
                        if codes[rr, cc] == h and n < flat.shape[0]:
                            seen = False
                            for q in range(n):
                                if flat[q] == k:
                                    seen = True
                                    break
                            if not seen:
                                flat[n] = k
                                n += 1
                    if exit_cell >= 0:
                        break
            if exit_cell < 0:
                status[s] = 3
                stop[s] = cur
                break
            # exit_cell has a lower neighbour, so the next iteration steps off.
            cur = exit_cell
            steps += n
        steps_out[s] = steps
    return status, stop, steps_out


# --------------------------------------------------------------------------- #
# Lakes                                                                        #
# --------------------------------------------------------------------------- #

def audit_lakes(plan, h: H.Hierarchy, codes: np.ndarray, wg: kvwater.WaterGrid,
                *, inflow_lakes: set, tin, tout,
                spill_tol_m: float = DEFAULT_SPILL_TOL_M,
                regulated_margin_m: float = DEFAULT_REGULATED_MARGIN_M,
                min_iou: float = DEFAULT_MIN_IOU) -> list:
    nodes = h.nodes
    step_m = (h.height_max - h.height_min) / 65535.0
    tol_m = spill_tol_m + step_m
    labels = h.labels
    island = (wg.lake_island if wg.lake_island is not None
              else np.zeros(wg.type.shape, dtype=bool))
    lake_id = wg.lake_id
    mask = (wg.type == kvwater.TYPE_LAKE) & ~island
    idx = np.flatnonzero(mask)
    ids = lake_id.ravel()[idx]
    order = np.argsort(ids, kind="stable")
    idx, ids = idx[order], ids[order]
    uniq, first, counts = np.unique(ids, return_index=True, return_counts=True)

    stamp = np.zeros(codes.shape, dtype=np.int32)
    stamp_val = 0
    parent = nodes["parent"]
    out = []
    matched_by_node: dict = {}
    for k, (lid, a, cnt) in enumerate(zip(uniq, first, counts)):
        lid = int(lid)
        info = (wg.lake_table or {}).get(lid, {}) or {}
        level = wg.lake_level(info)
        rec = {"lake_id": lid, "vatn_lnr": info.get("lopenr"), "navn": info.get("navn"),
               "mask_samples": int(cnt), "level_m": level,
               "level_source": info.get("level_source")}
        cells = idx[a:a + cnt].astype(np.int64)
        cx, cy = h.cell_xy(plan, cells[len(cells) // 2])
        rec["xy"] = [round(float(cx), 1), round(float(cy), 1)]
        if level is None or not np.isfinite(level):
            rec["status"] = "no_level"
            out.append(rec)
            continue
        leaf_ids, leaf_n = np.unique(labels.ravel()[cells], return_counts=True)
        leaf_n = np.where(leaf_ids == H.ROOT, 0, leaf_n)
        if leaf_n.sum() == 0:
            rec["status"] = "no_node"
            rec["reason"] = "every mask sample drains straight off the map"
            out.append(rec)
            continue
        leaf = int(leaf_ids[int(np.argmax(leaf_n))])
        level_code = float(h.m_to_code(level))
        tol_code = tol_m / max(step_m, 1e-9)
        cap = int(max(POOL_CAP_FACTOR * cnt, POOL_CAP_MIN))
        buf = np.empty(cap, dtype=np.int64)

        best = None
        v = leaf
        while v != H.ROOT and v != H.NONE:
            # The pool of v at the lake level, capped at v's own spill, plus the
            # tolerance: a lake's shoreline samples sit exactly ON its level (depth
            # 0), and on a small lake they are most of the mask. Flooding stays
            # inside v's subtree, so the tolerance cannot leak across the saddle.
            eff = min(level_code, float(nodes["spill_code"][v])) + tol_code
            # One stamp value per evaluation marks visited samples without ever
            # clearing the array.
            stamp_val += 1
            n, inter, capped = _pool_overlap(
                codes, labels, tin, tout, v, eff, cells, lake_id, np.uint16(lid),
                island, stamp, np.int32(stamp_val), cap, buf)
            iou = inter / max(n + cnt - inter, 1)
            if best is None or iou > best[1]:
                best = (v, iou, n, inter, capped)
            if capped or n > POOL_CAP_FACTOR * cnt:
                break
            # The pool at L cannot grow past a node that spills above L. A node
            # spilling AT L (within tolerance) is not a wall: parts of one lake
            # separated by a neck whose shore samples sit exactly on the waterline
            # merge there, so keep climbing.
            if float(nodes["spill_code"][v]) > level_code + tol_code:
                break
            v = int(parent[v])

        v, iou, n, inter, capped = best
        spill_m = float(nodes["spill_m"][v])
        floor_m = float(nodes["floor_m"][v])
        rec.update({"node_id": int(v), "iou": round(float(iou), 4),
                    "pool_samples": int(n), "pool_capped": bool(capped),
                    "floor_m": round(floor_m, 3),
                    "spill_m": None if not np.isfinite(spill_m) else round(spill_m, 3),
                    "spills_off_map": bool(nodes["flags"][v] & H.FLAG_SPILLS_OFF_MAP)})
        head_m = float(nodes["outflow_head_m"][v])
        rec["outflow_head_m"] = round(head_m, 3)
        needed_head = 0.0
        if iou < min_iou:
            rec["status"] = "no_node"
            rec["reason"] = f"best overlap {iou:.2f} < {min_iou}"
        elif np.isfinite(spill_m) and spill_m + tol_m < level <= spill_m + head_m + tol_m:
            rec["status"] = "held_by_outflow"
            rec["above_spill_m"] = round(level - spill_m, 3)
            needed_head = level - spill_m
        elif np.isfinite(spill_m) and level > spill_m + head_m + tol_m:
            rec["status"] = "level_above_spill"
            rec["excess_m"] = round(level - spill_m - head_m, 3)
            rec["above_spill_m"] = round(level - spill_m, 3)
            sx, sy = h.cell_xy(plan, int(nodes["spill_cell"][v]))
            rec["spill_xy"] = [round(float(sx), 1), round(float(sy), 1)]
            rec["overflow_to"] = int(nodes["overflow_to"][v])
            rec["spill_through"] = _spill_through(
                wg, (nodes["spill_cell"][v], nodes["outlet_cell"][v]),
                codes.shape[0], codes.shape[1], lid)
        elif (np.isfinite(spill_m) and spill_m > level + regulated_margin_m
              and lid in inflow_lakes):
            rec["status"] = "below_spill_regulated"
            rec["headroom_m"] = round(spill_m - level, 3)
        else:
            rec["status"] = "ok"

        if rec["status"] != "no_node":
            prev = matched_by_node.get(v)
            if prev is None or inter > prev[1]:
                matched_by_node[v] = (lid, inter, level, needed_head)
        out.append(rec)

    for v, (lid, _, level, needed_head) in matched_by_node.items():
        nodes["lake_id"][v] = min(lid, 65535)
        nodes["authored_level_m"][v] = level
        nodes["flags"][v] |= H.FLAG_AUTHORED_LAKE
        # The authored level is right: the node's outflow head carries it exactly.
        if needed_head > float(nodes["outflow_head_m"][v]):
            nodes["outflow_head_m"][v] = needed_head
    for r in out:
        if (r.get("node_id") is not None and r["status"] != "no_node"
                and matched_by_node.get(r["node_id"], (None,))[0] != r["lake_id"]):
            r["shares_node_with_lake"] = int(matched_by_node[r["node_id"]][0])
    return out


# --------------------------------------------------------------------------- #
# Rivers                                                                       #
# --------------------------------------------------------------------------- #

def audit_rivers(plan, h: H.Hierarchy, codes: np.ndarray, wg: kvwater.WaterGrid,
                 segments: list) -> list:
    from scipy.ndimage import binary_dilation

    # A walk that gets within two samples of a lake has reached it: an inflow
    # trench is carved below the lake level and ends against the lake's shoreline
    # samples, which sit exactly on the level, so its last sample is a pit that
    # fills to the lake level and joins the lake there.
    is_lake = binary_dilation(wg.type == kvwater.TYPE_LAKE,
                              structure=np.ones((3, 3), bool), iterations=2)
    corridor = binary_dilation(wg.type != kvwater.TYPE_LAND,
                               structure=np.ones((3, 3), bool))
    starts, ends, max_steps, keep = [], [], [], []
    for s in segments:
        ri, ci = rivernet._nearest_idx(plan, s.xy)
        in_lake = np.zeros(s.n, dtype=bool)
        for (a, b, _lid) in s.lake_spans:
            in_lake[a:b + 1] = True
        open_idx = np.flatnonzero(~in_lake)
        if open_idx.size < 2:
            continue
        v0 = int(open_idx[0])
        starts.append(int(ri[v0]) * plan.samples_x + int(ci[v0]))
        ends.append(int(ri[-1]) * plan.samples_x + int(ci[-1]))
        max_steps.append(3 * s.n + 10)
        keep.append(s)
    if not keep:
        return [], 0, 0
    nodes = h.nodes
    n_minor_pits = 0
    status, stop, steps = _walk_rivers(
        codes, corridor, is_lake, h.labels, nodes["floor_code"], nodes["spill_code"],
        h.leaf_count, np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64), np.asarray(max_steps, dtype=np.int64))
    out = []
    for s, st, cell in zip(keep, status, stop):
        if st == RIVER_OK:
            continue
        x, y = h.cell_xy(plan, int(cell))
        rec = {"segment_id": s.seg_id, "strekn_lnr": s.strekn_lnr, "order": s.order,
               "status": RIVER_STATUS[int(st)],
               "xy": [round(float(x), 1), round(float(y), 1)]}
        if st == RIVER_PIT:
            leaf = int(h.labels.ravel()[int(cell)])
            depth = float(nodes["spill_m"][leaf] - nodes["floor_m"][leaf])
            if depth < RIVER_PIT_LIST_MIN_DEPTH_M:
                n_minor_pits += 1
                continue
            rec["node_id"] = leaf
            rec["depth_m"] = round(depth, 3)
        out.append(rec)
    return out, len(keep), n_minor_pits


# --------------------------------------------------------------------------- #
# Unexplained depressions                                                      #
# --------------------------------------------------------------------------- #

def unexplained_depressions(plan, h: H.Hierarchy, tin, tout, *,
                            min_depth_m: float = UNEXPLAINED_MIN_DEPTH_M,
                            min_area_m2: float = UNEXPLAINED_MIN_AREA_M2,
                            limit: int = UNEXPLAINED_LIMIT) -> tuple[list, int, int]:
    nodes = h.nodes
    n = h.node_count
    area = nodes["subtree_cells"].astype(np.float64) * plan.spacing_m ** 2
    depth = (nodes["spill_m"] - nodes["floor_m"]).astype(np.float64)
    has = (nodes["flags"] & H.FLAG_HAS_SPILL) != 0
    lake = (nodes["flags"] & H.FLAG_AUTHORED_LAKE) != 0
    cand = has & ~lake & (depth >= min_depth_m) & (area >= min_area_m2)

    # Exclude anything containing a lake node or inside one: order nodes by tin and
    # count lake nodes in each [tin, tout) range with a prefix sum.
    by_tin = np.zeros(n + 1, dtype=np.int64)
    by_tin[tin[lake] + 1] = 1
    pref = np.cumsum(by_tin)
    contains_lake = (pref[tout] - pref[tin]) > 0
    inside_lake = np.zeros(n, dtype=bool)
    parent = nodes["parent"].astype(np.int64)
    for v in np.argsort(tin):              # parents before children
        p = parent[v]
        if p != H.NONE:
            inside_lake[v] = inside_lake[p] or lake[p]
    cand &= ~contains_lake & ~inside_lake

    # A depression that spills INTO a matched lake at or below that lake's level
    # fills to the lake and joins it — typically an inflow trench carved below the
    # lake level and ending against the lake's shoreline samples. The lake explains
    # it. The lake is found from `overflow_to`: an authored-lake ancestor of it, or
    # the highest-level authored lake inside its subtree.
    lake_ids = np.flatnonzero(lake)
    joins_lake = np.zeros(n, dtype=bool)
    if lake_ids.size:
        order = np.argsort(tin[lake_ids])
        lk_tin = tin[lake_ids][order]
        lk_lvl = nodes["authored_level_m"][lake_ids][order].astype(np.float64)
        tol = DEFAULT_SPILL_TOL_M + (h.height_max - h.height_min) / 65535.0
        for v in np.flatnonzero(cand):
            ov = int(nodes["overflow_to"][v])
            if ov == H.NONE or ov == H.ROOT:
                continue
            level = -np.inf
            a = ov
            while a != H.NONE:
                if lake[a]:
                    level = float(nodes["authored_level_m"][a])
                    break
                a = int(parent[a])
            lo, hi = np.searchsorted(lk_tin, [tin[ov], tout[ov]])
            if hi > lo:
                level = max(level, float(np.nanmax(lk_lvl[lo:hi])))
            if np.isfinite(level) and float(nodes["spill_m"][v]) <= level + tol:
                joins_lake[v] = True
    n_joins = int((cand & joins_lake).sum())
    cand &= ~joins_lake
    # Outermost only.
    outer = cand.copy()
    for v in np.flatnonzero(cand):
        p = parent[v]
        while p != H.NONE:
            if cand[p]:
                outer[v] = False
                break
            p = parent[p]
    ids = np.flatnonzero(outer)
    score = area[ids] * depth[ids]
    ids = ids[np.argsort(-score, kind="stable")]
    rows = []
    for v in ids[:limit]:
        fx, fy = h.cell_xy(plan, int(nodes["floor_cell"][v]))
        sx, sy = h.cell_xy(plan, int(nodes["spill_cell"][v]))
        rows.append({"node_id": int(v), "depth_m": round(float(depth[v]), 3),
                     "area_m2": round(float(area[v]), 1),
                     "floor_xy": [round(float(fx), 1), round(float(fy), 1)],
                     "spill_xy": [round(float(sx), 1), round(float(sy), 1)],
                     "floor_m": round(float(nodes["floor_m"][v]), 3),
                     "spill_m": round(float(nodes["spill_m"][v]), 3)})
    return rows, int(ids.size), n_joins


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def run_audit(plan: core.GridPlan, h: H.Hierarchy, codes: np.ndarray,
              wg: kvwater.WaterGrid, river_net,
              river_surface: Optional[np.ndarray] = None) -> dict:
    """`river_surface` is the water surface as exported (only its river samples are
    read); without it every outflow head is 0."""
    if river_surface is not None:
        H.set_outflow_heads(h, wg.type == kvwater.TYPE_RIVER, river_surface)
    tin, tout = H.subtree_intervals(h.nodes)
    inflow = {j.lake_id for j in (river_net.junctions if river_net else [])
              if j.kind == rivernet.JUNCTION_INFLOW}
    lakes = audit_lakes(plan, h, codes, wg, inflow_lakes=inflow, tin=tin, tout=tout)
    if river_net is not None and river_net.segments:
        rivers, walked, minor_pits = audit_rivers(plan, h, codes, wg,
                                                  river_net.segments)
    else:
        rivers, walked, minor_pits = [], 0, 0
    deps, dep_total, dep_joins = unexplained_depressions(plan, h, tin, tout)

    def count(rows, key="status"):
        c: dict = {}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        return c

    lake_status = count(lakes)
    river_status = count(rivers) if rivers else {}
    spill_through: dict = {}
    for r in lakes:
        if r["status"] == "level_above_spill":
            k = r.get("spill_through", "land")
            spill_through[k] = spill_through.get(k, 0) + 1
    summary = {
        "lakes": len(lakes),
        "lakes_by_status": lake_status,
        "lakes_above_spill_by_spill_through": spill_through,
        "lakes_sharing_a_node": sum(1 for r in lakes if "shares_node_with_lake" in r),
        "river_segments_walked": int(walked),
        "river_findings_by_status": river_status,
        "river_pits_below_list_depth": int(minor_pits),
        "unexplained_depressions": dep_total,
        "depressions_joining_a_lake": dep_joins,
    }
    return {
        "format": "kvterrain-water-audit/1",
        "crs": f"EPSG:{plan.epsg}",
        "coordinates": "UTM metres, sample positions",
        "thresholds": {
            "spill_tolerance_m": DEFAULT_SPILL_TOL_M,
            "spill_tolerance_note": "plus one height step (the packing quantum)",
            "regulated_margin_m": DEFAULT_REGULATED_MARGIN_M,
            "min_iou": DEFAULT_MIN_IOU,
            "river_pit_list_min_depth_m": RIVER_PIT_LIST_MIN_DEPTH_M,
            "unexplained_min_depth_m": UNEXPLAINED_MIN_DEPTH_M,
            "unexplained_min_area_m2": UNEXPLAINED_MIN_AREA_M2,
        },
        "notes": {
            "lakes": "status per authored lake: ok, held_by_outflow (above the "
                     "spill but within the outflow river's head; not a finding), "
                     "level_above_spill (above spill + outflow head; excess_m is "
                     "measured from there), below_spill_regulated (report only), "
                     "no_node, no_level",
            "rivers": "only findings are listed, and pits only from "
                      "river_pit_list_min_depth_m (shallower ones are counted in the "
                      "summary); steepest descent on the packed heights from each "
                      "segment's first open-channel vertex, no flow-direction atlas "
                      "yet",
            "depressions": "geometric only (no catchment filter yet); outermost "
                           "candidates, ranked by area x depth; a depression that "
                           "spills into a matched lake at or below its level joins "
                           "that lake and is only counted "
                           "(summary.depressions_joining_a_lake)",
        },
        "summary": summary,
        "lakes": lakes,
        "rivers": rivers,
        "depressions": deps,
    }


def write_audit(audit: dict, out_dir: str) -> dict:
    path = os.path.join(out_dir, AUDIT_FILE)
    with open(path, "w") as fh:
        json.dump(audit, fh, indent=1)
    return {"file": AUDIT_FILE, "bytes": os.path.getsize(path),
            "summary": audit["summary"]}
