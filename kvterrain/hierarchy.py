"""
kvterrain.hierarchy
===================

The DEPRESSION HIERARCHY of the terrain as shipped: every pit, the saddle it
spills over, where the water goes next, and which pit each sample drains into.
WATER_REDESIGN.md §4.2; the algorithm follows Barnes, Callaghan & Wickert (2020),
implemented here from scratch in Numba.

BUILT ON THE PACKED HEIGHTS
---------------------------
The hierarchy is built on the leaf level's u16 CODES, exactly the numbers
`heights.atlas` holds — not on the float heights they were packed from. Unpacking
is monotone, so a hierarchy on codes is the hierarchy the runtime gets from the
atlas, with no float comparison anywhere to disagree about. Quantisation creates
flats (one step is ~1.5 cm on Lierne), which is why flats are handled explicitly
and why the queue is deterministic.

THE ALGORITHM
-------------
1. PITS. A pit is a maximal 8-connected set of equal-code samples in which no
   sample has a strictly lower neighbour and none lies on the map edge. A flat
   that drains anywhere is not a pit — otherwise every quantisation flat on a
   slope would become a zero-depth node. Pits are numbered 1..P in raster order
   of their first sample; those numbers are the leaf node ids.

2. FLOOD. A bucket queue with one FIFO bucket per code, seeded with every pit
   sample and every map-edge sample, in raster order. Popping a sample labels its
   unlabelled neighbours with its own label and queues them at their own code.
   Every pushed code is >= the code being popped (a lower unlabelled neighbour
   would have been reached from its own pit or edge first), so the queue only
   ever moves forward.

3. MERGES, ONLINE. When a sample is popped next to an already-POPPED sample with a
   different label, the saddle between those two labels is the current code.
   Because pops come in non-decreasing code order, the first saddle seen between
   two regions is their lowest one, so the merges are Kruskal's algorithm in
   discovery order with a union-find — no saddle table, no sort:
     * two depressions meet  -> a new parent node; each child records the spill
       code, the spill sample on its side, the outlet sample across, and the
       other side's top node as `overflow_to`;
     * a depression meets the outside (label 0) -> it becomes a child of the root.

EVERY EDGE SAMPLE IS AN OUTLET (owner's decision, 2026-09-17). The root, node 0,
is "off the map": water reaching any edge sample leaves the export. An inland
export has no other sensible choice — with only sea-level edges draining, every
valley leaving the map would fill to its lowest pass. Lakes the boundary cuts
through are protected by the lake carve instead (`carve_lake_beds(edge_is_shore)`
ramps the bed back up to the waterline at the edge), and the audit reports any
lake that spills off the map anyway.

IDS. 0 is the root, 1..P the leaves, P+1.. internal nodes in merge order. Every
non-root node's parent id is greater than its own except for children of the
root, so ascending id order is a bottom-up walk.

OUTFLOW HEAD. A node's spill is only where water is STABLE up to. Where the spill
runs out through a river channel, the lake stands higher than that while water is
leaving: by the depth of the river at the spill point, since everything that
leaves has to get through there. `outflow_head_m` is that depth, so a node holds
water up to `spill_m + outflow_head_m`. It is the river surface as exported
(which at an outlet is raised to meet its lake) minus the spill, 0 where the
spill is on land (`set_outflow_heads`). For a matched authored lake the audit
raises it to `authored_level_m - spill_m` when the authored level lies within
the channel's head plus tolerance, so the authored level is reproduced exactly.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import core

try:
    import numba as nb
except ImportError as e:        # pragma: no cover — a clear message, not a NameError
    raise ImportError(
        "kvterrain.hierarchy needs numba (pip install -r requirements.txt)") from e

HIERARCHY_FILE = "hierarchy.bin"
LABELS_FILE = "labels.atlas"
LABELS_BYTES_PER_SAMPLE = 4
HYPSOMETRY_FILE = "hypsometry.bin"
HYPSOMETRY_MAGIC = b"KVHYPS01"
HYPSOMETRY_VERSION = 1
HYPSOMETRY_SAMPLES = 64
HYPSOMETRY_BELOW_SPILL_SAMPLES = 48
HYPSOMETRY_HEADER_FMT = "<8sIIIIII"
HYPSOMETRY_HEADER_BYTES = struct.calcsize(HYPSOMETRY_HEADER_FMT)
HYPSOMETRY_DTYPE = np.dtype([
    ("max_code", "<u2"),
    ("flags", "<u2"),
    ("area_m2", "<f4", (HYPSOMETRY_SAMPLES,)),
])
HYPSOMETRY_FLAG_VALID = 1 << 0

BIN_MAGIC = b"KVDEPH01"
BIN_VERSION = 1
HEADER_FMT = "<8sIIIIIIII"          # magic, version, header_bytes, record_bytes,
HEADER_BYTES = struct.calcsize(HEADER_FMT)   # node_count, leaf_count, samples_x,
                                             # samples_y, connectivity

NONE = 0xFFFFFFFF
ROOT = 0
CONNECTIVITY = 8

# One record per node, little-endian, 52 bytes. Readers take record_bytes from the
# header, so fields can be appended in a later version without breaking them.
NODE_DTYPE = np.dtype([
    ("parent", "<u4"),
    ("overflow_to", "<u4"),
    ("floor_cell", "<u4"),
    ("spill_cell", "<u4"),
    ("outlet_cell", "<u4"),
    ("cell_count", "<u4"),
    ("subtree_cells", "<u4"),
    ("floor_code", "<u2"),
    ("spill_code", "<u2"),
    ("floor_m", "<f4"),
    ("spill_m", "<f4"),
    ("authored_level_m", "<f4"),
    ("lake_id", "<u2"),
    ("flags", "<u2"),
    ("outflow_head_m", "<f4"),
])
assert NODE_DTYPE.itemsize == 52

FLAG_LEAF = 1 << 0
FLAG_HAS_SPILL = 1 << 1
FLAG_SPILLS_OFF_MAP = 1 << 2
FLAG_OUTLET_OCEAN = 1 << 3
FLAG_AUTHORED_LAKE = 1 << 4
FLAG_MINOR = 1 << 5
# This node's outflow head was capped by `cap_heads_below_lakes`. (Bit 6 is the
# runtime's own FLAG_EDITED_ESCAPE.)
FLAG_HEAD_CAPPED = 1 << 7

FLAG_NAMES = {
    "leaf": FLAG_LEAF,
    "has_spill": FLAG_HAS_SPILL,
    "spills_off_map": FLAG_SPILLS_OFF_MAP,
    "outlet_ocean": FLAG_OUTLET_OCEAN,
    "authored_lake": FLAG_AUTHORED_LAKE,
    "minor": FLAG_MINOR,
    "head_capped": FLAG_HEAD_CAPPED,
}

# Below either threshold a node is flagged MINOR: kept in the tree (a runtime walk
# needs every spill point to climb out of a pit), but not worth displaying.
DEFAULT_MINOR_DEPTH_M = 0.5
DEFAULT_MINOR_AREA_M2 = 250.0


# --------------------------------------------------------------------------- #
# Numba kernels                                                                #
# --------------------------------------------------------------------------- #

@nb.njit(cache=True)
def _find_pits(codes, SY, SX):
    """Label pit samples 1..P (NONE elsewhere). Returns (labels, P)."""
    N = SY * SX
    labels = np.full(N, NONE, dtype=np.uint32)
    seen = np.zeros(N, dtype=np.uint8)
    stack = np.empty(N, dtype=np.int32)
    P = 0
    for i in range(N):
        if seen[i]:
            continue
        h = codes[i]
        r0 = i // SX
        c0 = i - r0 * SX
        lower = False
        for dr in (-1, 0, 1):
            rr = r0 + dr
            if rr < 0 or rr >= SY:
                continue
            for dc in (-1, 0, 1):
                cc = c0 + dc
                if cc < 0 or cc >= SX or (dr == 0 and dc == 0):
                    continue
                if codes[rr * SX + cc] < h:
                    lower = True
        if lower:
            continue
        # Flood the equal-code flat this sample belongs to.
        seen[i] = 1
        stack[0] = i
        read = 0
        write = 1
        drains = False
        while read < write:
            j = stack[read]
            read += 1
            rj = j // SX
            cj = j - rj * SX
            if rj == 0 or rj == SY - 1 or cj == 0 or cj == SX - 1:
                drains = True
            for dr in (-1, 0, 1):
                rr = rj + dr
                if rr < 0 or rr >= SY:
                    continue
                for dc in (-1, 0, 1):
                    cc = cj + dc
                    if cc < 0 or cc >= SX or (dr == 0 and dc == 0):
                        continue
                    k = rr * SX + cc
                    hk = codes[k]
                    if hk < h:
                        drains = True
                    elif hk == h and seen[k] == 0:
                        seen[k] = 1
                        stack[write] = k
                        write += 1
        if not drains:
            P += 1
            for t in range(write):
                labels[stack[t]] = P
    return labels, P


@nb.njit(cache=True)
def _uf_find(uf, x):
    while uf[x] != x:
        uf[x] = uf[uf[x]]
        x = uf[x]
    return x


@nb.njit(cache=True)
def _flood(codes, labels, P, SY, SX):
    """Priority-Flood + online merges. Fills `labels` in place; returns node arrays."""
    N = SY * SX
    M = 2 * P + 2
    parent = np.full(M, NONE, dtype=np.uint32)
    overflow_to = np.full(M, NONE, dtype=np.uint32)
    floor_cell = np.full(M, NONE, dtype=np.uint32)
    spill_cell = np.full(M, NONE, dtype=np.uint32)
    outlet_cell = np.full(M, NONE, dtype=np.uint32)
    floor_code = np.zeros(M, dtype=np.uint16)
    spill_code = np.zeros(M, dtype=np.uint16)
    has_spill = np.zeros(M, dtype=np.uint8)
    uf = np.arange(M).astype(np.int64)

    head = np.full(65536, -1, dtype=np.int64)
    tail = np.full(65536, -1, dtype=np.int64)
    nxt = np.empty(N, dtype=np.int32)
    popped = np.zeros(N, dtype=np.uint8)

    # Seeds, raster order: edge samples join the root, pit samples keep their pit.
    root_floor = 65535
    root_floor_cell = -1
    for i in range(N):
        r = i // SX
        c = i - r * SX
        edge = r == 0 or r == SY - 1 or c == 0 or c == SX - 1
        lab = labels[i]
        if edge:
            labels[i] = ROOT
        elif lab == NONE:
            continue
        else:
            if floor_cell[lab] == NONE:
                floor_cell[lab] = i
                floor_code[lab] = codes[i]
        p = codes[i]
        nxt[i] = -1
        if tail[p] == -1:
            head[p] = i
        else:
            nxt[tail[p]] = i
        tail[p] = i

    next_id = P + 1
    for p in range(65536):
        while head[p] != -1:
            i = head[p]
            head[p] = nxt[i]
            if head[p] == -1:
                tail[p] = -1
            popped[i] = 1
            li = labels[i]
            if li == ROOT and codes[i] < root_floor:
                root_floor = codes[i]
                root_floor_cell = i
            r0 = i // SX
            c0 = i - r0 * SX
            for dr in (-1, 0, 1):
                rr = r0 + dr
                if rr < 0 or rr >= SY:
                    continue
                for dc in (-1, 0, 1):
                    cc = c0 + dc
                    if cc < 0 or cc >= SX or (dr == 0 and dc == 0):
                        continue
                    k = rr * SX + cc
                    lk = labels[k]
                    if lk == NONE:
                        labels[k] = li
                        q = codes[k]
                        if q < p:          # cannot happen; never let the queue run back
                            q = p
                        nxt[k] = -1
                        if tail[q] == -1:
                            head[q] = k
                        else:
                            nxt[tail[q]] = k
                        tail[q] = k
                    elif popped[k] == 1 and lk != li:
                        ra = _uf_find(uf, li)
                        rb = _uf_find(uf, lk)
                        if ra == rb:
                            continue
                        if ra == ROOT or rb == ROOT:
                            if ra == ROOT:
                                rn, own, other = rb, k, i
                            else:
                                rn, own, other = ra, i, k
                            parent[rn] = ROOT
                            overflow_to[rn] = ROOT
                            spill_code[rn] = p
                            has_spill[rn] = 1
                            spill_cell[rn] = own
                            outlet_cell[rn] = other
                            uf[rn] = ROOT
                        else:
                            m = next_id
                            next_id += 1
                            parent[ra] = m
                            parent[rb] = m
                            spill_code[ra] = p
                            spill_code[rb] = p
                            has_spill[ra] = 1
                            has_spill[rb] = 1
                            spill_cell[ra] = i
                            outlet_cell[ra] = k
                            overflow_to[ra] = rb
                            spill_cell[rb] = k
                            outlet_cell[rb] = i
                            overflow_to[rb] = ra
                            if (floor_code[ra] < floor_code[rb] or
                                    (floor_code[ra] == floor_code[rb] and
                                     floor_cell[ra] < floor_cell[rb])):
                                floor_code[m] = floor_code[ra]
                                floor_cell[m] = floor_cell[ra]
                            else:
                                floor_code[m] = floor_code[rb]
                                floor_cell[m] = floor_cell[rb]
                            uf[ra] = m
                            uf[rb] = m
                            uf[m] = m

    if root_floor_cell >= 0:
        floor_code[ROOT] = root_floor
        floor_cell[ROOT] = root_floor_cell

    n = next_id
    cell_count = np.zeros(n, dtype=np.uint32)
    for i in range(N):
        cell_count[labels[i]] += 1
    subtree = cell_count.copy()
    for v in range(1, n):
        if parent[v] != NONE:
            subtree[parent[v]] += subtree[v]

    return (n, parent[:n].copy(), overflow_to[:n].copy(), floor_cell[:n].copy(),
            spill_cell[:n].copy(), outlet_cell[:n].copy(), floor_code[:n].copy(),
            spill_code[:n].copy(), has_spill[:n].copy(), cell_count, subtree)


@nb.njit(cache=True)
def _downsample_labels(labels, codes):
    """One label pyramid step: corner-anchored 3x3 gather (parent i sits on child
    2i, edge-clamped), the label of the LOWEST child code, ties to the lower label."""
    SY, SX = labels.shape
    oy = (SY - 1) // 2 + 1
    ox = (SX - 1) // 2 + 1
    out = np.empty((oy, ox), dtype=np.uint32)
    for r in range(oy):
        for c in range(ox):
            best_h = 65536
            best_l = np.uint32(NONE)
            for dr in (-1, 0, 1):
                rr = min(max(2 * r + dr, 0), SY - 1)
                for dc in (-1, 0, 1):
                    cc = min(max(2 * c + dc, 0), SX - 1)
                    h = codes[rr, cc]
                    l = labels[rr, cc]
                    if h < best_h or (h == best_h and l < best_l):
                        best_h = h
                        best_l = l
            out[r, c] = best_l
    return out


# --------------------------------------------------------------------------- #
# Python API                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class Hierarchy:
    nodes: np.ndarray               # NODE_DTYPE, (node_count,)
    labels: np.ndarray              # u32 (SY, SX), leaf id per sample, 0 = drains off map
    leaf_count: int
    samples_x: int
    samples_y: int
    height_min: float
    height_max: float
    seconds: dict

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    def code_to_m(self, code):
        return self.height_min + (np.asarray(code, dtype=np.float64) / 65535.0) * (
            self.height_max - self.height_min)

    def m_to_code(self, metres):
        """Fractional code for a height in metres (NOT rounded)."""
        return (np.asarray(metres, dtype=np.float64) - self.height_min) / max(
            self.height_max - self.height_min, 1e-6) * 65535.0

    def cell_xy(self, plan: core.GridPlan, cell) -> tuple:
        cell = np.asarray(cell, dtype=np.int64)
        r = cell // self.samples_x
        c = cell - r * self.samples_x
        x = plan.origin_x + c * plan.spacing_m
        y = plan.origin_y + (self.samples_y - 1 - r) * plan.spacing_m
        return x, y


def build_hierarchy(
    codes: np.ndarray,
    height_min: float,
    height_max: float,
    *,
    spacing_m: float = 1.0,
    ocean: Optional[np.ndarray] = None,
    minor_depth_m: float = DEFAULT_MINOR_DEPTH_M,
    minor_area_m2: float = DEFAULT_MINOR_AREA_M2,
) -> Hierarchy:
    """
    Build the hierarchy on a (SY, SX) u16 code grid (the leaf level as packed).
    `ocean` (bool, same shape) sets FLAG_OUTLET_OCEAN on nodes whose outlet sample
    is sea.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint16)
    SY, SX = codes.shape
    if SY * SX >= NONE:
        raise ValueError("lattice too large for u32 cell indices")
    flat = codes.ravel()
    t0 = time.time()
    labels, P = _find_pits(flat, SY, SX)
    t1 = time.time()
    (n, parent, overflow_to, floor_cell, spill_cell, outlet_cell, floor_code,
     spill_code, has_spill, cell_count, subtree) = _flood(flat, labels, P, SY, SX)
    t2 = time.time()

    nodes = np.zeros(n, dtype=NODE_DTYPE)
    nodes["parent"] = parent
    nodes["overflow_to"] = overflow_to
    nodes["floor_cell"] = floor_cell
    nodes["spill_cell"] = spill_cell
    nodes["outlet_cell"] = outlet_cell
    nodes["cell_count"] = cell_count
    nodes["subtree_cells"] = subtree
    nodes["floor_code"] = floor_code
    nodes["spill_code"] = spill_code
    rng = height_max - height_min
    nodes["floor_m"] = (height_min + floor_code.astype(np.float64) / 65535.0 * rng)
    spill_m = height_min + spill_code.astype(np.float64) / 65535.0 * rng
    nodes["spill_m"] = np.where(has_spill == 1, spill_m, np.nan)
    nodes["authored_level_m"] = np.nan

    flags = np.zeros(n, dtype=np.uint16)
    ids = np.arange(n)
    flags[(ids >= 1) & (ids <= P)] |= FLAG_LEAF
    flags[has_spill == 1] |= FLAG_HAS_SPILL
    flags[(parent == ROOT)] |= FLAG_SPILLS_OFF_MAP
    if ocean is not None:
        oc = np.asarray(ocean, dtype=bool).ravel()
        has_out = outlet_cell != NONE
        is_oc = np.zeros(n, dtype=bool)
        is_oc[has_out] = oc[outlet_cell[has_out].astype(np.int64)]
        flags[is_oc] |= FLAG_OUTLET_OCEAN
    depth_m = (spill_code.astype(np.float64) - floor_code) / 65535.0 * rng
    area_m2 = subtree.astype(np.float64) * spacing_m * spacing_m
    minor = (has_spill == 1) & ((depth_m < minor_depth_m) | (area_m2 < minor_area_m2))
    flags[minor] |= FLAG_MINOR
    nodes["flags"] = flags

    return Hierarchy(nodes=nodes, labels=labels.reshape(SY, SX), leaf_count=int(P),
                     samples_x=int(SX), samples_y=int(SY),
                     height_min=float(height_min), height_max=float(height_max),
                     seconds={"pits": round(t1 - t0, 3), "flood": round(t2 - t1, 3)})


def set_outflow_heads(h: Hierarchy, is_river: np.ndarray,
                      river_surface: np.ndarray) -> None:
    """
    `outflow_head_m` for every node that spills through a river sample: the river
    surface there minus the spill — the river's depth at the spill point — never
    negative; 0 elsewhere. The higher of the two saddle samples counts.
    `river_surface` is quantised and decoded exactly as surface.atlas before any
    samples are read.  Heads therefore reproduce bit-for-bit when the runtime
    derives them from that atlas rather than from this in-memory float raster.

    Measured before the lake tie-in, with the channel's own ground as its level,
    509 Lierne lakes still stood a median 0.45 m above spill + head: at an outlet
    that ground is the flown lake surface, which sits below the authored level.
    The exported surface is raised to the lake there, which is what the river
    actually carries out of it.
    """
    from . import watersurface

    nodes = h.nodes
    has = (nodes["flags"] & FLAG_HAS_SPILL) != 0
    head = np.zeros(h.node_count, dtype=np.float64)
    riv = np.asarray(is_river, dtype=bool).ravel()
    packed = watersurface.pack_surface_u16(
        np.asarray(river_surface), h.height_min, h.height_max)
    lvl = watersurface.unpack_surface_u16(
        packed, h.height_min, h.height_max).ravel()
    spill = nodes["spill_m"].astype(np.float64)
    for key in ("spill_cell", "outlet_cell"):
        cell = nodes[key].astype(np.int64)
        ok = has & (cell != NONE)
        c = cell[ok]
        lv = np.where(riv[c], lvl[c], np.nan).astype(np.float64)
        cand = np.where(np.isfinite(lv), lv - spill[ok], 0.0)
        head[ok] = np.maximum(head[ok], cand)
    nodes["outflow_head_m"] = np.maximum(head, 0.0).astype(np.float32)


def lowest_authored_level_below(nodes: np.ndarray) -> np.ndarray:
    """Per node, the lowest `authored_level_m` of any authored lake in its subtree
    (itself excluded), +inf where there is none. Relies on a non-root parent's id
    being greater than its children's."""
    n = nodes.shape[0]
    low = np.full(n, np.inf)
    parent = nodes["parent"].astype(np.int64)
    authored = (nodes["flags"] & FLAG_AUTHORED_LAKE) != 0
    level = nodes["authored_level_m"].astype(np.float64)
    for v in range(1, n):
        p = parent[v]
        if p == ROOT or p == NONE or p >= n:
            continue
        mine = low[v]
        if authored[v] and np.isfinite(level[v]):
            mine = min(mine, level[v])
        if mine < low[p]:
            low[p] = mine
    return low


def capped_heads(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(head, capped): `outflow_head_m` with every node that is not an authored lake
    held no higher than the lowest authored lake inside it, and where that bit."""
    head = nodes["outflow_head_m"].astype(np.float64)
    low = lowest_authored_level_below(nodes)
    authored = (nodes["flags"] & FLAG_AUTHORED_LAKE) != 0
    room = np.maximum(low - nodes["spill_m"].astype(np.float64), 0.0)
    capped = (~authored) & np.isfinite(low) & (head > room)
    head = np.where(capped, room, head).astype(np.float32)
    return head, capped


def cap_heads_below_lakes(h: Hierarchy) -> dict:
    """
    A node that is not an authored lake may not hold water above an authored lake
    inside it: the runtime fills a node's whole subtree to the highest level on its
    ancestor path, so such a node floods that lake past its mapped shore.

    The head that did it is read at a river sample on the saddle, and on Lierne
    (2026-09-30) that sample was mostly not the river carrying water OUT of the
    node: an inflow's carved trench on the lake's rim, or a stray river sample
    beside a lake carrying a nearby river's higher surface. 195 unmatched
    ancestors then held up to 2.9 m above the lakes below them, and the runtime
    drew 392 lakes too high. Run after the lakes are matched (FLAG_AUTHORED_LAKE).
    Marks every capped node FLAG_HEAD_CAPPED.
    """
    nodes = h.nodes
    head, capped = capped_heads(nodes)
    before = nodes["outflow_head_m"].astype(np.float64)
    nodes["outflow_head_m"] = head
    nodes["flags"][capped] |= FLAG_HEAD_CAPPED
    cut = before[capped] - head[capped]
    return {"heads_capped_below_a_lake": int(capped.sum()),
            "cap_median_m": round(float(np.median(cut)), 3) if cut.size else 0.0,
            "cap_max_m": round(float(cut.max()), 3) if cut.size else 0.0}


def hypsometry_axis(floor_code: int, spill_code: int, max_code: int) -> np.ndarray:
    """The fixed absolute packed-height axis for one non-root node.

    Integer half-up interpolation is part of the wire contract: it avoids a
    language-specific floating-point rounding decision in rebuilt nodes.
    """
    floor = int(floor_code)
    spill = max(int(spill_code), floor)
    top = max(int(max_code), spill)
    out = np.empty(HYPSOMETRY_SAMPLES, dtype=np.uint16)
    low_n = HYPSOMETRY_BELOW_SPILL_SAMPLES
    den = low_n - 1
    for i in range(low_n):
        out[i] = (floor * (den - i) + spill * i + den // 2) // den
    high_n = HYPSOMETRY_SAMPLES - low_n
    for k in range(1, high_n + 1):
        out[low_n + k - 1] = (
            spill * (high_n - k) + top * k + high_n // 2) // high_n
    return out


@nb.njit(cache=True)
def _leaf_hypsometry(labels, codes, floor_code, spill_code, max_code,
                     leaf_count, sample_count):
    """Difference counts at the first axis sample >= each cell code."""
    diff = np.zeros((leaf_count + 1, sample_count), dtype=np.uint32)
    low_n = HYPSOMETRY_BELOW_SPILL_SAMPLES
    high_n = sample_count - low_n
    for i in range(labels.size):
        leaf = int(labels[i])
        if leaf <= 0 or leaf > leaf_count:
            continue
        code = int(codes[i])
        floor = int(floor_code[leaf])
        spill = max(int(spill_code[leaf]), floor)
        top = max(int(max_code[leaf]), spill)
        lo = 0
        hi = sample_count
        while lo < hi:
            mid = (lo + hi) // 2
            if mid < low_n:
                den = low_n - 1
                axis = (floor * (den - mid) + spill * mid + den // 2) // den
            else:
                k = mid - low_n + 1
                axis = (spill * (high_n - k) + top * k + high_n // 2) // high_n
            if axis < code:
                lo = mid + 1
            else:
                hi = mid
        if lo < sample_count:
            diff[leaf, lo] += 1
    for leaf in range(1, leaf_count + 1):
        for j in range(1, sample_count):
            diff[leaf, j] += diff[leaf, j - 1]
    return diff


def _curve_at(axis: np.ndarray, area: np.ndarray, z: int) -> float:
    """Linear interpolation; duplicate codes take their rightmost area."""
    j = int(np.searchsorted(axis, z, side="right"))
    if j == 0:
        return 0.0
    if j >= axis.size:
        return float(area[-1])
    x0, x1 = int(axis[j - 1]), int(axis[j])
    if x1 == x0:
        return float(area[j])
    t = (int(z) - x0) / float(x1 - x0)
    return float(area[j - 1]) + t * (float(area[j]) - float(area[j - 1]))


def build_hypsometry(h: Hierarchy, codes: np.ndarray,
                     spacing_m: float) -> np.ndarray:
    """Fixed-size, absolute-axis cumulative area curves for every non-root node.

    Leaves are measured from the shipped label/code grids. Internal nodes are
    deliberately composed from their immediate child curves, never rescanned,
    which is the same operation the runtime can use after an in-place rebuild.
    """
    flat_codes = np.ascontiguousarray(codes, dtype=np.uint16).ravel()
    flat_labels = np.ascontiguousarray(h.labels, dtype=np.uint32).ravel()
    if flat_codes.size != flat_labels.size:
        raise ValueError("hypsometry codes and labels must have the same shape")
    n, P = h.node_count, h.leaf_count
    nodes = h.nodes

    max_code = np.zeros(n, dtype=np.uint16)
    for leaf in range(1, P + 1):
        max_code[leaf] = nodes["floor_code"][leaf]
    np.maximum.at(max_code, flat_labels.astype(np.intp), flat_codes)
    parent = nodes["parent"].astype(np.int64)
    for v in range(1, n):
        p = int(parent[v])
        if p != NONE and max_code[v] > max_code[p]:
            max_code[p] = max_code[v]

    records = np.zeros(n, dtype=HYPSOMETRY_DTYPE)
    records["max_code"] = max_code
    records["flags"][1:] = HYPSOMETRY_FLAG_VALID
    leaf_counts = _leaf_hypsometry(
        flat_labels, flat_codes, nodes["floor_code"], nodes["spill_code"],
        max_code, P, HYPSOMETRY_SAMPLES)
    cell_area = float(spacing_m) * float(spacing_m)
    records["area_m2"][1:P + 1] = (
        leaf_counts[1:].astype(np.float64) * cell_area).astype(np.float32)
    del leaf_counts

    children: list[list[int]] = [[] for _ in range(n)]
    for v in range(1, n):
        p = int(parent[v])
        if p != NONE:
            children[p].append(v)

    for v in range(P + 1, n):
        if v == ROOT:
            continue
        axis = hypsometry_axis(
            nodes["floor_code"][v], nodes["spill_code"][v], max_code[v])
        total = np.zeros(HYPSOMETRY_SAMPLES, dtype=np.float64)
        for child in children[v]:
            child_axis = hypsometry_axis(
                nodes["floor_code"][child], nodes["spill_code"][child],
                max_code[child])
            child_area = records["area_m2"][child]
            for j, z in enumerate(axis):
                total[j] += _curve_at(child_axis, child_area, int(z))
        records["area_m2"][v] = total.astype(np.float32)
    return records


def subtree_intervals(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tin, tout) per node from a DFS, so `a` is in the subtree of `b` iff
    tin[b] <= tin[a] < tout[b]."""
    n = nodes.shape[0]
    parent = nodes["parent"].astype(np.int64)
    has = parent != NONE
    child = np.nonzero(has)[0]
    order = np.argsort(parent[child], kind="stable")
    child = child[order]
    starts = np.searchsorted(parent[child], np.arange(n + 1))
    return _dfs_intervals(child, starts, n)


@nb.njit(cache=True)
def _dfs_intervals(child, starts, n):
    tin = np.zeros(n, dtype=np.int64)
    tout = np.zeros(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    cursor = np.zeros(n, dtype=np.int64)
    # Node 0 is the only root: every other node has a parent.
    t = 1
    sp = 0
    stack[0] = ROOT
    cursor[ROOT] = starts[ROOT]
    while sp >= 0:
        v = stack[sp]
        if cursor[v] < starts[v + 1]:
            w = child[cursor[v]]
            cursor[v] += 1
            sp += 1
            stack[sp] = w
            tin[w] = t
            t += 1
            cursor[w] = starts[w]
        else:
            tout[v] = t
            sp -= 1
    return tin, tout


def build_label_pyramid(labels0: np.ndarray, code_levels: list) -> list:
    """Label pyramid: level L from level L-1's labels and L-1's PACKED heights (the
    heights.atlas codes at that level), so it can be recomputed from the export."""
    levels = [np.ascontiguousarray(labels0, dtype=np.uint32)]
    for lvl in range(1, len(code_levels)):
        levels.append(_downsample_labels(
            levels[-1], np.ascontiguousarray(code_levels[lvl - 1], dtype=np.uint16)))
    return levels


# --------------------------------------------------------------------------- #
# Writers                                                                      #
# --------------------------------------------------------------------------- #

def write_hypsometry_bin(records: np.ndarray, path: str) -> dict:
    records = np.ascontiguousarray(records, dtype=HYPSOMETRY_DTYPE)
    with open(path, "wb") as fh:
        fh.write(struct.pack(
            HYPSOMETRY_HEADER_FMT, HYPSOMETRY_MAGIC, HYPSOMETRY_VERSION,
            HYPSOMETRY_HEADER_BYTES, HYPSOMETRY_DTYPE.itemsize,
            int(records.shape[0]), HYPSOMETRY_SAMPLES,
            HYPSOMETRY_BELOW_SPILL_SAMPLES))
        records.tofile(fh)
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path)}


def read_hypsometry_bin(path: str) -> tuple[dict, np.ndarray]:
    with open(path, "rb") as fh:
        raw_header = fh.read(HYPSOMETRY_HEADER_BYTES)
        if len(raw_header) != HYPSOMETRY_HEADER_BYTES:
            raise ValueError("short hypsometry header")
        (magic, version, header_bytes, record_bytes, node_count,
         samples, below_spill) = struct.unpack(HYPSOMETRY_HEADER_FMT, raw_header)
        if magic != HYPSOMETRY_MAGIC:
            raise ValueError(f"bad hypsometry magic {magic!r}")
        if record_bytes != HYPSOMETRY_DTYPE.itemsize:
            raise ValueError(
                f"hypsometry record_bytes {record_bytes} != {HYPSOMETRY_DTYPE.itemsize}")
        fh.seek(header_bytes)
        raw = fh.read()
    expected = node_count * record_bytes
    if len(raw) != expected:
        raise ValueError(f"hypsometry payload is {len(raw)} bytes, expected {expected}")
    records = np.frombuffer(raw, dtype=HYPSOMETRY_DTYPE).copy()
    return {
        "version": version, "header_bytes": header_bytes,
        "record_bytes": record_bytes, "node_count": node_count,
        "samples_per_node": samples,
        "below_spill_samples": below_spill,
    }, records


def hypsometry_manifest(h: Hierarchy, records: np.ndarray, info: dict,
                        spacing_m: float) -> dict:
    return {
        "file": info["file"],
        "bytes": info["bytes"],
        "format": "kvterrain-hypsometry/1",
        "magic": HYPSOMETRY_MAGIC.decode("ascii"),
        "version": HYPSOMETRY_VERSION,
        "byte_order": "little_endian",
        "header_bytes": HYPSOMETRY_HEADER_BYTES,
        "header_layout": HYPSOMETRY_HEADER_FMT,
        "header_fields": ["magic", "version", "header_bytes", "record_bytes",
                          "node_count", "samples_per_node",
                          "below_spill_samples"],
        "record_bytes": HYPSOMETRY_DTYPE.itemsize,
        "record_fields": [
            {"name": "max_code", "dtype": "u16le", "offset": 0},
            {"name": "flags", "dtype": "u16le", "offset": 2,
             "bits": {"valid": HYPSOMETRY_FLAG_VALID}},
            {"name": "area_m2", "dtype": "64*f32le", "offset": 4},
        ],
        "node_count": h.node_count,
        "samples_per_node": HYPSOMETRY_SAMPLES,
        "below_spill_samples": HYPSOMETRY_BELOW_SPILL_SAMPLES,
        "upper_samples": HYPSOMETRY_SAMPLES - HYPSOMETRY_BELOW_SPILL_SAMPLES,
        "node_record": "record i belongs to hierarchy node i; root record 0 is invalid",
        "axis": {
            "kind": "absolute_heights_atlas_u16_code",
            "decode_m": "height_min_m + code/65535*(height_max_m-height_min_m)",
            "lower": "48 integer-half-up samples, inclusive, from floor_code to "
                     "spill_code",
            "upper": "16 integer-half-up samples after the lower block, from "
                     "spill_code toward max_code, inclusive of max_code",
            "integer_rounding": "round non-negative rational half up",
            "max_code": "maximum heights.atlas level-0 code carried by any leaf "
                        "label in this node's subtree; parent max is max(children)",
        },
        "value": "cumulative planimetric flooded area in square metres at or "
                 "below the sample's absolute elevation code",
        "interpolation": "piecewise linear in absolute height code; below the "
                         "first code area=0; above max_code area=last value; "
                         "duplicate codes take the rightmost area",
        "composition": "leaf curves are derived from heights.atlas level 0 and "
                       "labels.atlas level 0. Every non-root internal curve is "
                       "computed only as sum(interpolate(child,z)) over immediate "
                       "children at its own absolute axis, in ascending child id "
                       "order, accumulated as f64 then stored f32.",
        "cell_area_m2": float(spacing_m) * float(spacing_m),
        "valid_nodes": int(((records["flags"] & HYPSOMETRY_FLAG_VALID) != 0).sum()),
    }

def write_hierarchy_bin(h: Hierarchy, path: str) -> dict:
    with open(path, "wb") as fh:
        fh.write(struct.pack(HEADER_FMT, BIN_MAGIC, BIN_VERSION, HEADER_BYTES,
                             NODE_DTYPE.itemsize, h.node_count, h.leaf_count,
                             h.samples_x, h.samples_y, CONNECTIVITY))
        fh.write(h.nodes.tobytes())
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path)}


def read_hierarchy_bin(path: str) -> tuple[dict, np.ndarray]:
    with open(path, "rb") as fh:
        hdr = fh.read(HEADER_BYTES)
        (magic, version, header_bytes, record_bytes, node_count, leaf_count,
         sx, sy, conn) = struct.unpack(HEADER_FMT, hdr)
        if magic != BIN_MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        fh.seek(header_bytes)
        raw = fh.read()
    header = {"version": version, "header_bytes": header_bytes,
              "record_bytes": record_bytes, "node_count": node_count,
              "leaf_count": leaf_count, "samples_x": sx, "samples_y": sy,
              "connectivity": conn, "trailing_bytes": len(raw) - node_count * record_bytes}
    if record_bytes < NODE_DTYPE.itemsize:
        raise ValueError(f"record_bytes {record_bytes} < {NODE_DTYPE.itemsize}")
    rec = np.frombuffer(raw[:node_count * record_bytes], dtype=np.uint8).reshape(
        node_count, record_bytes)
    nodes = np.frombuffer(rec[:, :NODE_DTYPE.itemsize].tobytes(), dtype=NODE_DTYPE)
    return header, nodes


def export_labels_atlas(plan: core.GridPlan, label_levels: list, out_dir: str,
                        *, atlas_name: str = LABELS_FILE) -> dict:
    """Every label level sliced into tiles and concatenated, 4 bytes per sample,
    in the same level/row-major order as every other atlas."""
    TC = plan.tile_cells
    TS = TC + 1
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, atlas_name)
    tiles = 0
    with open(path, "wb") as fh:
        for lvl, arr in enumerate(label_levels):
            tx_n, ty_n = core.tiles_at_level(plan.leaf_tiles_x, plan.leaf_tiles_y, lvl)
            SY = arr.shape[0]
            for ty in range(ty_n):
                for tx in range(tx_n):
                    r0, c0, _ = core.north_up_tile_slice(SY, TC, tx, ty)
                    tile = np.ascontiguousarray(arr[r0:r0 + TS, c0:c0 + TS], dtype="<u4")
                    assert tile.shape == (TS, TS), f"bad labels slice L{lvl} {tx},{ty}"
                    tile.tofile(fh)
                    tiles += 1
    expect = core.atlas_total_bytes(plan.leaf_tiles_x, plan.leaf_tiles_y,
                                    plan.num_levels, TS, LABELS_BYTES_PER_SAMPLE)
    actual = os.path.getsize(path)
    if actual != expect:
        raise RuntimeError(f"dense labels atlas size mismatch: {actual} != {expect}")
    return {"atlas_file": atlas_name, "bytes": actual, "tiles_written": tiles}


def manifest_block(h: Hierarchy, *, hier_file: dict, labels: dict,
                   minor_depth_m: float, minor_area_m2: float,
                   ocean_level_m: float) -> dict:
    fl = h.nodes["flags"]
    nonroot = np.arange(h.node_count) != ROOT
    return {
        "format": "kvterrain-depression-hierarchy/1",
        "status": "node table, labels, composable absolute-axis hypsometry and "
                  "flow-derived node catchments; flow directions are a level-0 "
                  "sidecar atlas",
        "file": hier_file["file"],
        "bytes": hier_file["bytes"],
        "magic": BIN_MAGIC.decode("ascii"),
        "version": BIN_VERSION,
        "byte_order": "little_endian",
        "header_bytes": HEADER_BYTES,
        "header_fields": ["magic", "version", "header_bytes", "record_bytes",
                          "node_count", "leaf_count", "samples_x", "samples_y",
                          "connectivity"],
        "record_bytes": NODE_DTYPE.itemsize,
        "record_fields": [
            {"name": name, "dtype": NODE_DTYPE.fields[name][0].str,
             "offset": NODE_DTYPE.fields[name][1]} for name in NODE_DTYPE.names],
        "node_count": h.node_count,
        "leaf_count": h.leaf_count,
        "connectivity": CONNECTIVITY,
        "built_on": "heights.atlas level 0 as packed (u16 codes); floor_code and "
                    "spill_code compare exactly against it, *_m = height_min_m + "
                    "code / 65535 * (height_max_m - height_min_m)",
        "cell_index": "row * samples_x + col over the level-0 lattice; row 0 is north",
        "ids": "0 = root (off the map); 1..leaf_count = leaves (pits, raster order of "
               "their first sample); then internal nodes in merge order. A node's "
               "parent id is greater than its own except for children of the root",
        "none": NONE,
        "semantics": {
            "parent": "NONE on the root",
            "overflow_to": "the node on the other side of the saddle at the time of "
                           "the merge (its sibling); 0 for children of the root",
            "spill_cell": "the sample on this node's side of its lowest saddle",
            "outlet_cell": "the neighbouring sample across that saddle",
            "spill_code": "max(code(spill_cell), code(outlet_cell)); valid only "
                          "with the has_spill flag",
            "cell_count": "samples labelled with this node (non-zero only for leaves "
                          "and the root)",
            "subtree_cells": "samples in this node and all its descendants; area = "
                             "subtree_cells * leaf_spacing_m^2",
            "lake_id": "authored lake matched to this node (lakes.json lake_id), 0 = none",
            "outflow_head_m": "depth of water the outflow river carries over the "
                              "spill: the node holds water up to spill_m + "
                              "outflow_head_m. The river surface at the spill is "
                              "first encoded and decoded with surface.atlas's "
                              "[0,65534] packing, then reduced by spill_m (0 where "
                              "the spill is on land); "
                              "for a matched lake whose authored level lies within "
                              "that head plus tolerance, raised to authored_level_m "
                              "- spill_m",
            "edges": "every map-edge sample is an outlet into the root",
        },
        "flags": FLAG_NAMES,
        "minor_thresholds": {"depth_m": minor_depth_m, "area_m2": minor_area_m2,
                             "rule": "has_spill and (spill_m - floor_m < depth_m or "
                                     "subtree area < area_m2)"},
        "ocean_level_m": ocean_level_m,
        "stats": {
            "nodes_minor": int(((fl & FLAG_MINOR) != 0).sum()),
            "nodes_spilling_off_map": int(((fl & FLAG_SPILLS_OFF_MAP) != 0).sum()),
            "nodes_with_lake": int(((fl & FLAG_AUTHORED_LAKE) != 0).sum()),
            "cells_draining_off_map_directly": int(h.nodes["cell_count"][ROOT]),
            "max_depth_m": float(np.nanmax(
                np.where(nonroot, h.nodes["spill_m"] - h.nodes["floor_m"], np.nan)))
            if h.node_count > 1 else 0.0,
        },
        "labels": {
            "file": labels["atlas_file"],
            "bytes": labels["bytes"],
            "dtype": "u32le",
            "bytes_per_sample": LABELS_BYTES_PER_SAMPLE,
            "atlas_format": core.ATLAS_FORMAT,
            "row_order": "north_to_south",
            "value": "leaf node id the sample drains into; 0 = drains straight off "
                     "the map",
            "offsets": "same arithmetic as every atlas with bytes_per_sample = 4: a "
                       "label tile has the same TILE INDEX as its height twin but "
                       "not the same byte offset",
            "downsample": "level L sample = label of the child with the lowest "
                          "heights.atlas code at level L-1 in the corner-anchored "
                          "3x3 gather (edge-clamped), ties to the lower label",
        },
        "seconds": h.seconds,
    }
