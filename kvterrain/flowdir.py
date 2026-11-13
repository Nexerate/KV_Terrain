"""Deterministic D8 flow directions and hierarchy-node catchment areas.

Both products are derived from the packed level-0 height codes.  A runtime can
therefore reproduce the routing without making any floating-point terrain
comparisons, and can maintain catchments after a local edit by applying the same
flat convention to the rebuilt window.
"""

from __future__ import annotations

import os
import struct

import numpy as np

from . import core

try:
    import numba as nb
except ImportError as e:        # pragma: no cover
    raise ImportError(
        "kvterrain.flowdir needs numba (pip install -r requirements.txt)") from e


FLOWDIR_FILE = "flowdir.atlas"
FLOWDIR_BYTES_PER_SAMPLE = 1
FLOWDIR_TERMINAL = 0
FLOWDIR_UNASSIGNED = 255

# Clockwise from north.  Row 0 is north, so north decrements the row.
D8_DR = np.asarray([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int8)
D8_DC = np.asarray([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int8)
D8_NAMES = ("north", "north_east", "east", "south_east",
            "south", "south_west", "west", "north_west")

CATCHMENTS_FILE = "catchments.bin"
CATCHMENTS_MAGIC = b"KVCTCH01"
CATCHMENTS_VERSION = 1
CATCHMENTS_HEADER_FMT = "<8sIIII"
CATCHMENTS_HEADER_BYTES = struct.calcsize(CATCHMENTS_HEADER_FMT)
CATCHMENTS_RECORD_BYTES = 8


@nb.njit(cache=True)
def _build_flow_directions(codes, SY, SX):
    """Resolve every equal-height component into one or more downhill trees."""
    N = SY * SX
    flow = np.full(N, FLOWDIR_UNASSIGNED, dtype=np.uint8)
    seen = np.zeros(N, dtype=np.uint8)
    queue = np.empty(N, dtype=np.int32)

    for start in range(N):
        if seen[start] != 0:
            continue
        height = int(codes[start])

        # First collect this whole 8-connected equal-code component.  Components
        # begin at their lowest row-major member because `start` scans in order.
        seen[start] = 1
        queue[0] = start
        read = 0
        comp_n = 1
        while read < comp_n:
            i = int(queue[read])
            read += 1
            r = i // SX
            c = i - r * SX
            for d in range(8):
                rr = r + int(D8_DR[d])
                cc = c + int(D8_DC[d])
                if rr < 0 or rr >= SY or cc < 0 or cc >= SX:
                    continue
                k = rr * SX + cc
                if seen[k] == 0 and int(codes[k]) == height:
                    seen[k] = 1
                    queue[comp_n] = k
                    comp_n += 1

        # Compact the component's outlet seeds into the front of the same queue.
        # Edges terminate off-map. Other seeds point to their steepest strictly
        # lower neighbour; slope ties take the lowest D8 code.
        seed_n = 0
        pit = int(queue[0])
        for q in range(comp_n):
            i = int(queue[q])
            if i < pit:
                pit = i
            r = i // SX
            c = i - r * SX
            if r == 0 or r == SY - 1 or c == 0 or c == SX - 1:
                flow[i] = FLOWDIR_TERMINAL
                queue[seed_n] = i
                seed_n += 1
                continue

            best_d = -1
            best_drop = 0
            best_den = 1
            for d in range(8):
                rr = r + int(D8_DR[d])
                cc = c + int(D8_DC[d])
                k = rr * SX + cc
                neighbour = int(codes[k])
                if neighbour >= height:
                    continue
                drop = height - neighbour
                den = 1 if (d & 1) == 0 else 2
                lhs = drop * drop * best_den
                rhs = best_drop * best_drop * den
                if best_d < 0 or lhs > rhs or (lhs == rhs and d < best_d):
                    best_d = d
                    best_drop = drop
                    best_den = den
            if best_d >= 0:
                flow[i] = np.uint8(best_d + 1)
                queue[seed_n] = i
                seed_n += 1

        # A closed flat has no lower outlet. Its lowest row-major sample is the
        # sole terminal; this is also the hierarchy pit flat's canonical sink.
        if seed_n == 0:
            flow[pit] = FLOWDIR_TERMINAL
            queue[0] = pit
            seed_n = 1
        elif seed_n > 1:
            # Equal-distance flat ties are settled by the lowest outlet cell.
            queue[:seed_n].sort()

        # Multi-source BFS across the flat. Neighbours are visited in D8 code
        # order. A newly reached sample points back toward the popped sample.
        read = 0
        write = seed_n
        while read < write:
            i = int(queue[read])
            read += 1
            r = i // SX
            c = i - r * SX
            for d in range(8):
                rr = r + int(D8_DR[d])
                cc = c + int(D8_DC[d])
                if rr < 0 or rr >= SY or cc < 0 or cc >= SX:
                    continue
                k = rr * SX + cc
                if int(codes[k]) != height or flow[k] != FLOWDIR_UNASSIGNED:
                    continue
                opposite = ((d + 4) & 7) + 1
                flow[k] = np.uint8(opposite)
                queue[write] = k
                write += 1

    return flow


def build_flow_directions(codes: np.ndarray) -> np.ndarray:
    """Return a north-up u8 D8 grid derived solely from packed height codes."""
    a = np.ascontiguousarray(codes, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError("flow directions need a 2D packed-height grid")
    SY, SX = a.shape
    if a.size >= np.iinfo(np.uint32).max:
        raise ValueError("flow-direction lattice is too large for u32 cell indices")
    out = _build_flow_directions(a.ravel(), SY, SX).reshape(SY, SX)
    if (out == FLOWDIR_UNASSIGNED).any():
        raise RuntimeError("flat resolver left an unassigned sample")
    return out


@nb.njit(cache=True)
def _accumulate(flow, labels, parent, node_count, SY, SX):
    N = flow.size
    indegree = np.zeros(N, dtype=np.uint8)
    area = np.ones(N, dtype=np.uint32)
    queue = np.empty(N, dtype=np.int32)

    for i in range(N):
        d = int(flow[i])
        if d == FLOWDIR_TERMINAL:
            continue
        if d < 1 or d > 8:
            return np.zeros(node_count, dtype=np.uint64), 0, 0
        r = i // SX
        c = i - r * SX
        rr = r + int(D8_DR[d - 1])
        cc = c + int(D8_DC[d - 1])
        if rr < 0 or rr >= SY or cc < 0 or cc >= SX:
            return np.zeros(node_count, dtype=np.uint64), 0, 0
        indegree[rr * SX + cc] += 1

    write = 0
    for i in range(N):
        if indegree[i] == 0:
            queue[write] = i
            write += 1

    read = 0
    while read < write:
        i = int(queue[read])
        read += 1
        d = int(flow[i])
        if d == FLOWDIR_TERMINAL:
            continue
        r = i // SX
        c = i - r * SX
        k = ((r + int(D8_DR[d - 1])) * SX +
             c + int(D8_DC[d - 1]))
        area[k] += area[i]
        indegree[k] -= 1
        if indegree[k] == 0:
            queue[write] = k
            write += 1

    catchment = np.zeros(node_count, dtype=np.uint64)
    terminals = 0
    if write == N:
        for i in range(N):
            if flow[i] == FLOWDIR_TERMINAL:
                catchment[int(labels[i])] += np.uint64(area[i])
                terminals += 1
        # Leaf catchments are disjoint. Internal nodes are defined as the sum of
        # immediate children; root already contains the directly off-map trees.
        for v in range(1, node_count):
            catchment[int(parent[v])] += catchment[v]
    return catchment, write, terminals


def node_catchment_areas(flow: np.ndarray, labels: np.ndarray,
                         nodes: np.ndarray, spacing_m: float) -> tuple[np.ndarray, dict]:
    """Catchment area per hierarchy node from this exact flow-direction field."""
    dirs = np.ascontiguousarray(flow, dtype=np.uint8)
    labs = np.ascontiguousarray(labels, dtype=np.uint32)
    if dirs.shape != labs.shape or dirs.ndim != 2:
        raise ValueError("flow directions and hierarchy labels must share a 2D shape")
    SY, SX = dirs.shape
    cells, processed, terminals = _accumulate(
        dirs.ravel(), labs.ravel(), nodes["parent"], int(nodes.shape[0]), SY, SX)
    if processed != dirs.size:
        raise ValueError(
            f"flow directions contain a cycle or invalid pointer: {processed}/{dirs.size} processed")
    terminal_labels = labs.ravel()[dirs.ravel() == FLOWDIR_TERMINAL]
    terminal_count = np.bincount(
        terminal_labels.astype(np.intp), minlength=int(nodes.shape[0]))
    leaf_ids = np.flatnonzero((nodes["flags"] & 1) != 0)
    if leaf_ids.size and not (terminal_count[leaf_ids] == 1).all():
        bad = int(np.count_nonzero(terminal_count[leaf_ids] != 1))
        raise ValueError(f"{bad} hierarchy leaves do not have exactly one D8 terminal")
    area = cells.astype(np.float64) * (float(spacing_m) * float(spacing_m))
    return area.astype("<f8", copy=False), {
        "samples": int(dirs.size),
        "processed": int(processed),
        "terminals": int(terminals),
        "leaf_terminals": int(terminal_count[leaf_ids].sum()),
        "off_map_terminals": int(terminal_count[0]),
        "root_catchment_cells": int(cells[0]),
    }


def export_flowdir_atlas(plan: core.GridPlan, flow: np.ndarray, out_dir: str,
                         *, file_name: str = FLOWDIR_FILE) -> dict:
    """Write level 0 in the same north-up, row-major dense tile order as other atlases."""
    dirs = np.ascontiguousarray(flow, dtype=np.uint8)
    if dirs.shape != (plan.samples_y, plan.samples_x):
        raise ValueError(f"flow directions {dirs.shape} do not match the plan lattice")
    TC = plan.tile_cells
    TS = TC + 1
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    with open(path, "wb") as fh:
        for ty in range(plan.leaf_tiles_y):
            for tx in range(plan.leaf_tiles_x):
                r0, c0, _ = core.north_up_tile_slice(plan.samples_y, TC, tx, ty)
                tile = np.ascontiguousarray(dirs[r0:r0 + TS, c0:c0 + TS], dtype=np.uint8)
                if tile.shape != (TS, TS):
                    raise RuntimeError(f"bad flowdir tile {tile.shape} at {tx},{ty}")
                tile.tofile(fh)
    expected = (plan.leaf_tiles_x * plan.leaf_tiles_y * TS * TS)
    actual = os.path.getsize(path)
    if actual != expected:
        raise RuntimeError(f"flowdir atlas size mismatch: {actual} != {expected}")
    encoding = {"terminal_pit_or_off_map": FLOWDIR_TERMINAL}
    encoding.update({name: i + 1 for i, name in enumerate(D8_NAMES)})
    return {
        "file": file_name,
        "bytes": actual,
        "format": "dense_v1_level0",
        "dtype": "u8",
        "bytes_per_sample": FLOWDIR_BYTES_PER_SAMPLE,
        "levels": 1,
        "level0_only": True,
        "row_order": "north_to_south",
        "tile_order": "row_major; tile index = tile_y * leaf_tiles_x + tile_x",
        "tile_offset": "(tile_y * leaf_tiles_x + tile_x) * tile_samples^2",
        "encoding": encoding,
        "direction_order": "clockwise from north; row 0 is north",
        "terminal": "0 on every map-edge sample and on the lowest row-major "
                    "sample of each closed equal-code pit flat",
        "slope_rule": "steepest strictly downhill D8 slope; compare drop^2/distance^2 "
                      "as integers; ties take the lowest direction code",
        "flat_rule": "8-connected equal heights.atlas code component; seed map edges "
                     "and cells with a lower neighbour; sort seeds by row-major cell "
                     "index; multi-source FIFO breadth-first search, neighbours in "
                     "direction-code order; first reach wins. A component without a "
                     "seed uses its lowest row-major cell as its sole terminal.",
        "built_on": "heights.atlas level 0 packed u16 codes",
    }


def write_catchments_bin(area_m2: np.ndarray, path: str) -> dict:
    values = np.ascontiguousarray(area_m2, dtype="<f8")
    with open(path, "wb") as fh:
        fh.write(struct.pack(
            CATCHMENTS_HEADER_FMT, CATCHMENTS_MAGIC, CATCHMENTS_VERSION,
            CATCHMENTS_HEADER_BYTES, CATCHMENTS_RECORD_BYTES, int(values.size)))
        values.tofile(fh)
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path)}


def read_catchments_bin(path: str) -> tuple[dict, np.ndarray]:
    with open(path, "rb") as fh:
        raw_header = fh.read(CATCHMENTS_HEADER_BYTES)
        if len(raw_header) != CATCHMENTS_HEADER_BYTES:
            raise ValueError("short catchments header")
        magic, version, header_bytes, record_bytes, node_count = struct.unpack(
            CATCHMENTS_HEADER_FMT, raw_header)
        if magic != CATCHMENTS_MAGIC:
            raise ValueError(f"bad catchments magic {magic!r}")
        if record_bytes != CATCHMENTS_RECORD_BYTES:
            raise ValueError(f"catchments record_bytes {record_bytes} != 8")
        fh.seek(header_bytes)
        raw = fh.read()
    expected = int(node_count) * int(record_bytes)
    if len(raw) != expected:
        raise ValueError(f"catchments payload is {len(raw)} bytes, expected {expected}")
    return {
        "version": int(version), "header_bytes": int(header_bytes),
        "record_bytes": int(record_bytes), "node_count": int(node_count),
    }, np.frombuffer(raw, dtype="<f8").copy()


def catchments_manifest(node_count: int, info: dict, accumulation: dict,
                        spacing_m: float) -> dict:
    return {
        "file": info["file"],
        "bytes": info["bytes"],
        "format": "kvterrain-catchments/1",
        "magic": CATCHMENTS_MAGIC.decode("ascii"),
        "version": CATCHMENTS_VERSION,
        "byte_order": "little_endian",
        "header_bytes": CATCHMENTS_HEADER_BYTES,
        "header_layout": CATCHMENTS_HEADER_FMT,
        "header_fields": ["magic", "version", "header_bytes", "record_bytes",
                          "node_count"],
        "record_bytes": CATCHMENTS_RECORD_BYTES,
        "record": "one f64le catchment_m2; record i belongs to hierarchy node i",
        "node_count": int(node_count),
        "units": "m2",
        "cell_area_m2": float(spacing_m) * float(spacing_m),
        "source": FLOWDIR_FILE,
        "leaf_rule": "sum the unit source area of every level-0 sample whose D8 "
                     "chain terminates in that leaf's pit; terminal sample included; "
                     "membership is by flow, not by labels.atlas cell_count",
        "partition_cell_rule": "cell_count is not added and a node's Priority-Flood "
                               "partition is not unioned into the result; labels.atlas "
                               "is used only to map each terminal pit to its leaf node",
        "parent_rule": "sum immediate child catchments in ascending child id; "
                       "internal nodes have no additional own cells",
        "root_rule": "direct off-map terminal catchments plus every root child; "
                     "therefore root catchment is the entire level-0 lattice",
        "includes_terminal_sample": True,
        "includes_cell_count_as_separate_term": False,
        "parent_includes_children": True,
        "accumulation": accumulation,
    }
