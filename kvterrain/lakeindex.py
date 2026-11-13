"""
kvterrain.lakeindex
===================

Two small level-0 lake indexes the runtime would otherwise rebuild at startup by
scanning the whole lattice (3.5 s on a 16x16 km export):

``lake_cells.bin``    where every authored lake's water_id samples are, as runs.
``lake_samples.bin``  how many of each authored lake's samples every level-0
                      label holds (the base lake-sample index of the pin).

Both are POST-PROCESSING: they read the finished ``water_id.atlas`` and
``labels.atlas`` back out of an export directory and change nothing else, so they
can be (re)written on an existing export without re-exporting
(``kvterrain lake-indexes --export DIR``). ``process`` runs the same step last.

Wire layouts, all little-endian, cell = row * samples_x + col with row 0 north:

lake_cells.bin
    u32 lake_count
    per lake, ascending lake id:
        i32 lake_id, u32 run_count, run_count x (u32 first_cell, u32 length)
    A run is a maximal stretch of consecutive cells in ONE row whose water_id is
    ``lake_id_base + lake_id``; runs in raster order. Island cells stay lake-coded
    in water_id, so they are included (``lake_islands.bin`` lists them). Every
    lake id present in water_id is listed, authored level or not.

lake_samples.bin
    u32 record_count
    record_count x (u32 label, i32 lake_id, u32 count), sorted by label, lake_id
    count = level-0 samples with that label and water_id == lake_id_base +
    lake_id. Only lakes whose lakes.json ``authored_level_m`` > 0; labels ROOT (0)
    and NONE (0xFFFFFFFF) are skipped. The same samples `hierarchy.
    lake_sample_levels` reduces to a per-node minimum level.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Optional

import numpy as np

from . import waterid

LAKE_CELLS_FILE = "lake_cells.bin"
LAKE_SAMPLES_FILE = "lake_samples.bin"

LABEL_ROOT = 0
LABEL_NONE = 0xFFFFFFFF

_RUN_DTYPE = np.dtype([("first_cell", "<u4"), ("length", "<u4")])
_SAMPLE_DTYPE = np.dtype([("label", "<u4"), ("lake_id", "<i4"), ("count", "<u4")])


# --------------------------------------------------------------------------- #
# Building                                                                     #
# --------------------------------------------------------------------------- #

def _lake_cells(water_id_flat: np.ndarray, lake_id_base: int):
    """(cell indices ascending, their local lake ids) of every lake-coded sample."""
    idx = np.flatnonzero(water_id_flat >= lake_id_base)
    lake = water_id_flat[idx].astype(np.int32) - int(lake_id_base)
    return idx, lake


def build_lake_runs(water_id: np.ndarray, lake_id_base: int = waterid.LAKE_ID_BASE):
    """Per lake id (ascending): its runs as (first_cell, length) arrays.

    Returns ``(lake_ids, run_counts, first_cells, lengths)`` with the runs grouped
    by lake in lake order and in raster order within a lake.
    """
    wid = np.asarray(water_id)
    if wid.ndim != 2:
        raise ValueError("water_id must be a 2D level-0 grid")
    sx = int(wid.shape[1])
    if wid.size > np.iinfo(np.uint32).max:
        raise ValueError("level 0 is too large for u32 cell indices")
    idx, lake = _lake_cells(wid.ravel(order="C"), lake_id_base)
    if idx.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e.astype(np.int32), e.astype(np.uint32), e, e
    start = np.ones(idx.size, dtype=bool)
    start[1:] = ((idx[1:] != idx[:-1] + 1) | (lake[1:] != lake[:-1])
                 | (idx[1:] % sx == 0))
    s = np.flatnonzero(start)
    first = idx[s]
    length = np.diff(np.append(s, idx.size))
    run_lake = lake[s]
    del idx, lake, start, s
    # Stable, so raster order survives within each lake.
    order = np.argsort(run_lake, kind="stable")
    first, length, run_lake = first[order], length[order], run_lake[order]
    lake_ids, run_counts = np.unique(run_lake, return_counts=True)
    return lake_ids.astype(np.int32), run_counts.astype(np.uint32), first, length


def build_lake_samples(water_id: np.ndarray, labels: np.ndarray,
                       authored_lakes, lake_id_base: int = waterid.LAKE_ID_BASE
                       ) -> np.ndarray:
    """Records (label, lake_id, count) sorted by label then lake_id, over the
    lakes in `authored_lakes`, skipping labels ROOT and NONE."""
    wid = np.asarray(water_id).ravel(order="C")
    lab = np.asarray(labels).ravel(order="C")
    if wid.size != lab.size:
        raise ValueError(f"water_id has {wid.size} samples, labels {lab.size}")
    keep = np.zeros(65536, dtype=bool)
    ids = np.asarray(sorted(int(i) for i in authored_lakes), dtype=np.int64)
    ids = ids[(ids >= 0) & (ids + lake_id_base <= 65535)]
    keep[ids + int(lake_id_base)] = True
    idx = np.flatnonzero(keep[wid])
    lake = wid[idx].astype(np.uint64) - np.uint64(lake_id_base)
    label = lab[idx].astype(np.uint64)
    del idx
    ok = (label != LABEL_ROOT) & (label != LABEL_NONE)
    key = (label[ok] << np.uint64(16)) | lake[ok]
    del label, lake, ok
    uniq, counts = np.unique(key, return_counts=True)
    rec = np.empty(uniq.size, dtype=_SAMPLE_DTYPE)
    rec["label"] = (uniq >> np.uint64(16)).astype(np.uint32)
    rec["lake_id"] = (uniq & np.uint64(0xFFFF)).astype(np.int32)
    rec["count"] = counts.astype(np.uint32)
    return rec


# --------------------------------------------------------------------------- #
# Wire                                                                         #
# --------------------------------------------------------------------------- #

def write_lake_cells(path: str, lake_ids, run_counts, first_cells, lengths) -> int:
    """Write lake_cells.bin; returns bytes written."""
    runs = np.empty(len(first_cells), dtype=_RUN_DTYPE)
    runs["first_cell"] = first_cells
    runs["length"] = lengths
    bounds = np.concatenate([[0], np.cumsum(np.asarray(run_counts, dtype=np.int64))])
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", len(lake_ids)))
        for i, (lid, n) in enumerate(zip(lake_ids, run_counts)):
            fh.write(struct.pack("<iI", int(lid), int(n)))
            runs[bounds[i]:bounds[i + 1]].tofile(fh)
    return os.path.getsize(path)


def read_lake_cells(path: str) -> dict:
    """{lake_id: (first_cells, lengths)} from a lake_cells.bin, validated."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4:
        raise ValueError("short lake_cells count")
    n = struct.unpack_from("<I", raw, 0)[0]
    off = 4
    out = {}
    prev = None
    for _ in range(n):
        if off + 8 > len(raw):
            raise ValueError("lake_cells.bin ends inside a lake header")
        lid, runs = struct.unpack_from("<iI", raw, off)
        off += 8
        if prev is not None and lid <= prev:
            raise ValueError("lake ids are not strictly ascending")
        prev = lid
        end = off + runs * _RUN_DTYPE.itemsize
        if end > len(raw):
            raise ValueError(f"lake {lid} runs overrun the file")
        r = np.frombuffer(raw[off:end], dtype=_RUN_DTYPE)
        off = end
        if r.size > 1 and np.any(r["first_cell"][1:] <= r["first_cell"][:-1]):
            raise ValueError(f"lake {lid} runs are not in raster order")
        out[int(lid)] = (r["first_cell"].copy(), r["length"].copy())
    if off != len(raw):
        raise ValueError(f"{len(raw) - off} trailing bytes in lake_cells.bin")
    return out


def write_lake_samples(path: str, records: np.ndarray) -> int:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", int(records.size)))
        np.ascontiguousarray(records, dtype=_SAMPLE_DTYPE).tofile(fh)
    return os.path.getsize(path)


def read_lake_samples(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4:
        raise ValueError("short lake_samples count")
    n = struct.unpack_from("<I", raw, 0)[0]
    if len(raw) != 4 + n * _SAMPLE_DTYPE.itemsize:
        raise ValueError(f"lake_samples.bin is {len(raw)} bytes, expected "
                         f"{4 + n * _SAMPLE_DTYPE.itemsize}")
    rec = np.frombuffer(raw, dtype=_SAMPLE_DTYPE, offset=4).copy()
    key = (rec["label"].astype(np.uint64) << np.uint64(32)) | \
        rec["lake_id"].astype(np.uint32).astype(np.uint64)
    if key.size > 1 and np.any(key[1:] <= key[:-1]):
        raise ValueError("lake_samples records are not strictly sorted")
    return rec


# --------------------------------------------------------------------------- #
# The post-processing step                                                     #
# --------------------------------------------------------------------------- #

def authored_lake_ids(lakes_path: str) -> list:
    """Lake ids whose `authored_level_m` > 0 in a lakes.json."""
    with open(lakes_path) as f:
        doc = json.load(f)
    rows = doc.get("lakes", doc) if isinstance(doc, dict) else doc
    out = []
    for r in rows:
        if "lake_id" not in r:
            continue
        lv = r.get("authored_level_m")
        if lv is not None and np.isfinite(float(lv)) and float(lv) > 0:
            out.append(int(r["lake_id"]))
    return out


def write_lake_indexes(root: str, manifest: Optional[dict] = None) -> dict:
    """
    Read level 0 of water_id (and labels) back out of the export at `root`, write
    lake_cells.bin and lake_samples.bin beside lake_islands.bin, name them under
    `manifest["water_id"]` and rewrite manifest.json.

    `manifest` is the dict to update and dump; by default the one on disk. Pass
    the in-memory manifest when the caller will dump it again itself. Returns the
    two manifest blocks. lake_samples is skipped (and its block absent) when the
    export has no labels atlas.
    """
    from . import exports

    exp = exports.load(root)
    if manifest is not None:
        exp.manifest = manifest
    wid_block = exp.manifest.get("water_id")
    if not wid_block or not exp.atlas_path("water_id"):
        raise ValueError(f"{root} has no water_id atlas")
    base = int(wid_block.get("encoding", {}).get("lake_id_base", waterid.LAKE_ID_BASE))
    SY, SX = exp.level_shape(0)
    common = {
        "byte_order": "little_endian",
        "level": 0,
        "samples_x": int(SX),
        "samples_y": int(SY),
        "cell_index": "row * samples_x + col; row 0 is north",
        "lake_id_base": base,
    }

    water_id = exp.read_level("water_id", 0)
    lake_ids, run_counts, first, length = build_lake_runs(water_id, base)
    cells_bytes = write_lake_cells(os.path.join(exp.root, LAKE_CELLS_FILE),
                                   lake_ids, run_counts, first, length)
    blocks = {"lake_cells": {
        "file": LAKE_CELLS_FILE,
        "bytes": cells_bytes,
        "format": "u32 lake_count; per lake ascending: i32 lake_id, u32 run_count, "
                  "run_count x (u32 first_cell, u32 length)",
        **common,
        "lake_count": int(lake_ids.size),
        "run_count": int(first.size),
        "sample_count": int(length.sum()),
        "semantics": "a run is a maximal stretch of consecutive cells in one row "
                     "with water_id == lake_id_base + lake_id; runs in raster "
                     "order; island cells are lake-coded and included; every lake "
                     "id present in water_id is listed",
    }}
    del first, length

    lakes_file = (exp.manifest.get("water_vector", {}).get("lakes", {}).get("file")
                  or "lakes.json")
    lakes_path = os.path.join(exp.root, lakes_file)
    if exp.atlas_path("labels"):
        authored = authored_lake_ids(lakes_path) if os.path.exists(lakes_path) else []
        labels = exp.read_level("labels", 0)
        rec = build_lake_samples(water_id, labels, authored, base)
        del labels
        samples_bytes = write_lake_samples(
            os.path.join(exp.root, LAKE_SAMPLES_FILE), rec)
        blocks["lake_samples"] = {
            "file": LAKE_SAMPLES_FILE,
            "bytes": samples_bytes,
            "format": "u32 record_count; record_count x (u32 label, i32 lake_id, "
                      "u32 count) sorted by label then lake_id",
            **common,
            "record_count": int(rec.size),
            "sample_count": int(rec["count"].sum(dtype=np.int64)),
            "lakes": int(np.unique(rec["lake_id"]).size),
            "semantics": "count = level-0 samples with that labels.atlas label and "
                         "water_id == lake_id_base + lake_id; only lakes whose "
                         f"{lakes_file} authored_level_m > 0; labels ROOT (0) and "
                         "NONE (0xFFFFFFFF) skipped",
        }
    del water_id

    wid_block.update(blocks)
    with open(os.path.join(exp.root, exports.MANIFEST_NAME), "w") as f:
        json.dump(exp.manifest, f, indent=2)
    return blocks
