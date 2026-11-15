"""
kvterrain.tilepack
==================

The tiled, per-tile compressed atlas container, ``kvtiles_v1``, and a reader
that opens either it or the older uncompressed ``dense_v1`` layout.

An atlas is still the same sequence of tiles it always was — level 0 first,
then each coarser level, row-major within a level — so a tile's INDEX is
unchanged. Only how a tile's bytes are found changes: ``dense_v1`` finds tile i
at ``i * tile_raw_bytes``, while ``kvtiles_v1`` keeps an offset index.

kvtiles_v1, all little-endian::

    header (32 bytes)
        8s   magic             b"KVTILE01"
        u32  version           1
        u32  header_bytes      32
        u32  tile_count
        u32  tile_raw_bytes    bytes of one decoded tile (tile_samples^2 * bps)
        u32  bytes_per_sample
        u32  filter            0 = none, 1 = delta_split (u16 only)
    index, right after the header
        (tile_count + 1) x u64 ABSOLUTE file offsets. Tile i's payload is
        bytes [off[i], off[i+1]); off[tile_count] == file length.
    payloads
        lz4.block.compress(filtered, mode="high_compression", compression=9,
        store_size=False), or `filtered` itself when that would not be smaller.
        A payload whose length == tile_raw_bytes is stored, anything shorter
        is lz4.

Filter 1, delta_split, over a square tile of u16 samples a[r][c] in file row
order (north to south): d[0][0] = a[0][0]; d[r][0] = a[r][0] - a[r-1][0] for
r > 0; d[r][c] = a[r][c] - a[r][c-1] for c > 0; all mod 2^16. Then every low
byte of d in sample order, followed by every high byte.
"""

from __future__ import annotations

import math
import os
import struct
from typing import Optional, Union

import numpy as np

FORMAT = "kvtiles_v1"
DENSE_FORMAT = "dense_v1"

MAGIC = b"KVTILE01"
VERSION = 1
HEADER_FMT = "<8sIIIIII"
HEADER_BYTES = struct.calcsize(HEADER_FMT)          # 32

FILTER_NONE = 0
FILTER_DELTA_SPLIT = 1
FILTER_NAMES = {FILTER_NONE: "none", FILTER_DELTA_SPLIT: "delta_split"}

LZ4_MODE = "high_compression"
LZ4_LEVEL = 9

_DTYPE_BY_BYTES = {1: np.dtype("u1"), 2: np.dtype("<u2"), 4: np.dtype("<u4")}


def _lz4():
    try:
        import lz4.block
    except ImportError as e:                     # pragma: no cover - env problem
        raise ImportError("kvtiles_v1 atlases need the lz4 package "
                          "(pip install -r requirements.txt)") from e
    return lz4.block


def _side(tile_raw_bytes: int, bytes_per_sample: int) -> int:
    n = tile_raw_bytes // bytes_per_sample
    s = math.isqrt(n)
    if s * s * bytes_per_sample != tile_raw_bytes:
        raise ValueError(f"tile_raw_bytes {tile_raw_bytes} is not a square tile "
                         f"of {bytes_per_sample}-byte samples")
    return s


# --------------------------------------------------------------------------- #
# Filter 1: delta_split                                                        #
# --------------------------------------------------------------------------- #

def delta_split_encode(raw: bytes, side: int) -> bytes:
    a = np.frombuffer(raw, dtype="<u2").reshape(side, side)
    d = a.copy()
    d[:, 1:] = a[:, 1:] - a[:, :-1]              # uint16 arithmetic wraps mod 2^16
    d[1:, 0] = a[1:, 0] - a[:-1, 0]
    d = d.astype("<u2", copy=False).ravel()
    lo = (d & 0xFF).astype(np.uint8)
    hi = (d >> 8).astype(np.uint8)
    return lo.tobytes() + hi.tobytes()


def delta_split_decode(filtered: bytes, side: int) -> bytes:
    n = side * side
    b = np.frombuffer(filtered, dtype=np.uint8)
    if b.size != 2 * n:
        raise ValueError(f"delta_split payload is {b.size} bytes, expected {2 * n}")
    d = (b[:n].astype(np.uint64) | (b[n:].astype(np.uint64) << 8)).reshape(side, side)
    d[:, 0] = np.cumsum(d[:, 0]) & 0xFFFF
    a = np.cumsum(d, axis=1) & 0xFFFF
    return a.astype("<u2").tobytes()


def _apply_filter(raw: bytes, filt: int, side: int) -> bytes:
    if filt == FILTER_NONE:
        return raw
    if filt == FILTER_DELTA_SPLIT:
        return delta_split_encode(raw, side)
    raise ValueError(f"unknown filter {filt}")


def _undo_filter(filtered: bytes, filt: int, side: int) -> bytes:
    if filt == FILTER_NONE:
        return filtered
    if filt == FILTER_DELTA_SPLIT:
        return delta_split_decode(filtered, side)
    raise ValueError(f"unknown filter {filt}")


# --------------------------------------------------------------------------- #
# Writer                                                                       #
# --------------------------------------------------------------------------- #

class TileWriter:
    """
    Stream tiles into a kvtiles_v1 file, in tile-index order.

    The header and index are reserved up front and written on `close()`, which
    also refuses a file that did not receive exactly `tile_count` tiles. Use as a
    context manager; an exception inside the block closes the file without
    finalising it (and deletes it), so a half-written atlas never looks valid.
    """

    def __init__(self, path: str, tile_count: int, tile_raw_bytes: int,
                 bytes_per_sample: int, filter: int = FILTER_NONE):
        if bytes_per_sample not in _DTYPE_BY_BYTES:
            raise ValueError(f"bytes_per_sample must be 1, 2 or 4, not {bytes_per_sample}")
        if filter == FILTER_DELTA_SPLIT and bytes_per_sample != 2:
            raise ValueError("delta_split is defined for u16 tiles only")
        if filter not in FILTER_NAMES:
            raise ValueError(f"unknown filter {filter}")
        if tile_count < 0 or tile_count >= 2 ** 32:
            raise ValueError(f"tile_count {tile_count} out of range")
        self.path = path
        self.tile_count = int(tile_count)
        self.tile_raw_bytes = int(tile_raw_bytes)
        self.bytes_per_sample = int(bytes_per_sample)
        self.filter = int(filter)
        self.side = _side(self.tile_raw_bytes, self.bytes_per_sample)
        self._block = _lz4()
        self._offsets = []
        self.stored_raw = 0
        self._fh = open(path, "wb")
        self._data_start = HEADER_BYTES + 8 * (self.tile_count + 1)
        self._fh.write(b"\0" * self._data_start)
        self._pos = self._data_start
        self._closed = False

    def write(self, tile: Union[bytes, np.ndarray]) -> None:
        if len(self._offsets) >= self.tile_count:
            raise RuntimeError(f"{self.path}: more than {self.tile_count} tiles written")
        if isinstance(tile, np.ndarray):
            want = _DTYPE_BY_BYTES[self.bytes_per_sample]
            raw = np.ascontiguousarray(tile, dtype=want).tobytes()
        else:
            raw = bytes(tile)
        if len(raw) != self.tile_raw_bytes:
            raise ValueError(f"{self.path}: tile {len(self._offsets)} is {len(raw)} "
                             f"bytes, expected {self.tile_raw_bytes}")
        filtered = _apply_filter(raw, self.filter, self.side)
        payload = self._block.compress(filtered, mode=LZ4_MODE,
                                       compression=LZ4_LEVEL, store_size=False)
        if len(payload) >= self.tile_raw_bytes:
            payload = filtered
            self.stored_raw += 1
        self._offsets.append(self._pos)
        self._fh.write(payload)
        self._pos += len(payload)

    @property
    def tiles_written(self) -> int:
        return len(self._offsets)

    def close(self) -> int:
        """Finalise the file; returns its size in bytes."""
        if self._closed:
            return os.path.getsize(self.path)
        if len(self._offsets) != self.tile_count:
            self.abort()
            raise RuntimeError(f"{self.path}: {len(self._offsets)} tiles written, "
                               f"expected {self.tile_count}")
        index = np.array(self._offsets + [self._pos], dtype="<u8")
        self._fh.seek(0)
        self._fh.write(struct.pack(HEADER_FMT, MAGIC, VERSION, HEADER_BYTES,
                                   self.tile_count, self.tile_raw_bytes,
                                   self.bytes_per_sample, self.filter))
        self._fh.write(index.tobytes())
        self._fh.close()
        self._closed = True
        size = os.path.getsize(self.path)
        if size != self._pos:
            raise RuntimeError(f"{self.path}: file is {size} bytes, index ends at {self._pos}")
        return size

    def abort(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True
            try:
                os.remove(self.path)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False


# --------------------------------------------------------------------------- #
# Reader                                                                       #
# --------------------------------------------------------------------------- #

def is_kvtiles(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(len(MAGIC)) == MAGIC


class AtlasReader:
    """
    Random access to the tiles of an atlas in either layout, decided by magic.

    `tile_raw_bytes` and `bytes_per_sample` are what the manifest says a tile
    is. For dense_v1 they are the only way to find a tile; for kvtiles_v1 they
    are checked against the file header. `tile_count`, when given, is checked
    too (and, for dense_v1, the file size must be exactly that many tiles).
    """

    def __init__(self, path: str, tile_raw_bytes: int, bytes_per_sample: int,
                 tile_count: Optional[int] = None):
        self.path = path
        self.tile_raw_bytes = int(tile_raw_bytes)
        self.bytes_per_sample = int(bytes_per_sample)
        self.dtype = _DTYPE_BY_BYTES[self.bytes_per_sample]
        self.side = _side(self.tile_raw_bytes, self.bytes_per_sample)
        self.file_bytes = os.path.getsize(path)
        self._fh = open(path, "rb")
        try:
            head = self._fh.read(HEADER_BYTES)
            if head[:len(MAGIC)] == MAGIC:
                self._open_kvtiles(head)
            else:
                self._open_dense()
            if tile_count is not None and self.tile_count != int(tile_count):
                raise ValueError(f"{os.path.basename(path)} holds {self.tile_count} "
                                 f"tiles, expected {tile_count}")
        except Exception:
            self._fh.close()
            raise

    def _open_dense(self) -> None:
        self.format = DENSE_FORMAT
        self.filter = FILTER_NONE
        if self.file_bytes % self.tile_raw_bytes:
            raise ValueError(f"{os.path.basename(self.path)} is {self.file_bytes} bytes, "
                             f"not a whole number of {self.tile_raw_bytes}-byte tiles")
        self.tile_count = self.file_bytes // self.tile_raw_bytes
        self.offsets = None

    def _open_kvtiles(self, head: bytes) -> None:
        name = os.path.basename(self.path)
        if len(head) != HEADER_BYTES:
            raise ValueError(f"{name}: short kvtiles header")
        (_, version, header_bytes, tile_count, tile_raw_bytes, bps,
         filt) = struct.unpack(HEADER_FMT, head)
        if version != VERSION:
            raise ValueError(f"{name}: kvtiles version {version}, expected {VERSION}")
        if header_bytes < HEADER_BYTES:
            raise ValueError(f"{name}: header_bytes {header_bytes} < {HEADER_BYTES}")
        if tile_raw_bytes != self.tile_raw_bytes:
            raise ValueError(f"{name}: tile_raw_bytes {tile_raw_bytes}, manifest "
                             f"says {self.tile_raw_bytes}")
        if bps != self.bytes_per_sample:
            raise ValueError(f"{name}: bytes_per_sample {bps}, manifest says "
                             f"{self.bytes_per_sample}")
        if filt not in FILTER_NAMES or (filt == FILTER_DELTA_SPLIT and bps != 2):
            raise ValueError(f"{name}: bad filter {filt}")
        self.format = FORMAT
        self.filter = int(filt)
        self.tile_count = int(tile_count)
        self._fh.seek(header_bytes)
        raw = self._fh.read(8 * (self.tile_count + 1))
        if len(raw) != 8 * (self.tile_count + 1):
            raise ValueError(f"{name}: short tile index")
        off = np.frombuffer(raw, dtype="<u8").astype(np.int64)
        data_start = header_bytes + 8 * (self.tile_count + 1)
        sizes = np.diff(off)
        if off[0] != data_start:
            raise ValueError(f"{name}: first payload at {off[0]}, expected {data_start}")
        if off[-1] != self.file_bytes:
            raise ValueError(f"{name}: index ends at {off[-1]}, file is {self.file_bytes} bytes")
        if (sizes <= 0).any() or (sizes > self.tile_raw_bytes).any():
            raise ValueError(f"{name}: a tile payload is empty or larger than a raw tile")
        self.offsets = off

    def payload_bytes(self, i: int) -> int:
        if self.offsets is None:
            return self.tile_raw_bytes
        return int(self.offsets[i + 1] - self.offsets[i])

    def read_tile_bytes(self, i: int) -> bytes:
        if not 0 <= i < self.tile_count:
            raise IndexError(f"tile {i} out of range 0..{self.tile_count - 1}")
        if self.offsets is None:
            self._fh.seek(i * self.tile_raw_bytes)
            buf = self._fh.read(self.tile_raw_bytes)
            if len(buf) != self.tile_raw_bytes:
                raise ValueError(f"{os.path.basename(self.path)}: short read at tile {i}")
            return buf
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        self._fh.seek(start)
        payload = self._fh.read(end - start)
        if len(payload) == self.tile_raw_bytes:
            filtered = payload
        else:
            filtered = _lz4().decompress(payload, uncompressed_size=self.tile_raw_bytes)
            if len(filtered) != self.tile_raw_bytes:
                raise ValueError(f"{os.path.basename(self.path)}: tile {i} decoded to "
                                 f"{len(filtered)} bytes")
        return _undo_filter(filtered, self.filter, self.side)

    def read_tile(self, i: int) -> np.ndarray:
        """Tile i as a (side, side) array of the atlas's sample type."""
        return np.frombuffer(self.read_tile_bytes(i), dtype=self.dtype).reshape(
            self.side, self.side)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# Converting an existing export                                                #
# --------------------------------------------------------------------------- #

def filter_for(kind: str) -> int:
    """Heights get delta_split; every other atlas is stored unfiltered."""
    return FILTER_DELTA_SPLIT if kind == "height" else FILTER_NONE


def pack_atlas(src: str, dst: str, tile_raw_bytes: int, bytes_per_sample: int,
               tile_count: int, filter: int) -> dict:
    """Rewrite one atlas (either layout) as kvtiles_v1, then decode every tile of
    the new file and require it to be byte-identical to the source tile."""
    with AtlasReader(src, tile_raw_bytes, bytes_per_sample, tile_count) as r:
        with TileWriter(dst, tile_count, tile_raw_bytes, bytes_per_sample, filter) as w:
            for i in range(tile_count):
                w.write(r.read_tile_bytes(i))
            stored_raw = w.stored_raw
        with AtlasReader(dst, tile_raw_bytes, bytes_per_sample, tile_count) as back:
            for i in range(tile_count):
                if back.read_tile_bytes(i) != r.read_tile_bytes(i):
                    raise RuntimeError(f"{os.path.basename(dst)}: tile {i} does not "
                                       f"decode to the source bytes")
    return {"src_bytes": os.path.getsize(src), "dst_bytes": os.path.getsize(dst),
            "tiles": tile_count, "tiles_stored_raw": stored_raw,
            "filter": FILTER_NAMES[filter], "verified_tiles": tile_count}


def pack_export(src_dir: str, dst_dir: str, progress=None) -> dict:
    """
    Copy an export to `dst_dir` with every atlas the manifest names rewritten as
    kvtiles_v1 and verified tile for tile. Every other file (Unity .meta files
    included) is copied unchanged; manifest.json is rewritten to describe the
    new atlases. `src_dir` is only ever read.
    """
    import json
    import shutil
    import time
    from . import exports

    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    if dst_dir == src_dir or dst_dir.startswith(src_dir + os.sep):
        raise ValueError("destination must not be the source or inside it")
    if os.path.exists(dst_dir) and os.listdir(dst_dir):
        raise ValueError(f"destination {dst_dir} exists and is not empty")

    exp = exports.load(src_dir)
    man = json.loads(json.dumps(exp.manifest))
    atlas_files = {}
    for kind in exports.ATLAS_KINDS:
        p = exp.atlas_path(kind)
        if p:
            atlas_files[os.path.basename(p)] = kind

    os.makedirs(dst_dir, exist_ok=True)
    t0 = time.perf_counter()
    copied = []
    for base, dirs, names in os.walk(src_dir):
        rel = os.path.relpath(base, src_dir)
        out_base = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(out_base, exist_ok=True)
        for fn in names:
            if rel == "." and (fn in atlas_files or fn == exports.MANIFEST_NAME):
                continue
            shutil.copy2(os.path.join(base, fn), os.path.join(out_base, fn))
            copied.append(os.path.join(rel, fn) if rel != "." else fn)

    lx, ly = exp.leaf_tiles
    ts = exp.tile_samples
    from . import core
    packed = {}
    for fn, kind in atlas_files.items():
        t = time.perf_counter()
        bps = exp.bytes_per_sample(kind)
        levels = exp.atlas_levels(kind)
        rep = pack_atlas(os.path.join(src_dir, fn), os.path.join(dst_dir, fn),
                         core.atlas_tile_bytes(ts, bps), bps,
                         core.atlas_tile_count(lx, ly, levels), filter_for(kind))
        rep["seconds"] = round(time.perf_counter() - t, 2)
        packed[fn] = rep
        if progress:
            progress(fn, rep)

    describe_atlases(man, {fn: (kind, packed[fn]["dst_bytes"])
                           for fn, kind in atlas_files.items()})
    with open(os.path.join(dst_dir, exports.MANIFEST_NAME), "w") as f:
        json.dump(man, f, indent=2)

    return {"src": src_dir, "dst": dst_dir, "atlases": packed,
            "copied_files": len(copied),
            "seconds": round(time.perf_counter() - t0, 2)}


def describe_atlases(man: dict, files: dict) -> None:
    """Update a manifest in place for kvtiles_v1 atlases. `files` maps each atlas
    file name to (kind, file size in bytes)."""
    atlas = man.setdefault("atlas", {})
    atlas["format"] = FORMAT
    atlas["compression"] = compression_block()
    atlas["file_bytes"] = {fn: int(size) for fn, (_, size) in files.items()}
    atlas["tile_filter"] = {fn: FILTER_NAMES[filter_for(kind)]
                            for fn, (kind, _) in files.items()}
    if "leaf_tiles" in man and "num_levels" in man:
        from . import core
        atlas["tile_count"] = core.atlas_tile_count(*man["leaf_tiles"], man["num_levels"])
    for fn, (kind, size) in files.items():
        if kind == "surface" and isinstance(man.get("water_surface"), dict):
            man["water_surface"]["atlas_format"] = FORMAT
            man["water_surface"]["bytes"] = int(size)
        elif kind == "water_id" and isinstance(man.get("water_id"), dict):
            man["water_id"]["atlas_format"] = FORMAT
            man["water_id"]["bytes"] = int(size)
        elif kind == "labels":
            lab = (man.get("hierarchy") or {}).get("labels")
            if isinstance(lab, dict):
                lab["atlas_format"] = FORMAT
                lab["bytes"] = int(size)
        elif kind == "flowdir" and isinstance(man.get("flow_direction"), dict):
            man["flow_direction"].update(flowdir_layout(int(size)))


def compression_block() -> dict:
    return {
        "container": FORMAT,
        "codec": "lz4_block",
        "lz4_mode": LZ4_MODE,
        "lz4_level": LZ4_LEVEL,
        "header": {"bytes": HEADER_BYTES, "magic": MAGIC.decode("ascii"),
                   "version": VERSION, "layout": HEADER_FMT,
                   "fields": ["magic", "version", "header_bytes", "tile_count",
                              "tile_raw_bytes", "bytes_per_sample", "filter"]},
        "index": "(tile_count + 1) u64le absolute file offsets right after the "
                 "header; tile i is bytes [off[i], off[i+1]); off[tile_count] == "
                 "file length",
        "payload": "lz4 block without size prefix; a payload whose length == "
                   "tile_raw_bytes is stored uncompressed",
        "filters": {"0": "none",
                    "1": "delta_split (u16): d[0][0]=a[0][0]; d[r][0]=a[r][0]-a[r-1][0]; "
                         "d[r][c]=a[r][c]-a[r][c-1]; mod 2^16; then all low bytes of d "
                         "in sample order followed by all high bytes"},
        "tile_index": "unchanged: levels ascending, row-major within a level",
        "bytes_per_sample": "raw bytes per decoded sample",
    }


def flowdir_layout(file_bytes: int) -> dict:
    return {
        "bytes": int(file_bytes),
        "format": FORMAT + "_level0",
        "tile_offset": "tile index = tile_y * leaf_tiles_x + tile_x; the tile's "
                       "byte range comes from the file's kvtiles_v1 offset index",
    }
