"""kvtiles_v1: container layout, filters, stored-raw path, and the dense_v1 reader."""

import os
import struct

import numpy as np
import pytest

from kvterrain import core, tilepack as T

DT = {1: "u1", 2: "<u2", 4: "<u4"}


def _write(path, tiles, bps, filt):
    raw_bytes = tiles[0].nbytes
    with T.TileWriter(str(path), len(tiles), raw_bytes, bps, filt) as w:
        for t in tiles:
            w.write(t)
    return w


def _read_all(path, raw_bytes, bps, count):
    with T.AtlasReader(str(path), raw_bytes, bps, count) as r:
        return [r.read_tile(i).copy() for i in range(count)]


@pytest.mark.parametrize("bps,filt", [(2, T.FILTER_NONE), (2, T.FILTER_DELTA_SPLIT),
                                      (4, T.FILTER_NONE), (1, T.FILTER_NONE)])
def test_random_round_trip(tmp_path, bps, filt):
    rng = np.random.default_rng(bps * 10 + filt)
    side = 33
    hi = {1: 2 ** 8, 2: 2 ** 16, 4: 2 ** 32}[bps]
    tiles = [rng.integers(0, hi, size=(side, side), dtype=np.uint64).astype(DT[bps])
             for _ in range(7)]
    # Smooth tiles too, so the lz4 path is taken as well as the stored path.
    ramp = (np.add.outer(np.arange(side), np.arange(side)) * 3 % hi).astype(DT[bps])
    tiles += [ramp, ramp.copy()]
    p = tmp_path / "a.atlas"
    _write(p, tiles, bps, filt)
    got = _read_all(p, tiles[0].nbytes, bps, len(tiles))
    for a, b in zip(tiles, got):
        assert a.tobytes() == b.tobytes()


def test_header_and_index_layout(tmp_path):
    side = 9
    tiles = [np.full((side, side), v, "<u2") for v in (0, 1, 65535)]
    p = tmp_path / "h.atlas"
    _write(p, tiles, 2, T.FILTER_DELTA_SPLIT)
    data = p.read_bytes()
    magic, ver, hb, count, raw, bps, filt = struct.unpack("<8sIIIIII", data[:32])
    assert (magic, ver, hb, count, raw, bps, filt) == (b"KVTILE01", 1, 32, 3,
                                                       side * side * 2, 2, 1)
    off = np.frombuffer(data[32:32 + 8 * 4], "<u8")
    assert off[0] == 32 + 8 * 4
    assert off[-1] == len(data)
    assert (np.diff(off) > 0).all()


def test_constant_tiles_compress_hard(tmp_path):
    side = 257
    tiles = [np.full((side, side), v, "<u2") for v in (0, 1234, 65535)]
    p = tmp_path / "c.atlas"
    w = _write(p, tiles, 2, T.FILTER_DELTA_SPLIT)
    assert w.stored_raw == 0
    assert os.path.getsize(p) < 3 * side * side * 2 // 50
    got = _read_all(p, side * side * 2, 2, 3)
    for a, b in zip(tiles, got):
        assert np.array_equal(a, b)


def test_incompressible_tiles_are_stored_raw(tmp_path):
    rng = np.random.default_rng(5)
    side = 65
    tiles = [rng.integers(0, 256, size=(side, side), dtype=np.uint8) for _ in range(4)]
    p = tmp_path / "r.atlas"
    w = _write(p, tiles, 1, T.FILTER_NONE)
    assert w.stored_raw == 4
    with T.AtlasReader(str(p), side * side, 1, 4) as r:
        assert all(r.payload_bytes(i) == side * side for i in range(4))
        for i, t in enumerate(tiles):
            assert np.array_equal(r.read_tile(i), t)


def test_stored_raw_with_delta_filter_is_the_filtered_bytes(tmp_path):
    rng = np.random.default_rng(6)
    side = 33
    t = rng.integers(0, 65536, size=(side, side), dtype=np.uint32).astype("<u2")
    p = tmp_path / "s.atlas"
    w = _write(p, [t], 2, T.FILTER_DELTA_SPLIT)
    assert w.stored_raw == 1
    data = p.read_bytes()
    assert data[32 + 16:] == T.delta_split_encode(t.tobytes(), side)
    assert np.array_equal(_read_all(p, t.nbytes, 2, 1)[0], t)


def test_delta_split_definition_and_wraparound():
    a = np.array([[0, 65535, 0],
                  [65535, 0, 1],
                  [1, 65535, 65535]], "<u2")
    enc = T.delta_split_encode(a.tobytes(), 3)
    d = (np.frombuffer(enc[:9], np.uint8).astype(np.uint16)
         | (np.frombuffer(enc[9:], np.uint8).astype(np.uint16) << 8)).reshape(3, 3)
    want = np.zeros((3, 3), np.int64)
    want[0, 0] = a[0, 0]
    for r in range(1, 3):
        want[r, 0] = int(a[r, 0]) - int(a[r - 1, 0])
    for r in range(3):
        for c in range(1, 3):
            want[r, c] = int(a[r, c]) - int(a[r, c - 1])
    assert np.array_equal(d, (want % 65536).astype(np.uint16))
    assert T.delta_split_decode(enc, 3) == a.tobytes()


def test_writer_refuses_a_short_or_long_file(tmp_path):
    t = np.zeros((5, 5), "<u2")
    p = tmp_path / "short.atlas"
    with pytest.raises(RuntimeError):
        with T.TileWriter(str(p), 2, t.nbytes, 2) as w:
            w.write(t)
    assert not p.exists()
    with T.TileWriter(str(p), 1, t.nbytes, 2) as w:
        w.write(t)
        with pytest.raises(RuntimeError):
            w.write(t)
    with pytest.raises(ValueError):
        with T.TileWriter(str(tmp_path / "bad.atlas"), 1, t.nbytes, 2) as w:
            w.write(np.zeros((4, 4), "<u2"))
    with pytest.raises(ValueError):
        T.TileWriter(str(tmp_path / "f.atlas"), 1, 25, 1, T.FILTER_DELTA_SPLIT)


def test_reader_checks_header_against_manifest(tmp_path):
    t = np.zeros((5, 5), "<u2")
    p = tmp_path / "x.atlas"
    _write(p, [t, t], 2, T.FILTER_NONE)
    with pytest.raises(ValueError):
        T.AtlasReader(str(p), t.nbytes, 2, 3)               # wrong tile count
    with pytest.raises(ValueError):
        T.AtlasReader(str(p), 7 * 7 * 2, 2)                 # wrong tile size
    data = bytearray(p.read_bytes())
    p.write_bytes(bytes(data[:-1]))                        # truncated payload
    with pytest.raises(ValueError):
        T.AtlasReader(str(p), t.nbytes, 2)


def test_dense_reader_and_pack_atlas(tmp_path):
    rng = np.random.default_rng(9)
    ts, lx, ly, levels = 17, 4, 2, 3
    count = core.atlas_tile_count(lx, ly, levels)
    assert count == 8 + 2 + 1
    tiles = [(np.cumsum(rng.integers(-3, 4, size=(ts, ts)), axis=1) + 30000)
             .astype("<u2") for _ in range(count)]
    dense = tmp_path / "dense.atlas"
    dense.write_bytes(b"".join(t.tobytes() for t in tiles))
    with T.AtlasReader(str(dense), ts * ts * 2, 2, count) as r:
        assert r.format == "dense_v1"
        for lvl in range(levels):
            tx_n, ty_n = core.tiles_at_level(lx, ly, lvl)
            for y in range(ty_n):
                for x in range(tx_n):
                    i = core.atlas_tile_index(lx, ly, lvl, x, y)
                    off = core.atlas_tile_offset(lx, ly, levels, ts, lvl, x, y)
                    assert off == i * ts * ts * 2
                    assert np.array_equal(r.read_tile(i), tiles[i])
    packed = tmp_path / "packed.atlas"
    rep = T.pack_atlas(str(dense), str(packed), ts * ts * 2, 2, count,
                       T.FILTER_DELTA_SPLIT)
    assert rep["verified_tiles"] == count
    assert rep["dst_bytes"] < rep["src_bytes"]
    assert [t.tobytes() for t in _read_all(packed, ts * ts * 2, 2, count)] == \
        [t.tobytes() for t in tiles]
