"""Level-0 lake indexes: lake_cells.bin and lake_samples.bin."""

import struct

import numpy as np

from kvterrain import lakeindex


def _grid():
    B = 16
    w = np.zeros((3, 5), dtype=np.uint16)
    w[0, 1:4] = B + 2          # one run of lake 2
    w[0, 4] = B + 1            # adjacent, other lake -> own run
    w[1, 0] = B + 1            # row wrap must split from row 0's last cell
    w[1, 2] = B + 2
    w[2, 0:5] = B + 2          # full row
    w[2, 2] = 1                # river splits it
    labels = np.zeros((3, 5), dtype=np.uint32)
    labels[0] = [0, 7, 7, 9, 7]
    labels[1] = [7, 0, 0xFFFFFFFF, 0, 0]
    labels[2] = [9, 9, 9, 7, 7]
    return w, labels


def test_runs_split_on_rows_and_codes(tmp_path):
    w, _ = _grid()
    ids, counts, first, length = lakeindex.build_lake_runs(w)
    assert ids.tolist() == [1, 2]
    assert counts.tolist() == [2, 4]
    assert first.tolist() == [4, 5, 1, 7, 10, 13]
    assert length.tolist() == [1, 1, 3, 1, 2, 2]
    path = str(tmp_path / lakeindex.LAKE_CELLS_FILE)
    nbytes = lakeindex.write_lake_cells(path, ids, counts, first, length)
    assert nbytes == 4 + 2 * 8 + 6 * 8
    raw = open(path, "rb").read()
    assert struct.unpack_from("<IiIII", raw) == (2, 1, 2, 4, 1)
    back = lakeindex.read_lake_cells(path)
    assert back[2][0].tolist() == [1, 7, 10, 13]
    for lid, (f, n) in back.items():
        assert int(n.sum()) == int((w == 16 + lid).sum())


def test_samples_skip_root_none_and_unauthored(tmp_path):
    w, labels = _grid()
    rec = lakeindex.build_lake_samples(w, labels, [2])
    assert [tuple(map(int, r)) for r in rec] == [(7, 2, 4), (9, 2, 3)]
    path = str(tmp_path / lakeindex.LAKE_SAMPLES_FILE)
    assert lakeindex.write_lake_samples(path, rec) == 4 + 2 * 12
    assert np.array_equal(lakeindex.read_lake_samples(path), rec)
