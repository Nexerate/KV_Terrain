"""Water identity atlas sidecars."""

import struct

import numpy as np

from kvterrain import waterid


def test_lake_islands_round_trip_is_sorted_row_major(tmp_path):
    island = np.zeros((4, 6), dtype=bool)
    island[3, 0] = True
    island[0, 5] = True
    island[1, 2] = True

    info = waterid.export_lake_islands(island, str(tmp_path))
    path = tmp_path / waterid.LAKE_ISLANDS_FILE
    raw = path.read_bytes()

    assert struct.unpack_from("<I", raw)[0] == 3
    assert np.array_equal(
        waterid.read_lake_islands(str(path)),
        np.array([5, 8, 18], dtype=np.uint32))
    assert info == {
        "file": "lake_islands.bin",
        "bytes": 16,
        "format": "u32_count_then_u32_cell_indices",
        "byte_order": "little_endian",
        "level": 0,
        "count": 3,
        "samples_x": 6,
        "samples_y": 4,
        "cell_index": "row * samples_x + col; row 0 is north",
        "water_id_semantics": "listed cells remain lake-coded in water_id.atlas",
    }
