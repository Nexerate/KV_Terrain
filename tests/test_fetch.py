import os
import threading

import numpy as np
import pytest

from kvterrain import core, dataset as ds_mod, fetch as kvfetch

SERVER = "https://example.invalid/DTM/ImageServer"


def _plan(cells=64, spacing=10.0):
    return core.GridPlan(epsg=25833, spacing_m=spacing, origin_x=400000.0,
                         origin_y=7150000.0, tile_cells=cells, leaf_tiles_x=4,
                         leaf_tiles_y=3, num_levels=3)


def world(u, e, sx, sy, nx, ny, sp):
    """A fetcher whose heights depend only on world position (the demo one centres
    its hill on each request), returned north-up like the ImageServer."""
    xs = sx + np.arange(nx) * sp
    ys = sy + np.arange(ny) * sp
    X, Y = np.meshgrid(xs, ys)
    return (300 + 0.01 * (X - 400000) + 0.02 * (Y - 7150000)
            + np.sin(X / 70.0) * np.cos(Y / 50.0)).astype(np.float32)[::-1]


def _whole(plan):
    """The region in one request — the reference every chunking must reproduce."""
    return core.assemble_region(plan, world, SERVER, max_fetch_px=100_000)


@pytest.mark.parametrize("n,max_px", [(8193, 2048), (8193, 4096), (513, 4096),
                                      (4097, 1024), (5000, 2048), (2048, 2048)])
def test_chunk_spans_cover_exactly_and_leave_no_sliver(n, max_px):
    spans = core.chunk_spans(n, max_px)
    assert spans[0][0] == 0 and sum(s for _, s in spans) == n
    assert all(a + s == b for (a, s), (b, _) in zip(spans, spans[1:]))
    sizes = [s for _, s in spans]
    assert max(sizes) <= max_px * (1 + core.CHUNK_SLACK_FRACTION)
    assert max(sizes) - min(sizes) <= 1          # balanced: no 1-px requests
    # A 2^k + 1 lattice (what plan_grid produces) is one request per max_px.
    if n == 8193 and max_px == 2048:
        assert len(spans) == 4


def test_parallel_assembly_matches_a_single_request():
    plan = _plan()
    ref = _whole(plan)
    got = core.assemble_region(plan, world, SERVER,
                               max_fetch_px=64, workers=4)
    assert np.array_equal(got, ref)


def test_a_failing_chunk_is_split_and_fetched_in_quarters():
    plan = _plan()

    def flaky(u, e, sx, sy, nx, ny, sp):
        if nx > 40 or ny > 40:                   # the "proxy" refuses big requests
            raise RuntimeError("502 Proxy Error")
        return world(u, e, sx, sy, nx, ny, sp)

    stats = {}
    got = core.assemble_region(plan, flaky, SERVER, max_fetch_px=80, workers=3,
                               min_split_px=16, stats=stats)
    assert np.array_equal(got, _whole(plan))
    assert stats["splits"] > 0 and stats["failed_requests"] == stats["splits"]


def test_an_interrupted_fetch_resumes_from_its_cached_chunks(tmp_path):
    plan = _plan()
    cache = str(tmp_path / "cache")
    calls = []
    lock = threading.Lock()

    def dies_on_the_last_chunk(u, e, sx, sy, nx, ny, sp):
        with lock:
            calls.append((sx, sy))
        if sx > plan.origin_x + 150 * plan.spacing_m and \
                sy > plan.origin_y + 100 * plan.spacing_m:
            raise RuntimeError("connection reset")
        return world(u, e, sx, sy, nx, ny, sp)

    with pytest.raises(RuntimeError, match="resumes"):
        core.assemble_region(plan, dies_on_the_last_chunk, SERVER, max_fetch_px=64,
                             min_split_px=10_000, retry_backoff_s=(0.0,), cache_dir=cache)
    cached = len(os.listdir(cache))
    assert cached == len(core.fetch_chunks(plan, 64)) - 1

    calls.clear()
    stats = {}
    got = core.assemble_region(plan, world, SERVER,
                               max_fetch_px=64, workers=2, cache_dir=cache,
                               stats=stats)
    assert np.array_equal(got, _whole(plan))
    assert stats["from_cache"] == cached and stats["requests"] == 1


def test_a_successful_fetch_deletes_its_resume_cache(tmp_path, monkeypatch):
    plan = _plan()
    monkeypatch.setattr(kvfetch, "live_height_fetcher",
                        lambda kind, session=None: world)
    res = kvfetch.run_fetch(plan, name="t", root=str(tmp_path), server_url=SERVER,
                            include_water=False, make_preview=False, auto_name=False,
                            max_fetch_px=64)
    assert np.array_equal(res.dataset.heights(), _whole(plan))
    cache = kvfetch.cache_dir_for(str(tmp_path), plan, SERVER)
    assert not os.path.exists(cache)
    assert res.dataset.manifest["timing"]["height_requests"]["chunks"] == \
        len(core.fetch_chunks(plan, 64))
    # ...and the cache root does not show up as a dataset.
    assert [d.slug for d in ds_mod.list_datasets(str(tmp_path))] == ["t"]


def test_resume_reuses_the_quarters_of_a_chunk_that_was_split(tmp_path):
    plan = _plan()
    cache = str(tmp_path / "cache")
    big = []

    def refuses_big_then_dies(u, e, sx, sy, nx, ny, sp):
        if nx > 40:
            big.append(1)
            raise RuntimeError("502 Proxy Error")
        if sx > plan.origin_x + 200 * plan.spacing_m:
            raise RuntimeError("connection reset")
        return world(u, e, sx, sy, nx, ny, sp)

    with pytest.raises(RuntimeError):
        core.assemble_region(plan, refuses_big_then_dies, SERVER, max_fetch_px=64,
                             min_split_px=16, retry_backoff_s=(), cache_dir=cache)
    stats = {}
    got = core.assemble_region(plan, world, SERVER, max_fetch_px=64,
                               min_split_px=16, cache_dir=cache, stats=stats)
    assert np.array_equal(got, _whole(plan))
    # Quarters finished before the failure are read back, not refetched as part
    # of their whole chunk.
    assert stats["from_cache"] > len(core.fetch_chunks(plan, 64))
