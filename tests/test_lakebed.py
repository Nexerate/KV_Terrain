import numpy as np
from scipy.ndimage import binary_dilation

from kvterrain import dataset as ds_mod, lakebed

SP = 5.0
LEVEL = 300.0


def _valley(west_slope=0.8, east_slope=0.1, n=240, radius=60, island=True):
    """An elliptical lake in a valley: steep west wall, gentle east shore."""
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = n // 2
    r = np.hypot((xx - cx) / 1.5, yy - cy)
    lake = r < radius
    isl = (np.hypot(xx - cx - 25, yy - cy) < 6) if island else np.zeros_like(lake)
    ground = LEVEL + np.where(xx < cx, west_slope, east_slope) * np.maximum(r - radius, 0) * SP
    ground = np.where(isl, LEVEL + 4.0, ground).astype(np.float32)
    height = np.where(lake & ~isl, LEVEL, ground).astype(np.float32)
    t = np.where(lake, lakebed.TYPE_LAKE_VALUE, 0).astype(np.uint8)
    surf = np.where(lake, LEVEL, np.nan).astype(np.float32)
    return height, t, surf, ground, lake, isl


def _build(height, t, surf, ground, isl, **kw):
    params = kw.pop("params", lakebed.LakeBedParams())
    return lakebed.build_lake_beds(height, t, SP, surface_moh=surf, ground_m=ground,
                                   island=isl, params=params, edge_is_shore=True,
                                   shore_8_connected=True, **kw)


def test_shoreline_stays_on_the_waterline_and_nothing_reaches_the_surface():
    h, t, surf, g, lake, isl = _valley()
    out, rep = _build(h, t, surf, g, isl,
                      params=lakebed.LakeBedParams(noise_amplitude=0.9))
    water = lake & ~isl
    depth = np.where(water, LEVEL - out, np.nan)
    ring = water & binary_dilation(~water, structure=np.ones((3, 3), bool))
    assert np.allclose(depth[ring], 0.0)
    # Every interior sample is under water, however hard the noise pushes.
    assert (depth[water & ~ring] > 0).all()
    assert np.array_equal(out[~water], h[~water])     # land and islands untouched
    assert rep["tiers"]["synthetic"] == 1


def test_steep_surroundings_make_deep_lakes_and_the_deep_leans_to_the_cliff():
    params = lakebed.LakeBedParams(noise_amplitude=0.0)
    steep = _valley(0.8, 0.8, island=False)
    flat = _valley(0.05, 0.05, island=False)
    d_steep = LEVEL - _build(*steep[:4], steep[5], params=params)[0]
    d_flat = LEVEL - _build(*flat[:4], flat[5], params=params)[0]
    assert d_steep[steep[4]].max() > 5 * d_flat[flat[4]].max()

    h, t, surf, g, lake, isl = _valley(0.8, 0.1, island=False)
    d = np.where(lake, LEVEL - _build(h, t, surf, g, isl, params=params)[0], -1)
    _, col = np.unravel_index(np.argmax(d), d.shape)
    assert col < h.shape[1] // 2          # west, the steep side
    # ...and the bed keeps deepening toward the middle: never a flat floor.
    row = d[h.shape[0] // 2]
    inner = row[(row > 0)]
    assert len(np.unique(np.round(inner, 3))) > 0.9 * inner.size


def test_the_bed_rises_to_meet_an_island():
    params = lakebed.LakeBedParams(noise_amplitude=0.0)
    h, t, surf, g, lake, isl = _valley()
    d = LEVEL - _build(h, t, surf, g, isl, params=params)[0]
    c = h.shape[0] // 2
    # Moving from open water toward the island (centred 25 px east of the lake
    # centre) the depth falls to the waterline at its shore.
    assert d[c, c + 12] > d[c, c + 17] > d[c, c + 19] >= 0
    assert np.array_equal(h[isl], _build(h, t, surf, g, isl, params=params)[0][isl])


def test_noise_is_anchored_to_the_world_not_the_window():
    x = np.arange(0, 2000, 7.0)
    a = lakebed.world_noise(x[None, :], np.full((1, x.size), 500.0),
                            wavelength_m=400, octaves=5)
    b = lakebed.world_noise(x[None, 50:], np.full((1, x.size - 50), 500.0),
                            wavelength_m=400, octaves=5)
    assert np.array_equal(a[:, 50:], b)
    assert a.min() >= -1 and a.max() <= 1 and a.std() > 0.1


def _survey_lake():
    h, t, surf, g, lake, isl = _valley(island=False)
    lake_id = np.where(lake, 1, 0).astype(np.uint16)
    table = {1: {"lopenr": 77, "navn": "Testvatnet"}}
    return h, t, surf, g, lake, isl, lake_id, table


def test_surveyed_lake_follows_its_contours():
    h, t, surf, g, lake, isl, lake_id, table = _survey_lake()
    H = h.shape[0]
    c = H // 2
    # A survey measured 1 m BELOW today's level: depths are re-referenced.
    rows = np.array([c, c, c, c - 20, c + 20])
    cols = np.array([c - 60, c - 30, c, c, c])
    true = np.array([6.0, 20.0, 35.0, 12.0, 12.0])
    xy = np.stack([cols * SP, (H - 1 - rows) * SP], axis=1).astype(float)
    sv = lakebed.LakeSurvey(vatnlnr=77, max_depth_m=40, mean_depth_m=15,
                            level_m=LEVEL - 1.0, xy=xy, depth_m=true)
    out, rep = _build(h, t, surf, g, isl, lake_id=lake_id, lake_table=table,
                      surveys={77: sv},
                      params=lakebed.LakeBedParams(noise_amplitude=0.0))
    d = LEVEL - out
    got = d[rows, cols]
    assert np.allclose(got, true + 1.0, rtol=0.2), got
    entry = rep["surveyed_lakes"][0]
    assert entry["tier"] == "surveyed" and entry["level_offset_m"] == 1.0
    assert rep["tiers"]["surveyed"] == 1 and rep["tiers"]["synthetic"] == 0


def test_summary_only_survey_hits_max_and_mean():
    h, t, surf, g, lake, isl, lake_id, table = _survey_lake()
    sv = lakebed.LakeSurvey(vatnlnr=77, max_depth_m=90.0, mean_depth_m=30.0,
                            level_m=LEVEL)
    out, rep = _build(h, t, surf, g, isl, lake_id=lake_id, lake_table=table,
                      surveys={77: sv})
    d = (LEVEL - out)[lake]
    assert abs(d.max() - 90.0) < 1e-3
    assert abs(d.mean() - 30.0) < 0.5
    assert rep["surveyed_lakes"][0]["tier"] == "summary"


def test_surveys_can_be_switched_off():
    h, t, surf, g, lake, isl, lake_id, table = _survey_lake()
    sv = lakebed.LakeSurvey(vatnlnr=77, max_depth_m=90.0, level_m=LEVEL)
    _, rep = _build(h, t, surf, g, isl, lake_id=lake_id, lake_table=table,
                    surveys={77: sv},
                    params=lakebed.LakeBedParams(use_surveys=False))
    assert rep["tiers"] == {"surveyed": 0, "summary": 0, "synthetic": 1}


def test_parse_surveys_joins_layers_by_lake_number():
    raw = {
        "surveys": [{"properties": {"vatnlnr": 5, "innsjonavn": "A", "maksdyp_m": 12,
                                    "middeldyp_m": 4, "oppmaltvannstand_m": 101,
                                    "oppmaltaar": 1980}}],
        "contours": [{"properties": {"vatnlnr": 5, "dybde_m": 2},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[0, 0], [20, 0]]}}],
        "points": [{"properties": {"vatnlnr": 5, "dybde_m": 11.5},
                    "geometry": {"type": "Point", "coordinates": [3, 4]}}],
        "shores": [],
    }
    sv = lakebed.parse_surveys(raw, step_m=5.0)[5]
    assert (sv.name, sv.max_depth_m, sv.mean_depth_m, sv.level_m, sv.year) == \
        ("A", 12.0, 4.0, 101.0, 1980)
    assert sv.depth_m.size == 6            # 5 densified contour vertices + 1 point
    assert sv.depth_m[-1] == 11.5


def test_dataset_stores_bathymetry_and_old_datasets_have_none(tmp_path):
    from kvterrain import core
    plan = core.plan_grid(12.6, 64.42, 12.61, 64.425, spacing_m=10.0, tile_cells=32)
    lat = ds_mod.Lattice.from_plan(plan)
    hts = np.zeros((lat.samples_y, lat.samples_x), np.float32)
    old = ds_mod.write(str(tmp_path / "old"), lat, hts, name="old")
    assert not old.has_bathymetry and old.bathymetry_geojson() is None
    raw = {"surveys": [{"properties": {"vatnlnr": 1}}], "contours": [],
           "points": [], "shores": []}
    new = ds_mod.write_bathymetry(ds_mod.load(old.root), raw)
    again = ds_mod.load(new.root)
    assert again.has_bathymetry
    assert again.bathymetry_geojson()["surveys"] == raw["surveys"]
    assert again.summary()["lake_surveys"] == 1


def test_a_lake_split_across_bodies_is_scaled_once_as_a_whole():
    # One NVE lake cut in two by a dry causeway: two bodies, one survey. Each
    # fragment scaled to the survey max on its own would put 90 m in both.
    h, t, surf, g, lake, isl = _valley(island=False)
    c = h.shape[1] // 2
    causeway = np.zeros_like(lake)
    causeway[:, c + 30:c + 33] = True
    lake2 = lake & ~causeway
    t = np.where(lake2, lakebed.TYPE_LAKE_VALUE, 0).astype(np.uint8)
    surf = np.where(lake2, LEVEL, np.nan).astype(np.float32)
    h = np.where(lake2, LEVEL, np.maximum(g, LEVEL + 1)).astype(np.float32)
    lake_id = np.where(lake2, 1, 0).astype(np.uint16)
    sv = lakebed.LakeSurvey(vatnlnr=77, max_depth_m=90.0, level_m=LEVEL)
    out, rep = _build(h, t, surf, h, isl, lake_id=lake_id,
                      lake_table={1: {"lopenr": 77}}, surveys={77: sv})
    d = LEVEL - out
    west, east = lake2 & (np.arange(h.shape[1]) < c + 30), \
        lake2 & (np.arange(h.shape[1]) > c + 32)
    assert rep["bodies"] == 2 and rep["surveyed_lakes"][0]["bodies"] == 2
    assert abs(max(d[west].max(), d[east].max()) - 90.0) < 1e-3
    assert d[east].max() < 60.0            # the small fragment stays shallower


def test_a_lake_mostly_outside_the_region_is_not_forced_to_the_survey_max():
    h, t, surf, g, lake, isl, lake_id, _ = _survey_lake()
    area = float(lake.sum()) * SP * SP * 4       # only a quarter is in the region
    sv = lakebed.LakeSurvey(vatnlnr=77, max_depth_m=500.0, mean_depth_m=200.0,
                            level_m=LEVEL)
    out, rep = _build(h, t, surf, g, isl, lake_id=lake_id,
                      lake_table={1: {"lopenr": 77, "area_m2": area}},
                      surveys={77: sv})
    e = rep["surveyed_lakes"][0]
    assert e["tier"] == "synthetic" and e["in_region_share"] == 0.25
    assert (LEVEL - out)[lake].max() < 500.0
