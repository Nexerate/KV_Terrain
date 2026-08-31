"""
kvterrain.process
=================

Stage two of two: **take a fetched region and turn it into the export.**

Nothing here touches the network. Every input is either a `kvterrain.dataset` on
disk or a number you chose, and the whole stage is therefore free to re-run: this
is where lake carving, river rasterisation, level estimation, the water surface,
the class/identity raster, the polyline network and the R16 packing happen, and
it is the part you iterate on.

The output format is UNCHANGED and is not ours to change — `manifest.json`,
`heights.atlas`, `surface.atlas`, `water_id.atlas`, `rivers.bin`, `lakes.json`,
`junctions.json`, byte for byte as before the split. The intermediate dataset
format is ours; this one belongs to the engine that consumes it.

Order of operations
-------------------

The sequence below is load-bearing and the comments on each step say why. In
short: repair voids before anything reads the bed, snapshot the uncarved bed
before anything carves it, resolve every lake's level exactly once before
anything reads one, and take the river water level from the ground BEFORE the
trench is cut under it. Steps that look reorderable are not.

This code was `core.run_export`'s second half. It moved here so that the fetch
could stop before it and a dataset could start after it — the sequence itself is
unchanged, and `core.run_export` still exists as the one-shot fetch+process path
that calls straight into `run_process`.

Progress
--------

`progress(fraction, label)`, matching `kvterrain.fetch`. Stage boundaries are
weighted by rough share of wall clock, and every stage's real duration is
reported back on the result as `stage_seconds` — which is the number to look at
when you want to know what is actually costing you a minute.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np

from . import core

Progress = Callable[[float, str], None]

# (key, label, share of the bar). Shares are rough wall-clock weights measured on
# a 20x20 km / 5 m block; they only need to be right enough that the bar does not
# sit at 3% for a minute and then jump.
STAGES = (
    ("rasterise",      "rasterising rivers & lakes",        0.10),
    ("void_repair",    "repairing lake voids",              0.05),
    ("shore_snap",     "snapping shorelines to flat water", 0.06),
    ("lake_levels",    "resolving lake levels",             0.06),
    ("river_surface",  "levelling river channels",          0.12),
    ("carve_rivers",   "carving river beds",                0.10),
    ("carve_lakes",    "carving lake beds",                 0.08),
    ("surface_pyramid", "building the water-surface pyramid", 0.06),
    ("water_id",       "classifying water & lake identity", 0.06),
    ("river_network",  "building the river polyline network", 0.14),
    ("pack",           "packing tiles & writing the atlas", 0.17),
)


class _Stages:
    """Walks STAGES, reporting a fraction and recording how long each one took."""

    def __init__(self, progress: Optional[Progress], skip: frozenset = frozenset()):
        self._progress = progress
        self.seconds: dict = {}
        self._t0 = None
        self._key = None
        active = [s for s in STAGES if s[0] not in skip]
        total = sum(s[2] for s in active) or 1.0
        self._bounds = {}
        acc = 0.0
        for key, label, share in active:
            self._bounds[key] = (acc / total, (acc + share) / total, label)
            acc += share

    def begin(self, key: str) -> None:
        self.end()
        lo, _, label = self._bounds.get(key, (0.0, 0.0, key))
        self._key, self._t0 = key, time.time()
        if self._progress:
            self._progress(lo, label)

    def end(self) -> None:
        if self._key is not None:
            self.seconds[self._key] = round(time.time() - self._t0, 3)
            self._key = None

    def done(self) -> None:
        self.end()
        if self._progress:
            self._progress(1.0, "done")


def run_process(
    plan: core.GridPlan,
    heights: np.ndarray,
    out_dir: str,
    *,
    features=None,
    source_kind: str = "DTM",
    nodata_fill_m: float = 0.0,
    height_min: Optional[float] = None,
    height_max: Optional[float] = None,
    include_water: bool = False,
    water_opts: Optional[dict] = None,
    progress: Optional[Progress] = None,
) -> core.PackResult:
    """
    Build the export in `out_dir` from an already-fetched height lattice.

    `heights` is the raw assembled lattice — float32, NaN for nodata, north-up,
    shaped `(plan.samples_y, plan.samples_x)`. It is COPIED before anything else:
    the carves write in place, and the caller's array is very often a dataset's
    read-only memory map that must survive to be processed again.
    """
    from . import water, bathymetry, watersurface, waterid, rivernet

    if heights.shape != (plan.samples_y, plan.samples_x):
        raise ValueError(
            f"heights {heights.shape} do not match the plan's lattice "
            f"{(plan.samples_y, plan.samples_x)} — wrong tile_cells, or a dataset "
            f"and a plan that were never the same region")

    leaf = np.array(heights, dtype=np.float32, copy=True)

    skip = frozenset() if include_water else frozenset(
        k for k, _, _ in STAGES if k != "pack")
    stages = _Stages(progress, skip=skip)

    surface_levels = None
    water_id_levels = None
    water_leaf = None
    river_net = None
    water_surface = None
    water_params: dict = {}
    lake_count = 0
    carve_depth_m = 0.0
    lake_ramp_m = 0.0
    lake_min_depth_m = 0.0
    lake_slope = 0.0
    snap_report: dict = {}
    emit_geojson = False

    if include_water:
        opts = dict(water_opts or {})
        # `features`/`fetcher` are fetch-stage concerns that older callers passed
        # through this dict. Accept `features` as an override, drop `fetcher`:
        # this stage never goes to the network.
        features = opts.pop("features", None) if features is None else features
        opts.pop("fetcher", None)
        if features is None:
            raise ValueError(
                "include_water=True but no water features were supplied. "
                "run_process does not fetch — pass features= (or process a "
                "dataset that has water).")

        # Lake carve: deepen inward toward a maximum REAL depth over a long ramp,
        # so how deep a lake gets is decided by how big it is.
        carve_depth_m = float(opts.pop("lake_max_depth_m", bathymetry.DEFAULT_CARVE_DEPTH_M))
        lake_slope = float(opts.pop("lake_shore_slope", bathymetry.DEFAULT_SHORE_SLOPE))
        lake_ramp_m = float(opts.pop("lake_ramp_radius_m", 0.0))
        if "lake_bevel_px" in opts:        # legacy alias -> ramp width in metres
            lake_ramp_m = max(1.0, float(opts.pop("lake_bevel_px"))) * plan.spacing_m
        lake_min_depth_m = float(opts.pop("lake_min_depth_m", bathymetry.DEFAULT_MIN_DEPTH_M))
        lake_snap_px = int(opts.pop("lake_snap_px", water.DEFAULT_LAKE_SNAP_PX))
        depth_scale = float(opts.pop("river_depth_scale", 1.0))
        bank_tol_m = float(opts.pop("river_bank_tolerance_m",
                                    bathymetry.DEFAULT_BANK_TOLERANCE_M))
        ocean_level_m = float(opts.pop("ocean_level_m", waterid.DEFAULT_OCEAN_LEVEL_M))
        vertex_stride_m = float(opts.pop("river_vertex_stride_m", plan.spacing_m))
        emit_geojson = bool(opts.pop("emit_geojson", False))
        estimate_lake_levels = bool(opts.pop("estimate_lake_levels", True))
        perimeter_cap = bool(opts.pop("lake_perimeter_cap", True))
        fill_lake_holes = bool(opts.pop("fill_lake_holes", True))

        # Rasterise-only keys. `include_main_rivers` is a FETCH decision and is
        # meaningless here (the hovedelv layer is either in the dataset or it
        # isn't), so it is accepted and dropped rather than treated as an error.
        raster_opts = {}
        for k in ("width_by_order", "width_scale", "all_touched_rivers"):
            if k in opts:
                raster_opts[k] = opts.pop(k)
        opts.pop("include_main_rivers", None)

        stages.begin("rasterise")
        water_leaf = water.rasterize_water(
            plan, features, fill_lake_holes=fill_lake_holes, **raster_opts)
        lake_count = len(water_leaf.lake_table)

        # 1. VOID REPAIR, before anything reads the bed. Kartverket returns a void
        #    lake interior as a finite 0/sentinel rather than NaN, so an isfinite
        #    test misses it entirely. Left in place it becomes the export's
        #    height_min and the shore ring keeps a fake pit right at the waterline.
        #    The carve below hides interior voids as a side effect but cannot reach
        #    the ring outside the polygon, and does nothing for lakes with neither
        #    an NVE hoyde nor a usable shoreline. See bathymetry's module docstring.
        stages.begin("void_repair")
        leaf = bathymetry.fill_lake_surface(leaf, water_leaf.type)

        # 1a2. SNAP each lake to its own flat water surface in the DTM. The NVE
        #      outline and the LiDAR block are independent products, so a one- or
        #      two-texel ring of the lake's own water plane falls outside the
        #      polygon and would render as a raised rim tracing the true shoreline
        #      just outside the water. See snap_lakes_to_flat_water.
        stages.begin("shore_snap")
        snap_report = water.snap_lakes_to_flat_water(
            water_leaf, leaf, max_px=lake_snap_px)

        # 1b. SNAPSHOT the repaired bed before ANY carve. Two consumers need it:
        #     the lake level estimator (which must read the LiDAR water surface,
        #     not a bowl we dug) and the river water level, whose whole model is
        #     "the water sits at the lowest ground the channel has here".
        leaf_uncarved = leaf.copy()

        # 1c. RESOLVE EVERY LAKE'S LEVEL, once, before anything reads one. NVE
        #     `hoyde` VERBATIM where NVE publishes one — it is the authored level
        #     and the DTM does not get a vote on it. Only for a lake without one is
        #     a level read off the (now repaired, still uncarved) DTM inside the
        #     polygon, and that estimate is capped by the ground ringing the lake.
        #     Everything downstream — the carve, the surface raster, the river
        #     tie-in, lakes.json — goes through WaterGrid.lake_level, so they cannot
        #     drift apart.
        stages.begin("lake_levels")
        level_report = water.apply_estimated_levels(
            water_leaf, leaf_uncarved, enabled=estimate_lake_levels,
            island=water_leaf.lake_island, perimeter_cap=perimeter_cap)
        lake_surf = water_leaf.lake_surface_moh()

        # 2. THE RIVER WATER LEVEL, from the uncarved ground: the lowest ground
        #    across each channel, levelled over that channel's own width, then
        #    raised (never lowered) to a lake's surface where the two meet. This
        #    has to come BEFORE the carve, because the carve stops at the waterline
        #    this defines — otherwise the trench is cut wherever the rasterised
        #    buffer went, cliff faces included.
        stages.begin("river_surface")
        river_surf = watersurface.river_surface_moh(
            plan, features, water_leaf, leaf_uncarved,
            lake_surface=lake_surf,
            width_by_order=raster_opts.get("width_by_order"),
            width_scale=raster_opts.get("width_scale", 1.0),
            bank_tolerance_m=bank_tol_m)

        # 2b. Carve the river channels. The water surface sits on the ground, so
        #     this trench is the entire water column — see carve_river_beds.
        stages.begin("carve_rivers")
        leaf = bathymetry.carve_river_beds(
            leaf, water_leaf.type, water_leaf.weight, plan.spacing_m,
            level=river_surf,
            bank_tolerance_m=bank_tol_m,
            depth_scale=depth_scale,
            width_by_order=raster_opts.get("width_by_order"),
            width_scale=raster_opts.get("width_scale", 1.0),
        )

        # 2c. SNAPSHOT the river-carved but LAKE-uncarved bed, for the polylines.
        #     The lake carve writes a fabricated bowl (carve_depth_m under the
        #     authored surface) inside every lake polygon. That bowl is a display
        #     device — it exists so the water plane does not z-fight the
        #     LiDAR-flattened lake surface — and it is NOT terrain. The runtime
        #     burns polyline `z` straight into bedConditioned, so a `z` sampled
        #     from the carved array cuts the solver's routing grid to a fabricated
        #     depth at exactly the points where a channel hands off to a lake
        #     basin. The river trench, by contrast, IS the channel the burn is
        #     trying to open, so `z` is sampled after it. Sampling this array does
        #     not weaken the polyline/raster agreement, because `level` is read
        #     verbatim out of `water_surface` (see build_river_network) rather than
        #     recomputed from `z`.
        leaf_bed_uncarved = leaf.copy()

        # 3. Carve the lake bowls so the water plane does not z-fight the
        #    LiDAR-flattened lake surface. Safe for the solver only because the lake
        #    level is authored and pinned — see carve_lake_beds.
        #
        #    `estimate_missing` is tied to the SAME switch that governs the surface
        #    raster. It must never be true while the raster is empty: the carve's
        #    private shore-estimate fallback is exactly how lakes without an NVE
        #    hoyde ended up as 20 m dry pits. Either both know a level, or neither
        #    touches the lake.
        stages.begin("carve_lakes")
        leaf = bathymetry.carve_lake_beds(
            leaf,
            water_leaf.type,
            plan.spacing_m,
            surface_moh=lake_surf,
            carve_depth_m=carve_depth_m,
            shore_slope=lake_slope,
            ramp_m=lake_ramp_m or None,
            min_depth_m=lake_min_depth_m,
            island=water_leaf.lake_island,
            estimate_missing=False,
        )

        # 4. Unified water surface (lakes = authored level, rivers = the channel
        #    level from step 2), then its water-only pyramid.
        stages.begin("surface_pyramid")
        water_surface = watersurface.combine_water_surface(lake_surf, river_surf)
        surface_levels = watersurface.build_surface_pyramid(water_surface, plan.num_levels)

        # 5. Class + authored lake identity. Ocean is flood-filled inward from the
        #    map edges, not thresholded, so inland sub-sea-level ground (including
        #    the bowls just carved) is never mislabelled sea.
        stages.begin("water_id")
        ocean = waterid.ocean_mask_from_edges(
            leaf, ocean_level_m,
            exclude=(water_leaf.type != water.TYPE_LAND))
        water_id_leaf = waterid.build_water_id(water_leaf, ocean)
        water_id_levels = waterid.build_water_id_pyramid(water_id_leaf, plan.num_levels)

        # 6. The polylines. `z` is sampled from the river-carved, lake-UNCARVED bed
        #    (step 2b) because the runtime burns it; `level` is read verbatim from
        #    `water_surface`, so polyline level still equals the raster surface
        #    pixel-for-pixel.
        stages.begin("river_network")
        river_net = rivernet.build_river_network(
            plan, features, water_leaf, leaf_bed_uncarved,
            water_surface=water_surface,
            vertex_stride_m=vertex_stride_m, depth_scale=depth_scale)

        water_params = {
            "river_width_scale": raster_opts.get("width_scale", 1.0),
            "river_depth_scale": depth_scale,
            "river_bank_tolerance_m": bank_tol_m,
            "river_vertex_stride_m": vertex_stride_m,
            "lake_carve_depth_m": carve_depth_m,
            "lake_shore_slope": lake_slope,
            "lake_shore_ramp_m": lake_ramp_m or (carve_depth_m / max(lake_slope, 1e-6)),
            "lake_min_depth_m": lake_min_depth_m,
            "lake_snap_px": lake_snap_px,
            "lake_holes_filled": fill_lake_holes,
            "ocean_level_m": ocean_level_m,
            "include_main_rivers": bool(getattr(features, "main_rivers", None)),
            "estimate_lake_levels": estimate_lake_levels,
            "lake_perimeter_cap": perimeter_cap,
        }

    stages.begin("pack")
    levels = core.build_pyramid(leaf, plan.num_levels)
    res = core.export_tiles(plan, levels, out_dir,
                            nodata_fill_m=nodata_fill_m,
                            height_min=height_min, height_max=height_max,
                            source_kind=source_kind)
    res.water_grid = water_leaf
    res.river_net = river_net
    res.water_id_leaf = water_id_levels[0] if water_id_levels else None
    res.coarse_level = levels[-1]
    res.terrain_leaf = leaf
    res.water_surface = water_surface

    if surface_levels is not None:
        from . import watersurface, waterid, rivernet

        # The surface atlas packs on the SAME [height_min, height_max] as the height atlas,
        # so it can only be written now that export_tiles has finalised that range.
        wsurf = watersurface.export_surface_tiles(
            plan, surface_levels, out_dir, res.height_min, res.height_max)
        # Name the surface atlas in the header so the runtime opens the second handle.
        res.manifest["atlas"]["surface_file"] = wsurf["atlas_file"]
        wsurf["lake_count"] = lake_count
        wsurf["lake_levels"] = level_report
        wsurf["lake_bathymetry"] = {
            "enabled": True,
            "carve_depth_m": float(carve_depth_m),
            "shore_slope_m_per_m": float(lake_slope),
            "shore_ramp_m": float(lake_ramp_m or (carve_depth_m / max(lake_slope, 1e-6))),
            "min_depth_m": float(lake_min_depth_m),
            "method": "linear_ramp_below_known_surface",
            "ramp_anchor": "waterline",
            "ramp_profile_m": "a STRAIGHT ramp: depth = carve_depth_m * clip((metres "
                              "from the waterline) / shore_ramp_m, 0, 1), flat from "
                              "there inward. shore_ramp_m = carve_depth_m / "
                              "shore_slope_m_per_m, so at slope 1.0 a 20 m bed takes "
                              "20 m of shore. A body too small to reach carve_depth_m "
                              "has its profile scaled up just far enough to reach "
                              "min_depth_m at its deepest sample",
            "shoreline_snap": snap_report,
            "islands": ("lake polygon holes are rasterised as lake so the surface "
                        "runs through them, but their DTM height is left alone and "
                        "they act as shore for the ramp; depth = max(0, level - "
                        "terrain) hides them"),
            "rationale": "LiDAR returns the lake SURFACE as terrain height, so "
                         "without a carve the water plane and the terrain are "
                         "coincident. Safe for the solver: ramped, never breaks "
                         "the rim, and the lake level is authored and pinned.",
        }
        wsurf["river_bathymetry"] = {
            "enabled": True,
            "depth_scale": float(water_params.get("river_depth_scale", 1.0)),
            "method": "trench_under_the_channel_scaled_by_modelled_width",
            "bank_tolerance_m": float(water_params.get("river_bank_tolerance_m", 0.0)),
            "level": "the ground under the channel's own centreline, spread flat "
                     "across the samples that centreline seeded — so the water "
                     "surface is level ACROSS a channel and descends ALONG it",
            "profile": "depth = depth_by_order(order) * smoothstep(distance from "
                       "bank / modelled channel half-width), cut in full while the "
                       "sample is within bank_tolerance_m of the water level and "
                       "tapering to nothing at twice that; a one-sample channel "
                       "takes full depth as a step",
            "rationale": "the river water surface is the uncarved ground under the "
                         "channel, so the trench is the entire water column. Carving "
                         "down rather than raising the surface keeps a river in its "
                         "landscape instead of on top of it; levelling across the "
                         "channel and stopping the carve at the waterline keeps it "
                         "off the cliffs beside it.",
        }
        res.manifest["water_surface"] = wsurf

        wid = waterid.export_water_id_tiles(plan, water_id_levels, out_dir)
        res.manifest["atlas"]["water_id_file"] = wid["atlas_file"]
        res.manifest["water_id"] = wid

        res.manifest["water_vector"] = rivernet.export_river_network(
            river_net, plan, out_dir, emit_geojson=emit_geojson)

        res.manifest["generator"] = rivernet.generator_metadata(plan, {
            "source_kind": source_kind,
            "nodata_fill_m": nodata_fill_m,
            "height_min_m": res.height_min,
            "height_max_m": res.height_max,
            "tile_cells": plan.tile_cells,
            "water": water_params,
        })

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(res.manifest, f, indent=2)
    else:
        from . import rivernet
        res.manifest["generator"] = rivernet.generator_metadata(plan, {
            "source_kind": source_kind,
            "nodata_fill_m": nodata_fill_m,
            "height_min_m": res.height_min,
            "height_max_m": res.height_max,
            "tile_cells": plan.tile_cells,
            "water": None,
        })
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(res.manifest, f, indent=2)

    stages.done()
    res.stage_seconds = stages.seconds
    return res


# --------------------------------------------------------------------------- #
# Dataset entry point                                                          #
# --------------------------------------------------------------------------- #

def process_dataset(
    ds,
    out_dir: str,
    *,
    tile_cells: Optional[int] = None,
    nodata_fill_m: float = 0.0,
    height_min: Optional[float] = None,
    height_max: Optional[float] = None,
    include_water: Optional[bool] = None,
    water_opts: Optional[dict] = None,
    progress: Optional[Progress] = None,
) -> core.PackResult:
    """
    Process a `kvterrain.dataset.Dataset` into a finished export.

    `tile_cells` defaults to the tiling the fetch was planned with, but any value
    in `ds.lattice.valid_tile_cells()` re-tiles the SAME samples — no refetch.
    `include_water` defaults to whatever the dataset actually holds; asking for
    water from a dataset fetched without it is an error rather than a silent
    dry export.
    """
    if include_water is None:
        include_water = ds.has_water
    if include_water and not ds.has_water:
        raise ValueError(
            f"dataset {ds.name!r} was fetched without water, so there are no NVE "
            f"vectors to process. Re-fetch it with water on.")

    plan = ds.plan(tile_cells)

    def stage(frac: float, label: str) -> None:
        if progress:
            progress(frac, label)

    features = None
    if include_water:
        # Parsing is ~seconds against a fetch of ~minutes, and doing it here rather
        # than at fetch time is what lets a field-map fix reach old datasets.
        stage(0.0, "parsing stored NVE vectors")
        features = ds.water_features()

    res = run_process(
        plan, ds.heights(), out_dir,
        features=features,
        source_kind=ds.source_kind,
        nodata_fill_m=nodata_fill_m,
        height_min=height_min, height_max=height_max,
        include_water=include_water,
        water_opts=water_opts,
        progress=progress,
    )

    # Record which dataset this came from, so an export can always be traced back
    # to the fetch it was built from (and reproduced from it).
    res.manifest.setdefault("generator", {})["dataset"] = {
        "name": ds.name,
        "slug": ds.slug,
        "path": ds.root,
        "created_utc": ds.created_utc,
        "format": ds.manifest.get("format"),
        "source": ds.source_kind,
        "demo": ds.is_demo,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(res.manifest, f, indent=2)
    return res
