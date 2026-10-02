"""
kvterrain.process
=================

Stage two of two: **take a fetched region and turn it into the export.**

Nothing here touches the network. Every input is either a `kvterrain.dataset` on
disk or a number you chose, and the whole stage is therefore free to re-run: this
is where lake carving, river rasterisation, level estimation, the water surface,
the class/identity raster, the polyline network and the R16 packing happen, and
it is the part you iterate on.

The export is a contract with the engine that consumes it. The water redesign
deliberately changed its contents and added products; `rivers.bin` is now version
2 with Q, hydraulic width and hierarchy attachment. The river bed is enforced downhill, lakes cut by
the map edge ramp back up to their level there, and a published lake level the
terrain contradicts is replaced, so `heights.atlas` (and what is carved from it)
differs from earlier exports. The new products beside the old ones are:
`labels.atlas`, `flowdir.atlas`, `hierarchy.bin`, `hypsometry.bin`,
`catchments.bin`, `water_audit.json` and `lake_islands.bin`. With
`downhill_river_bed`, `lake_edge_is_shore`, `lake_shore_8_connected` and
`hierarchy` remain independently switchable; the v2 river graph contract is not.

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
    ("river_bed",      "tracing rivers & enforcing a downhill bed", 0.12),
    ("river_surface",  "levelling river channels",          0.12),
    ("carve_rivers",   "carving river beds",                0.10),
    ("carve_lakes",    "carving lake beds",                 0.08),
    ("surface_pyramid", "building the water-surface pyramid", 0.06),
    ("water_id",       "classifying water & lake identity", 0.06),
    ("river_network",  "sampling the river polyline network", 0.04),
    ("hierarchy",      "building the depression hierarchy", 0.04),
    ("water_audit",    "auditing water against the hierarchy", 0.05),
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
        self._order = [s[0] for s in active]
        self._n = 0
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
        self._n = self._order.index(key) + 1 if key in self._order else 0
        if self._progress:
            self._progress(lo, f"[{self._n}/{len(self._order)}] {label}")

    def within(self, frac: float, note: str = "") -> None:
        """Move the bar INSIDE the current stage. A stage that runs for minutes
        with a frozen label is indistinguishable from a hang, which is exactly
        how a slow river-network pass used to read."""
        if not self._progress or self._key is None:
            return
        lo, hi, label = self._bounds.get(self._key, (0.0, 1.0, self._key))
        text = f"[{self._n}/{len(self._order)}] {label}"
        if note:
            text += f" — {note}"
        self._progress(lo + (hi - lo) * min(max(frac, 0.0), 1.0), text)

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
    survey_layers: Optional[dict] = None,
    n50_layer: Optional[dict] = None,
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

    `survey_layers` is the dataset's raw NVE lake-survey layers
    (`Dataset.bathymetry_geojson`), or None when the dataset predates them — the
    lake beds are then synthetic everywhere, and the manifest says so.
    """
    from . import (water, bathymetry, watersurface, waterid, rivernet, riverbed,
                   lakebed, rivermouth, shoreline, riverwidth, thalweg,
                   riverbranch)

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
    lake_bed_model = lakebed.DEFAULT_MODEL
    lake_bed_params = None
    lake_bed_report: dict = {}
    mouth_params = None
    shore_params = None
    shores = None
    width_params = None
    width_report: dict = {}
    snap_params = None
    thalweg_report: dict = {}
    side_channels = False
    river_depth_model = "discharge"
    depth_by_order_compat = None
    snap_report: dict = {}
    emit_geojson = False
    hier = None
    hypso = None
    flow_dirs = None
    catchment_area = None
    catchment_report = None
    codes0 = None
    audit = None
    bed_report: dict = {"enabled": False}
    build_hier = False
    ocean_level_m = 0.0

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
        # The lake bed model. `terrain` (default) continues the surrounding land
        # under the water and uses NVE surveys where they exist; `ramp` is the
        # legacy 1:1 ramp to carve_depth_m above.
        lake_bed_model = str(opts.pop("lake_bed_model", lakebed.DEFAULT_MODEL))
        if lake_bed_model not in lakebed.MODELS:
            raise ValueError(f"lake_bed_model must be one of {lakebed.MODELS}, "
                             f"not {lake_bed_model!r}")
        lbp = lakebed.LakeBedParams(min_depth_m=lake_min_depth_m)
        for key, attr in (("lake_slope_ratio", "slope_ratio"),
                          ("lake_bed_curvature", "curvature"),
                          ("lake_ring_fraction", "ring_fraction"),
                          ("lake_bed_noise", "noise_amplitude"),
                          ("lake_bed_noise_wavelength_m", "noise_wavelength_m"),
                          ("lake_bed_seed", "noise_seed"),
                          ("lake_bed_max_depth_m", "max_depth_m"),
                          ("lake_surveys", "use_surveys")):
            if key in opts:
                val = opts.pop(key)
                if val is not None:
                    setattr(lbp, attr, type(getattr(lbp, attr))(val)
                            if getattr(lbp, attr) is not None else float(val))
        lake_bed_params = lbp
        mouth_params = rivermouth.MouthParams(
            enabled=bool(opts.pop("river_mouths", True)))
        shore_params = shoreline.ShoreParams(
            enabled=bool(opts.pop("lake_shoreline", True)))
        width_params = riverwidth.WidthParams(
            enabled=bool(opts.pop("river_n50_widths", True)))
        snap_params = thalweg.SnapParams(
            enabled=bool(opts.pop("river_thalweg_snap", True)))
        side_channels = bool(opts.pop("river_side_channels", True))
        if survey_layers is None:
            survey_layers = opts.pop("bathymetry", None)
        else:
            opts.pop("bathymetry", None)
        lake_snap_px = int(opts.pop("lake_snap_px", water.DEFAULT_LAKE_SNAP_PX))
        depth_scale = float(opts.pop("river_depth_scale", 1.0))
        # How deep a river channel is: from its modelled discharge (default), or
        # the legacy table by stream order.
        river_depth_model = str(opts.pop("river_depth_model", "discharge"))
        if river_depth_model not in ("discharge", "order"):
            raise ValueError("river_depth_model must be 'discharge' or 'order', "
                             f"not {river_depth_model!r}")
        bank_tol_m = float(opts.pop("river_bank_tolerance_m",
                                    bathymetry.DEFAULT_BANK_TOLERANCE_M))
        ocean_level_m = float(opts.pop("ocean_level_m", waterid.DEFAULT_OCEAN_LEVEL_M))
        vertex_stride_m = float(opts.pop("river_vertex_stride_m", plan.spacing_m))
        runoff_m_per_year = float(opts.pop(
            "river_runoff_m_per_year",
            rivernet.DEFAULT_EFFECTIVE_RUNOFF_M_PER_YEAR))
        emit_geojson = bool(opts.pop("emit_geojson", False))
        estimate_lake_levels = bool(opts.pop("estimate_lake_levels", True))
        perimeter_cap = bool(opts.pop("lake_perimeter_cap", True))
        # None switches the check off; the default is the owner's p90 + 2 m.
        max_above_shore = opts.pop("lake_level_max_above_shore_m",
                                   water.DEFAULT_LEVEL_MAX_ABOVE_SHORE_M)
        max_above_shore = None if max_above_shore is None else float(max_above_shore)
        max_below_scan = opts.pop("lake_level_max_below_scan_m",
                                  water.DEFAULT_LEVEL_MAX_BELOW_SCAN_M)
        max_below_scan = None if max_below_scan is None else float(max_below_scan)
        max_above_scan = opts.pop("lake_level_max_above_scan_m",
                                  water.DEFAULT_LEVEL_MAX_ABOVE_SCAN_M)
        max_above_scan = None if max_above_scan is None else float(max_above_scan)
        twin_scan = opts.pop("lake_level_twin_max_scan_diff_m",
                             water.DEFAULT_LEVEL_TWIN_MAX_SCAN_DIFF_M)
        twin_scan = None if twin_scan is None else float(twin_scan)
        fill_lake_holes = bool(opts.pop("fill_lake_holes", True))
        # The water redesign's first milestone. All three default ON; turning all
        # three off reproduces the pre-redesign export byte for byte.
        downhill_bed = bool(opts.pop("downhill_river_bed", True))
        lake_edge_is_shore = bool(opts.pop("lake_edge_is_shore", True))
        lake_shore_8 = bool(opts.pop("lake_shore_8_connected", True))
        build_hier = bool(opts.pop("hierarchy", True))

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
            island=water_leaf.lake_island, perimeter_cap=perimeter_cap,
            max_above_shore_m=max_above_shore, max_below_scan_m=max_below_scan,
            max_above_scan_m=max_above_scan, twin_max_scan_diff_m=twin_scan)
        lake_surf = water_leaf.lake_surface_moh()

        # 1d. THE SHORELINE: the scan decides where the water ends. Lake samples the
        #     scan puts above the level become land (or islands); only the scanned
        #     water plane gets a modelled bed. Terrain model only, and before the
        #     rivers, so everything downstream sees the lake as it will ship.
        if lake_bed_model == lakebed.MODEL_TERRAIN:
            shores = shoreline.split_shores(water_leaf, leaf_uncarved,
                                            spacing_m=plan.spacing_m,
                                            params=shore_params)
            lake_surf = water_leaf.lake_surface_moh()

        # 2. TRACE the polylines, attach modelled mean-annual discharge to the
        #    authored graph, and re-rasterise ONLY the rivers at w=f(Q). Lakes,
        #    their snapped shores, ids, levels and island mask stay untouched.
        stages.begin("river_bed")
        traced = rivernet.trace_river_network(
            plan, features, water_leaf, leaf_uncarved,
            vertex_stride_m=vertex_stride_m, progress=stages.within)
        hydrology_report = rivernet.assign_discharge_and_width(
            traced, effective_runoff_m_per_year=runoff_m_per_year,
            width_scale=raster_opts.get("width_scale", 1.0),
            depth_from_q=(river_depth_model == "discharge"))
        # 2'. The VISUAL width per vertex: N50's river areas and width classes where
        #     the dataset has them, the modelled width elsewhere (see riverwidth).
        #     The hydraulic width stays on the segment for rivers.bin.
        centres: dict = {}
        vertex_w, width_report = riverwidth.vertex_widths(
            traced.segments, n50_layer, params=width_params, centres=centres)
        for s in traced.segments:
            s.vertex_width_m = vertex_w.get(s.seg_id)
        # 2''. Each stream onto the channel the scan shows: NVE's lines are N50's,
        #      often a valley side away from the bottom (see thalweg). Everything
        #      below — raster, level, profiles, rivers.bin — follows the moved lines.
        thalweg_report = thalweg.snap_segments(
            plan, traced.segments, leaf_uncarved, water_leaf, params=snap_params,
            centres=centres)
        # 2'''. No width stamped around a line may spill past its N50 area.
        areas = riverwidth.water_areas(n50_layer) if width_params.enabled else None
        width_report["fit_to_areas"] = riverwidth.fit_to_areas(
            traced.segments, areas, margin_m=plan.spacing_m)
        # 2iv. The side channels: arms and delta mouths N50 maps that no line runs
        #      through (see riverbranch). Each is a segment of its own from here
        #      on, raster, carve and rivers.bin alike.
        width_report["side_channels"] = (riverbranch.add_side_channels(
            plan, traced, areas, water_leaf,
            width_scale=raster_opts.get("width_scale", 1.0),
            depth_from_q=(river_depth_model == "discharge"))
            if side_channels and areas is not None else {"enabled": False})
        water.rerasterize_rivers(plan, water_leaf, traced.segments,
                                 clip_water=(areas.buffer(plan.spacing_m)
                                             if areas is not None else None))

        # 2a. THE RIVER WATER LEVEL, from the uncarved ground: the lowest ground
        #    across each channel, levelled over that channel's own width, then
        #    raised (never lowered) to a lake's surface where the two meet. This
        #    has to come BEFORE the carve, because the carve stops at the waterline
        #    this defines — otherwise the trench is cut wherever the rasterised
        #    buffer went, cliff faces included.
        stages.begin("river_surface")
        river_level = watersurface.river_surface_moh(
            plan, features, water_leaf, leaf_uncarved,
            width_by_order=raster_opts.get("width_by_order"),
            width_scale=raster_opts.get("width_scale", 1.0),
            bank_tolerance_m=bank_tol_m,
            centrelines=([s.xy for s in traced.segments]
                         if snap_params.enabled or centres else None))
        lake_tie = watersurface.lake_tie_in(water_leaf, lake_surf)
        # The level as it was before any downhill enforcement. The carve's bank
        # taper measures against THIS, so lowering a level cannot switch the
        # carve off under it.
        river_surf_measured = watersurface.apply_lake_tie_in(river_level, lake_tie)

        bed_target = None
        shaped_profiles = None
        river_surf = river_surf_measured

        # 2b. Shape and carve together.  The bank is a bilinear read just outside
        #     the runtime stamp, so its four corners can include river pixels that
        #     this very carve lowers.  One pre-carve ceiling therefore cannot
        #     guarantee the post-carve bank invariant.  Re-evaluate against the
        #     resulting terrain and rebuild from the same uncarved snapshot; a
        #     bounded correction pass closes that feedback without accumulating
        #     carve depth.
        stages.begin("carve_rivers")
        bank_ground = leaf_uncarved
        passes = riverbed.BANK_PROFILE_PASSES if downhill_bed else 1
        for bank_pass in range(passes):
            if downhill_bed:
                low_level, bed_target, bed_report, shaped_profiles = riverbed.downhill_river_bed(
                    plan, traced, water_leaf, leaf_uncarved, river_level,
                    depth_scale=depth_scale,
                    width_by_order=raster_opts.get("width_by_order"),
                    width_scale=raster_opts.get("width_scale", 1.0),
                    bank_ground=bank_ground,
                    starting_profiles=(shaped_profiles if bank_pass else None),
                    soften_gentle_drops=(bank_pass == passes - 1),
                    return_profiles=True)
                river_surf = watersurface.apply_lake_tie_in(low_level, lake_tie)
                del low_level

            leaf = bathymetry.carve_river_beds(
                leaf_uncarved, water_leaf.type, water_leaf.weight, plan.spacing_m,
                level=river_surf_measured,
                bed_target=bed_target,
                surface_target=river_surf if downhill_bed else None,
                bank_tolerance_m=bank_tol_m,
                depth_scale=depth_scale,
                width_by_order=raster_opts.get("width_by_order"),
                width_scale=raster_opts.get("width_scale", 1.0),
                width_m=water_leaf.width_m,
                depth_m=water_leaf.depth_m,
            )

            if downhill_bed:
                # ...and along each polyline's own sample path, which the
                # cross-section carve cannot reach where a stream rasterised to
                # a gappy mask.  Its dry samples also participate in the next
                # pass's bank ceiling.
                bed_report.update(riverbed.burn_downhill_paths(
                    plan, traced, water_leaf, leaf))
                bank_ground = leaf
                bed_report["bank_profile_passes"] = int(passes)
        del river_level, lake_tie
        # Descent measured on the samples the polylines actually cross, with the
        # enforcement on or off, so the two runs can be compared.
        bed_report.update(riverbed.pixel_descent(plan, traced, leaf))

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

        # 3. Carve the lake beds. The water plane must not z-fight the
        #    LiDAR-flattened lake surface, and — since lakes can drain at runtime —
        #    the bed must also pass for ground once exposed: the `terrain` model
        #    continues the land around each lake under it, with NVE surveys where
        #    they exist (see lakebed). Safe for the solver only because the lake
        #    level is authored and pinned, the shore ring stays on the waterline and
        #    no rim is lowered — see carve_lake_beds.
        #
        #    `estimate_missing` is tied to the SAME switch that governs the surface
        #    raster. It must never be true while the raster is empty: the carve's
        #    private shore-estimate fallback is exactly how lakes without an NVE
        #    hoyde ended up as 20 m dry pits. Either both know a level, or neither
        #    touches the lake.
        stages.begin("carve_lakes")
        if lake_bed_model == lakebed.MODEL_RAMP:
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
                edge_is_shore=lake_edge_is_shore,
                shore_8_connected=lake_shore_8,
            )
        else:
            surveys = lakebed.parse_surveys(survey_layers, step_m=plan.spacing_m)
            # The model builds the scanned water plane only, from its anchor; the
            # rest of each lake is scanned ground and keeps its height.
            plane_type = np.where(shores.plane, water.TYPE_LAKE,
                                  water.TYPE_LAND).astype(water_leaf.type.dtype)
            leaf, lake_bed_report = lakebed.build_lake_beds(
                leaf, plane_type, plan.spacing_m,
                surface_moh=shores.anchor,
                ground_m=leaf_uncarved,
                lake_id=water_leaf.lake_id,
                lake_table=water_leaf.lake_table,
                island=water_leaf.lake_island,
                surveys=surveys,
                world_origin=(plan.origin_x, plan.origin_y),
                params=lake_bed_params,
                edge_is_shore=lake_edge_is_shore,
                shore_8_connected=lake_shore_8,
                progress=stages.within,
            )
            lake_bed_report["dataset_has_bathymetry"] = survey_layers is not None
            del plane_type
            # The lake's own ring no longer sits on the waterline; hold it there
            # only where the lake would otherwise drain (see shoreline).
            shores.report["sealed_samples"] = shoreline.seal_shores(
                leaf, water_leaf, lake_surf, edge_is_shore=lake_edge_is_shore)
            lake_bed_report["shoreline"] = shores.report

            # 3b. Carry each river's channel on into its lake: the bed just built
            #     treats a mouth as shore, which left a line on the waterline right
            #     across every channel. Depths are read off the river-carved,
            #     lake-uncarved bed. See rivermouth.
            mouths = rivermouth.find_mouths(
                plan, traced.segments, leaf_bed_uncarved, lake_surf,
                spacing_m=plan.spacing_m, depth_scale=depth_scale)
            raised: list = []
            leaf, lake_bed_report["river_mouths"] = rivermouth.carve_river_mouths(
                leaf, mouths, plan=plan, water_type=water_leaf.type,
                lake_id=water_leaf.lake_id, island=water_leaf.lake_island,
                params=mouth_params, raised_out=raised)
            # An outlet's riffle raises river samples. The polylines' `z` is read
            # from the lake-uncarved snapshot, whose river samples otherwise equal
            # `leaf`'s, so they must see the riffle too. The water going over it is
            # the lake's, at the lake's level: that is the surface the outflow
            # head is read from (exporter and runtime alike).
            riffle_level = {}
            for rr, cc, lvl in raised:
                leaf_bed_uncarved[rr, cc] = leaf[rr, cc]
                cur = river_surf[rr, cc]
                river_surf[rr, cc] = np.where(np.isfinite(cur),
                                              np.maximum(cur, lvl), lvl)
                for i in (rr * plan.samples_x + cc).tolist():
                    riffle_level[i] = max(riffle_level.get(i, -np.inf), lvl)
            # ...and so must the polylines' `level`, which is read from the shaped
            # profiles rather than back from the raster.
            if riffle_level and shaped_profiles:
                for seg in traced.segments:
                    prof = shaped_profiles.get(seg.seg_id)
                    if prof is None or not seg.lake_spans:
                        continue
                    ri, ci = rivernet._nearest_idx(plan, seg.xy)
                    flat = ri * plan.samples_x + ci
                    for k in np.flatnonzero(np.isin(flat, list(riffle_level))):
                        lv = riffle_level[int(flat[k])]
                        if not np.isfinite(prof[2][k]) or prof[2][k] < lv:
                            prof[2][k] = lv

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
        #    (step 2b) because the runtime burns it; open-channel `level` comes
        #    directly from the shaped vertices that were spread into the raster,
        #    avoiding a lossy nearest-pixel readback. Lake spans stay authored.
        stages.begin("river_network")
        river_net = rivernet.build_river_network(
            plan, features, water_leaf, leaf_bed_uncarved,
            water_surface=water_surface,
            shaped_profiles=shaped_profiles, drawn_ground=leaf,
            vertex_stride_m=vertex_stride_m, depth_scale=depth_scale,
            traced=traced, progress=stages.within)
        river_net.report["hydrology"] = hydrology_report
        river_net.report["width_model"] = {
            "formula": "width_m = clamp(a * discharge_m3_s^b, min_m, max_m) * scale",
            "quantity": "full channel width",
            "units": "metres",
            "evaluation": "evaluate in float64 from the stored f32 discharge, then store f32",
            "a": rivernet.DEFAULT_WIDTH_COEFFICIENT,
            "b": rivernet.DEFAULT_WIDTH_EXPONENT,
            "min_m": rivernet.DEFAULT_MIN_WIDTH_M,
            "max_m": rivernet.DEFAULT_MAX_WIDTH_M,
            "scale": raster_opts.get("width_scale", 1.0),
        }
        if river_depth_model == "discharge":
            river_net.report["depth_model"] = {
                "formula": "depth_m = clamp(c * discharge_m3_s^f, min_m, max_m) * scale",
                "quantity": "channel carve depth below the river's water level",
                "units": "metres",
                "c": rivernet.DEFAULT_DEPTH_COEFFICIENT,
                "f": rivernet.DEFAULT_DEPTH_EXPONENT,
                "min_m": rivernet.DEFAULT_MIN_DEPTH_M,
                "max_m": rivernet.DEFAULT_MAX_DEPTH_M,
                "scale": depth_scale,
                "note": "f differs from the runtime's DEPTH_EXPONENT (0.4) on "
                        "purpose: a steeper curve over this tile's discharge range. "
                        "water_surface.river_carve_depth_by_order_m now holds the "
                        "median of this depth per stream order, for readers that "
                        "still look depth up by order.",
            }
            by_order: dict = {}
            for s in traced.segments:
                if s.depth_m:
                    by_order.setdefault(min(max(int(s.order), 1), 8), []).append(s.depth_m)
            depth_by_order_compat = {
                str(o): round(float(np.median(by_order[o])) * depth_scale, 3)
                if o in by_order else round(rivernet.DEFAULT_MIN_DEPTH_M * depth_scale, 3)
                for o in range(1, 9)}
        else:
            depth_by_order_compat = None
        del leaf_uncarved, leaf_bed_uncarved, bed_target, river_surf_measured, shaped_profiles

        # 7. THE DEPRESSION HIERARCHY, on the heights exactly as they will be
        #    packed. Nothing below this point may touch `leaf`: the hierarchy
        #    describes these codes and no others (WATER_REDESIGN.md §3).
        if build_hier:
            from . import flowdir, hierarchy, wateraudit
            stages.begin("hierarchy")
            hmin_p, hmax_p = core.resolve_height_range(leaf, height_min, height_max)
            codes0 = core.pack_leaf_codes(leaf, hmin_p, hmax_p, nodata_fill_m)
            hier = hierarchy.build_hierarchy(
                codes0, hmin_p, hmax_p, spacing_m=plan.spacing_m, ocean=ocean)
            river_net.report["hierarchy_attachment"] = \
                rivernet.attach_hierarchy_nodes(plan, river_net, hier)
            hypso = hierarchy.build_hypsometry(
                hier, codes0, spacing_m=plan.spacing_m)
            flow_dirs = flowdir.build_flow_directions(codes0)
            catchment_area, catchment_report = flowdir.node_catchment_areas(
                flow_dirs, hier.labels, hier.nodes, spacing_m=plan.spacing_m)

            # 8. Match authored lakes to nodes and audit rivers and depressions.
            #    Writes each matched lake into its node, so it runs before packing.
            stages.begin("water_audit")
            # The river surface as exported is what an outflow carries over a
            # lake's spill: its depth there is the lake's outflow head.
            audit = wateraudit.run_audit(plan, hier, codes0, water_leaf, river_net,
                                         river_surface=water_surface)

        water_params = {
            "river_width_scale": raster_opts.get("width_scale", 1.0),
            "river_depth_scale": depth_scale,
            "river_depth_model": river_depth_model,
            "river_bank_tolerance_m": bank_tol_m,
            "river_vertex_stride_m": vertex_stride_m,
            "river_effective_runoff_m_per_year": runoff_m_per_year,
            "lake_carve_depth_m": carve_depth_m,
            "lake_shore_slope": lake_slope,
            "lake_shore_ramp_m": lake_ramp_m or (carve_depth_m / max(lake_slope, 1e-6)),
            "lake_min_depth_m": lake_min_depth_m,
            "lake_bed_model": lake_bed_model,
            "lake_bed": (lake_bed_params.to_dict()
                         if lake_bed_model == lakebed.MODEL_TERRAIN else None),
            "river_mouths": (mouth_params.to_dict()
                             if lake_bed_model == lakebed.MODEL_TERRAIN else None),
            "lake_shoreline": (shore_params.to_dict()
                               if lake_bed_model == lakebed.MODEL_TERRAIN else None),
            "lake_snap_px": lake_snap_px,
            "lake_holes_filled": fill_lake_holes,
            "ocean_level_m": ocean_level_m,
            "include_main_rivers": bool(getattr(features, "main_rivers", None)),
            "estimate_lake_levels": estimate_lake_levels,
            "lake_perimeter_cap": perimeter_cap,
            "lake_level_max_above_shore_m": max_above_shore,
            "downhill_river_bed": downhill_bed,
            "lake_edge_is_shore": lake_edge_is_shore,
            "lake_shore_8_connected": lake_shore_8,
            "hierarchy": build_hier,
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
    res.hierarchy = hier
    res.water_audit = audit

    if hier is not None:
        from . import flowdir, hierarchy, wateraudit
        if (res.height_min, res.height_max) != (hier.height_min, hier.height_max):
            raise RuntimeError("hierarchy was built on a different packing range "
                               "than the atlas was written with")
        # Label pyramid: each coarse level from the one below and THAT level's
        # packed heights, i.e. exactly what heights.atlas holds there.
        # One code grid per level (the last is never gathered from, but keeps
        # the list aligned with the pyramid).
        code_levels = [codes0] + [
            core.pack_leaf_codes(levels[l], res.height_min, res.height_max,
                                 nodata_fill_m)
            for l in range(1, plan.num_levels)]
        label_levels = hierarchy.build_label_pyramid(hier.labels, code_levels)
        del code_levels
        labels_info = hierarchy.export_labels_atlas(plan, label_levels, out_dir)
        del label_levels
        hier_info = hierarchy.write_hierarchy_bin(
            hier, os.path.join(out_dir, hierarchy.HIERARCHY_FILE))
        hypso_info = hierarchy.write_hypsometry_bin(
            hypso, os.path.join(out_dir, hierarchy.HYPSOMETRY_FILE))
        flow_info = flowdir.export_flowdir_atlas(plan, flow_dirs, out_dir)
        catch_info = flowdir.write_catchments_bin(
            catchment_area, os.path.join(out_dir, flowdir.CATCHMENTS_FILE))
        res.manifest["atlas"]["labels_file"] = labels_info["atlas_file"]
        res.manifest["atlas"]["bytes_per_sample"][labels_info["atlas_file"]] = \
            hierarchy.LABELS_BYTES_PER_SAMPLE
        res.manifest["atlas"]["flowdir_file"] = flow_info["file"]
        res.manifest["atlas"]["bytes_per_sample"][flow_info["file"]] = \
            flowdir.FLOWDIR_BYTES_PER_SAMPLE
        res.manifest["hierarchy"] = hierarchy.manifest_block(
            hier, hier_file=hier_info, labels=labels_info,
            minor_depth_m=hierarchy.DEFAULT_MINOR_DEPTH_M,
            minor_area_m2=hierarchy.DEFAULT_MINOR_AREA_M2,
            ocean_level_m=ocean_level_m)
        res.manifest["hierarchy"]["hypsometry"] = hierarchy.hypsometry_manifest(
            hier, hypso, hypso_info, spacing_m=plan.spacing_m)
        res.manifest["hierarchy"]["catchments"] = flowdir.catchments_manifest(
            hier.node_count, catch_info, catchment_report,
            spacing_m=plan.spacing_m)
        res.manifest["flow_direction"] = flow_info
        res.manifest["water_audit"] = wateraudit.write_audit(audit, out_dir)

    if surface_levels is not None:
        from . import watersurface, waterid, rivernet

        # The surface atlas packs on the SAME [height_min, height_max] as the height atlas,
        # so it can only be written now that export_tiles has finalised that range.
        wsurf = watersurface.export_surface_tiles(
            plan, surface_levels, out_dir, res.height_min, res.height_max)
        # Name the surface atlas in the header so the runtime opens the second handle.
        res.manifest["atlas"]["surface_file"] = wsurf["atlas_file"]
        res.manifest["atlas"]["bytes_per_sample"][wsurf["atlas_file"]] = 2
        wsurf["lake_count"] = lake_count
        wsurf["lake_levels"] = level_report
        wsurf["discharge"] = river_net.report.get("hydrology", {})
        wsurf["width_model"] = river_net.report.get("width_model", {})
        if depth_by_order_compat is not None:
            wsurf["river_carve_depth_by_order_m"] = depth_by_order_compat
            wsurf["depth_model"] = river_net.report.get("depth_model", {})
        wsurf["visual_widths"] = {
            **width_report,
            "dataset_has_n50": n50_layer is not None,
            "method": "per-vertex visual width: the N50 river polygon's width across "
                      "the centreline where one exists (median- and mean-smoothed "
                      "along the river, tapered back over taper_m, mask clipped to "
                      "the polygon plus one sample), N50 width class 2 caps the "
                      "hydraulic width at 3 m and class 3 holds it to 3-15 m, "
                      "elsewhere the hydraulic width; written to river_widths.bin",
        }
        wsurf["thalweg_snap"] = {
            **thalweg_report,
            "method": "each stream vertex moved along its normal, at most "
                      "max_offset_m, to where the ground stands least above its "
                      "lower bank; a shortest path per segment with costs for "
                      "offset and sideways change, downstream-first so "
                      "confluences stay joined; lake vertices and rivers wider "
                      "than 2*max_half_width_m are moved to the middle of their "
                      "N50 river area instead. rivers.bin "
                      "carries the moved vertices; geom_hash still identifies "
                      "the mapped run",
        }
        wsurf["lake_bathymetry"] = _lake_bathymetry_block(
            lake_bed_model, lake_bed_report, snap_report, water_params) \
            if lake_bed_model == "terrain" else {
            "enabled": True,
            "model": "ramp",
            "carve_depth_m": float(carve_depth_m),
            "shore_slope_m_per_m": float(lake_slope),
            "shore_ramp_m": float(lake_ramp_m or (carve_depth_m / max(lake_slope, 1e-6))),
            "min_depth_m": float(lake_min_depth_m),
            "edge_is_shore": bool(water_params.get("lake_edge_is_shore", False)),
            "shore_8_connected": bool(water_params.get("lake_shore_8_connected", False)),
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
            "downhill_bed": bed_report,
            "level": "the uncarved centreline ground capped at the lowest dry bank, "
                     "made downstream-only and smoothed by a symmetric box mean; "
                     "accordant at confluences and graded onto authored lake levels, "
                     "then spread flat across the channel",
            "profile": "depth = depth_by_order(order) * smoothstep(distance from "
                       "bank / modelled channel half-width), cut in full while the "
                       "sample is within bank_tolerance_m of the water level and "
                       "tapering to nothing at twice that; a one-sample channel "
                       "takes full depth as a step",
            "rationale": "the bank ceiling prevents a water sheet sloping off a low "
                         "channel edge; symmetric smoothing removes running-minimum "
                         "stairs without fitting or raising; the bed target follows "
                         "every lowering so the shaped surface still fills its trench.",
        }
        res.manifest["water_surface"] = wsurf

        wid = waterid.export_water_id_tiles(plan, water_id_levels, out_dir)
        wid["lake_islands"] = waterid.export_lake_islands(
            water_leaf.lake_island, out_dir)
        res.manifest["atlas"]["water_id_file"] = wid["atlas_file"]
        res.manifest["atlas"]["bytes_per_sample"][wid["atlas_file"]] = 2
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

def _lake_bathymetry_block(model: str, rep: dict, snap_report: dict,
                           water_params: dict) -> dict:
    """The manifest's account of the `terrain` lake-bed model."""
    return {
        "enabled": True,
        "model": model,
        "method": "terrain_continuation_with_nve_surveys",
        "profile": "depth = S * dmax * b * (1 - exp(-k * q / b)), q = distance "
                   "from shore / the body's max distance (centreline rounded), "
                   "S = secant slope of the land ring around the lake blended over "
                   "the water, k = slope_ratio, b = curvature; then depth *= "
                   "(1 + noise_amplitude * n) with world-anchored fractal noise n "
                   "in [-1, 1]",
        "tiers": {
            "surveyed": "NVE depth contours/points: the synthetic shape times a "
                        "smooth multiplier fitted through the survey depths "
                        "(re-referenced from the survey water level to today's)",
            "summary": "NVE max/mean depth only: the synthetic shape scaled to the "
                       "max, with a profile exponent that reproduces the mean",
            "synthetic": "no survey: the shape alone, scaled up to min_depth_m "
                         "if a body is too small to reach it",
        },
        "edge_is_shore": bool(water_params.get("lake_edge_is_shore", False)),
        "shore_8_connected": bool(water_params.get("lake_shore_8_connected", False)),
        "shoreline_snap": snap_report,
        "islands": "island samples keep their DTM height and count as shore",
        "river_mouths_method": "each river's channel continued into its lake along "
                               "its polyline, only ever lowering the lake bed: a "
                               "delta at inflows (the river's depth, capped at its "
                               "order's, fading out over ~12 channel widths), a "
                               "funnel at outlets holding the river's depth until "
                               "deeper lake bed. The shore ring stays on the "
                               "waterline except the throats: an inlet's opens "
                               "unless the flood below the level escapes; an "
                               "outlet's opens onto a riffle, the river's first "
                               "samples raised to a crest above the river bed and "
                               "above the rim of any hollow a real share of the "
                               "lake drains into, with the lake's level as the "
                               "river surface there. The lake's spill is then that "
                               "river sample and its outflow head carries the "
                               "level (see kvterrain/rivermouth.py)",
        "rationale": "lakes can drain at runtime, so the bed must pass for ground: "
                     "it continues the land around it, never breaks the waterline "
                     "(multiplicative noise cannot move a zero or cross it), and "
                     "uses real bathymetry wherever NVE has surveyed the lake",
        **rep,
    }


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

    started = time.time()
    res = run_process(
        plan, ds.heights(), out_dir,
        features=features,
        survey_layers=ds.bathymetry_geojson() if include_water else None,
        n50_layer=ds.n50_rivers() if include_water else None,
        source_kind=ds.source_kind,
        nodata_fill_m=nodata_fill_m,
        height_min=height_min, height_max=height_max,
        include_water=include_water,
        water_opts=water_opts,
        progress=progress,
    )

    # Record which dataset this came from, so an export can always be traced back
    # to the fetch it was built from (and reproduced from it). Slug, not path:
    # `kvterrain process --dataset <slug>` finds it again, while an absolute path
    # would bake one machine's directory layout into a file that ships onward and
    # would go stale the moment the dataset moved.
    res.manifest.setdefault("generator", {})["dataset"] = {
        "name": ds.name,
        "slug": ds.slug,
        "created_utc": ds.created_utc,
        "format": ds.manifest.get("format"),
        "source": ds.source_kind,
        "demo": ds.is_demo,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(res.manifest, f, indent=2)

    # Log the run beside the DATASET, not the export: `process` overwrites the
    # export in place, so the export only ever remembers the last run's settings.
    # The question you actually have two runs later — "was the shoreline better at
    # slope 0.8 or 1.0?" — needs the history, and it has to outlive the export.
    from . import runlog
    runlog.append(ds.root, runlog.summarise_run(
        manifest=res.manifest,
        tile_cells=plan.tile_cells,
        include_water=include_water,
        water_opts=water_opts,
        nodata_fill_m=nodata_fill_m,
        height_min=height_min, height_max=height_max,
        out_dir=out_dir,
        seconds=time.time() - started,
        stage_seconds=res.stage_seconds,
    ))
    return res
