"""
kvterrain command-line interface.

The tool comes apart at the dataset boundary, and so does this CLI:

    kvterrain fetch    --bbox ... --spacing 5          # network -> data/datasets/<name>
    kvterrain datasets                                 # what have I already fetched?
    kvterrain process  --dataset <name> --out out/     # dataset -> the Unity export

`build` is still there and still does both in one pass with nothing stored in
between, for when you genuinely only want the region once. If you expect to
retune a carve, fetch once and process many times — that is the whole point of
the split.
"""
import argparse
import json
import os
import sys


from . import core

# The synthetic surface used to live here AND in app.py, two copies that could
# disagree about what an offline dry run even builds. It is now one function.
from .fetch import synthetic_height_fetcher as _demo_fetcher


def _bar(prefix: str):
    """A (fraction, label) progress printer for the two stage runners."""
    state = {"last": ""}

    def prog(frac: float, label: str) -> None:
        line = f"\r  {prefix} {frac * 100:5.1f}%  {label:<78.78s}"
        state["last"] = line
        print(line, end="", flush=True)
    return prog


def _resolve_dataset(spec: str, root=None):
    """Accept a path to a dataset directory, or the slug/name of one under `root`."""
    from . import dataset as ds_mod

    if os.path.isdir(spec) and os.path.exists(
            os.path.join(spec, ds_mod.MANIFEST_NAME)):
        return ds_mod.load(spec)
    root = root or ds_mod.DEFAULT_ROOT
    cand = os.path.join(root, spec)
    if os.path.exists(os.path.join(cand, ds_mod.MANIFEST_NAME)):
        return ds_mod.load(cand)
    available = ds_mod.list_datasets(root)
    for d in available:
        if spec in (d.slug, d.name):
            return d
    names = ", ".join(d.slug for d in available) or "(none)"
    raise SystemExit(f"no dataset {spec!r} in {root}. Available: {names}")


def _water_opts(a) -> dict:
    """The process-stage water settings, shared by `build` and `process`."""
    opts = {
        "include_main_rivers": not getattr(a, "no_main_rivers", False),
        "width_scale": a.river_width_scale,
        "lake_ramp_radius_m": a.lake_ramp_radius,
        "lake_max_depth_m": a.lake_max_depth,
        "lake_min_depth_m": a.lake_min_depth,
        "lake_shore_slope": a.lake_shore_slope,
        "lake_snap_px": a.lake_snap_px,
        "river_depth_scale": a.river_depth_scale,
        "river_bank_tolerance_m": a.river_bank_tolerance,
        "fill_lake_holes": not a.keep_lake_holes,
        "ocean_level_m": a.ocean_level,
        "emit_geojson": a.emit_geojson,
        "estimate_lake_levels": a.estimate_lake_levels,
        "lake_perimeter_cap": a.lake_perimeter_cap,
        "lake_level_max_above_shore_m": (a.lake_level_max_above_shore
                                         if a.correct_lake_levels else None),
        "downhill_river_bed": a.downhill_river_bed,
        "lake_edge_is_shore": a.lake_edge_is_shore,
        "lake_shore_8_connected": a.lake_shore_8_connected,
        "hierarchy": a.hierarchy,
    }
    if a.river_vertex_stride is not None:
        opts["river_vertex_stride_m"] = a.river_vertex_stride
    return opts


# --------------------------------------------------------------------------- #
# fetch — stage one                                                            #
# --------------------------------------------------------------------------- #

def _ds_mod():
    """`dataset` imported for argparse's choices, without paying for it at import."""
    from . import dataset
    return dataset


def cmd_fetch(a):
    from . import fetch as kvfetch, dataset as ds_mod

    lon_min, lat_min, lon_max, lat_max = a.bbox
    plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
                          spacing_m=a.spacing, tile_cells=a.tile_cells, epsg=a.epsg)
    print(f"CRS EPSG:{plan.epsg}  region {plan.width_m:.0f}x{plan.height_m:.0f} m  "
          f"{plan.samples_x}x{plan.samples_y} samples at {plan.spacing_m:g} m")

    res = kvfetch.run_fetch(
        plan,
        name=a.name,
        root=a.root,
        dest=a.dest,
        source_kind=a.source,
        max_fetch_px=a.max_px,
        include_water=a.water,
        include_main_rivers=not a.no_main_rivers,
        compression=a.compress or ds_mod.DEFAULT_COMPRESSION,
        demo=a.demo,
        progress=_bar("fetch"),
        request_bbox_lonlat=(lon_min, lat_min, lon_max, lat_max),
    )
    d = res.dataset
    s = d.summary()
    print(f"\n\nfetched {s['name']!r} -> {d.root}")
    print(f"  {s['area_km2']:,.1f} km²  ({s['width_m']:,.0f} x {s['height_m']:,.0f} m)  "
          f"{s['spacing_m']:g} m spacing  EPSG:{s['epsg']}")
    print(f"  heights {s['samples'][0]}x{s['samples'][1]} samples, "
          f"{s['coverage_pct']:.2f}% covered, "
          f"range [{s['height_min_m']:.1f}, {s['height_max_m']:.1f}] m")
    if s["has_water"]:
        print(f"  water: {s['rivers']:,} elvenett + {s['main_rivers']:,} hovedelv "
              f"segments, {s['lakes']:,} lake polygons")
    else:
        print("  water: not fetched")
    hb = d.manifest["heights"]
    print(f"  {s['bytes_total'] / 1e6:,.1f} MB on disk  "
          f"({res.seconds_heights:.0f}s heights + {res.seconds_water:.0f}s water)")
    print(f"  lattice stored '{s['compression']}': {hb['bytes'] / 1e6:,.1f} MB "
          f"vs {hb['uncompressed_bytes'] / 1e6:,.1f} MB raw "
          f"({100 * hb['bytes'] / max(hb['uncompressed_bytes'], 1):.0f}%)")
    print(f"  tile_cells at fetch {s['tile_cells']}; this lattice can also be "
          f"processed as {s['valid_tile_cells']}")
    print(f"\nnext:  kvterrain process --dataset {s['slug']} --out out/")


# --------------------------------------------------------------------------- #
# datasets — what is already on disk                                           #
# --------------------------------------------------------------------------- #

def cmd_datasets(a):
    from . import dataset as ds_mod

    root = a.root or ds_mod.DEFAULT_ROOT
    found = ds_mod.list_datasets(root)
    if not found:
        print(f"no datasets in {root}\n\nfetch one:  kvterrain fetch --bbox "
              f"LONMIN LATMIN LONMAX LATMAX --spacing 5")
        return
    print(f"{len(found)} dataset(s) in {root}\n")
    for d in found:
        s = d.summary()
        water = (f"{s['rivers']:,}r/{s['lakes']:,}l" if s["has_water"] else "no water")
        print(f"  {s['slug']:<24} {s['name']:<22} {s['area_km2']:>8,.1f} km²  "
              f"{s['spacing_m']:>5g} m  {s['source']:<4} {water:<14} "
              f"{s['bytes_total'] / 1e6:>8,.1f} MB  "
              f"{s['created_utc'].replace('T', ' ')[:16]}"
              + ("  [demo]" if s["demo"] else ""))
    if a.verbose:
        for d in found:
            print(f"\n=== {d.slug} ===")
            print(json.dumps(d.summary(), indent=2, default=str))


# --------------------------------------------------------------------------- #
# process — stage two                                                          #
# --------------------------------------------------------------------------- #

def cmd_process(a):
    from . import process as kvprocess

    d = _resolve_dataset(a.dataset, a.root)
    s = d.summary()
    want_water = d.has_water and not a.no_water
    tile_cells = a.tile_cells or d.default_tile_cells
    plan = d.plan(tile_cells)
    print(f"dataset {s['name']!r} ({d.root})")
    print(f"  CRS EPSG:{plan.epsg}  region {plan.width_m:.0f}x{plan.height_m:.0f} m  "
          f"leaf {plan.leaf_tiles_x}x{plan.leaf_tiles_y} tiles  "
          f"{plan.num_levels} levels  {plan.total_tiles()} tiles total")
    if not want_water:
        print("  water: OFF" + ("" if d.has_water else " (this dataset has none)"))

    res = kvprocess.process_dataset(
        d, a.out,
        tile_cells=tile_cells,
        nodata_fill_m=a.nodata_fill,
        height_min=a.hmin, height_max=a.hmax,
        include_water=want_water,
        water_opts=_water_opts(a) if want_water else None,
        progress=_bar("process"),
    )
    print()
    _report_export(a, res, want_water)
    if res.stage_seconds:
        slowest = sorted(res.stage_seconds.items(), key=lambda kv: -kv[1])[:4]
        print("\n  stage timings (slowest first): "
              + ", ".join(f"{k} {v:.1f}s" for k, v in slowest))


def cmd_build(a):
    """Fetch and process in one pass, storing nothing in between.

    Kept for one-shot builds. When you expect to process the same region more
    than once, `fetch` then `process` instead — that is what the split is for.
    """
    lon_min, lat_min, lon_max, lat_max = a.bbox
    plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
                          spacing_m=a.spacing, tile_cells=a.tile_cells, epsg=a.epsg)
    print(f"CRS EPSG:{plan.epsg}  region {plan.width_m:.0f}x{plan.height_m:.0f} m  "
          f"leaf {plan.leaf_tiles_x}x{plan.leaf_tiles_y} tiles  "
          f"{plan.num_levels} levels  {plan.total_tiles()} tiles total")

    def prog(d, t, m):
        print(f"\r  {m:<60.60s}", end="", flush=True)

    fetcher = _demo_fetcher if a.demo else None
    server = "(demo)" if a.demo else None

    water_opts = None
    if a.water:
        from . import water as _water
        water_opts = _water_opts(a)
        if a.demo:
            water_opts["fetcher"] = _water.synthetic_water_features

    res = core.run_export(plan, a.out, source_kind=a.source,
                          fetcher=fetcher, server_url=server,
                          max_fetch_px=a.max_px,
                          nodata_fill_m=a.nodata_fill,
                          height_min=a.hmin, height_max=a.hmax,
                          progress=prog,
                          include_water=a.water, water_opts=water_opts)
    print()
    _report_export(a, res, a.water)


def _report_export(a, res, want_water: bool) -> None:
    """The end-of-build summary, shared by `build` and `process`."""
    print(f"packed {res.tiles_written} tiles to {a.out}  "
          f"range [{res.height_min:.1f}, {res.height_max:.1f}] m")
    atlas = res.manifest["atlas"]
    line = (f"  + {atlas['height_file']} "
            f"({atlas['height_bytes'] / 1e6:.1f} MB, {atlas['tile_bytes']}-byte tiles)")
    if atlas.get("surface_file"):
        line += f" + {atlas['surface_file']}"
    print(line)
    if want_water:
        wm = res.manifest.get("water_surface", {})
        print(f"  + water surface: {wm.get('lake_count', 0)} lakes, "
              f"packed in {atlas['surface_file']} (© NVE)")
        ll = wm.get("lake_levels")
        if ll:
            print(f"    levels: {ll['levels_from_nve']} from NVE hoyde (used "
                  f"verbatim), {ll['levels_estimated']} estimated from the LiDAR "
                  f"water surface ({ll.get('levels_capped_to_perimeter', 0)} of "
                  f"those capped at the surrounding terrain, max drop "
                  f"{ll.get('levels_cap_max_drop_m', 0.0):.2f} m), "
                  f"{ll['levels_unresolved']} unresolved (left uncarved)")
            if ll.get("levels_corrected"):
                print(f"    corrected: {ll['levels_corrected']} published levels stood "
                      f"more than {ll['correction']['max_above_shore_m']:g} m above "
                      f"their shore and were replaced by the DTM estimate (max drop "
                      f"{ll['levels_corrected_max_drop_m']:.1f} m)")
        wid = res.manifest.get("water_id", {})
        if wid:
            print(f"  + {wid['atlas_file']} (u16 class + lake id; "
                  f"lake code = {wid['encoding']['lake_id_base']} + lake_id)")
        wv = res.manifest.get("water_vector", {})
        if wv:
            r, lk, jn = wv["rivers"], wv["lakes"], wv["junctions"]
            print(f"  + {r['file']}: {r['segments']} segments, {r['vertices']} "
                  f"vertices ({r['bytes'] / 1e6:.1f} MB), upstream→downstream")
            print(f"  + {lk['file']}: {lk['count']} lakes "
                  f"({lk['with_authored_level']} with an authored level)")
            print(f"  + {jn['file']}: {jn['count']} junctions "
                  f"({jn['inflow']} inflow, {jn['outflow']} outflow)")
            if wv.get("rivers_geojson"):
                print(f"  + {wv['rivers_geojson']['file']} (debug sidecar)")
            _print_water_report(wv["validation"])
        snap = wm.get("lake_bathymetry", {}).get("shoreline_snap")
        if snap:
            print(f"    shoreline snap: {snap['samples_added']} samples across "
                  f"{snap['lakes_grown']} lakes were still the lake's own flat water "
                  f"surface outside its polygon and are now part of it")
        lb = wm.get("lake_bathymetry", {})
        print(f"  + lake beds carved on a straight ramp to {a.lake_max_depth:.1f} m "
              f"over {lb.get('shore_ramp_m', 0.0):.0f} m of shore "
              f"({a.lake_shore_slope:g} m per m, min {a.lake_min_depth:.1f} m)")
        print(f"  + river beds carved under the channel by stream order "
              f"(× {a.river_depth_scale:g}); the surface is level across each "
              f"channel and holds water within {a.river_bank_tolerance:.1f} m of it")


def _print_water_report(rep: dict) -> None:
    """Validation is advisory: these numbers are reported, never used to correct."""
    print("\n  water network validation (advisory; bilinear z, reads the banks too):")
    print(f"    descent      : {rep['descent_rising_vertices']} of "
          f"{rep['descent_vertices_checked']} vertices rise downstream "
          f"({rep['descent_rising_pct']:.2f}%), worst "
          f"{rep['descent_worst_rise_m']:.2f} m, across "
          f"{rep['descent_segments_with_rise']} segments")
    print(f"    flow dir     : {rep['flowdir_disagreements']} of "
          f"{rep['flowdir_segments_checked']} segments disagree with Z "
          f"({rep['flowdir_disagreement_pct']:.2f}%) — "
          f"{rep['flowdir_disagreement_pct_lake_touching']:.2f}% of lake-touching, "
          f"{rep['flowdir_disagreement_pct_non_lake']:.2f}% of the rest")
    print(f"    connectivity : {rep['connectivity_orphan_segments']} orphans, "
          f"{rep['connectivity_source_segments']} sources, "
          f"{rep['connectivity_sink_segments']} sinks")
    if rep["flowdir_segments_checked"] and (
            rep["flowdir_disagreement_pct_lake_touching"] >
            2.0 * max(rep["flowdir_disagreement_pct_non_lake"], 1e-6)):
        print("    NOTE: disagreement is concentrated in lake-touching segments, "
              "which suggests a real ordering issue rather than DTM noise.")


def cmd_validate(a):
    """Spot-check packed leaf heights against Kartverket's open point API. The
    checking is in `exports.check_against_point_api`; this only prints it."""
    from . import exports as kvexports

    rep = kvexports.check_against_point_api(kvexports.load(a.out), n=a.n)
    for s in rep["samples"]:
        print(f"  ({s['x']:.0f},{s['y']:.0f})  packed {s['packed_m']:7.2f}  "
              f"api {s['api_m']:7.2f}  d={s['abs_error_m']:.2f} m")
    for err in rep["errors"]:
        print(f"  {err}")
    if rep["median_abs_error_m"] is not None:
        print(f"\nmedian |error| = {rep['median_abs_error_m']:.2f} m  "
              f"(includes R16 quantisation ~{rep['quantisation_m']:.3f} m)")


def cmd_validate_atlas(a):
    """Verify each dense atlas against the manifest header: exact file size, and
    that a sample of computed tile offsets each land on a full tileBytes block.

    The checking lives in `exports.check_atlas` so the UI runs the same code —
    this function only prints the report it hands back."""
    from . import exports as kvexports

    exp = kvexports.load(a.out)
    rep = kvexports.check_atlas(exp, samples_per_level=a.n)

    for entry in rep["atlases"]:
        if entry.get("error"):
            print(f"[{entry['kind']}] {entry['error']}")
            continue
        print(f"[{entry['kind']}] {entry['file']}: {entry['bytes']} bytes, expected "
              f"{entry['expected_bytes']} ({entry['bytes_per_sample']} bytes/sample) -> "
              f"{'OK' if entry['size_ok'] else 'MISMATCH (grid not dense!)'}")
        if not entry["size_ok"]:
            continue
        for sr in entry["short_reads"]:
            print(f"    L{sr['level']} {sr['x']},{sr['y']}: short read at offset "
                  f"{sr['offset']}")
        print(f"    sampled {entry['offsets_checked']} tile offsets, "
              f"{'all full-length' if not entry['short_reads'] else 'SHORT READS ABOVE'}")

    print("RESULT:", "PASS" if rep["ok"] else "FAIL")
    sys.exit(0 if rep["ok"] else 1)


def cmd_validate_water(a):
    """
    Re-read the exported water products and check them against the manifest:
    rivers.bin parses cleanly and its counts match, every segment link resolves,
    every junction names a lake that exists, and the water_id atlas is dense with
    the right size. Advisory checks (descent, flow direction) are replayed from
    the stored report rather than recomputed — the tool never corrects them, so
    the numbers only need surfacing.

    As with the atlas check, the checking itself is in `exports.check_water`.
    """
    from . import exports as kvexports

    exp = kvexports.load(a.out)
    rep = kvexports.check_water(exp)
    if rep.get("skipped"):
        print(f"no 'water_vector' block in manifest — {rep['skipped']}")
        sys.exit(1)

    r = rep["rivers"]
    if r:
        print(f"[rivers] {r['file']}: v{r['version']} EPSG:{r['epsg']} "
              f"stride {r['stride_m']:g} m, {r['segments']} segments, "
              f"{r['vertices']} vertices")
        if not rep["errors"]:
            print(f"    parsed cleanly; {r['vertices_walked']} vertices walked, "
                  f"all downstream links resolve")

    lk = rep["lakes"]
    if lk:
        print(f"[lakes] {lk['with_ids']} lakes, {lk['estimated_level']} with an "
              f"estimated level, {lk['without_level']} without any level")
    jn = rep["junctions"]
    if jn:
        print(f"[junctions] {jn['count']} junctions, "
              f"{jn['naming_unknown_lake']} naming a lake not in lakes.json")
    wid = rep.get("water_id")
    if wid:
        print(f"[water_id] {wid['bytes']} bytes, expected {wid['expected_bytes']} "
              f"-> {'OK' if wid['ok'] else 'MISMATCH'}")

    for note in rep.get("notes", []):
        print(f"    NOTE: {note}")
    for err in rep["errors"]:
        print(f"    ERROR: {err}")
    if rep.get("validation"):
        _print_water_report(rep["validation"])

    print("\nRESULT:", "PASS" if rep["ok"] else "FAIL")
    sys.exit(0 if rep["ok"] else 1)


def cmd_validate_hierarchy(a):
    """
    Check hierarchy.bin and labels.atlas against the manifest and heights.atlas:
    tree structure, stored codes, saddle placement, label counts, and every
    coarse label level recomputed from the one below. The checking is in
    `exports.check_hierarchy`.
    """
    from . import exports as kvexports

    rep = kvexports.check_hierarchy(kvexports.load(a.out))
    if rep.get("skipped"):
        print(rep["skipped"])
        sys.exit(1)
    st = rep.get("stats") or {}
    if st:
        print(f"[hierarchy] {st['nodes']} nodes, {st['leaves']} leaves, "
              f"{st['minor']} minor, {st['with_lake']} matched to a lake, "
              f"{st['spilling_off_map']} spilling off the map")
    summ = rep.get("audit_summary")
    if summ:
        print(f"[audit] lakes {summ.get('lakes_by_status')}; rivers "
              f"{summ.get('river_findings_by_status')} of "
              f"{summ.get('river_segments_walked')} walked; "
              f"{summ.get('unexplained_depressions')} unexplained depressions")
    for err in rep["errors"]:
        print(f"    ERROR: {err}")
    print("\nRESULT:", "PASS" if rep["ok"] else "FAIL")
    sys.exit(0 if rep["ok"] else 1)


def cmd_describe_services(a):
    """
    Print the live schema of every NVE layer the water pipeline reads, and show
    how each logical field resolves against it.

    This is the tool to reach for when a field stops matching: it distinguishes
    "NVE renamed it" from "our candidate spelling was always wrong" in one call,
    without running an export.
    """
    from . import water as kvwater

    bad = False
    for info in kvwater.describe_water_services():
        print(f"\n=== {info['label']}  {info['url']} ===")
        if info.get("error"):
            print(f"  UNREACHABLE: {info['error']}")
            bad = True
            continue
        print(f"  name={info['name']!r}  geometry={info['geometry_type']}  "
              f"maxRecordCount={info['max_record_count']}")
        if a.fields:
            print("  fields (NAME is what GeoJSON properties use; alias is not):")
            for f in info["fields"]:
                print(f"    {f['name']:<28} {f['type']:<16} alias={f['alias']}")
        print("  logical field resolution:")
        for logical, actual in sorted(info["resolved"].items()):
            if actual:
                print(f"    {logical:<14} -> {actual}")
            elif logical in info["critical"]:
                print(f"    {logical:<14} -> UNMATCHED  *** affects exported geometry ***")
                bad = True
            else:
                note = f"  ({info['note']})" if info.get("note") else ""
                print(f"    {logical:<14} -> absent, expected{note}")

    print("\nRESULT:", "FAIL — see UNMATCHED above" if bad else "PASS")
    sys.exit(1 if bad else 0)


def _add_area_args(p):
    """What defines the lattice — i.e. what forces a new fetch if you change it."""
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"))
    p.add_argument("--spacing", type=float, default=5.0,
                   help="leaf sample spacing in metres — the finest LOD. Don't ask "
                        "for finer than the native data (DTM1 = 1 m).")
    p.add_argument("--tile-cells", type=int, default=128, dest="tile_cells",
                   help="cells per tile edge (samples = cells + 1). Used to pad the "
                        "region out to whole power-of-two tiles; a fetched dataset "
                        "can still be re-tiled to any smaller power of two at "
                        "process time without refetching.")
    p.add_argument("--epsg", type=int, default=None)
    p.add_argument("--source", choices=["DTM", "DOM"], default="DTM")
    p.add_argument("--max-px", type=int, default=core.DEFAULT_MAX_FETCH_PX, dest="max_px",
                   help="largest exportImage request, in pixels per axis")
    p.add_argument("--demo", action="store_true",
                   help="synthetic surface + synthetic rivers/lakes, no network")


def _add_process_args(p):
    """Everything that only reads already-fetched data, and is therefore free to
    re-run. These are the knobs the split exists to let you turn."""
    p.add_argument("--nodata-fill", type=float, default=0.0, dest="nodata_fill")
    p.add_argument("--hmin", type=float, default=None,
                   help="fix the R16 vertical range low end (default: data min)")
    p.add_argument("--hmax", type=float, default=None,
                   help="fix the R16 vertical range high end (default: data max)")
    p.add_argument("--river-width-scale", type=float, default=1.0,
                   dest="river_width_scale")
    p.add_argument("--river-depth-scale", type=float, default=1.0, dest="river_depth_scale",
                   help="scales how deep the channel trench is carved under each "
                        "river (per stream order); >1 = deeper/more opaque rivers. "
                        "The water surface itself always sits on the terrain.")
    p.add_argument("--river-bank-tolerance", type=float, default=2.0,
                   dest="river_bank_tolerance",
                   help="how far (metres) a channel sample may stand above its "
                        "river's water level and still hold water. Bounds how far "
                        "water can climb a bank or a cliff face the rasterised "
                        "channel lapped onto; lower it if you still see water on "
                        "rock, raise it if channels look too narrow.")
    p.add_argument("--keep-lake-holes", action="store_true", dest="keep_lake_holes",
                   help="do NOT fill lake polygon holes. By default islands are "
                        "covered by the water surface (their own terrain hides it) "
                        "because the polygon/raster misalignment otherwise leaves a "
                        "one-texel dry moat around every island.")
    p.add_argument("--lake-shore-slope", type=float, default=1.0, dest="lake_shore_slope",
                   help="metres of lake depth gained per metre of shore, on a "
                        "STRAIGHT ramp. At the default 1.0 a --lake-max-depth of 20 m "
                        "is reached 20 m from shore (four texels at 5 m spacing). "
                        "Lower = gentler sides over a longer ramp.")
    p.add_argument("--lake-ramp-radius", type=float, default=0.0, dest="lake_ramp_radius",
                   help="set the ramp length in metres directly, overriding "
                        "--lake-shore-slope. 0 (default) derives it from the slope.")
    p.add_argument("--lake-max-depth", type=float, default=20.0, dest="lake_max_depth",
                   help="maximum synthetic lake depth in metres, reached only by "
                        "lakes big enough to ramp that far")
    p.add_argument("--lake-min-depth", type=float, default=2.0, dest="lake_min_depth",
                   help="depth every lake reaches at its deepest sample, however "
                        "small it is, so ponds don't render as dry ground")
    p.add_argument("--lake-snap-px", type=int, default=2, dest="lake_snap_px",
                   help="how many texels outside its polygon a lake may claim samples "
                        "that are still its own flat water surface in the DTM. The NVE "
                        "outline and the LiDAR block do not align to the pixel, and the "
                        "leftover ring renders as a raised rim tracing the true "
                        "shoreline. 0 disables.")
    p.add_argument("--ocean-level", type=float, default=0.0, dest="ocean_level",
                   help="sea level in metres above sea level for the Ocean class. "
                        "Kartverket heights are m.o.h., so 0 is real sea level; the "
                        "consuming world's ocean plane height is a runtime concern. "
                        "Ocean is flood-filled from the map edges, not thresholded.")
    p.add_argument("--river-vertex-stride-m", type=float, default=None,
                   dest="river_vertex_stride",
                   help="spacing to densify river polylines to before sampling Z "
                        "(default: leaf spacing). Denser follows the channel more "
                        "closely; coarser shrinks rivers.bin.")
    p.add_argument("--estimate-lake-levels", action=argparse.BooleanOptionalAction,
                   default=True, dest="estimate_lake_levels",
                   help="fill in a water level for lakes NVE gives no 'hoyde' for, "
                        "by reading the LiDAR water surface inside the polygon "
                        "(default: on). With --no-estimate-lake-levels those lakes "
                        "are left uncarved and dry rather than guessed at.")
    p.add_argument("--lake-perimeter-cap", action=argparse.BooleanOptionalAction,
                   default=True, dest="lake_perimeter_cap",
                   help="cap an ESTIMATED lake level at the height of the land ring "
                        "just outside the polygon, so a lake NVE gives no 'hoyde' "
                        "for cannot end up standing above the terrain around it "
                        "(default: on). A published hoyde is never capped.")
    p.add_argument("--correct-lake-levels", action=argparse.BooleanOptionalAction,
                   default=True, dest="correct_lake_levels",
                   help="replace a published NVE lake level that stands more than "
                        "--lake-level-max-above-shore above the 90th percentile of "
                        "the land ringing the lake with the DTM estimate "
                        "(default: on). The raw value stays in lakes.json as "
                        "hoyde_moh.")
    p.add_argument("--lake-level-max-above-shore", type=float, default=2.0,
                   dest="lake_level_max_above_shore",
                   help="metres a published lake level may stand above the 90th "
                        "percentile of its shore ring before it counts as wrong "
                        "(default: 2)")
    p.add_argument("--downhill-river-bed", action=argparse.BooleanOptionalAction,
                   default=True, dest="downhill_river_bed",
                   help="enforce, before carving, that no river bed or water level "
                        "rises downstream: a running minimum over the densified "
                        "polylines, carried across links and through lakes "
                        "(default: on). This changes heights.atlas.")
    p.add_argument("--lake-edge-is-shore", action=argparse.BooleanOptionalAction,
                   default=True, dest="lake_edge_is_shore",
                   help="treat the map edge as shore when carving a lake the "
                        "export boundary cuts through, so the bed ramps back up to "
                        "the waterline there and the lake does not drain off the "
                        "map in the depression hierarchy (default: on)")
    p.add_argument("--lake-shore-8-connected", action=argparse.BooleanOptionalAction,
                   default=True, dest="lake_shore_8_connected",
                   help="put every lake sample with a dry neighbour, diagonal ones "
                        "included, on the waterline, so the 8-connected hierarchy "
                        "cannot drain a lake through a corner of its shore "
                        "(default: on)")
    p.add_argument("--hierarchy", action=argparse.BooleanOptionalAction,
                   default=True, dest="hierarchy",
                   help="build the depression hierarchy and the water audit: "
                        "labels.atlas, hierarchy.bin, water_audit.json "
                        "(default: on)")
    p.add_argument("--emit-geojson", action="store_true", dest="emit_geojson",
                   help="also write rivers.geojson (debug sidecar for QGIS; "
                        "rivers.bin is the runtime format)")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="kvterrain",
        description="Kartverket height (+ NVE water) -> Unity quad-tree terrain. "
                    "Fetch once (`fetch`), process as often as you like "
                    "(`process`), or do both in one pass (`build`).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- stage one ------------------------------------------------------- #
    f = sub.add_parser("fetch", help="fetch a region and store it as a dataset")
    _add_area_args(f)
    f.add_argument("--name", default=None,
                   help="name for the dataset (default: the municipality under the "
                        "region's centre, from Kartverket's kommuneinfo API)")
    f.add_argument("--root", default=None,
                   help=f"where datasets live (default: {'data/datasets'} in the "
                        f"project)")
    f.add_argument("--dest", default=None,
                   help="write to exactly this directory instead of deriving one "
                        "from --name under --root")
    f.add_argument("--water", action=argparse.BooleanOptionalAction, default=True,
                   help="also fetch the NVE rivers/lakes that intersect the region "
                        "(default: on — the vectors are cheap next to the raster, "
                        "and `process --no-water` can always ignore them later)")
    f.add_argument("--no-main-rivers", action="store_true", dest="no_main_rivers",
                   help="skip the hovedelv layer, which is the second NVE query and "
                        "the only thing that upgrades trunk-river size class")
    f.add_argument("--compress", default=None, dest="compress",
                   choices=sorted(_ds_mod().COMPRESSION_MODES),
                   help="how the height lattice is stored (default: lossless — a "
                        "byte shuffle then gzip, ~60%% of raw, exact). 'none' keeps "
                        "a plain .npy; 'mm'/'cm' round to that step first and save "
                        "roughly half, which is still finer than Kartverket's own "
                        "DTM accuracy but is LOSSY.")
    f.set_defaults(func=cmd_fetch)

    dl = sub.add_parser("datasets", help="list the datasets already fetched")
    dl.add_argument("--root", default=None)
    dl.add_argument("--verbose", action="store_true", help="dump each summary as JSON")
    dl.set_defaults(func=cmd_datasets)

    # ---- stage two ------------------------------------------------------- #
    pr = sub.add_parser("process", help="build the export from a fetched dataset")
    pr.add_argument("--dataset", required=True,
                    help="dataset directory, or the slug/name of one under --root")
    pr.add_argument("--root", default=None)
    pr.add_argument("--out", required=True)
    pr.add_argument("--tile-cells", type=int, default=None, dest="tile_cells",
                    help="re-tile the SAME samples (default: whatever the fetch "
                         "used). Any power of two the lattice divides by — see "
                         "`kvterrain datasets --verbose`.")
    pr.add_argument("--no-water", action="store_true", dest="no_water",
                    help="ignore the dataset's water vectors and build height only")
    _add_process_args(pr)
    pr.set_defaults(func=cmd_process)

    # ---- both at once ---------------------------------------------------- #
    b = sub.add_parser("build", help="fetch + process in one pass, storing nothing "
                                     "in between")
    _add_area_args(b)
    b.add_argument("--out", required=True)
    b.add_argument("--water", action="store_true",
                   help="also fetch and process NVE rivers/lakes")
    b.add_argument("--no-main-rivers", action="store_true", dest="no_main_rivers")
    _add_process_args(b)
    b.set_defaults(func=cmd_build)

    # ---- checks ---------------------------------------------------------- #
    v = sub.add_parser("validate", help="spot-check packed leaf heights vs point API")
    v.add_argument("--out", required=True)
    v.add_argument("--n", type=int, default=12)
    v.set_defaults(func=cmd_validate)

    va = sub.add_parser("validate-atlas",
                        help="check dense atlas size + byte-for-byte vs per-tile files")
    va.add_argument("--out", required=True)
    va.add_argument("--n", type=int, default=8,
                    help="tiles sampled per level for the byte-for-byte check")
    va.set_defaults(func=cmd_validate_atlas)

    ds = sub.add_parser("describe-services",
                        help="print the live NVE layer schemas and show how each "
                             "logical field resolves against them")
    ds.add_argument("--fields", action="store_true",
                    help="also list every field with its type and alias")
    ds.set_defaults(func=cmd_describe_services)

    vw = sub.add_parser("validate-water",
                        help="check rivers.bin / lakes.json / junctions.json / "
                             "water_id.atlas against the manifest")
    vw.add_argument("--out", required=True)
    vw.set_defaults(func=cmd_validate_water)

    vh = sub.add_parser("validate-hierarchy",
                        help="check hierarchy.bin / labels.atlas against the "
                             "manifest and heights.atlas")
    vh.add_argument("--out", required=True)
    vh.set_defaults(func=cmd_validate_hierarchy)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main(sys.argv[1:])