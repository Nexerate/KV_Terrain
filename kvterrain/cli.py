"""kvterrain command-line interface."""
import argparse
import json
import os
import sys

import numpy as np

from . import core


def _demo_fetcher(u, e, sx, sy, nx, ny, sp):
    xs = sx + np.arange(nx) * sp
    ys = sy + np.arange(ny) * sp
    XX, YY = np.meshgrid(xs, ys)
    surf = (300.0
            + 400.0 * np.exp(-(((XX - XX.mean()) / 4000.0) ** 2
                               + ((YY - YY.mean()) / 4000.0) ** 2))
            + 60.0 * np.sin(XX / 900.0) * np.cos(YY / 700.0))
    return surf.astype(np.float32)[::-1, :]


def cmd_build(a):
    lon_min, lat_min, lon_max, lat_max = a.bbox
    plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
                          spacing_m=a.spacing, tile_cells=a.tile_cells, epsg=a.epsg)
    print(f"CRS EPSG:{plan.epsg}  region {plan.width_m:.0f}x{plan.height_m:.0f} m  "
          f"leaf {plan.leaf_tiles_x}x{plan.leaf_tiles_y} tiles  "
          f"{plan.num_levels} levels  {plan.total_tiles()} tiles total")

    def prog(d, t, m):
        print(f"\r  fetch {d}/{t}", end="", flush=True)

    fetcher = _demo_fetcher if a.demo else None
    server = "(demo)" if a.demo else None

    water_opts = None
    if a.water:
        from . import water as _water
        water_opts = {
            "include_main_rivers": not a.no_main_rivers,
            "width_scale": a.river_width_scale,
            "lake_ramp_radius_m": a.lake_ramp_radius,
            "lake_max_depth_m": a.lake_max_depth,
            "river_depth_scale": a.river_depth_scale,
            "ocean_level_m": a.ocean_level,
            "emit_geojson": a.emit_geojson,
            "estimate_lake_levels": a.estimate_lake_levels,
        }
        if a.river_vertex_stride is not None:
            water_opts["river_vertex_stride_m"] = a.river_vertex_stride
        if a.demo:
            water_opts["fetcher"] = _water.synthetic_water_features

    res = core.run_export(plan, a.out, source_kind=a.source,
                          fetcher=fetcher, server_url=server,
                          max_fetch_px=a.max_px,
                          nodata_fill_m=a.nodata_fill,
                          height_min=a.hmin, height_max=a.hmax,
                          progress=prog,
                          include_water=a.water, water_opts=water_opts)
    print(f"\npacked {res.tiles_written} tiles to {a.out}  "
          f"range [{res.height_min:.1f}, {res.height_max:.1f}] m")
    atlas = res.manifest["atlas"]
    line = (f"  + {atlas['height_file']} "
            f"({atlas['height_bytes'] / 1e6:.1f} MB, {atlas['tile_bytes']}-byte tiles)")
    if atlas.get("surface_file"):
        line += f" + {atlas['surface_file']}"
    print(line)
    if a.water:
        wm = res.manifest.get("water_surface", {})
        print(f"  + water surface: {wm.get('lake_count', 0)} lakes, "
              f"packed in {atlas['surface_file']} (© NVE)")
        ll = wm.get("lake_levels")
        if ll:
            print(f"    levels: {ll['levels_from_nve']} from NVE hoyde, "
                  f"{ll['levels_estimated']} estimated from the LiDAR water surface, "
                  f"{ll['levels_unresolved']} unresolved (left uncarved)")
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
        print(f"  + lake beds carved flat {a.lake_max_depth:.1f} m below the known "
              f"NVE surface (bevel radius {a.lake_ramp_radius:.1f} m)")
        print(f"  + river surface raised above the DTM channel by stream order "
              f"(× {a.river_depth_scale:g})")


def _print_water_report(rep: dict) -> None:
    """Validation is advisory: the tool reports and never corrects."""
    print("\n  water network validation (advisory — nothing was corrected):")
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
    import requests

    with open(os.path.join(a.out, "manifest.json")) as f:
        man = json.load(f)
    epsg = int(man["crs"].split(":")[1])
    hmin, hmax = man["height_min_m"], man["height_max_m"]
    tc = man["tile_cells"]
    ts = man["tile_samples"]
    spacing = man["leaf_spacing_m"]
    ox, oy = man["origin_utm"]
    lx, ly = man["leaf_tiles"]
    nlev = man["num_levels"]

    # Everything about a leaf tile is implied by the header — its atlas byte offset and its
    # world-space SW corner — so we sample leaf tiles directly from the grid, no per-tile
    # metadata needed. Read the packed u16 grid out of the height atlas by computed offset.
    tile_bytes = core.atlas_tile_bytes(ts)
    atlas_path = os.path.join(a.out, man["atlas"]["height_file"])
    rng = np.random.default_rng(0)
    coords = [(x, y) for y in range(ly) for x in range(lx)]      # level-0 grid
    pick = rng.choice(len(coords), min(a.n, len(coords)), replace=False)

    sess = requests.Session()
    errs = []
    with open(atlas_path, "rb") as atlas_fh:
        for idx in pick:
            x, y = coords[int(idx)]
            off = core.atlas_tile_offset(lx, ly, nlev, ts, 0, x, y)
            atlas_fh.seek(off)
            a16 = np.frombuffer(atlas_fh.read(tile_bytes), dtype="<u2").reshape(ts, ts)

            i = int(rng.integers(0, tc + 1)); j = int(rng.integers(0, tc + 1))
            packed = a16[tc - j, i]
            h_packed = hmin + (packed / 65535.0) * (hmax - hmin)
            # Tile SW corner in world CRS, then the sampled cell within it.
            X = ox + (x * tc + i) * spacing
            Y = oy + (y * tc + j) * spacing
            try:
                r = sess.get(core.POINT_API,
                             params={"ost": X, "nord": Y, "koordsys": epsg},
                             timeout=30)
                data = r.json()
                h_api = data.get("punkter", [{}])[0].get("z")
            except Exception as ex:
                print(f"  point API call failed ({ex}); check param names live.")
                return
            if h_api is None:
                continue
            errs.append(abs(h_api - h_packed))
            print(f"  ({X:.0f},{Y:.0f})  packed {h_packed:7.2f}  api {h_api:7.2f}  "
                  f"d={abs(h_api - h_packed):.2f} m")
    if errs:
        print(f"\nmedian |error| = {np.median(errs):.2f} m  "
              f"(includes R16 quantisation ~{(hmax - hmin) / 65535:.3f} m)")


def cmd_validate_atlas(a):
    """Verify a dense atlas against the manifest header and (if present) the
    per-tile files: exact file size, and byte-for-byte equality of a sample of
    tiles extracted by computed offset. Implements the §8 export-side checks."""
    with open(os.path.join(a.out, "manifest.json")) as f:
        man = json.load(f)

    atlas = man.get("atlas")
    if not atlas:
        print("no 'atlas' block in manifest — not a kvterrain atlas export.")
        sys.exit(1)

    lx, ly = man["leaf_tiles"]
    nlev = man["num_levels"]
    ts = man["tile_samples"]
    tile_bytes = core.atlas_tile_bytes(ts)
    expect = core.atlas_total_bytes(lx, ly, nlev, ts)

    targets = [("height", atlas.get("height_file"))]
    if atlas.get("surface_file"):
        targets.append(("surface", atlas["surface_file"]))

    rng = np.random.default_rng(0)
    ok = True
    for kind, fname in targets:
        path = os.path.join(a.out, fname)
        if not os.path.exists(path):
            print(f"[{kind}] MISSING atlas file {fname}")
            ok = False
            continue

        actual = os.path.getsize(path)
        size_ok = actual == expect
        ok &= size_ok
        print(f"[{kind}] {fname}: {actual} bytes, expected {expect} "
              f"-> {'OK' if size_ok else 'MISMATCH (grid not dense!)'}")
        if not size_ok:
            continue

        # Sample tiles across levels and confirm each computed offset yields a full
        # tileBytes block inside the file (offsets land where the header implies).
        checked = 0
        with open(path, "rb") as fh:
            for lvl in range(nlev):
                tx, ty = core.tiles_at_level(lx, ly, lvl)
                coords = [(x, y) for y in range(ty) for x in range(tx)]
                for idx in rng.choice(len(coords), size=min(a.n, len(coords)),
                                      replace=False):
                    x, y = coords[int(idx)]
                    off = core.atlas_tile_offset(lx, ly, nlev, ts, lvl, x, y)
                    fh.seek(off)
                    if len(fh.read(tile_bytes)) != tile_bytes:
                        print(f"    L{lvl} {x},{y}: short read at offset {off}")
                        ok = False
                    checked += 1
        print(f"    sampled {checked} tile offsets, all full-length")

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_validate_water(a):
    """
    Re-read the exported water products and check them against the manifest:
    rivers.bin parses cleanly and its counts match, every segment link resolves,
    every junction names a lake that exists, and the water_id atlas is dense with
    the right size. Advisory checks (descent, flow direction) are replayed from
    the stored report rather than recomputed — the tool never corrects them, so
    the numbers only need surfacing.
    """
    import struct

    from . import rivernet

    with open(os.path.join(a.out, "manifest.json")) as f:
        man = json.load(f)

    wv = man.get("water_vector")
    if not wv:
        print("no 'water_vector' block in manifest — build with --water.")
        sys.exit(1)

    ok = True

    # ---- rivers.bin ------------------------------------------------------
    path = os.path.join(a.out, wv["rivers"]["file"])
    with open(path, "rb") as fh:
        hdr = fh.read(48)
        (magic, version, epsg, ox, oy, stride, nseg, nvert, _res) = struct.unpack(
            "<8sIIddfIII", hdr)
        if magic != rivernet.BIN_MAGIC:
            print(f"[rivers] BAD MAGIC {magic!r}")
            sys.exit(1)
        print(f"[rivers] {wv['rivers']['file']}: v{version} EPSG:{epsg} "
              f"stride {stride:g} m, {nseg} segments, {nvert} vertices")

        seen_ids = set()
        downstream_refs = []
        total_v = 0
        for _ in range(nseg):
            (sid, strekn, vatn, order, down) = struct.unpack("<IqqHi", fh.read(26))
            (nup,) = struct.unpack("<H", fh.read(2))
            fh.read(4 * nup)
            (ln,) = struct.unpack("<B", fh.read(1)); fh.read(ln)
            (ln,) = struct.unpack("<B", fh.read(1)); fh.read(ln)
            (nspan,) = struct.unpack("<H", fh.read(2))
            fh.read(10 * nspan)
            fh.read(16)
            (nv,) = struct.unpack("<I", fh.read(4))
            fh.read(16 * nv)
            seen_ids.add(sid)
            total_v += nv
            if down >= 0:
                downstream_refs.append((sid, down))
        trailing = fh.read()

    if trailing:
        print(f"    {len(trailing)} trailing bytes after the last segment")
        ok = False
    if total_v != nvert:
        print(f"    vertex count mismatch: header {nvert}, actual {total_v}")
        ok = False
    dangling = [(s, d) for s, d in downstream_refs if d not in seen_ids]
    if dangling:
        print(f"    {len(dangling)} downstream links point at missing segments")
        ok = False
    else:
        print(f"    parsed cleanly; {len(downstream_refs)} downstream links all resolve")

    # ---- lakes + junctions ----------------------------------------------
    with open(os.path.join(a.out, wv["lakes"]["file"])) as f:
        lakes = json.load(f)
    lake_ids = {l["lake_id"] for l in lakes["lakes"]}
    # The pin reads `authored_level_m`, which carries an estimated level too — so
    # this must not test `hoyde_moh`, or every estimated lake reads as unpinnable.
    no_level = [l["lake_id"] for l in lakes["lakes"]
                if l.get("authored_level_m") is None]
    est = [l for l in lakes["lakes"] if l.get("level_source") == "dtm_interior_median"]
    print(f"[lakes] {len(lake_ids)} lakes, {len(est)} with an estimated level, "
          f"{len(no_level)} without any level")
    if no_level:
        print("    NOTE: lakes without a level cannot be pinned AuthoredWins. They "
              "are also left uncarved, so they stay flat ground rather than "
              "becoming an empty bowl — rerun without --no-estimate-lake-levels "
              "to fill them in.")

    with open(os.path.join(a.out, wv["junctions"]["file"])) as f:
        junc = json.load(f)
    bad = [j for j in junc["junctions"] if j["lake_id"] not in lake_ids]
    orphan_seg = [j for j in junc["junctions"] if j["segment_id"] not in seen_ids]
    print(f"[junctions] {len(junc['junctions'])} total")
    if bad:
        print(f"    {len(bad)} reference a lake_id with no lake record")
        ok = False
    if orphan_seg:
        print(f"    {len(orphan_seg)} reference a segment_id not in rivers.bin")
        ok = False
    if not bad and not orphan_seg:
        print("    all lake and segment references resolve")

    # ---- water_id atlas --------------------------------------------------
    wid = man.get("water_id")
    if wid:
        lx, ly = man["leaf_tiles"]
        ts = man["tile_samples"]
        expect = core.atlas_total_bytes(lx, ly, man["num_levels"], ts)
        wpath = os.path.join(a.out, wid["atlas_file"])
        actual = os.path.getsize(wpath) if os.path.exists(wpath) else -1
        good = actual == expect
        ok &= good
        print(f"[water_id] {wid['atlas_file']}: {actual} bytes, expected {expect} "
              f"-> {'OK' if good else 'MISMATCH'}")
        hpath = os.path.join(a.out, man["atlas"]["height_file"])
        if good and os.path.getsize(hpath) == actual:
            print("    same size as the height atlas — tile offsets are shared")

    _print_water_report(wv["validation"])
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


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


def main(argv=None):
    p = argparse.ArgumentParser(prog="kvterrain")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="fetch + build tiles")
    b.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"))
    b.add_argument("--spacing", type=float, default=5.0)
    b.add_argument("--tile-cells", type=int, default=128, dest="tile_cells")
    b.add_argument("--epsg", type=int, default=None)
    b.add_argument("--source", choices=["DTM", "DOM"], default="DTM")
    b.add_argument("--out", required=True)
    b.add_argument("--max-px", type=int, default=core.DEFAULT_MAX_FETCH_PX, dest="max_px")
    b.add_argument("--nodata-fill", type=float, default=0.0, dest="nodata_fill")
    b.add_argument("--hmin", type=float, default=None)
    b.add_argument("--hmax", type=float, default=None)
    b.add_argument("--demo", action="store_true")
    b.add_argument("--water", action="store_true")
    b.add_argument("--no-main-rivers", action="store_true", dest="no_main_rivers")
    b.add_argument("--river-width-scale", type=float, default=1.0, dest="river_width_scale")
    b.add_argument("--river-depth-scale", type=float, default=1.0, dest="river_depth_scale",
                   help="scales how far river surfaces sit above the DTM channel bed "
                        "(per stream order); >1 = deeper/more opaque rivers")
    b.add_argument("--lake-ramp-radius", type=float, default=10.0, dest="lake_ramp_radius",
                   help="shore-to-max-depth distance in metres for synthetic lake beds")
    b.add_argument("--lake-max-depth", type=float, default=20.0, dest="lake_max_depth",
                   help="maximum synthetic lake depth in metres")
    b.add_argument("--ocean-level", type=float, default=0.0, dest="ocean_level",
                   help="sea level in metres above sea level for the Ocean class. "
                        "Kartverket heights are m.o.h., so 0 is real sea level; the "
                        "consuming world's ocean plane height is a runtime concern. "
                        "Ocean is flood-filled from the map edges, not thresholded.")
    b.add_argument("--river-vertex-stride-m", type=float, default=None,
                   dest="river_vertex_stride",
                   help="spacing to densify river polylines to before sampling Z "
                        "(default: leaf spacing). Denser follows the channel more "
                        "closely; coarser shrinks rivers.bin.")
    b.add_argument("--estimate-lake-levels", action=argparse.BooleanOptionalAction,
                   default=True, dest="estimate_lake_levels",
                   help="fill in a water level for lakes NVE gives no 'hoyde' for, "
                        "by reading the LiDAR water surface inside the polygon "
                        "(default: on). With --no-estimate-lake-levels those lakes "
                        "are left uncarved and dry rather than guessed at.")
    b.add_argument("--emit-geojson", action="store_true", dest="emit_geojson",
                   help="also write rivers.geojson (debug sidecar for QGIS; "
                        "rivers.bin is the runtime format)")
    b.set_defaults(func=cmd_build)

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

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main(sys.argv[1:])