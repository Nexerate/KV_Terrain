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
        }
        if a.demo:
            water_opts["fetcher"] = _water.synthetic_water_features

    res = core.run_export(plan, a.out, source_kind=a.source,
                          fetcher=fetcher, server_url=server,
                          max_fetch_px=a.max_px,
                          nodata_fill_m=a.nodata_fill,
                          height_min=a.hmin, height_max=a.hmax,
                          progress=prog,
                          include_water=a.water, water_opts=water_opts)
    print(f"\nwrote {res.tiles_written} tiles to {a.out}  "
          f"range [{res.height_min:.1f}, {res.height_max:.1f}] m")
    if a.water:
        wm = res.manifest.get("water", {})
        print(f"  + water: {wm.get('lake_count', 0)} lakes, "
              f".water tiles alongside every .r16 (© NVE)")
        print(f"  + synthetic lake beds: radius {a.lake_ramp_radius:.1f} m, "
              f"max depth {a.lake_max_depth:.1f} m")


def cmd_validate(a):
    import requests

    with open(os.path.join(a.out, "manifest.json")) as f:
        man = json.load(f)
    epsg = int(man["crs"].split(":")[1])
    hmin, hmax = man["height_min_m"], man["height_max_m"]
    tc = man["tile_cells"]
    spacing = man["leaf_spacing_m"]

    leaf = next(l for l in man["levels"] if l["level"] == 0)
    rng = np.random.default_rng(0)
    sample_tiles = rng.choice(len(leaf["tiles"]), min(a.n, len(leaf["tiles"])), replace=False)
    sess = requests.Session()
    errs = []
    for ti in sample_tiles:
        t = leaf["tiles"][int(ti)]
        a16 = np.fromfile(os.path.join(a.out, t["file"]), dtype="<u2").reshape(tc + 1, tc + 1)
        i = int(rng.integers(0, tc + 1)); j = int(rng.integers(0, tc + 1))
        packed = a16[tc - j, i]
        h_packed = hmin + (packed / 65535.0) * (hmax - hmin)
        X = t["bbox_utm"][0] + i * spacing
        Y = t["bbox_utm"][1] + j * spacing
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
    b.add_argument("--lake-ramp-radius", type=float, default=10.0, dest="lake_ramp_radius",
                   help="shore-to-max-depth distance in metres for synthetic lake beds")
    b.add_argument("--lake-max-depth", type=float, default=20.0, dest="lake_max_depth",
                   help="maximum synthetic lake depth in metres")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="spot-check packed leaf heights vs point API")
    v.add_argument("--out", required=True)
    v.add_argument("--n", type=int, default=12)
    v.set_defaults(func=cmd_validate)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main(sys.argv[1:])