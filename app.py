"""
kvterrain — Streamlit UI
========================

Draw a rectangle over Norway, pick a resolution, and build a quad-tree pyramid of
R16 height tiles for a Unity terrain system. Data © Kartverket (CC BY 4.0).

Run:
    streamlit run app.py
"""
import io
import json
import os
import zipfile
import tempfile
import numpy as np
import streamlit as st

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from kvterrain import core

st.set_page_config(page_title="Kartverket → Unity terrain", layout="wide")
st.title("Kartverket height → Unity quad-tree terrain")
st.caption("Draw a rectangle, fetch Norwegian DTM/DOM, build R16 tiles. "
           "Data © Kartverket (CC BY 4.0).")

# --------------------------------------------------------------------------- #
# Sidebar: parameters                                                          #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Parameters")
    source_kind = st.selectbox("Surface", ["DTM (bare earth)", "DOM (surface)"], 0)
    source_kind = "DTM" if source_kind.startswith("DTM") else "DOM"

    spacing_m = st.number_input("Leaf spacing (m / texel)", 0.25, 50.0, 5.0, 0.25,
                                help="Don't request finer than native data (DTM1 = 1 m).")
    tile_cells = st.selectbox("Tile cells (samples = cells + 1)", [64, 128, 256], 2)

    epsg_choice = st.selectbox(
        "UTM zone (EUREF89)", ["auto", "25832 (W/S)", "25833 (most)", "25835 (NE)"], 0)
    epsg = None if epsg_choice == "auto" else int(epsg_choice.split()[0])

    max_fetch_px = st.select_slider("Max fetch px / request",
                                    [1024, 2048, 4096, 8192], 2048)

    st.divider()
    st.subheader("Height packing")
    pack_mode = st.radio("R16 vertical range", ["Auto (data min/max)", "Fixed"], 0)
    if pack_mode == "Fixed":
        hmin = st.number_input("height_min (m)", -500.0, 3000.0, 0.0, 10.0)
        hmax = st.number_input("height_max (m)", -500.0, 3000.0, 2500.0, 10.0)
    else:
        hmin = hmax = None
    nodata_fill = st.number_input("Fill for nodata (m)", -500.0, 3000.0, 0.0, 1.0)

    st.divider()
    st.subheader("Water (NVE)")
    want_water = st.toggle("Fetch rivers & lakes", value=True,
                           help="Also fetch NVE Elvenett + Innsjødatabase and write a "
                                ".water mask tile beside every height tile.")
    main_rivers = st.toggle("Include main rivers (hovedelv) for size", value=True,
                            disabled=not want_water,
                            help="Second pass that upgrades main-river size class.")
    river_width_scale = st.slider("River width scale", 0.25, 4.0, 1.0, 0.25,
                                  disabled=not want_water,
                                  help="Scales modelled channel widths → how many "
                                       "pixels each river seeds.")
    lake_ramp_radius = st.slider("Lake shore bevel (m)", 0.5, 40.0, 10.0, 0.5,
                                 disabled=not want_water,
                                 help="Width of the shore bevel where the lake bed "
                                      "ramps from the waterline down to full depth.")
    lake_max_depth = st.slider("Lake carve depth (m, real)", 1.0, 40.0, 20.0, 1.0,
                               disabled=not want_water,
                               help="Real metres the flat lake bed sits below the known "
                                    "NVE surface. Must exceed the renderer's opaque "
                                    "threshold AFTER any height compression (e.g. 20 m "
                                    "real → 4 units at 1:5).")

    st.divider()
    demo = st.toggle("Demo mode (synthetic, no network)", value=False,
                     help="Build from a fake surface to test the pipeline/UI offline. "
                          "Also uses a synthetic river+lake when water is on.")

# --------------------------------------------------------------------------- #
# Map with draw control                                                        #
# --------------------------------------------------------------------------- #
col_map, col_info = st.columns([3, 2])

with col_map:
    m = folium.Map(location=[61.3, 8.3], zoom_start=6, tiles="OpenStreetMap")
    Draw(
        export=False,
        draw_options={"rectangle": True, "polygon": False, "polyline": False,
                      "circle": False, "marker": False, "circlemarker": False},
        edit_options={"edit": False},
    ).add_to(m)
    map_state = st_folium(m, height=540, width=None,
                          returned_objects=["last_active_drawing", "all_drawings"])


def _bbox_from_drawing(state):
    draw = (state or {}).get("last_active_drawing") or None
    if not draw:
        drawings = (state or {}).get("all_drawings") or []
        draw = drawings[-1] if drawings else None
    if not draw:
        return None
    coords = draw["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


bbox = _bbox_from_drawing(map_state)

# --------------------------------------------------------------------------- #
# Plan preview                                                                 #
# --------------------------------------------------------------------------- #
with col_info:
    st.subheader("Plan")
    if bbox is None:
        st.info("Draw a rectangle on the map (top-left toolbar) to begin.")
        plan = None
    else:
        lon_min, lat_min, lon_max, lat_max = bbox
        plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
                              spacing_m=spacing_m, tile_cells=tile_cells, epsg=epsg)
        fetch_chunks = (
            -(-plan.samples_x // max_fetch_px) * -(-plan.samples_y // max_fetch_px))
        approx_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 2 / 1e6

        st.markdown(
            f"""
- **CRS:** EPSG:{plan.epsg}
- **Region (padded):** {plan.width_m:,.0f} × {plan.height_m:,.0f} m
- **Leaf grid:** {plan.leaf_tiles_x} × {plan.leaf_tiles_y} tiles
  ({plan.samples_x} × {plan.samples_y} samples)
- **Pyramid levels:** {plan.num_levels} (root → leaves)
- **Total tiles:** {plan.total_tiles()}  ·  **≈ {approx_mb:,.1f} MB** on disk
- **Fetch requests:** {fetch_chunks} (≤ {max_fetch_px}px each)
""")
        ox, oy = plan.origin_x, plan.origin_y
        st.caption(f"SW origin (UTM): {ox:,.1f}, {oy:,.1f}")
        if want_water:
            water_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 5 / 1e6
            st.caption(f"+ water: {plan.total_tiles()} .water tiles "
                       f"(≈ {water_mb:,.1f} MB, 5 bytes/sample) from NVE")
            st.caption(f"+ lake beds carved flat {lake_max_depth:.0f} m below the "
                       f"known NVE surface ({lake_ramp_radius:.0f} m shore bevel)")
        if plan.fetch_pixels() > 60_000 * 60_000:
            st.warning("Very large area — consider the bulk DTM1 download instead.")

# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #
st.divider()
go = st.button("Build terrain tiles", type="primary", disabled=(plan is None))

if go and plan is not None:
    prog = st.progress(0.0, text="starting…")

    def on_progress(done, total, msg):
        total = max(int(total), 1)
        prog.progress(min(float(done) / float(total), 1.0), text=f"Fetching {msg}")

    if demo:
        def fetcher(u, e, sx, sy, nx, ny, sp):
            xs = sx + np.arange(nx) * sp
            ys = sy + np.arange(ny) * sp
            XX, YY = np.meshgrid(xs, ys)
            surf = (300.0
                    + 400.0 * np.exp(-(((XX - XX.mean()) / 4000.0) ** 2
                                       + ((YY - YY.mean()) / 4000.0) ** 2))
                    + 60.0 * np.sin(XX / 900.0) * np.cos(YY / 700.0))
            return surf.astype(np.float32)[::-1, :]
        server_url = "(demo)"
    else:
        fetcher = None
        server_url = None

    try:
        out_dir = tempfile.mkdtemp(prefix="kvterrain_")

        leaf = core.assemble_region(
            plan,
            fetcher or (lambda u, e, sx, sy, nx, ny, sp:
                        core.export_image_fetch(u, e, sx, sy, nx, ny, sp,
                                                source_kind=source_kind)),
            server_url or core.IMAGESERVER[source_kind],
            max_fetch_px=max_fetch_px, progress=on_progress)

        # ---- water pass runs BEFORE the height pyramid is built -----------
        # Lakes/rivers must exist so the bed can be carved and the water-surface
        # field computed against `leaf` before build_pyramid() locks it in.
        water_levels = None
        water_leaf = None
        surface_levels = None
        if want_water:
            prog.progress(0.99, text="fetching + rasterising rivers & lakes…")
            from kvterrain import water as kvwater
            from kvterrain import bathymetry
            from kvterrain import watersurface as kvws

            feats = (kvwater.synthetic_water_features(plan) if demo
                     else kvwater.fetch_water_features(
                         plan, include_main_rivers=main_rivers))
            water_leaf = kvwater.rasterize_water(
                plan, feats, width_scale=river_width_scale)

            lake_surf = water_leaf.lake_surface_moh()          # NVE hoyde
            leaf = bathymetry.carve_lake_beds(
                leaf,
                water_leaf.type,
                plan.spacing_m,
                surface_moh=lake_surf,
                carve_depth_m=lake_max_depth,
            )
            # Unified per-pixel water surface (lakes = hoyde, rivers = DTM-estimated).
            river_surf = kvws.river_surface_moh(
                plan, feats, water_leaf, leaf, lake_surface=lake_surf)
            water_surface = kvws.combine_water_surface(lake_surf, river_surf)
            surface_levels = kvws.build_surface_pyramid(water_surface, plan.num_levels)

            water_levels = kvwater.build_water_pyramid(water_leaf, plan.num_levels)

        prog.progress(1.0, text="building pyramid + slicing tiles…")
        levels = core.build_pyramid(leaf, plan.num_levels)
        res = core.export_tiles(plan, levels, out_dir,
                                nodata_fill_m=nodata_fill,
                                height_min=hmin, height_max=hmax,
                                source_kind=source_kind)

        if water_levels is not None:
            from kvterrain import water as kvwater
            wres = kvwater.export_water_tiles(plan, water_levels, out_dir,
                                              width_scale=river_width_scale)
            res.manifest["water"] = wres.water_manifest
            res.manifest["water"]["synthetic_lake_bathymetry"] = {
                "enabled": True,
                "carve_depth_m": float(lake_max_depth),
                "method": "flat_below_known_surface_with_shore_bevel",
            }
        if surface_levels is not None:
            from kvterrain import watersurface as kvws
            res.manifest["water_surface"] = kvws.export_surface_tiles(
                plan, surface_levels, out_dir, res.height_min, res.height_max)
        if water_levels is not None or surface_levels is not None:
            with open(os.path.join(out_dir, "manifest.json"), "w") as _f:
                json.dump(res.manifest, _f, indent=2)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    zf.write(fp, os.path.relpath(fp, out_dir))
        buf.seek(0)

        st.success(f"Built {res.tiles_written} tiles across {plan.num_levels} levels. "
                   f"Vertical range [{res.height_min:.1f}, {res.height_max:.1f}] m.")

        coarse = levels[-1]
        norm = np.clip((coarse - res.height_min) /
                       max(res.height_max - res.height_min, 1e-6), 0, 1)
        prev_cols = st.columns(2) if water_levels is not None else [st]
        prev_cols[0].image(np.nan_to_num(norm), caption="Height root (coarsest)",
                           clamp=True, width=260)

        if water_levels is not None:
            wc = water_levels[-1]
            rgb = np.zeros((*wc.type.shape, 3), dtype=np.float32)
            rgb[wc.type == kvwater.TYPE_LAKE] = (0.10, 0.35, 0.85)
            rmask = wc.type == kvwater.TYPE_RIVER
            inten = 0.4 + 0.6 * (wc.weight.astype(np.float32) / 8.0)
            rgb[..., 0][rmask] = 0.0
            rgb[..., 1][rmask] = 0.7 * inten[rmask]
            rgb[..., 2][rmask] = 1.0 * inten[rmask]
            prev_cols[1].image(np.clip(rgb, 0, 1), caption="Water root (blue=lake, cyan=river)",
                               clamp=True, width=260)
            wm = res.manifest["water"]
            st.caption(f"Water: {wm['lake_count']} lake(s); .water tiles "
                       f"(5 bytes/sample: type,weight,flow,lake_id) beside every "
                       f".r16. {kvwater.WATER_ATTRIBUTION}.")
            st.caption(f"Water surface: per-pixel .wsurf tiles (moh); lakes from NVE "
                       f"hoyde, rivers DTM-estimated. Bed carved {lake_max_depth:.0f} m "
                       f"below surface ({lake_ramp_radius:.0f} m bevel).")

        st.download_button("Download tiles + manifest (zip)", buf,
                           file_name="kvterrain_tiles.zip", mime="application/zip")
        with st.expander("manifest.json"):
            st.json(res.manifest)

    except Exception as e:
        st.error(f"Build failed: {e}")
        st.caption("If this is a live fetch, the sandbox/network may not reach "
                   "hoydedata.no. Try Demo mode to verify the pipeline, then run "
                   "the tool from a machine with internet access.")

# """
# kvterrain — Streamlit UI
# ========================

# Draw a rectangle over Norway, pick a resolution, and build a quad-tree pyramid of
# R16 height tiles for a Unity terrain system. Data © Kartverket (CC BY 4.0).

# Run:
#     streamlit run app.py
# """
# import io
# import json
# import os
# import zipfile
# import tempfile
# import numpy as np
# import streamlit as st

# import folium
# from folium.plugins import Draw
# from streamlit_folium import st_folium

# from kvterrain import core

# st.set_page_config(page_title="Kartverket → Unity terrain", layout="wide")
# st.title("Kartverket height → Unity quad-tree terrain")
# st.caption("Draw a rectangle, fetch Norwegian DTM/DOM, build R16 tiles. "
#            "Data © Kartverket (CC BY 4.0).")

# # --------------------------------------------------------------------------- #
# # Sidebar: parameters                                                          #
# # --------------------------------------------------------------------------- #
# with st.sidebar:
#     st.header("Parameters")
#     source_kind = st.selectbox("Surface", ["DTM (bare earth)", "DOM (surface)"], 0)
#     source_kind = "DTM" if source_kind.startswith("DTM") else "DOM"

#     spacing_m = st.number_input("Leaf spacing (m / texel)", 0.25, 50.0, 5.0, 0.25,
#                                 help="Don't request finer than native data (DTM1 = 1 m).")
#     tile_cells = st.selectbox("Tile cells (samples = cells + 1)", [64, 128, 256], 1)

#     epsg_choice = st.selectbox(
#         "UTM zone (EUREF89)", ["auto", "25832 (W/S)", "25833 (most)", "25835 (NE)"], 0)
#     epsg = None if epsg_choice == "auto" else int(epsg_choice.split()[0])

#     max_fetch_px = st.select_slider("Max fetch px / request",
#                                     [1024, 2048, 4096, 8192], 4096)

#     st.divider()
#     st.subheader("Height packing")
#     pack_mode = st.radio("R16 vertical range", ["Auto (data min/max)", "Fixed"], 0)
#     if pack_mode == "Fixed":
#         hmin = st.number_input("height_min (m)", -500.0, 3000.0, 0.0, 10.0)
#         hmax = st.number_input("height_max (m)", -500.0, 3000.0, 2500.0, 10.0)
#     else:
#         hmin = hmax = None
#     nodata_fill = st.number_input("Fill for nodata (m)", -500.0, 3000.0, 0.0, 1.0)

#     st.divider()
#     st.subheader("Water (NVE)")
#     want_water = st.toggle("Fetch rivers & lakes", value=False,
#                            help="Also fetch NVE Elvenett + Innsjødatabase and write a "
#                                 ".water mask tile beside every height tile.")
#     main_rivers = st.toggle("Include main rivers (hovedelv) for size", value=True,
#                             disabled=not want_water,
#                             help="Second pass that upgrades main-river size class.")
#     river_width_scale = st.slider("River width scale", 0.25, 4.0, 1.0, 0.25,
#                                   disabled=not want_water,
#                                   help="Scales modelled channel widths → how many "
#                                        "pixels each river seeds.")
#     lake_ramp_radius = st.slider("Lake shore ramp (m)", 0.5, 20.0, 4.0, 0.5,
#                                  disabled=not want_water,
#                                  help="Distance from shoreline until synthetic lake bed "
#                                       "reaches full depth.")
#     lake_max_depth = st.slider("Lake max depth (m)", 0.5, 20.0, 4.0, 0.5,
#                                disabled=not want_water,
#                                help="Maximum synthetic depth subtracted from lake surface "
#                                     "height before the terrain pyramid is built.")

#     st.divider()
#     demo = st.toggle("Demo mode (synthetic, no network)", value=False,
#                      help="Build from a fake surface to test the pipeline/UI offline. "
#                           "Also uses a synthetic river+lake when water is on.")

# # --------------------------------------------------------------------------- #
# # Map with draw control                                                        #
# # --------------------------------------------------------------------------- #
# col_map, col_info = st.columns([3, 2])

# with col_map:
#     m = folium.Map(location=[61.3, 8.3], zoom_start=6, tiles="OpenStreetMap")
#     Draw(
#         export=False,
#         draw_options={"rectangle": True, "polygon": False, "polyline": False,
#                       "circle": False, "marker": False, "circlemarker": False},
#         edit_options={"edit": False},
#     ).add_to(m)
#     map_state = st_folium(m, height=540, width=None,
#                           returned_objects=["last_active_drawing", "all_drawings"])


# def _bbox_from_drawing(state):
#     draw = (state or {}).get("last_active_drawing") or None
#     if not draw:
#         drawings = (state or {}).get("all_drawings") or []
#         draw = drawings[-1] if drawings else None
#     if not draw:
#         return None
#     coords = draw["geometry"]["coordinates"][0]
#     lons = [c[0] for c in coords]
#     lats = [c[1] for c in coords]
#     return min(lons), min(lats), max(lons), max(lats)


# bbox = _bbox_from_drawing(map_state)

# # --------------------------------------------------------------------------- #
# # Plan preview                                                                 #
# # --------------------------------------------------------------------------- #
# with col_info:
#     st.subheader("Plan")
#     if bbox is None:
#         st.info("Draw a rectangle on the map (top-left toolbar) to begin.")
#         plan = None
#     else:
#         lon_min, lat_min, lon_max, lat_max = bbox
#         plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
#                               spacing_m=spacing_m, tile_cells=tile_cells, epsg=epsg)
#         fetch_chunks = (
#             -(-plan.samples_x // max_fetch_px) * -(-plan.samples_y // max_fetch_px))
#         approx_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 2 / 1e6

#         st.markdown(
#             f"""
# - **CRS:** EPSG:{plan.epsg}
# - **Region (padded):** {plan.width_m:,.0f} × {plan.height_m:,.0f} m
# - **Leaf grid:** {plan.leaf_tiles_x} × {plan.leaf_tiles_y} tiles
#   ({plan.samples_x} × {plan.samples_y} samples)
# - **Pyramid levels:** {plan.num_levels} (root → leaves)
# - **Total tiles:** {plan.total_tiles()}  ·  **≈ {approx_mb:,.1f} MB** on disk
# - **Fetch requests:** {fetch_chunks} (≤ {max_fetch_px}px each)
# """)
#         ox, oy = plan.origin_x, plan.origin_y
#         st.caption(f"SW origin (UTM): {ox:,.1f}, {oy:,.1f}")
#         if want_water:
#             water_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 5 / 1e6
#             st.caption(f"+ water: {plan.total_tiles()} .water tiles "
#                        f"(≈ {water_mb:,.1f} MB, 5 bytes/sample) from NVE")
#             st.caption(f"+ synthetic lake beds: {lake_ramp_radius:.1f} m shore ramp, "
#                        f"{lake_max_depth:.1f} m max depth")
#         if plan.fetch_pixels() > 60_000 * 60_000:
#             st.warning("Very large area — consider the bulk DTM1 download instead.")

# # --------------------------------------------------------------------------- #
# # Build                                                                        #
# # --------------------------------------------------------------------------- #
# st.divider()
# go = st.button("Build terrain tiles", type="primary", disabled=(plan is None))

# if go and plan is not None:
#     prog = st.progress(0.0, text="starting…")

#     def on_progress(done, total, msg):
#         total = max(int(total), 1)
#         prog.progress(min(float(done) / float(total), 1.0), text=f"Fetching {msg}")

#     if demo:
#         def fetcher(u, e, sx, sy, nx, ny, sp):
#             xs = sx + np.arange(nx) * sp
#             ys = sy + np.arange(ny) * sp
#             XX, YY = np.meshgrid(xs, ys)
#             surf = (300.0
#                     + 400.0 * np.exp(-(((XX - XX.mean()) / 4000.0) ** 2
#                                        + ((YY - YY.mean()) / 4000.0) ** 2))
#                     + 60.0 * np.sin(XX / 900.0) * np.cos(YY / 700.0))
#             return surf.astype(np.float32)[::-1, :]
#         server_url = "(demo)"
#     else:
#         fetcher = None
#         server_url = None

#     try:
#         out_dir = tempfile.mkdtemp(prefix="kvterrain_")

#         leaf = core.assemble_region(
#             plan,
#             fetcher or (lambda u, e, sx, sy, nx, ny, sp:
#                         core.export_image_fetch(u, e, sx, sy, nx, ny, sp,
#                                                 source_kind=source_kind)),
#             server_url or core.IMAGESERVER[source_kind],
#             max_fetch_px=max_fetch_px, progress=on_progress)

#         # ---- water pass runs BEFORE the height pyramid is built -----------
#         # This is the fix: lake polygons must exist so the synthetic lakebed
#         # can be carved into `leaf` before build_pyramid() locks it in.
#         water_levels = None
#         water_leaf = None
#         if want_water:
#             prog.progress(0.99, text="fetching + rasterising rivers & lakes…")
#             from kvterrain import water as kvwater
#             from kvterrain import bathymetry

#             feats = kvwater.synthetic_water_features(plan) if demo else None
#             water_leaf = kvwater.rasterize_water(
#                 plan,
#                 feats or kvwater.fetch_water_features(
#                     plan, include_main_rivers=main_rivers),
#                 width_scale=river_width_scale)

#             # leaf = bathymetry.carve_lake_beds(
#             #     leaf,
#             #     water_leaf.type,
#             #     plan.spacing_m,
#             #     ramp_radius_m=lake_ramp_radius,
#             #     max_depth_m=lake_max_depth,
#             # )
#             leaf = bathymetry.carve_lake_beds(
#                 leaf, water_leaf.type, plan.spacing_m,
#                 surface_moh=water_leaf.lake_surface_moh(),   # <- NVE hoyde
#                 carve_depth_m=lake_max_depth)                # set the UI default to 20
#             water_levels = kvwater.build_water_pyramid(water_leaf, plan.num_levels)

#         prog.progress(1.0, text="building pyramid + slicing tiles…")
#         levels = core.build_pyramid(leaf, plan.num_levels)
#         res = core.export_tiles(plan, levels, out_dir,
#                                 nodata_fill_m=nodata_fill,
#                                 height_min=hmin, height_max=hmax,
#                                 source_kind=source_kind)

#         if water_levels is not None:
#             from kvterrain import water as kvwater
#             wres = kvwater.export_water_tiles(plan, water_levels, out_dir,
#                                               width_scale=river_width_scale)
#             res.manifest["water"] = wres.water_manifest
#             res.manifest["water"]["synthetic_lake_bathymetry"] = {
#                 "enabled": True,
#                 "ramp_radius_m": float(lake_ramp_radius),
#                 "max_depth_m": float(lake_max_depth),
#                 "method": "distance_to_shore_smoothstep",
#             }
#             with open(os.path.join(out_dir, "manifest.json"), "w") as _f:
#                 json.dump(res.manifest, _f, indent=2)

#         buf = io.BytesIO()
#         with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
#             for root, _, files in os.walk(out_dir):
#                 for fn in files:
#                     fp = os.path.join(root, fn)
#                     zf.write(fp, os.path.relpath(fp, out_dir))
#         buf.seek(0)

#         st.success(f"Built {res.tiles_written} tiles across {plan.num_levels} levels. "
#                    f"Vertical range [{res.height_min:.1f}, {res.height_max:.1f}] m.")

#         coarse = levels[-1]
#         norm = np.clip((coarse - res.height_min) /
#                        max(res.height_max - res.height_min, 1e-6), 0, 1)
#         prev_cols = st.columns(2) if water_levels is not None else [st]
#         prev_cols[0].image(np.nan_to_num(norm), caption="Height root (coarsest)",
#                            clamp=True, width=260)

#         if water_levels is not None:
#             wc = water_levels[-1]
#             rgb = np.zeros((*wc.type.shape, 3), dtype=np.float32)
#             rgb[wc.type == kvwater.TYPE_LAKE] = (0.10, 0.35, 0.85)
#             rmask = wc.type == kvwater.TYPE_RIVER
#             inten = 0.4 + 0.6 * (wc.weight.astype(np.float32) / 8.0)
#             rgb[..., 0][rmask] = 0.0
#             rgb[..., 1][rmask] = 0.7 * inten[rmask]
#             rgb[..., 2][rmask] = 1.0 * inten[rmask]
#             prev_cols[1].image(np.clip(rgb, 0, 1), caption="Water root (blue=lake, cyan=river)",
#                                clamp=True, width=260)
#             wm = res.manifest["water"]
#             st.caption(f"Water: {wm['lake_count']} lake(s); .water tiles "
#                        f"(5 bytes/sample: type,weight,flow,lake_id) beside every "
#                        f".r16. {kvwater.WATER_ATTRIBUTION}.")
#             st.caption(f"Lake bathymetry: synthetic distance-field ramp, "
#                        f"{lake_ramp_radius:.1f} m to full depth, {lake_max_depth:.1f} m max.")

#         st.download_button("Download tiles + manifest (zip)", buf,
#                            file_name="kvterrain_tiles.zip", mime="application/zip")
#         with st.expander("manifest.json"):
#             st.json(res.manifest)

#     except Exception as e:
#         st.error(f"Build failed: {e}")
#         st.caption("If this is a live fetch, the sandbox/network may not reach "
#                    "hoydedata.no. Try Demo mode to verify the pipeline, then run "
#                    "the tool from a machine with internet access.")

# """
# kvterrain — Streamlit UI
# ========================

# Draw a rectangle over Norway, pick a resolution, and build a quad-tree pyramid of
# R16 height tiles for a Unity terrain system. Data © Kartverket (CC BY 4.0).

# Run:
#     streamlit run app.py
# """
# import io
# import json
# import os
# import zipfile
# import tempfile
# import numpy as np
# import streamlit as st

# import folium
# from folium.plugins import Draw
# from streamlit_folium import st_folium

# from kvterrain import core

# st.set_page_config(page_title="Kartverket → Unity terrain", layout="wide")
# st.title("Kartverket height → Unity quad-tree terrain")
# st.caption("Draw a rectangle, fetch Norwegian DTM/DOM, build R16 tiles. "
#            "Data © Kartverket (CC BY 4.0).")

# # --------------------------------------------------------------------------- #
# # Sidebar: parameters                                                          #
# # --------------------------------------------------------------------------- #
# with st.sidebar:
#     st.header("Parameters")
#     source_kind = st.selectbox("Surface", ["DTM (bare earth)", "DOM (surface)"], 0)
#     source_kind = "DTM" if source_kind.startswith("DTM") else "DOM"

#     spacing_m = st.number_input("Leaf spacing (m / texel)", 0.25, 50.0, 5.0, 0.25,
#                                 help="Don't request finer than native data (DTM1 = 1 m).")
#     tile_cells = st.selectbox("Tile cells (samples = cells + 1)", [64, 128, 256], 1)

#     epsg_choice = st.selectbox(
#         "UTM zone (EUREF89)", ["auto", "25832 (W/S)", "25833 (most)", "25835 (NE)"], 0)
#     epsg = None if epsg_choice == "auto" else int(epsg_choice.split()[0])

#     max_fetch_px = st.select_slider("Max fetch px / request",
#                                     [1024, 2048, 4096, 8192], 4096)

#     st.divider()
#     st.subheader("Height packing")
#     pack_mode = st.radio("R16 vertical range", ["Auto (data min/max)", "Fixed"], 0)
#     if pack_mode == "Fixed":
#         hmin = st.number_input("height_min (m)", -500.0, 3000.0, 0.0, 10.0)
#         hmax = st.number_input("height_max (m)", -500.0, 3000.0, 2500.0, 10.0)
#     else:
#         hmin = hmax = None
#     nodata_fill = st.number_input("Fill for nodata (m)", -500.0, 3000.0, 0.0, 1.0)

#     st.divider()
#     st.subheader("Water (NVE)")
#     want_water = st.toggle("Fetch rivers & lakes", value=False,
#                            help="Also fetch NVE Elvenett + Innsjødatabase and write a "
#                                 ".water mask tile beside every height tile.")
#     main_rivers = st.toggle("Include main rivers (hovedelv) for size", value=True,
#                             disabled=not want_water,
#                             help="Second pass that upgrades main-river size class.")
#     river_width_scale = st.slider("River width scale", 0.25, 4.0, 1.0, 0.25,
#                                   disabled=not want_water,
#                                   help="Scales modelled channel widths → how many "
#                                        "pixels each river seeds.")
#     lake_ramp_radius = st.slider("Lake shore bevel (m)", 0.5, 40.0, 10.0, 0.5,
#                                  disabled=not want_water,
#                                  help="Width of the shore bevel where the lake bed "
#                                       "ramps from the waterline down to full depth.")
#     lake_max_depth = st.slider("Lake carve depth (m, real)", 1.0, 40.0, 20.0, 1.0,
#                                disabled=not want_water,
#                                help="Real metres the flat lake bed sits below the known "
#                                     "NVE surface. Must exceed the renderer's opaque "
#                                     "threshold AFTER any height compression (e.g. 20 m "
#                                     "real → 4 units at 1:5).")

#     st.divider()
#     demo = st.toggle("Demo mode (synthetic, no network)", value=False,
#                      help="Build from a fake surface to test the pipeline/UI offline. "
#                           "Also uses a synthetic river+lake when water is on.")

# # --------------------------------------------------------------------------- #
# # Map with draw control                                                        #
# # --------------------------------------------------------------------------- #
# col_map, col_info = st.columns([3, 2])

# with col_map:
#     m = folium.Map(location=[61.3, 8.3], zoom_start=6, tiles="OpenStreetMap")
#     Draw(
#         export=False,
#         draw_options={"rectangle": True, "polygon": False, "polyline": False,
#                       "circle": False, "marker": False, "circlemarker": False},
#         edit_options={"edit": False},
#     ).add_to(m)
#     map_state = st_folium(m, height=540, width=None,
#                           returned_objects=["last_active_drawing", "all_drawings"])


# def _bbox_from_drawing(state):
#     draw = (state or {}).get("last_active_drawing") or None
#     if not draw:
#         drawings = (state or {}).get("all_drawings") or []
#         draw = drawings[-1] if drawings else None
#     if not draw:
#         return None
#     coords = draw["geometry"]["coordinates"][0]
#     lons = [c[0] for c in coords]
#     lats = [c[1] for c in coords]
#     return min(lons), min(lats), max(lons), max(lats)


# bbox = _bbox_from_drawing(map_state)

# # --------------------------------------------------------------------------- #
# # Plan preview                                                                 #
# # --------------------------------------------------------------------------- #
# with col_info:
#     st.subheader("Plan")
#     if bbox is None:
#         st.info("Draw a rectangle on the map (top-left toolbar) to begin.")
#         plan = None
#     else:
#         lon_min, lat_min, lon_max, lat_max = bbox
#         plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
#                               spacing_m=spacing_m, tile_cells=tile_cells, epsg=epsg)
#         fetch_chunks = (
#             -(-plan.samples_x // max_fetch_px) * -(-plan.samples_y // max_fetch_px))
#         approx_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 2 / 1e6

#         st.markdown(
#             f"""
# - **CRS:** EPSG:{plan.epsg}
# - **Region (padded):** {plan.width_m:,.0f} × {plan.height_m:,.0f} m
# - **Leaf grid:** {plan.leaf_tiles_x} × {plan.leaf_tiles_y} tiles
#   ({plan.samples_x} × {plan.samples_y} samples)
# - **Pyramid levels:** {plan.num_levels} (root → leaves)
# - **Total tiles:** {plan.total_tiles()}  ·  **≈ {approx_mb:,.1f} MB** on disk
# - **Fetch requests:** {fetch_chunks} (≤ {max_fetch_px}px each)
# """)
#         ox, oy = plan.origin_x, plan.origin_y
#         st.caption(f"SW origin (UTM): {ox:,.1f}, {oy:,.1f}")
#         if want_water:
#             water_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 5 / 1e6
#             st.caption(f"+ water: {plan.total_tiles()} .water tiles "
#                        f"(≈ {water_mb:,.1f} MB, 5 bytes/sample) from NVE")
#             st.caption(f"+ lake beds carved flat {lake_max_depth:.0f} m below the "
#                        f"known NVE surface ({lake_ramp_radius:.0f} m shore bevel)")
#         if plan.fetch_pixels() > 60_000 * 60_000:
#             st.warning("Very large area — consider the bulk DTM1 download instead.")

# # --------------------------------------------------------------------------- #
# # Build                                                                        #
# # --------------------------------------------------------------------------- #
# st.divider()
# go = st.button("Build terrain tiles", type="primary", disabled=(plan is None))

# if go and plan is not None:
#     prog = st.progress(0.0, text="starting…")

#     def on_progress(done, total, msg):
#         total = max(int(total), 1)
#         prog.progress(min(float(done) / float(total), 1.0), text=f"Fetching {msg}")

#     if demo:
#         def fetcher(u, e, sx, sy, nx, ny, sp):
#             xs = sx + np.arange(nx) * sp
#             ys = sy + np.arange(ny) * sp
#             XX, YY = np.meshgrid(xs, ys)
#             surf = (300.0
#                     + 400.0 * np.exp(-(((XX - XX.mean()) / 4000.0) ** 2
#                                        + ((YY - YY.mean()) / 4000.0) ** 2))
#                     + 60.0 * np.sin(XX / 900.0) * np.cos(YY / 700.0))
#             return surf.astype(np.float32)[::-1, :]
#         server_url = "(demo)"
#     else:
#         fetcher = None
#         server_url = None

#     try:
#         out_dir = tempfile.mkdtemp(prefix="kvterrain_")

#         leaf = core.assemble_region(
#             plan,
#             fetcher or (lambda u, e, sx, sy, nx, ny, sp:
#                         core.export_image_fetch(u, e, sx, sy, nx, ny, sp,
#                                                 source_kind=source_kind)),
#             server_url or core.IMAGESERVER[source_kind],
#             max_fetch_px=max_fetch_px, progress=on_progress)

#         # ---- water pass runs BEFORE the height pyramid is built -----------
#         # Lakes/rivers must exist so the bed can be carved and the water-surface
#         # field computed against `leaf` before build_pyramid() locks it in.
#         water_levels = None
#         water_leaf = None
#         surface_levels = None
#         if want_water:
#             prog.progress(0.99, text="fetching + rasterising rivers & lakes…")
#             from kvterrain import water as kvwater
#             from kvterrain import bathymetry
#             from kvterrain import watersurface as kvws

#             feats = (kvwater.synthetic_water_features(plan) if demo
#                      else kvwater.fetch_water_features(
#                          plan, include_main_rivers=main_rivers))
#             water_leaf = kvwater.rasterize_water(
#                 plan, feats, width_scale=river_width_scale)

#             lake_surf = water_leaf.lake_surface_moh()          # NVE hoyde
#             leaf = bathymetry.carve_lake_beds(
#                 leaf,
#                 water_leaf.type,
#                 plan.spacing_m,
#                 surface_moh=lake_surf,
#                 carve_depth_m=lake_max_depth,
#             )
#             # Unified per-pixel water surface (lakes = hoyde, rivers = DTM-estimated).
#             river_surf = kvws.river_surface_moh(
#                 plan, feats, water_leaf, leaf, lake_surface=lake_surf)
#             water_surface = kvws.combine_water_surface(lake_surf, river_surf)
#             surface_levels = kvws.build_surface_pyramid(water_surface, plan.num_levels)

#             water_levels = kvwater.build_water_pyramid(water_leaf, plan.num_levels)

#         prog.progress(1.0, text="building pyramid + slicing tiles…")
#         levels = core.build_pyramid(leaf, plan.num_levels)
#         res = core.export_tiles(plan, levels, out_dir,
#                                 nodata_fill_m=nodata_fill,
#                                 height_min=hmin, height_max=hmax,
#                                 source_kind=source_kind)

#         if water_levels is not None:
#             from kvterrain import water as kvwater
#             wres = kvwater.export_water_tiles(plan, water_levels, out_dir,
#                                               width_scale=river_width_scale)
#             res.manifest["water"] = wres.water_manifest
#             res.manifest["water"]["synthetic_lake_bathymetry"] = {
#                 "enabled": True,
#                 "carve_depth_m": float(lake_max_depth),
#                 "method": "flat_below_known_surface_with_shore_bevel",
#             }
#         if surface_levels is not None:
#             from kvterrain import watersurface as kvws
#             res.manifest["water_surface"] = kvws.export_surface_tiles(
#                 plan, surface_levels, out_dir, res.height_min, res.height_max)
#         if water_levels is not None or surface_levels is not None:
#             with open(os.path.join(out_dir, "manifest.json"), "w") as _f:
#                 json.dump(res.manifest, _f, indent=2)

#         buf = io.BytesIO()
#         with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
#             for root, _, files in os.walk(out_dir):
#                 for fn in files:
#                     fp = os.path.join(root, fn)
#                     zf.write(fp, os.path.relpath(fp, out_dir))
#         buf.seek(0)

#         st.success(f"Built {res.tiles_written} tiles across {plan.num_levels} levels. "
#                    f"Vertical range [{res.height_min:.1f}, {res.height_max:.1f}] m.")

#         coarse = levels[-1]
#         norm = np.clip((coarse - res.height_min) /
#                        max(res.height_max - res.height_min, 1e-6), 0, 1)
#         prev_cols = st.columns(2) if water_levels is not None else [st]
#         prev_cols[0].image(np.nan_to_num(norm), caption="Height root (coarsest)",
#                            clamp=True, width=260)

#         if water_levels is not None:
#             wc = water_levels[-1]
#             rgb = np.zeros((*wc.type.shape, 3), dtype=np.float32)
#             rgb[wc.type == kvwater.TYPE_LAKE] = (0.10, 0.35, 0.85)
#             rmask = wc.type == kvwater.TYPE_RIVER
#             inten = 0.4 + 0.6 * (wc.weight.astype(np.float32) / 8.0)
#             rgb[..., 0][rmask] = 0.0
#             rgb[..., 1][rmask] = 0.7 * inten[rmask]
#             rgb[..., 2][rmask] = 1.0 * inten[rmask]
#             prev_cols[1].image(np.clip(rgb, 0, 1), caption="Water root (blue=lake, cyan=river)",
#                                clamp=True, width=260)
#             wm = res.manifest["water"]
#             st.caption(f"Water: {wm['lake_count']} lake(s); .water tiles "
#                        f"(5 bytes/sample: type,weight,flow,lake_id) beside every "
#                        f".r16. {kvwater.WATER_ATTRIBUTION}.")
#             st.caption(f"Water surface: per-pixel .wsurf tiles (moh); lakes from NVE "
#                        f"hoyde, rivers DTM-estimated. Bed carved {lake_max_depth:.0f} m "
#                        f"below surface ({lake_ramp_radius:.0f} m bevel).")

#         st.download_button("Download tiles + manifest (zip)", buf,
#                            file_name="kvterrain_tiles.zip", mime="application/zip")
#         with st.expander("manifest.json"):
#             st.json(res.manifest)

#     except Exception as e:
#         st.error(f"Build failed: {e}")
#         st.caption("If this is a live fetch, the sandbox/network may not reach "
#                    "hoydedata.no. Try Demo mode to verify the pipeline, then run "
#                    "the tool from a machine with internet access.")

# # """
# # kvterrain — Streamlit UI
# # ========================

# # Draw a rectangle over Norway, pick a resolution, and build a quad-tree pyramid of
# # R16 height tiles for a Unity terrain system. Data © Kartverket (CC BY 4.0).

# # Run:
# #     streamlit run app.py
# # """
# # import io
# # import json
# # import os
# # import zipfile
# # import tempfile
# # import numpy as np
# # import streamlit as st

# # import folium
# # from folium.plugins import Draw
# # from streamlit_folium import st_folium

# # from kvterrain import core

# # st.set_page_config(page_title="Kartverket → Unity terrain", layout="wide")
# # st.title("Kartverket height → Unity quad-tree terrain")
# # st.caption("Draw a rectangle, fetch Norwegian DTM/DOM, build R16 tiles. "
# #            "Data © Kartverket (CC BY 4.0).")

# # # --------------------------------------------------------------------------- #
# # # Sidebar: parameters                                                          #
# # # --------------------------------------------------------------------------- #
# # with st.sidebar:
# #     st.header("Parameters")
# #     source_kind = st.selectbox("Surface", ["DTM (bare earth)", "DOM (surface)"], 0)
# #     source_kind = "DTM" if source_kind.startswith("DTM") else "DOM"

# #     spacing_m = st.number_input("Leaf spacing (m / texel)", 0.25, 50.0, 5.0, 0.25,
# #                                 help="Don't request finer than native data (DTM1 = 1 m).")
# #     tile_cells = st.selectbox("Tile cells (samples = cells + 1)", [64, 128, 256], 1)

# #     epsg_choice = st.selectbox(
# #         "UTM zone (EUREF89)", ["auto", "25832 (W/S)", "25833 (most)", "25835 (NE)"], 0)
# #     epsg = None if epsg_choice == "auto" else int(epsg_choice.split()[0])

# #     max_fetch_px = st.select_slider("Max fetch px / request",
# #                                     [1024, 2048, 4096, 8192], 4096)

# #     st.divider()
# #     st.subheader("Height packing")
# #     pack_mode = st.radio("R16 vertical range", ["Auto (data min/max)", "Fixed"], 0)
# #     if pack_mode == "Fixed":
# #         hmin = st.number_input("height_min (m)", -500.0, 3000.0, 0.0, 10.0)
# #         hmax = st.number_input("height_max (m)", -500.0, 3000.0, 2500.0, 10.0)
# #     else:
# #         hmin = hmax = None
# #     nodata_fill = st.number_input("Fill for nodata (m)", -500.0, 3000.0, 0.0, 1.0)

# #     st.divider()
# #     st.subheader("Water (NVE)")
# #     want_water = st.toggle("Fetch rivers & lakes", value=False,
# #                            help="Also fetch NVE Elvenett + Innsjødatabase and write a "
# #                                 ".water mask tile beside every height tile.")
# #     main_rivers = st.toggle("Include main rivers (hovedelv) for size", value=True,
# #                             disabled=not want_water,
# #                             help="Second pass that upgrades main-river size class.")
# #     river_width_scale = st.slider("River width scale", 0.25, 4.0, 1.0, 0.25,
# #                                   disabled=not want_water,
# #                                   help="Scales modelled channel widths → how many "
# #                                        "pixels each river seeds.")
# #     lake_ramp_radius = st.slider("Lake shore ramp (m)", 0.5, 20.0, 4.0, 0.5,
# #                                  disabled=not want_water,
# #                                  help="Distance from shoreline until synthetic lake bed "
# #                                       "reaches full depth.")
# #     lake_max_depth = st.slider("Lake max depth (m)", 0.5, 20.0, 4.0, 0.5,
# #                                disabled=not want_water,
# #                                help="Maximum synthetic depth subtracted from lake surface "
# #                                     "height before the terrain pyramid is built.")

# #     st.divider()
# #     demo = st.toggle("Demo mode (synthetic, no network)", value=False,
# #                      help="Build from a fake surface to test the pipeline/UI offline. "
# #                           "Also uses a synthetic river+lake when water is on.")

# # # --------------------------------------------------------------------------- #
# # # Map with draw control                                                        #
# # # --------------------------------------------------------------------------- #
# # col_map, col_info = st.columns([3, 2])

# # with col_map:
# #     m = folium.Map(location=[61.3, 8.3], zoom_start=6, tiles="OpenStreetMap")
# #     Draw(
# #         export=False,
# #         draw_options={"rectangle": True, "polygon": False, "polyline": False,
# #                       "circle": False, "marker": False, "circlemarker": False},
# #         edit_options={"edit": False},
# #     ).add_to(m)
# #     map_state = st_folium(m, height=540, width=None,
# #                           returned_objects=["last_active_drawing", "all_drawings"])


# # def _bbox_from_drawing(state):
# #     draw = (state or {}).get("last_active_drawing") or None
# #     if not draw:
# #         drawings = (state or {}).get("all_drawings") or []
# #         draw = drawings[-1] if drawings else None
# #     if not draw:
# #         return None
# #     coords = draw["geometry"]["coordinates"][0]
# #     lons = [c[0] for c in coords]
# #     lats = [c[1] for c in coords]
# #     return min(lons), min(lats), max(lons), max(lats)


# # bbox = _bbox_from_drawing(map_state)

# # # --------------------------------------------------------------------------- #
# # # Plan preview                                                                 #
# # # --------------------------------------------------------------------------- #
# # with col_info:
# #     st.subheader("Plan")
# #     if bbox is None:
# #         st.info("Draw a rectangle on the map (top-left toolbar) to begin.")
# #         plan = None
# #     else:
# #         lon_min, lat_min, lon_max, lat_max = bbox
# #         plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
# #                               spacing_m=spacing_m, tile_cells=tile_cells, epsg=epsg)
# #         fetch_chunks = (
# #             -(-plan.samples_x // max_fetch_px) * -(-plan.samples_y // max_fetch_px))
# #         approx_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 2 / 1e6

# #         st.markdown(
# #             f"""
# # - **CRS:** EPSG:{plan.epsg}
# # - **Region (padded):** {plan.width_m:,.0f} × {plan.height_m:,.0f} m
# # - **Leaf grid:** {plan.leaf_tiles_x} × {plan.leaf_tiles_y} tiles
# #   ({plan.samples_x} × {plan.samples_y} samples)
# # - **Pyramid levels:** {plan.num_levels} (root → leaves)
# # - **Total tiles:** {plan.total_tiles()}  ·  **≈ {approx_mb:,.1f} MB** on disk
# # - **Fetch requests:** {fetch_chunks} (≤ {max_fetch_px}px each)
# # """)
# #         ox, oy = plan.origin_x, plan.origin_y
# #         st.caption(f"SW origin (UTM): {ox:,.1f}, {oy:,.1f}")
# #         if want_water:
# #             water_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 5 / 1e6
# #             st.caption(f"+ water: {plan.total_tiles()} .water tiles "
# #                        f"(≈ {water_mb:,.1f} MB, 5 bytes/sample) from NVE")
# #             st.caption(f"+ synthetic lake beds: {lake_ramp_radius:.1f} m shore ramp, "
# #                        f"{lake_max_depth:.1f} m max depth")
# #         if plan.fetch_pixels() > 60_000 * 60_000:
# #             st.warning("Very large area — consider the bulk DTM1 download instead.")

# # # --------------------------------------------------------------------------- #
# # # Build                                                                        #
# # # --------------------------------------------------------------------------- #
# # st.divider()
# # go = st.button("Build terrain tiles", type="primary", disabled=(plan is None))

# # if go and plan is not None:
# #     prog = st.progress(0.0, text="starting…")

# #     def on_progress(done, total, msg):
# #         total = max(int(total), 1)
# #         prog.progress(min(float(done) / float(total), 1.0), text=f"Fetching {msg}")

# #     if demo:
# #         def fetcher(u, e, sx, sy, nx, ny, sp):
# #             xs = sx + np.arange(nx) * sp
# #             ys = sy + np.arange(ny) * sp
# #             XX, YY = np.meshgrid(xs, ys)
# #             surf = (300.0
# #                     + 400.0 * np.exp(-(((XX - XX.mean()) / 4000.0) ** 2
# #                                        + ((YY - YY.mean()) / 4000.0) ** 2))
# #                     + 60.0 * np.sin(XX / 900.0) * np.cos(YY / 700.0))
# #             return surf.astype(np.float32)[::-1, :]
# #         server_url = "(demo)"
# #     else:
# #         fetcher = None
# #         server_url = None

# #     try:
# #         out_dir = tempfile.mkdtemp(prefix="kvterrain_")

# #         leaf = core.assemble_region(
# #             plan,
# #             fetcher or (lambda u, e, sx, sy, nx, ny, sp:
# #                         core.export_image_fetch(u, e, sx, sy, nx, ny, sp,
# #                                                 source_kind=source_kind)),
# #             server_url or core.IMAGESERVER[source_kind],
# #             max_fetch_px=max_fetch_px, progress=on_progress)

# #         # ---- water pass runs BEFORE the height pyramid is built -----------
# #         # This is the fix: lake polygons must exist so the synthetic lakebed
# #         # can be carved into `leaf` before build_pyramid() locks it in.
# #         water_levels = None
# #         water_leaf = None
# #         if want_water:
# #             prog.progress(0.99, text="fetching + rasterising rivers & lakes…")
# #             from kvterrain import water as kvwater
# #             from kvterrain import bathymetry

# #             feats = kvwater.synthetic_water_features(plan) if demo else None
# #             water_leaf = kvwater.rasterize_water(
# #                 plan,
# #                 feats or kvwater.fetch_water_features(
# #                     plan, include_main_rivers=main_rivers),
# #                 width_scale=river_width_scale)

# #             # leaf = bathymetry.carve_lake_beds(
# #             #     leaf,
# #             #     water_leaf.type,
# #             #     plan.spacing_m,
# #             #     ramp_radius_m=lake_ramp_radius,
# #             #     max_depth_m=lake_max_depth,
# #             # )
# #             leaf = bathymetry.carve_lake_beds(
# #                 leaf, water_leaf.type, plan.spacing_m,
# #                 surface_moh=water_leaf.lake_surface_moh(),   # <- NVE hoyde
# #                 carve_depth_m=lake_max_depth)                # set the UI default to 20
# #             water_levels = kvwater.build_water_pyramid(water_leaf, plan.num_levels)

# #         prog.progress(1.0, text="building pyramid + slicing tiles…")
# #         levels = core.build_pyramid(leaf, plan.num_levels)
# #         res = core.export_tiles(plan, levels, out_dir,
# #                                 nodata_fill_m=nodata_fill,
# #                                 height_min=hmin, height_max=hmax,
# #                                 source_kind=source_kind)

# #         if water_levels is not None:
# #             from kvterrain import water as kvwater
# #             wres = kvwater.export_water_tiles(plan, water_levels, out_dir,
# #                                               width_scale=river_width_scale)
# #             res.manifest["water"] = wres.water_manifest
# #             res.manifest["water"]["synthetic_lake_bathymetry"] = {
# #                 "enabled": True,
# #                 "ramp_radius_m": float(lake_ramp_radius),
# #                 "max_depth_m": float(lake_max_depth),
# #                 "method": "distance_to_shore_smoothstep",
# #             }
# #             with open(os.path.join(out_dir, "manifest.json"), "w") as _f:
# #                 json.dump(res.manifest, _f, indent=2)

# #         buf = io.BytesIO()
# #         with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
# #             for root, _, files in os.walk(out_dir):
# #                 for fn in files:
# #                     fp = os.path.join(root, fn)
# #                     zf.write(fp, os.path.relpath(fp, out_dir))
# #         buf.seek(0)

# #         st.success(f"Built {res.tiles_written} tiles across {plan.num_levels} levels. "
# #                    f"Vertical range [{res.height_min:.1f}, {res.height_max:.1f}] m.")

# #         coarse = levels[-1]
# #         norm = np.clip((coarse - res.height_min) /
# #                        max(res.height_max - res.height_min, 1e-6), 0, 1)
# #         prev_cols = st.columns(2) if water_levels is not None else [st]
# #         prev_cols[0].image(np.nan_to_num(norm), caption="Height root (coarsest)",
# #                            clamp=True, width=260)

# #         if water_levels is not None:
# #             wc = water_levels[-1]
# #             rgb = np.zeros((*wc.type.shape, 3), dtype=np.float32)
# #             rgb[wc.type == kvwater.TYPE_LAKE] = (0.10, 0.35, 0.85)
# #             rmask = wc.type == kvwater.TYPE_RIVER
# #             inten = 0.4 + 0.6 * (wc.weight.astype(np.float32) / 8.0)
# #             rgb[..., 0][rmask] = 0.0
# #             rgb[..., 1][rmask] = 0.7 * inten[rmask]
# #             rgb[..., 2][rmask] = 1.0 * inten[rmask]
# #             prev_cols[1].image(np.clip(rgb, 0, 1), caption="Water root (blue=lake, cyan=river)",
# #                                clamp=True, width=260)
# #             wm = res.manifest["water"]
# #             st.caption(f"Water: {wm['lake_count']} lake(s); .water tiles "
# #                        f"(5 bytes/sample: type,weight,flow,lake_id) beside every "
# #                        f".r16. {kvwater.WATER_ATTRIBUTION}.")
# #             st.caption(f"Lake bathymetry: synthetic distance-field ramp, "
# #                        f"{lake_ramp_radius:.1f} m to full depth, {lake_max_depth:.1f} m max.")

# #         st.download_button("Download tiles + manifest (zip)", buf,
# #                            file_name="kvterrain_tiles.zip", mime="application/zip")
# #         with st.expander("manifest.json"):
# #             st.json(res.manifest)

# #     except Exception as e:
# #         st.error(f"Build failed: {e}")
# #         st.caption("If this is a live fetch, the sandbox/network may not reach "
# #                    "hoydedata.no. Try Demo mode to verify the pipeline, then run "
# #                    "the tool from a machine with internet access.")