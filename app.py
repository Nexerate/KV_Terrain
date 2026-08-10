"""
kvterrain — Streamlit UI
========================

Draw a rectangle over Norway, pick a resolution, and build a quad-tree pyramid of
R16 height tiles for a Unity terrain system. Data © Kartverket (CC BY 4.0).

Run:
    streamlit run app.py
"""
import io
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
                           help="Also fetch NVE Elvenett + Innsjødatabase and pack a "
                                "per-pixel water surface into surface.atlas.")
    main_rivers = st.toggle("Include main rivers (hovedelv) for size", value=True,
                            disabled=not want_water,
                            help="Second pass that upgrades main-river size class.")
    river_width_scale = st.slider("River width scale", 0.25, 4.0, 1.0, 0.25,
                                  disabled=not want_water,
                                  help="Scales modelled channel widths → how many "
                                       "pixels each river seeds.")
    river_depth_scale = st.slider("River surface raise ×", 0.25, 4.0, 1.0, 0.25,
                                  disabled=not want_water,
                                  help="Scales how far each river's surface sits above "
                                       "the DTM channel bed (by stream order). Rivers "
                                       "aren't carved, so this is what gives them visible "
                                       "depth/opacity — bigger fills the incised channel "
                                       "and reads more solid. Sized like the lake carve so "
                                       "it survives the renderer's height compression.")
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
    estimate_lake_levels = st.toggle("Estimate missing lake levels", value=True,
                                     disabled=not want_water,
                                     help="NVE leaves 'hoyde' blank on a lot of small "
                                          "lakes — a quarter of them in Lierne. LiDAR "
                                          "reports the water surface as terrain, so the "
                                          "level is read from the DTM inside the polygon "
                                          "and used as if it were authored. Off: those "
                                          "lakes stay flat, dry ground.")
    ocean_level = st.number_input("Ocean level (m.o.h.)", -50.0, 50.0, 0.0, 1.0,
                                  disabled=not want_water,
                                  help="Kartverket heights are metres above sea level, "
                                       "so 0 is real sea level. Ocean is flood-filled "
                                       "inward from the map edges, not thresholded, so "
                                       "inland below-sea-level ground is never "
                                       "mislabelled as sea.")

    st.divider()
    st.subheader("River polylines")
    st.caption("The burn needs flow direction, connectivity and the ramp profile — "
               "none of which survive rasterisation. These are emitted as vectors "
               "beside the rasters and loaded once at runtime, not streamed.")
    river_vertex_stride = st.slider("Vertex stride (m, 0 = leaf spacing)",
                                    0.0, 32.0, 0.0, 1.0, disabled=not want_water,
                                    help="Polylines are densified to this spacing "
                                         "before the DTM is sampled beneath them. "
                                         "Denser follows the real channel more "
                                         "closely; coarser shrinks rivers.bin.")
    emit_geojson = st.toggle("Also write rivers.geojson", value=False,
                             disabled=not want_water,
                             help="Debug sidecar for QGIS. rivers.bin is the runtime "
                                  "format — this is for when you need to LOOK at a "
                                  "validation warning.")

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
            wsurf_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 2 / 1e6
            st.caption(f"+ water surface: surface.atlas "
                       f"(≈ {wsurf_mb:,.1f} MB, u16) from NVE")
            st.caption(f"+ water classes + lake ids: water_id.atlas "
                       f"(≈ {wsurf_mb:,.1f} MB, u16) — same tile offsets as the "
                       f"height atlas")
            st.caption("+ river polylines: rivers.bin + lakes.json + junctions.json "
                       "(vectors, loaded once at runtime — not streamed)")
            st.caption(f"+ lake beds carved flat {lake_max_depth:.0f} m below the "
                       f"known NVE surface ({lake_ramp_radius:.0f} m shore bevel)")
            st.caption(f"+ river surfaces raised above the DTM channel by stream "
                       f"order (× {river_depth_scale:g})")
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

        # ONE call. This module used to re-implement the whole build sequence —
        # assemble, void-repair, carve, river surface, combine, pyramid, export —
        # as a second copy of core.run_export. The two copies had already drifted
        # apart, and every change to the pipeline had to be written twice and kept
        # in sync by hand. All of the Streamlit code stays here; only the pipeline
        # duplication is gone. run_export hands the water products back on its
        # result so the previews below still work.
        water_opts = None
        if want_water:
            water_opts = {
                "include_main_rivers": main_rivers,
                "width_scale": river_width_scale,
                "river_depth_scale": river_depth_scale,
                "lake_ramp_radius_m": lake_ramp_radius,
                "lake_max_depth_m": lake_max_depth,
                "ocean_level_m": ocean_level,
                "emit_geojson": emit_geojson,
                "estimate_lake_levels": estimate_lake_levels,
            }
            if river_vertex_stride > 0:
                water_opts["river_vertex_stride_m"] = river_vertex_stride
            if demo:
                from kvterrain import water as kvwater
                water_opts["fetcher"] = kvwater.synthetic_water_features

        res = core.run_export(
            plan, out_dir,
            source_kind=source_kind,
            fetcher=fetcher,
            server_url=server_url,
            max_fetch_px=max_fetch_px,
            nodata_fill_m=nodata_fill,
            height_min=hmin, height_max=hmax,
            progress=on_progress,
            include_water=want_water,
            water_opts=water_opts,
        )
        prog.progress(1.0, text="packing…")

        water_leaf = res.water_grid
        lake_count = len((water_leaf.lake_table if water_leaf else {}) or {})

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    zf.write(fp, os.path.relpath(fp, out_dir))
        buf.seek(0)

        st.success(f"Built {res.tiles_written} tiles across {plan.num_levels} levels. "
                   f"Vertical range [{res.height_min:.1f}, {res.height_max:.1f}] m.")

        coarse = res.coarse_level
        norm = np.clip((coarse - res.height_min) /
                       max(res.height_max - res.height_min, 1e-6), 0, 1)
        have_water = want_water and water_leaf is not None
        prev_cols = st.columns(2) if have_water else [st]
        prev_cols[0].image(np.nan_to_num(norm), caption="Height root (coarsest)",
                           clamp=True, width=260)

        if have_water:
            from kvterrain import water as kvwater
            # Preview straight from the leaf masks (the categorical pyramid is gone);
            # stride down so the thumbnail stays small on big regions.
            stride = max(1, max(water_leaf.type.shape) // 480)
            wtype = water_leaf.type[::stride, ::stride]
            wweight = water_leaf.weight[::stride, ::stride]
            rgb = np.zeros((*wtype.shape, 3), dtype=np.float32)
            rgb[wtype == kvwater.TYPE_LAKE] = (0.10, 0.35, 0.85)
            rmask = wtype == kvwater.TYPE_RIVER
            inten = 0.4 + 0.6 * (wweight.astype(np.float32) / 8.0)
            rgb[..., 0][rmask] = 0.0
            rgb[..., 1][rmask] = 0.7 * inten[rmask]
            rgb[..., 2][rmask] = 1.0 * inten[rmask]
            prev_cols[1].image(np.clip(rgb, 0, 1), caption="Water (blue=lake, cyan=river)",
                               clamp=True, width=260)
            st.caption(f"Water surface: {lake_count} lake(s); per-pixel surface packed "
                       f"in surface.atlas (u16 m.o.h.). Lakes use NVE hoyde; rivers "
                       f"sit on the DTM channel raised by stream order "
                       f"(× {river_depth_scale:g}). {kvwater.WATER_ATTRIBUTION}.")
            st.caption(f"Lake beds carved flat {lake_max_depth:.0f} m below the known "
                       f"NVE surface ({lake_ramp_radius:.0f} m shore bevel).")

            wv = res.manifest.get("water_vector")
            if wv:
                r, lk, jn = wv["rivers"], wv["lakes"], wv["junctions"]
                st.caption(
                    f"Vectors: {r['segments']} river segments / {r['vertices']} "
                    f"vertices ({r['bytes'] / 1e6:.1f} MB, upstream→downstream, "
                    f"bed Z + surface level per vertex) · {lk['count']} lake "
                    f"records ({lk['with_authored_level']} with an authored level) "
                    f"· {jn['count']} junctions ({jn['inflow']} in / "
                    f"{jn['outflow']} out) · classes in water_id.atlas.")

                rep = wv["validation"]
                with st.expander("Water network validation (advisory)"):
                    st.markdown(
                        f"""
- **Descent:** {rep['descent_rising_vertices']:,} of
  {rep['descent_vertices_checked']:,} vertices rise going downstream
  ({rep['descent_rising_pct']:.2f} %), worst
  {rep['descent_worst_rise_m']:.2f} m, across
  {rep['descent_segments_with_rise']:,} segments.
- **Flow direction:** {rep['flowdir_disagreements']:,} of
  {rep['flowdir_segments_checked']:,} segments
  ({rep['flowdir_disagreement_pct']:.2f} %) run uphill by sampled Z —
  {rep['flowdir_disagreement_pct_lake_touching']:.2f} % of lake-touching
  segments vs {rep['flowdir_disagreement_pct_non_lake']:.2f} % of the rest.
- **Connectivity:** {rep['connectivity_orphan_segments']:,} orphans,
  {rep['connectivity_source_segments']:,} sources,
  {rep['connectivity_sink_segments']:,} sinks.
""")
                    st.caption("Nothing here is corrected. Source vertex order stays "
                               "authoritative and river Z is left alone; the runtime "
                               "burn's running-minimum enforces descent. Disagreement "
                               "concentrated in lake-touching segments would suggest a "
                               "real ordering problem rather than DTM noise.")

        st.download_button("Download tiles + manifest (zip)", buf,
                           file_name="kvterrain_tiles.zip", mime="application/zip")
        with st.expander("manifest.json"):
            st.json(res.manifest)

    except Exception as e:
        st.error(f"Build failed: {e}")
        st.caption("If this is a live fetch, the sandbox/network may not reach "
                   "hoydedata.no. Try Demo mode to verify the pipeline, then run "
                   "the tool from a machine with internet access.")