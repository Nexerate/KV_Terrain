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
    river_depth_scale = st.slider("River carve depth ×", 0.25, 4.0, 1.0, 0.25,
                                  disabled=not want_water,
                                  help="Scales the trench carved UNDER each river (by "
                                       "stream order). A river's water surface always "
                                       "sits on the terrain, so this trench is the whole "
                                       "water column — bigger reads more solid. Sized "
                                       "like the lake carve so it survives the renderer's "
                                       "height compression.")
    river_bank_tolerance = st.slider("River bank tolerance (m)", 0.5, 8.0, 2.0, 0.5,
                                     disabled=not want_water,
                                     help="How far a channel sample may stand above its "
                                          "river's water level and still hold water. The "
                                          "water surface is level across a channel, so "
                                          "this is what stops a thin film of water "
                                          "climbing a cliff the rasterised channel ran "
                                          "past. Lower = water stays in the channel; "
                                          "higher = wider, more forgiving channels.")
    lake_shore_slope = st.slider("Lake shore slope (m depth / m shore)",
                                 0.1, 2.0, 1.0, 0.1, disabled=not want_water,
                                 help="The lake bed descends on a STRAIGHT ramp at this "
                                      "slope until it reaches the carve depth. At 1.0 a "
                                      "20 m carve is reached 20 m from shore — four "
                                      "texels at 5 m spacing. Lower = gentler sides.")
    lake_max_depth = st.slider("Lake carve depth (m, real)", 1.0, 40.0, 20.0, 1.0,
                               disabled=not want_water,
                               help="Deepest the lake bed goes below the authored "
                                    "surface, reached only by lakes big enough to ramp "
                                    "that far. Must exceed the renderer's opaque "
                                    "threshold AFTER any height compression (e.g. 20 m "
                                    "real → 4 units at 1:5).")
    lake_min_depth = st.slider("Lake minimum depth (m)", 0.0, 10.0, 2.0, 0.5,
                               disabled=not want_water,
                               help="Every lake reaches at least this depth at its "
                                    "deepest sample, however small it is, so ponds "
                                    "don't render as dry ground.")
    lake_snap_px = st.slider("Shoreline snap (texels)", 0, 4, 2, 1,
                             disabled=not want_water,
                             help="How far outside its polygon a lake may claim samples "
                                  "that are still its own flat water surface in the DTM. "
                                  "The NVE outline and the LiDAR block don't align to the "
                                  "pixel, and the leftover ring renders as a raised rim "
                                  "tracing the true shoreline just outside the water.")
    fill_lake_holes = st.toggle("Cover lake islands with the water surface", value=True,
                                disabled=not want_water,
                                help="Islands (polygon holes) keep their own terrain, "
                                     "which hides the water again — but the surface runs "
                                     "underneath continuously, which removes the one-texel "
                                     "dry moat the polygon/raster misalignment leaves "
                                     "around every island.")
    estimate_lake_levels = st.toggle("Estimate missing lake levels", value=True,
                                     disabled=not want_water,
                                     help="NVE leaves 'hoyde' blank on a lot of small "
                                          "lakes — a quarter of them in Lierne. LiDAR "
                                          "reports the water surface as terrain, so the "
                                          "level is read from the DTM inside the polygon "
                                          "and used as if it were authored. Off: those "
                                          "lakes stay flat, dry ground.")
    lake_perimeter_cap = st.toggle("Cap estimated levels at the surrounding terrain",
                                   value=True, disabled=not want_water,
                                   help="For lakes NVE gives no 'hoyde' for, cap the "
                                        "estimated level at the height of the land ring "
                                        "just outside the polygon, so the lake cannot "
                                        "end up standing above the terrain around it. A "
                                        "published hoyde is always used exactly as "
                                        "published and is never capped.")
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
def _bbox_from_drawing(state):
    """(lon_min, lat_min, lon_max, lat_max) of the current rectangle, or None.

    `all_drawings` is preferred over `last_active_drawing` because it is what
    reflects an EDIT: after the rectangle is dragged or resized, the active
    drawing can still be the shape as it was first drawn.
    """
    drawings = (state or {}).get("all_drawings") or []
    draw = drawings[-1] if drawings else ((state or {}).get("last_active_drawing"))
    if not draw:
        return None
    coords = draw["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def _plan_outline_latlon(plan):
    """The plan's padded UTM extent as a lat/lon ring, for drawing on the map."""
    from pyproj import Transformer

    tr = Transformer.from_crs(f"EPSG:{plan.epsg}", "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = plan.bbox_utm
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return [(lat, lon) for lon, lat in (tr.transform(x, y) for x, y in ring)]


# THE MAP SPEC MUST NOT CHANGE BETWEEN RERUNS. streamlit-folium keys the component
# on a hash of the map's generated leaflet JS (`generate_js_hash`), so anything that
# alters that JS — adding a layer, or feeding the user's current centre/zoom back
# into folium.Map — remounts the iframe, and a remount wipes the rectangle the user
# drew and resets the view. The padded-extent outline is therefore passed through
# `feature_group_to_add`, which st_folium sends as a SEPARATE argument and applies
# to the live map, and the view is left to the component to keep.
st.session_state.setdefault("bbox", None)

# Plan for the rectangle as it stood at the end of the last run: this is what the
# padded-extent overlay is drawn from.
plan = None
if st.session_state.bbox:
    plan = core.plan_grid(*st.session_state.bbox, spacing_m=spacing_m,
                          tile_cells=tile_cells, epsg=epsg)

col_map, col_info = st.columns([3, 2])

with col_map:
    m = folium.Map(location=[61.3, 8.3], zoom_start=6, tiles="OpenStreetMap")
    Draw(
        export=False,
        draw_options={"rectangle": True, "polygon": False, "polyline": False,
                      "circle": False, "marker": False, "circlemarker": False},
        # Editing is ON so a rectangle can be dragged and resized after it is
        # drawn: pick the toolbar's edit (pencil) tool, drag the rectangle or its
        # corner handles, then Save. That is the point of showing the padded
        # extent — you can watch it overshoot a border or the data coverage and
        # slide the rectangle until it doesn't, instead of deleting and redrawing
        # by eye.
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    overlay = None
    if plan is not None:
        overlay = folium.FeatureGroup(name="export_extent")
        overlay.add_child(folium.Polygon(
            _plan_outline_latlon(plan),
            color="#e8590c", weight=2, dash_array="6,5", fill=True,
            fill_opacity=0.06, fill_color="#e8590c",
            tooltip=(f"Exported area: {plan.width_m:,.0f} × {plan.height_m:,.0f} m "
                     f"({plan.leaf_tiles_x}×{plan.leaf_tiles_y} tiles) — padded out "
                     f"from your rectangle to whole power-of-two tiles")))

    map_state = st_folium(
        m, key="draw_map", height=540, width=None,
        feature_group_to_add=overlay,
        returned_objects=["last_active_drawing", "all_drawings"])

# A new or edited rectangle needs one rerun for the overlay to be rebuilt around
# it. This cannot loop: the map spec is identical across the rerun, so the
# component keeps its state and returns the same rectangle.
bbox = _bbox_from_drawing(map_state)
if bbox != st.session_state.bbox:
    st.session_state.bbox = bbox
    st.rerun()

# --------------------------------------------------------------------------- #
# Plan preview                                                                 #
# --------------------------------------------------------------------------- #
with col_info:
    st.subheader("Plan")
    if plan is None:
        st.info("Draw a rectangle on the map (top-left toolbar) to begin. "
                "The dashed orange outline that appears is the area actually "
                "exported — bigger than what you draw, because the grid is padded "
                "out to whole power-of-two tiles. Use the toolbar's edit tool to "
                "drag or resize the rectangle until that outline sits where you "
                "want it.")
    else:
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
        x1, y1 = ox + plan.width_m, oy + plan.height_m
        st.caption(f"Exported extent (UTM {plan.epsg}): "
                   f"{ox:,.0f}, {oy:,.0f} → {x1:,.0f}, {y1:,.0f}")

        # How much of the exported area is padding, and on which sides — the thing
        # you need in order to nudge the rectangle off a border.
        from pyproj import Transformer as _T
        _tr = _T.from_crs("EPSG:4326", f"EPSG:{plan.epsg}", always_xy=True)
        _dx = [c[0] for c in (_tr.transform(st.session_state.bbox[0], st.session_state.bbox[1]),
                              _tr.transform(st.session_state.bbox[2], st.session_state.bbox[3]))]
        _dy = [c[1] for c in (_tr.transform(st.session_state.bbox[0], st.session_state.bbox[1]),
                              _tr.transform(st.session_state.bbox[2], st.session_state.bbox[3]))]
        pad_w = plan.width_m - (max(_dx) - min(_dx))
        pad_h = plan.height_m - (max(_dy) - min(_dy))
        st.caption(f"Padding beyond your rectangle: +{max(pad_w, 0):,.0f} m E–W, "
                   f"+{max(pad_h, 0):,.0f} m N–S (the padding is added north and "
                   f"east of the SW origin, which snaps down to the sample grid)")
        if want_water:
            wsurf_mb = plan.total_tiles() * (tile_cells + 1) ** 2 * 2 / 1e6
            st.caption(f"+ water surface: surface.atlas "
                       f"(≈ {wsurf_mb:,.1f} MB, u16) from NVE")
            st.caption(f"+ water classes + lake ids: water_id.atlas "
                       f"(≈ {wsurf_mb:,.1f} MB, u16) — same tile offsets as the "
                       f"height atlas")
            st.caption("+ river polylines: rivers.bin + lakes.json + junctions.json "
                       "(vectors, loaded once at runtime — not streamed)")
            st.caption(f"+ lake beds carved on a straight ramp to {lake_max_depth:.0f} m "
                       f"over {lake_max_depth/max(lake_shore_slope,1e-6):.0f} m of shore "
                       f"(min {lake_min_depth:.1f} m)")
            st.caption(f"+ river beds carved under the channel by stream order "
                       f"(× {river_depth_scale:g}); the surface is level across each "
                       f"channel (bank tolerance {river_bank_tolerance:.1f} m)")
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
                "river_bank_tolerance_m": river_bank_tolerance,
                "fill_lake_holes": fill_lake_holes,
                "lake_shore_slope": lake_shore_slope,
                "lake_max_depth_m": lake_max_depth,
                "lake_min_depth_m": lake_min_depth,
                "lake_snap_px": lake_snap_px,
                "ocean_level_m": ocean_level,
                "emit_geojson": emit_geojson,
                "estimate_lake_levels": estimate_lake_levels,
                "lake_perimeter_cap": lake_perimeter_cap,
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
                       f"in surface.atlas (u16 m.o.h.). Lakes use their authored level; "
                       f"river surfaces sit on the terrain, with the channel carved "
                       f"beneath by stream order (× {river_depth_scale:g}). "
                       f"{kvwater.WATER_ATTRIBUTION}.")
            st.caption(f"Lake beds carved on a straight ramp to {lake_max_depth:.0f} m "
                       f"over {lake_max_depth/max(lake_shore_slope,1e-6):.0f} m of shore "
                       f"(min {lake_min_depth:.1f} m).")

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