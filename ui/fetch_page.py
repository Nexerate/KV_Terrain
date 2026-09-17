"""
Stage one, on screen: draw a rectangle, fetch it, store it.

This page is the only one that talks to the network, and it deliberately offers
nothing else — no carve settings, no packing options, no preview of an export.
Its whole job is to turn an area of Norway into a dataset on disk that the
process page can then chew on as many times as you like.
"""
from __future__ import annotations

import time

import streamlit as st

from kvterrain import core, dataset as ds_mod, fetch as kvfetch
from . import widgets as W


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


def render() -> None:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium

    st.title("Fetch")
    st.caption("Draw a rectangle over Norway, pull the Kartverket height lattice and "
               "the NVE water vectors that intersect it, and store the result as a "
               "dataset. Nothing here carves, rasterises or packs anything — that is "
               "the Process page, and it never needs the network again.")

    # ---------------- sidebar: what defines the fetch ---------------------- #
    with st.sidebar:
        st.header("Fetch parameters")
        st.caption("Everything on this page changes what has to come off the "
                   "network. Carve and packing settings live on the Process page.")

        source_kind = st.selectbox("Surface", ["DTM (bare earth)", "DOM (surface)"], 0)
        source_kind = "DTM" if source_kind.startswith("DTM") else "DOM"

        spacing_m = st.number_input(
            "Leaf spacing (m / texel)", 0.25, 50.0, 5.0, 0.25,
            help="The finest LOD in the export, and the lattice everything else is "
                 "derived from. Don't request finer than the native data "
                 "(DTM1 = 1 m). This is the one setting you cannot change later "
                 "without re-fetching.")
        tile_cells = st.selectbox(
            "Tile cells (samples = cells + 1)", [64, 128, 256], 1,
            help="Used to pad the region out to whole power-of-two tiles. The "
                 "Process page can re-tile the same samples to any smaller power "
                 "of two without a re-fetch, so this mostly decides how much the "
                 "region is padded.")

        epsg_choice = st.selectbox(
            "UTM zone (EUREF89)", ["auto", "25832 (W/S)", "25833 (most)", "25835 (NE)"], 0)
        epsg = None if epsg_choice == "auto" else int(epsg_choice.split()[0])

        max_fetch_px = st.select_slider(
            "Max fetch px / request", [1024, 2048, 4096, 8192], 2048,
            help="How the region is chunked into exportImage calls. Smaller = more "
                 "requests but less to lose on a timeout.")

        st.divider()
        st.subheader("Water (NVE)")
        want_water = st.toggle(
            "Fetch rivers & lakes", value=True,
            help="Pulls NVE Elvenett + Innsjødatabase for the region and stores the "
                 "raw vectors. Cheap next to the height raster, and the Process page "
                 "can always ignore them — but they cannot be added later without "
                 "re-fetching, so leaving this on is nearly always right.")
        main_rivers = st.toggle(
            "Include main rivers (hovedelv)", value=True, disabled=not want_water,
            help="A second NVE query. Its geometry is never exported as polylines; "
                 "it exists so the raster burn upgrades trunk-river size class.")

        st.divider()
        st.subheader("Storage")
        modes = list(ds_mod.COMPRESSION_MODES)
        compression = st.selectbox(
            "Height lattice on disk", modes,
            index=modes.index(ds_mod.DEFAULT_COMPRESSION),
            format_func=lambda m: f"{m} — {ds_mod.COMPRESSION_MODES[m][1]}",
            help="float32 elevations barely gzip (~17%) because the mantissa "
                 "bytes are noise. Shuffling the bytes into planes first gets it "
                 "to ~60% of raw, losslessly, and costs milliseconds to read "
                 "back. The quantised modes round first and save roughly half — "
                 "still finer than Kartverket's own DTM accuracy, but lossy.")
        if ds_mod.COMPRESSION_MODES[compression][0] is not None:
            st.caption("⚠️ Lossy. The stored lattice will not round-trip exactly. "
                       "Fine for terrain you are going to carve anyway, but it is "
                       "a one-way door for this dataset.")

        st.divider()
        demo = st.toggle(
            "Demo mode (synthetic, no network)", value=False,
            help="Builds the dataset from a synthetic surface and a synthetic "
                 "river + lake. The vectors still go through the real raw-GeoJSON "
                 "parse, so an offline dry run exercises the same code a live one "
                 "does.")

        st.divider()
        root = st.text_input(
            "Dataset folder", value=ds_mod.DEFAULT_ROOT,
            help="Where fetched datasets are stored. The Process page reads this "
                 "same folder.")

    # ---------------- map ---------------------------------------------------- #
    # THE MAP SPEC MUST NOT CHANGE BETWEEN RERUNS. streamlit-folium keys the
    # component on a hash of the map's generated leaflet JS (`generate_js_hash`),
    # so anything that alters that JS — adding a layer, or feeding the user's
    # current centre/zoom back into folium.Map — remounts the iframe, and a
    # remount wipes the rectangle the user drew and resets the view. The
    # padded-extent outline is therefore passed through `feature_group_to_add`,
    # which st_folium sends as a SEPARATE argument and applies to the live map,
    # and the view is left to the component to keep.
    st.session_state.setdefault("fetch_bbox", None)

    plan = None
    if st.session_state.fetch_bbox:
        plan = core.plan_grid(*st.session_state.fetch_bbox, spacing_m=spacing_m,
                              tile_cells=tile_cells, epsg=epsg)

    col_map, col_info = st.columns([3, 2])

    with col_map:
        m = folium.Map(location=[61.3, 8.3], zoom_start=6, tiles="OpenStreetMap")
        Draw(
            export=False,
            draw_options={"rectangle": True, "polygon": False, "polyline": False,
                          "circle": False, "marker": False, "circlemarker": False},
            # Editing is ON so a rectangle can be dragged and resized after it is
            # drawn: pick the toolbar's edit (pencil) tool, drag the rectangle or
            # its corner handles, then Save. That is the point of showing the
            # padded extent — you can watch it overshoot a border or the data
            # coverage and slide the rectangle until it doesn't, instead of
            # deleting and redrawing by eye.
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        overlay = None
        if plan is not None:
            overlay = folium.FeatureGroup(name="export_extent")
            overlay.add_child(folium.Polygon(
                ds_mod.Lattice.from_plan(plan).outline_latlon(),
                color="#e8590c", weight=2, dash_array="6,5", fill=True,
                fill_opacity=0.06, fill_color="#e8590c",
                tooltip=(f"Fetched area: {plan.width_m:,.0f} × {plan.height_m:,.0f} m "
                         f"— padded out from your rectangle to whole power-of-two "
                         f"tiles")))

        map_state = st_folium(
            m, key="draw_map", height=540, width=None,
            feature_group_to_add=overlay,
            returned_objects=["last_active_drawing", "all_drawings"])

    # A new or edited rectangle needs one rerun for the overlay to be rebuilt
    # around it. This cannot loop: the map spec is identical across the rerun, so
    # the component keeps its state and returns the same rectangle.
    bbox = _bbox_from_drawing(map_state)
    if bbox != st.session_state.fetch_bbox:
        st.session_state.fetch_bbox = bbox
        st.rerun()

    # ---------------- plan preview ------------------------------------------ #
    with col_info:
        st.subheader("What will be fetched")
        if plan is None:
            st.info("Draw a rectangle on the map (top-left toolbar) to begin. "
                    "The dashed orange outline that appears is the area actually "
                    "fetched — bigger than what you draw, because the grid is "
                    "padded out to whole power-of-two tiles. Use the toolbar's "
                    "edit tool to drag or resize the rectangle until that outline "
                    "sits where you want it.")
        else:
            lat = ds_mod.Lattice.from_plan(plan)
            chunks = (-(-plan.samples_x // max_fetch_px)
                      * -(-plan.samples_y // max_fetch_px))
            heights_mb = plan.samples_x * plan.samples_y * 4 / 1e6

            # Metric values stay SHORT: st.metric truncates rather than wraps, and
            # "32,769 x 24,577" silently becomes "32,769 x 1..." on a big region —
            # exactly the region where the number matters. Long facts go in the list.
            c1, c2 = st.columns(2)
            c1.metric("Area", f"{lat.area_km2:,.0f} km²")
            c2.metric("Height lattice", f"{heights_mb:,.0f} MB")
            c1.metric("Fetch requests", f"{chunks:,}")
            c2.metric("Pyramid levels", f"{plan.num_levels}")

            st.markdown(
                f"""
- **Samples:** {plan.samples_x:,} × {plan.samples_y:,} at {plan.spacing_m:g} m
- **CRS:** EPSG:{plan.epsg} · **Region:** {plan.width_m:,.0f} × {plan.height_m:,.0f} m
- **Extent (UTM):** {W.fmt_extent(lat)}
- **Tiling at fetch:** {plan.leaf_tiles_x} × {plan.leaf_tiles_y} tiles of {tile_cells} cells
- **Re-tileable to:** {lat.valid_tile_cells()} cells, no re-fetch
""")
            if heights_mb > 1500:
                st.warning(
                    f"That is ≈ {heights_mb / 1000:,.1f} GB of float32 heights in one "
                    f"dataset, and {chunks:,} separate requests to hoydedata.no. "
                    f"Consider a smaller rectangle or a coarser spacing.")

            # How much of the fetched area is padding, and on which sides — the
            # thing you need in order to nudge the rectangle off a border.
            from pyproj import Transformer as _T
            _tr = _T.from_crs("EPSG:4326", f"EPSG:{plan.epsg}", always_xy=True)
            b = st.session_state.fetch_bbox
            _p0 = _tr.transform(b[0], b[1])
            _p1 = _tr.transform(b[2], b[3])
            pad_w = plan.width_m - abs(_p1[0] - _p0[0])
            pad_h = plan.height_m - abs(_p1[1] - _p0[1])
            st.caption(f"Padding beyond your rectangle: +{max(pad_w, 0):,.0f} m E–W, "
                       f"+{max(pad_h, 0):,.0f} m N–S (added north and east of the SW "
                       f"origin, which snaps down to the sample grid)")
            if plan.fetch_pixels() > 60_000 * 60_000:
                st.warning("Very large area — consider the bulk DTM1 download instead.")

    # ---------------- name + go ---------------------------------------------- #
    st.divider()
    name_col, btn_col = st.columns([3, 1])
    with name_col:
        name = st.text_input(
            "Dataset name",
            value=st.session_state.get("fetch_name", ""),
            placeholder=("Demo" if demo else
                         "leave blank to name it after the municipality at the centre"),
            help="Shown in the Process page's picker. Blank asks Kartverket's "
                 "kommuneinfo API what municipality the region's centre falls in.")
    with btn_col:
        st.write("")
        st.write("")
        go = st.button("Fetch dataset", type="primary", disabled=(plan is None),
                       width="stretch")

    if go and plan is not None:
        st.session_state["fetch_name"] = name
        bar = st.progress(0.0, text="starting…")
        started = time.time()

        def on_progress(frac, label):
            bar.progress(min(max(frac, 0.0), 1.0),
                         text=f"{label}  ·  {time.time() - started:.0f}s elapsed")

        try:
            res = kvfetch.run_fetch(
                plan,
                name=name.strip() or None,
                root=root,
                source_kind=source_kind,
                max_fetch_px=max_fetch_px,
                include_water=want_water,
                include_main_rivers=main_rivers,
                compression=compression,
                demo=demo,
                progress=on_progress,
                request_bbox_lonlat=st.session_state.fetch_bbox,
            )
        except Exception as e:      # noqa: BLE001 — surfaced, not swallowed
            bar.empty()
            st.error(f"Fetch failed: {e}")
            st.caption("If this was a live fetch, the network may not reach "
                       "hoydedata.no or kart.nve.no. Demo mode verifies the "
                       "pipeline offline.")
            return

        bar.progress(1.0, text=f"done in {res.seconds_total:.0f}s")
        d = res.dataset
        hb = d.manifest["heights"]
        if hb.get("uncompressed_bytes"):
            st.caption(
                f"Lattice stored '{d.compression}': "
                f"{W.fmt_bytes(hb['bytes'])} on disk vs "
                f"{W.fmt_bytes(hb['uncompressed_bytes'])} raw "
                f"({100 * hb['bytes'] / hb['uncompressed_bytes']:.0f}%)"
                + ("" if hb.get("lossless") else " — LOSSY"))
        st.success(
            f"Stored **{d.name}** in `{d.root}` — "
            f"{res.seconds_heights:.0f}s of heights + {res.seconds_water:.0f}s of "
            f"water, {W.fmt_bytes(d.bytes_total)} on disk. "
            f"Open **Process** to build an export from it, as many times as you like.")
        # Let the process page land on what was just fetched.
        st.session_state["process_selected"] = d.root
        W.dataset_card(d)

    # ---------------- what is already here ----------------------------------- #
    st.divider()
    existing = ds_mod.list_datasets(root)
    st.subheader(f"Datasets in this folder ({len(existing)})")
    if not existing:
        st.caption(f"Nothing in `{root}` yet.")
    else:
        st.caption("Check here before fetching — the whole point of the split is "
                   "not to download the same block twice.")
        for d in existing:
            st.write("· " + W.dataset_caption(d.summary()))
        W.dataset_map(existing)

    W.attribution_footer()
