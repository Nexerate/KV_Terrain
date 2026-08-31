"""
Stage two, on screen: pick a fetched dataset, tune the pipeline, watch it run.

No map dominates this page, because the region is already decided — it was
decided on the Fetch page and frozen on disk. What matters here is what the
pipeline DID to that region, so the space goes to the pictures that answer it:

* **Depth** (`water_surface − terrain`) is the panel to watch. It is exactly what
  the runtime renders, so a lake that comes out as a dry pan or a river that
  climbs a cliff shows up here and nowhere else.
* **Carve delta** (`fetched − processed`) shows how much ground was actually
  removed, which is the direct read-out of the carve sliders.
* The **detail inspector** shows any window at native resolution, because a
  shoreline judged from a 900-px thumbnail of a 4097-px lattice is not judged.

Everything on this page is free to re-run. That is the entire point of the split.
"""
from __future__ import annotations

import io
import os
import time
import zipfile

import numpy as np
import streamlit as st

from kvterrain import dataset as ds_mod, preview as kvpreview, process as kvprocess
from . import widgets as W

PREVIEW_PX = 900          # long edge of the overview panels
DETAIL_PX = 512           # default window of the native-resolution inspector


# --------------------------------------------------------------------------- #
# Settings                                                                     #
# --------------------------------------------------------------------------- #

def _sidebar(ds: ds_mod.Dataset) -> dict:
    """Every post-processing knob. All of them re-run for free."""
    s = ds.summary()
    with st.sidebar:
        st.header("Post-processing")
        st.caption("Nothing here touches the network. Change anything, run again.")

        valid = s["valid_tile_cells"] or [s["tile_cells"]]
        tile_cells = st.selectbox(
            "Tile cells", valid, index=valid.index(s["tile_cells"])
            if s["tile_cells"] in valid else len(valid) - 1,
            help="Re-tiles the SAME fetched samples. Fewer cells per tile = more, "
                 "smaller tiles and one more pyramid level.")

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
        st.subheader("Water")
        if not ds.has_water:
            st.info("This dataset has no NVE vectors — re-fetch the region with "
                    "water on to enable these.")
            want_water = False
        else:
            want_water = st.toggle("Process rivers & lakes", value=True)

        d = not want_water
        river_width_scale = st.slider(
            "River width scale", 0.25, 4.0, 1.0, 0.25, disabled=d,
            help="Scales modelled channel widths → how many pixels each river seeds.")
        river_depth_scale = st.slider(
            "River carve depth ×", 0.25, 4.0, 1.0, 0.25, disabled=d,
            help="Scales the trench carved UNDER each river (by stream order). A "
                 "river's water surface always sits on the terrain, so this trench "
                 "is the whole water column — bigger reads more solid. Sized like "
                 "the lake carve so it survives the renderer's height compression.")
        river_bank_tolerance = st.slider(
            "River bank tolerance (m)", 0.5, 8.0, 2.0, 0.5, disabled=d,
            help="How far a channel sample may stand above its river's water level "
                 "and still hold water. The water surface is level across a "
                 "channel, so this is what stops a thin film of water climbing a "
                 "cliff the rasterised channel ran past. Lower = water stays in the "
                 "channel; higher = wider, more forgiving channels.")
        lake_shore_slope = st.slider(
            "Lake shore slope (m depth / m shore)", 0.1, 2.0, 1.0, 0.1, disabled=d,
            help="The lake bed descends on a STRAIGHT ramp at this slope until it "
                 "reaches the carve depth. At 1.0 a 20 m carve is reached 20 m from "
                 "shore — four texels at 5 m spacing. Lower = gentler sides.")
        lake_max_depth = st.slider(
            "Lake carve depth (m, real)", 1.0, 40.0, 20.0, 1.0, disabled=d,
            help="Deepest the lake bed goes below the authored surface, reached only "
                 "by lakes big enough to ramp that far. Must exceed the renderer's "
                 "opaque threshold AFTER any height compression (e.g. 20 m real → "
                 "4 units at 1:5).")
        lake_min_depth = st.slider(
            "Lake minimum depth (m)", 0.0, 10.0, 2.0, 0.5, disabled=d,
            help="Every lake reaches at least this depth at its deepest sample, "
                 "however small it is, so ponds don't render as dry ground.")
        lake_snap_px = st.slider(
            "Shoreline snap (texels)", 0, 4, 2, 1, disabled=d,
            help="How far outside its polygon a lake may claim samples that are "
                 "still its own flat water surface in the DTM. The NVE outline and "
                 "the LiDAR block don't align to the pixel, and the leftover ring "
                 "renders as a raised rim tracing the true shoreline just outside "
                 "the water.")
        fill_lake_holes = st.toggle(
            "Cover lake islands with the water surface", value=True, disabled=d,
            help="Islands (polygon holes) keep their own terrain, which hides the "
                 "water again — but the surface runs underneath continuously, which "
                 "removes the one-texel dry moat the polygon/raster misalignment "
                 "leaves around every island.")
        estimate_lake_levels = st.toggle(
            "Estimate missing lake levels", value=True, disabled=d,
            help="NVE leaves 'hoyde' blank on a lot of small lakes — a quarter of "
                 "them in Lierne. LiDAR reports the water surface as terrain, so the "
                 "level is read from the DTM inside the polygon and used as if it "
                 "were authored. Off: those lakes stay flat, dry ground.")
        lake_perimeter_cap = st.toggle(
            "Cap estimated levels at the surrounding terrain", value=True, disabled=d,
            help="For lakes NVE gives no 'hoyde' for, cap the estimated level at the "
                 "height of the land ring just outside the polygon, so the lake "
                 "cannot end up standing above the terrain around it. A published "
                 "hoyde is always used exactly as published and is never capped.")
        ocean_level = st.number_input(
            "Ocean level (m.o.h.)", -50.0, 50.0, 0.0, 1.0, disabled=d,
            help="Kartverket heights are metres above sea level, so 0 is real sea "
                 "level. Ocean is flood-filled inward from the map edges, not "
                 "thresholded, so inland below-sea-level ground is never "
                 "mislabelled as sea.")

        st.divider()
        st.subheader("River polylines")
        st.caption("The burn needs flow direction, connectivity and the ramp profile "
                   "— none of which survive rasterisation. These are emitted as "
                   "vectors beside the rasters and loaded once at runtime.")
        river_vertex_stride = st.slider(
            "Vertex stride (m, 0 = leaf spacing)", 0.0, 32.0, 0.0, 1.0, disabled=d,
            help="Polylines are densified to this spacing before the DTM is sampled "
                 "beneath them. Denser follows the real channel more closely; "
                 "coarser shrinks rivers.bin.")
        emit_geojson = st.toggle(
            "Also write rivers.geojson", value=False, disabled=d,
            help="Debug sidecar for QGIS. rivers.bin is the runtime format — this is "
                 "for when you need to LOOK at a validation warning.")

    water_opts = None
    if want_water:
        water_opts = {
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

    return {
        "tile_cells": tile_cells,
        "nodata_fill_m": nodata_fill,
        "height_min": hmin,
        "height_max": hmax,
        "include_water": want_water,
        "water_opts": water_opts,
    }


# --------------------------------------------------------------------------- #
# Result rendering                                                             #
# --------------------------------------------------------------------------- #

def _overview_panels(state: dict) -> None:
    spacing = state["spacing_m"]
    s = kvpreview.stride_for(state["terrain"].shape, PREVIEW_PX)
    terr = state["terrain"][::s, ::s]
    raw = state["raw"][::s, ::s]
    base = kvpreview.terrain_rgb(terr, spacing * s,
                                 hmin=state["hmin"], hmax=state["hmax"])

    specs = [(base, f"Terrain after processing — hillshaded, 1:{s} of the lattice")]

    if state.get("wtype") is not None:
        wtype = state["wtype"][::s, ::s]
        weight = state["weight"][::s, ::s]
        specs.append((kvpreview.water_class_rgb(wtype, weight, base=base * 0.45),
                      "Water classes — blue lake, cyan river (brighter = higher "
                      "stream order)"))

    if state.get("surface") is not None:
        surf = state["surface"][::s, ::s]
        depth = np.where(np.isfinite(surf), surf - terr, np.nan)
        rgb, dmax = kvpreview.depth_rgb(depth, base=base * 0.55)
        specs.append((rgb, f"Water depth = surface − terrain (0 → {dmax:.1f} m). "
                           f"This is what the runtime renders."))
        specs.append((kvpreview.surface_rgb(surf, base=base * 0.35),
                      "Water-surface elevation (m.o.h.), ramped over the water's own "
                      "range"))

    delta = raw - terr
    if np.nanmax(np.abs(np.nan_to_num(delta))) > 1e-3:
        rgb, dscale = kvpreview.delta_rgb(delta, base=base * 0.55)
        specs.append((rgb, f"Ground changed = fetched − processed, ±{dscale:.1f} m. "
                           f"Blue removed (carves), orange added (lake void repair "
                           f"and the shoreline snap)."))

    W.panel_grid(specs, columns=2)


def _detail_inspector(state: dict) -> None:
    """Any window of the lattice at NATIVE resolution — the only way to judge a
    shoreline or a one-texel channel honestly."""
    H, W_ = state["terrain"].shape
    st.caption(f"The overview panels above are strided down from {W_:,} × {H:,} "
               f"samples. This one is 1:1.")

    c1, c2, c3 = st.columns([2, 2, 1])
    win = c3.select_slider("Window (samples)", [256, 384, 512, 768, 1024], DETAIL_PX)
    win = int(min(win, H, W_))
    cx = c1.slider("Centre — east", win // 2, max(W_ - win // 2, win // 2), W_ // 2)
    cy = c2.slider("Centre — north", win // 2, max(H - win // 2, win // 2), H // 2)

    # Slider y is measured from the NORTH edge, matching the row order of every
    # array in this pipeline, so "north" on the slider is up in the picture.
    r0 = int(np.clip(cy - win // 2, 0, max(H - win, 0)))
    c0 = int(np.clip(cx - win // 2, 0, max(W_ - win, 0)))
    sl = (slice(r0, r0 + win), slice(c0, c0 + win))

    terr = state["terrain"][sl]
    base = kvpreview.terrain_rgb(terr, state["spacing_m"])
    specs = [(base, f"Terrain · rows {r0}–{r0 + win}, cols {c0}–{c0 + win}")]
    if state.get("surface") is not None:
        surf = state["surface"][sl]
        depth = np.where(np.isfinite(surf), surf - terr, np.nan)
        rgb, dmax = kvpreview.depth_rgb(depth, base=base * 0.55)
        specs.append((rgb, f"Depth (0 → {dmax:.2f} m)"))
    W.panel_grid(specs, columns=2)

    ox = state["origin_x"] + c0 * state["spacing_m"]
    oy = state["origin_y"] + (H - 1 - (r0 + win)) * state["spacing_m"]
    st.caption(f"SW corner of this window in UTM {state['epsg']}: "
               f"{ox:,.0f}, {oy:,.0f} · {win * state['spacing_m']:,.0f} m across")


def _reports(state: dict) -> None:
    man = state["manifest"]
    wv = man.get("water_vector")
    wm = man.get("water_surface", {})

    tabs = st.tabs(["Lake levels", "Network validation", "Stage timings", "manifest.json"])

    with tabs[0]:
        ll = wm.get("lake_levels")
        if not ll:
            st.caption("No water in this run.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lakes", f"{wm.get('lake_count', 0):,}")
            c2.metric("From NVE hoyde", f"{ll['levels_from_nve']:,}",
                      help="Used verbatim. The DTM does not get a vote on a "
                           "published level.")
            c3.metric("Estimated", f"{ll['levels_estimated']:,}",
                      help="Read off the LiDAR water surface inside the polygon, "
                           "because NVE published no hoyde.")
            c4.metric("Unresolved", f"{ll['levels_unresolved']:,}",
                      help="Neither source gave a level. These are left UNCARVED — "
                           "carving them is how lakes became dry pits.")
            capped = ll.get("levels_capped_to_perimeter", 0)
            if capped:
                st.caption(f"{capped:,} estimated levels were capped at the "
                           f"surrounding terrain (max drop "
                           f"{ll.get('levels_cap_max_drop_m', 0.0):.2f} m).")
            if ll["levels_unresolved"]:
                st.warning(f"{ll['levels_unresolved']:,} lakes have no level from "
                           f"either source and were left as flat dry ground. Turn "
                           f"on *Estimate missing lake levels* to fill them from "
                           f"the DTM.")
            snap = wm.get("lake_bathymetry", {}).get("shoreline_snap")
            if snap:
                st.caption(f"Shoreline snap: {snap['samples_added']:,} samples across "
                           f"{snap['lakes_grown']:,} lakes were still the lake's own "
                           f"flat water surface outside its polygon and are now part "
                           f"of it.")

    with tabs[1]:
        if not wv:
            st.caption("No water in this run.")
        else:
            r, lk, jn = wv["rivers"], wv["lakes"], wv["junctions"]
            c1, c2, c3 = st.columns(3)
            c1.metric("River segments", f"{r['segments']:,}")
            c2.metric("Vertices", f"{r['vertices']:,}")
            c3.metric("Junctions", f"{jn['count']:,}",
                      help=f"{jn['inflow']:,} inflow / {jn['outflow']:,} outflow")
            rep = wv["validation"]
            st.markdown(
                f"""
- **Descent:** {rep['descent_rising_vertices']:,} of
  {rep['descent_vertices_checked']:,} vertices rise going downstream
  ({rep['descent_rising_pct']:.2f} %), worst {rep['descent_worst_rise_m']:.2f} m,
  across {rep['descent_segments_with_rise']:,} segments.
- **Flow direction:** {rep['flowdir_disagreements']:,} of
  {rep['flowdir_segments_checked']:,} segments
  ({rep['flowdir_disagreement_pct']:.2f} %) run uphill by sampled Z —
  {rep['flowdir_disagreement_pct_lake_touching']:.2f} % of lake-touching segments
  vs {rep['flowdir_disagreement_pct_non_lake']:.2f} % of the rest.
- **Connectivity:** {rep['connectivity_orphan_segments']:,} orphans,
  {rep['connectivity_source_segments']:,} sources,
  {rep['connectivity_sink_segments']:,} sinks.
""")
            st.caption("Nothing here is corrected. Source vertex order stays "
                       "authoritative and river Z is left alone; the runtime burn's "
                       "running-minimum enforces descent. Disagreement concentrated "
                       "in lake-touching segments would suggest a real ordering "
                       "problem rather than DTM noise.")

    with tabs[2]:
        secs = state.get("stage_seconds") or {}
        if not secs:
            st.caption("No timings recorded.")
        else:
            labels = {k: lbl for k, lbl, _ in kvprocess.STAGES}
            import pandas as pd
            df = pd.DataFrame(
                {"seconds": list(secs.values())},
                index=[labels.get(k, k) for k in secs],
            ).sort_values("seconds", ascending=False)
            st.bar_chart(df, horizontal=True)
            st.caption(f"Total {sum(secs.values()):.1f}s in the pipeline. Reach for "
                       f"this when you want to know what a re-run actually costs.")

    with tabs[3]:
        st.json(man)


def _history() -> None:
    hist = st.session_state.get("process_history") or []
    if len(hist) < 2:
        return
    import pandas as pd
    with st.expander(f"Run history ({len(hist)}) — what changed between runs"):
        st.dataframe(pd.DataFrame(hist), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Page                                                                         #
# --------------------------------------------------------------------------- #

def render() -> None:
    st.title("Process")
    st.caption("Pick a fetched dataset, tune the pipeline, run it. Every setting on "
               "this page is free to change — nothing here goes back to the network.")

    root = st.session_state.get("dataset_root", ds_mod.DEFAULT_ROOT)
    root = st.text_input("Dataset folder", value=root,
                         help="Where the Fetch page stores datasets.")
    st.session_state["dataset_root"] = root

    datasets = ds_mod.list_datasets(root)
    if not datasets:
        st.info(f"No datasets in `{root}`. Fetch one on the **Fetch** page first — "
                f"or run `kvterrain fetch --bbox … --spacing 5 --demo` for a "
                f"synthetic one to try this page on.")
        W.attribution_footer()
        return

    # ---- pick ------------------------------------------------------------- #
    roots = [d.root for d in datasets]
    prev = st.session_state.get("process_selected")
    idx = roots.index(prev) if prev in roots else 0
    chosen_root = st.selectbox(
        "Dataset", roots, index=idx,
        format_func=lambda r: W.dataset_caption(
            datasets[roots.index(r)].summary()))
    st.session_state["process_selected"] = chosen_root
    ds = datasets[roots.index(chosen_root)]

    W.dataset_card(ds)
    with st.expander("Where it is"):
        W.dataset_map(datasets, highlight=ds)

    opts = _sidebar(ds)

    # ---- run -------------------------------------------------------------- #
    st.divider()
    default_out = os.path.join(ds_mod.PROJECT_ROOT, "exports", ds.slug)
    out_col, btn_col = st.columns([3, 1])
    out_dir = out_col.text_input(
        "Output folder", value=st.session_state.get("process_out", default_out),
        help="Where the Unity export is written: manifest.json, heights.atlas, "
             "surface.atlas, water_id.atlas, rivers.bin, lakes.json, junctions.json. "
             "Re-running overwrites it.")
    st.session_state["process_out"] = out_dir
    btn_col.write("")
    btn_col.write("")
    go = btn_col.button("Run post-processing", type="primary", width="stretch")

    if go:
        plan = ds.plan(opts["tile_cells"])
        st.caption(f"Building {plan.total_tiles():,} tiles across {plan.num_levels} "
                   f"levels ({plan.leaf_tiles_x}×{plan.leaf_tiles_y} leaves of "
                   f"{plan.tile_cells} cells).")
        bar = st.progress(0.0, text="starting…")
        started = time.time()

        def on_progress(frac, label):
            bar.progress(min(max(frac, 0.0), 1.0),
                         text=f"{label}  ·  {time.time() - started:.0f}s elapsed")

        try:
            os.makedirs(out_dir, exist_ok=True)
            res = kvprocess.process_dataset(
                ds, out_dir,
                tile_cells=opts["tile_cells"],
                nodata_fill_m=opts["nodata_fill_m"],
                height_min=opts["height_min"], height_max=opts["height_max"],
                include_water=opts["include_water"],
                water_opts=opts["water_opts"],
                progress=on_progress,
            )
        except Exception as e:      # noqa: BLE001 — surfaced, not swallowed
            bar.empty()
            st.error(f"Processing failed: {e}")
            st.exception(e)
            return

        elapsed = time.time() - started
        bar.progress(1.0, text=f"done in {elapsed:.0f}s")

        wg = res.water_grid
        st.session_state["process_result"] = {
            "dataset": ds.root,
            "out_dir": out_dir,
            "manifest": res.manifest,
            "tiles": res.tiles_written,
            "hmin": res.height_min,
            "hmax": res.height_max,
            "spacing_m": plan.spacing_m,
            "origin_x": plan.origin_x,
            "origin_y": plan.origin_y,
            "epsg": plan.epsg,
            "elapsed": elapsed,
            "stage_seconds": res.stage_seconds,
            "terrain": res.terrain_leaf,
            "surface": res.water_surface,
            "wtype": wg.type if wg is not None else None,
            "weight": wg.weight if wg is not None else None,
            "raw": np.asarray(ds.heights()),
        }
        wo = opts["water_opts"] or {}
        st.session_state.setdefault("process_history", []).append({
            "run": len(st.session_state.get("process_history", [])) + 1,
            "dataset": ds.name,
            "tile_cells": opts["tile_cells"],
            "water": bool(opts["include_water"]),
            "lake_depth_m": wo.get("lake_max_depth_m"),
            "lake_slope": wo.get("lake_shore_slope"),
            "lake_min_m": wo.get("lake_min_depth_m"),
            "snap_px": wo.get("lake_snap_px"),
            "river_width": wo.get("width_scale"),
            "river_depth": wo.get("river_depth_scale"),
            "bank_tol_m": wo.get("river_bank_tolerance_m"),
            "seconds": round(elapsed, 1),
        })

    # ---- results ----------------------------------------------------------- #
    state = st.session_state.get("process_result")
    if not state or state["dataset"] != ds.root:
        if state:
            st.info("The result below is from a different dataset — run this one to "
                    "replace it.")
        W.attribution_footer()
        return

    st.divider()
    man = state["manifest"]
    atlas = man["atlas"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tiles", f"{state['tiles']:,}")
    c2.metric("Vertical range", f"{state['hmin']:.0f} – {state['hmax']:.0f} m")
    c3.metric("Atlas", W.fmt_bytes(atlas["height_bytes"]),
              help="heights.atlas; surface.atlas and water_id.atlas are the same size")
    c4.metric("Took", f"{state['elapsed']:.0f} s")
    st.success(f"Written to `{state['out_dir']}`")

    st.subheader("What the pipeline did")
    _overview_panels(state)

    with st.expander("Detail inspector — native resolution", expanded=False):
        _detail_inspector(state)

    st.subheader("Reports")
    _reports(state)
    _history()

    # ---- download ----------------------------------------------------------- #
    st.divider()
    if st.button("Package this export as a zip"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for base, _, files in os.walk(state["out_dir"]):
                for fn in files:
                    fp = os.path.join(base, fn)
                    zf.write(fp, os.path.relpath(fp, state["out_dir"]))
        buf.seek(0)
        st.download_button("Download tiles + manifest (zip)", buf,
                           file_name=f"{ds.slug}_export.zip", mime="application/zip")
        st.caption("The files are already on disk at the output folder above — this "
                   "is only for moving them somewhere else.")

    W.attribution_footer()
