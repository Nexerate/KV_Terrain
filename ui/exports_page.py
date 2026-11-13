"""
The third page: open a finished export, look at it, and check it.

Everything shown here is read back out of the exported FILES — the previews are
assembled tile by tile through `core.atlas_tile_offset`, the same arithmetic a
consumer has to implement. That makes this page a test as much as a viewer: a
picture that comes out right is evidence the header and the bytes agree, which a
preview drawn from the in-memory arrays could never be.

Three questions it answers:

  * What is in this export, and which fetch and which settings produced it?
  * Does it look right?
  * Is it intact — dense atlases, a parseable river network, resolvable links —
    and does it still agree with Kartverket at a world coordinate?
"""
from __future__ import annotations


import numpy as np
import streamlit as st

from kvterrain import exports as kvexports, preview as kvpreview
from . import widgets as W

PREVIEW_PX = 1100


def _summary_card(exp: kvexports.Export) -> None:
    s = exp.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Area", f"{s['area_km2']:,.0f} km²")
    c2.metric("Tiles", f"{s['tiles_total']:,}")
    c3.metric("Levels", f"{s['num_levels']}")
    c4.metric("On disk", W.fmt_bytes(s["bytes_total"]))

    lx, ly = s["leaf_tiles"]
    st.markdown(
        f"""
- **Built from dataset:** {f"`{s['dataset']}`" if s['dataset'] else
  "_not recorded — built before provenance, or by `build` in one pass_"}
- **Generated:** {W.fmt_when(s['generated_utc'])} · **Source:** {s['source']}
- **Leaf grid:** {lx} × {ly} tiles of {s['tile_cells']} cells at
  {s['spacing_m']:g} m · **CRS:** EPSG:{s['epsg']}
- **Vertical range:** {s['height_min_m']:.1f} – {s['height_max_m']:.1f} m
  (the R16 range every tile at every level is packed against)
- **Region:** {s['width_m']:,.0f} × {s['height_m']:,.0f} m
- **Path:** `{s['root']}`
""")
    if s["has_water"]:
        st.caption(f"Water: {s['lakes']:,} lakes, {s['river_segments']:,} river "
                   f"segments.")
    else:
        st.caption("Height only — this export has no water products.")


def _settings(exp: kvexports.Export) -> None:
    params = (exp.manifest.get("generator", {}) or {}).get("parameters", {}) or {}
    water = params.get("water")
    left, right = st.columns(2)
    with left:
        st.markdown("**Packing**")
        st.json({k: v for k, v in params.items() if k != "water"}, expanded=True)
    with right:
        st.markdown("**Water**")
        if water:
            st.json(water, expanded=True)
        else:
            st.caption("Built without water.")


def _previews(exp: kvexports.Export) -> None:
    lvl = exp.preview_level(PREVIEW_PX)
    SY, SX = exp.level_shape(lvl)
    spacing = float(exp.manifest["leaf_spacing_m"]) * (2 ** lvl)
    st.caption(f"Read back from the atlas files at pyramid level {lvl} "
               f"({SX:,} × {SY:,} samples, {spacing:g} m per sample), assembled "
               f"tile by tile through the manifest's own offset arithmetic.")

    try:
        h = exp.read_heights(lvl)
    except Exception as e:      # noqa: BLE001 — a bad atlas must say so, not crash
        st.error(f"Could not read the height atlas: {e}")
        return

    hmin, hmax = exp.height_range
    base = kvpreview.terrain_rgb(h, spacing, hmin=hmin, hmax=hmax)
    specs = [(base, f"Terrain, unpacked from heights.atlas over "
                    f"[{hmin:.0f}, {hmax:.0f}] m")]

    if exp.atlas_path("water_id"):
        try:
            cls, lake_id = exp.read_water_id(lvl)
            # decode_water_id's enum is Dry=0, Lake=1, River=2, Ocean=3, which is
            # NOT the raster's TYPE_* numbering — remap rather than trusting the
            # two to coincide.
            rgb = base * 0.45
            rgb[cls == 1] = kvpreview.COLOR_LAKE
            rgb[cls == 2] = kvpreview.COLOR_RIVER
            rgb[cls == 3] = kvpreview.COLOR_OCEAN
            n_lakes = int(len(np.unique(lake_id[cls == 1])))
            specs.append((np.clip(rgb, 0, 1),
                          f"Classes from water_id.atlas — {n_lakes} distinct lake "
                          f"ids at this level"))
        except Exception as e:      # noqa: BLE001
            st.warning(f"water_id atlas unreadable: {e}")

    if exp.atlas_path("surface"):
        try:
            surf = exp.read_surface(lvl)
            depth = np.where(np.isfinite(surf), surf - h, np.nan)
            rgb, dmax = kvpreview.depth_rgb(depth, base=base * 0.55)
            specs.append((rgb, f"Depth = surface.atlas − heights.atlas "
                               f"(0 → {dmax:.1f} m) — computed exactly the way the "
                               f"runtime will"))
            specs.append((kvpreview.surface_rgb(surf, base=base * 0.35),
                          "Water-surface elevation from surface.atlas"))
        except Exception as e:      # noqa: BLE001
            st.warning(f"surface atlas unreadable: {e}")

    W.panel_grid(specs, columns=2)


def _checks(exp: kvexports.Export) -> None:
    st.caption("The offline checks run automatically — they only read the files. "
               "The ground-truth check needs the network, so it is a button.")

    with st.spinner("checking atlases…"):
        rep = kvexports.check_atlas(exp)
    (st.success if rep["ok"] else st.error)(
        f"Atlases: {'PASS' if rep['ok'] else 'FAIL'} — every file must be exactly "
        f"the size its bytes per sample implies ({rep['expected_bytes']:,} bytes "
        f"for a 2-byte atlas). A mismatch means the tile grid was not dense, and a "
        f"reader using pure offset arithmetic would misalign every tile past the gap.")
    for a in rep["atlases"]:
        if a.get("error"):
            st.write(f"- **{a['kind']}** — {a['error']}")
            continue
        mark = "✅" if a["ok"] else "❌"
        st.write(f"- {mark} **{a['kind']}** `{a['file']}` — {a['bytes']:,} bytes "
                 f"({a['bytes_per_sample']} bytes/sample), "
                 f"{a['offsets_checked']} sampled offsets, "
                 f"{len(a['short_reads'])} short reads")

    st.divider()
    with st.spinner("checking the water products…"):
        wrep = kvexports.check_water(exp)
    if wrep.get("skipped"):
        st.info(f"Water checks skipped — {wrep['skipped']}.")
    else:
        (st.success if wrep["ok"] else st.error)(
            f"Water products: {'PASS' if wrep['ok'] else 'FAIL'}")
        r, lk, jn = wrep["rivers"], wrep["lakes"], wrep["junctions"]
        if r:
            st.write(f"- ✅ **rivers.bin** v{r['version']} — {r['segments']:,} "
                     f"segments, {r.get('vertices_walked', 0):,} vertices walked "
                     f"against a header claiming {r['vertices']:,}, "
                     f"{r.get('dangling_downstream', 0)} dangling downstream links")
        if lk:
            st.write(f"- ✅ **lakes.json** — {lk['count']:,} lakes, "
                     f"{lk['estimated_level']:,} with an estimated level, "
                     f"{lk['without_level']:,} with none")
        if jn:
            st.write(f"- ✅ **junctions.json** — {jn['count']:,} junctions, "
                     f"{jn['naming_unknown_lake']} naming a lake not in lakes.json")
        for note in wrep.get("notes", []):
            st.warning(note)
        for err in wrep["errors"]:
            st.error(err)

    st.divider()
    st.markdown("**Depression hierarchy**")
    if not exp.has_hierarchy:
        st.info("No hierarchy in this export.")
    else:
        st.caption("Reads all of level 0 from heights.atlas and labels.atlas, checks "
                   "the node table against them, and recomputes every coarse label "
                   "level — a few seconds on a large export, so it is a button.")
        if st.button("Check hierarchy.bin and labels.atlas", key="hier_go"):
            with st.spinner("checking the hierarchy…"):
                hrep = kvexports.check_hierarchy(exp)
            (st.success if hrep["ok"] else st.error)(
                f"Hierarchy: {'PASS' if hrep['ok'] else 'FAIL'}")
            stt = hrep.get("stats") or {}
            if stt:
                st.write(f"- {stt['nodes']:,} nodes, {stt['leaves']:,} leaves, "
                         f"{stt['minor']:,} minor, {stt['with_lake']:,} matched to a "
                         f"lake, {stt['spilling_off_map']:,} children of the root")
            for err in hrep["errors"]:
                st.error(err)

    st.divider()
    st.markdown("**Ground truth (needs the network)**")
    st.caption("Samples random leaf points and asks Kartverket's point API what "
               "the height there really is. Note the packed heights are the "
               "**carved** bed while the API returns untouched DTM, so a sample "
               "landing in water is *supposed* to sit lower — each row says which "
               "class it hit, and the dry-only median is the number that actually "
               "measures alignment.")
    n = st.slider("Points to sample", 4, 40, 12, 2, key="gt_n")
    if st.button("Check against Kartverket", key="gt_go"):
        bar = st.progress(0.0, text="starting…")
        try:
            gt = kvexports.check_against_point_api(
                exp, n=n, progress=lambda f, m: bar.progress(f, text=m))
        except Exception as e:      # noqa: BLE001
            bar.empty()
            st.error(f"Ground-truth check failed: {e}")
            return
        bar.empty()
        for err in gt["errors"]:
            st.error(err)
        if not gt["samples"]:
            st.warning("No samples came back with a height.")
            return

        import pandas as pd
        df = pd.DataFrame(gt["samples"])
        dry = df[df["water_class"] == "dry"]["abs_error_m"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Median error (dry)",
                  f"{dry.median():.2f} m" if len(dry) else "—",
                  help="The honest alignment number: dry samples only.")
        c2.metric("Median error (all)", f"{gt['median_abs_error_m']:.2f} m")
        c3.metric("R16 quantisation", f"{gt['quantisation_m']:.3f} m",
                  help="The floor: no export can beat its own packing precision.")
        st.dataframe(df, width="stretch", hide_index=True)


def _files(exp: kvexports.Export) -> None:
    import pandas as pd
    rows = [{"file": n, "bytes": b, "size": W.fmt_bytes(b)} for n, b in exp.files()]
    st.dataframe(pd.DataFrame(rows)[["file", "size", "bytes"]],
                 width="stretch", hide_index=True)


def render() -> None:
    st.title("Exports")
    st.caption("Open a finished export, look at what actually landed on disk, and "
               "check it. Everything here is read back out of the exported files "
               "through the same offset arithmetic a consumer has to implement — "
               "so a preview that comes out right is evidence the format is right.")

    root = st.session_state.get("export_root", kvexports.DEFAULT_ROOT)
    root = st.text_input("Exports folder", value=root,
                         help="Where the Process page writes by default.")
    st.session_state["export_root"] = root

    found = kvexports.list_exports(root)
    if not found:
        st.info(f"No exports in `{root}`. Build one on the **Process** page, or "
                f"with `kvterrain process --dataset <name> --out {root}/<name>`.")
        W.attribution_footer()
        return

    roots = [e.root for e in found]
    prev = st.session_state.get("export_selected")
    idx = roots.index(prev) if prev in roots else 0
    chosen = st.selectbox(
        "Export", roots, index=idx,
        format_func=lambda r: _caption(found[roots.index(r)]))
    st.session_state["export_selected"] = chosen
    exp = found[roots.index(chosen)]

    _summary_card(exp)

    tabs = st.tabs(["Preview", "Checks", "Settings used", "Files", "manifest.json"])
    with tabs[0]:
        _previews(exp)
    with tabs[1]:
        _checks(exp)
    with tabs[2]:
        _settings(exp)
    with tabs[3]:
        _files(exp)
    with tabs[4]:
        st.json(exp.manifest)

    W.attribution_footer()


def _caption(exp: kvexports.Export) -> str:
    s = exp.summary()
    ds = f" ← {s['dataset']}" if s["dataset"] else ""
    water = f"{s['lakes']:,} lakes" if s["has_water"] else "no water"
    return (f"{s['name']}{ds} — {s['area_km2']:,.0f} km² @ {s['spacing_m']:g} m · "
            f"{s['tiles_total']:,} tiles · {water} · "
            f"{W.fmt_bytes(s['bytes_total'])} · {W.fmt_when(s['generated_utc'])}")
