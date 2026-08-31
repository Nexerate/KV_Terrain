"""
Shared Streamlit pieces: formatting, dataset cards, map overlays, image panels.

Nothing here makes a pipeline decision. If a function in this file starts
deciding something about the data, it belongs in `kvterrain/` instead.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from kvterrain import dataset as ds_mod


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #

def fmt_bytes(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024
    return f"{n:,.1f} GB"


def fmt_when(iso: str) -> str:
    """'2026-08-31T21:34:15+00:00' -> '2026-08-31 21:34'."""
    return (iso or "").replace("T", " ")[:16]


def fmt_extent(lat) -> str:
    x0, y0, x1, y1 = lat.bbox_utm
    return f"{x0:,.0f}, {y0:,.0f} → {x1:,.0f}, {y1:,.0f}"


# --------------------------------------------------------------------------- #
# Images                                                                       #
# --------------------------------------------------------------------------- #

def show_rgb(container, rgb: np.ndarray, caption: str, *, width=None) -> None:
    """Float RGB in [0,1] -> an st.image. uint8 explicitly, so Streamlit never has
    to guess at the range and silently rescale a panel differently from its
    neighbour."""
    arr = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    container.image(arr, caption=caption, width=width or "stretch")


def panel_grid(specs: list, *, columns: int = 2) -> None:
    """`specs` is a list of (rgb_array, caption). Laid out `columns` per row."""
    for i in range(0, len(specs), columns):
        cols = st.columns(columns)
        for col, (rgb, cap) in zip(cols, specs[i:i + columns]):
            show_rgb(col, rgb, cap)


# --------------------------------------------------------------------------- #
# Dataset presentation                                                         #
# --------------------------------------------------------------------------- #

def dataset_caption(s: dict) -> str:
    """One dense line for a selectbox: enough to tell two fetches apart."""
    water = (f"{s['rivers']:,}r/{s['lakes']:,}l" if s["has_water"] else "no water")
    demo = " · DEMO" if s["demo"] else ""
    return (f"{s['name']} — {s['area_km2']:,.0f} km² @ {s['spacing_m']:g} m · "
            f"{s['source']} · {water} · {fmt_bytes(s['bytes_total'])} · "
            f"{fmt_when(s['created_utc'])}{demo}")


def dataset_card(ds: ds_mod.Dataset, *, show_preview: bool = True) -> None:
    """The 'what did I actually fetch' panel: a picture and the numbers behind it."""
    s = ds.summary()
    lat = ds.lattice

    left, right = st.columns([2, 3])

    with left:
        p = ds.preview_path()
        if show_preview and p:
            st.image(p, caption="Fetched terrain (hillshade) with NVE water drawn over it",
                     width="stretch")
        elif show_preview:
            st.info("No preview stored for this dataset.")

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Area", f"{s['area_km2']:,.0f} km²")
        c2.metric("Resolution", f"{s['spacing_m']:g} m")
        c3.metric("On disk", fmt_bytes(s["bytes_total"]))
        # Short values only — st.metric truncates, and a big lattice's sample count
        # is precisely the value you do not want quietly cut to "32,769 x 1...".
        c1, c2, c3 = st.columns(3)
        c1.metric("Height range",
                  f"{s['height_min_m']:.0f}–{s['height_max_m']:.0f} m"
                  if s["height_min_m"] is not None else "—")
        c2.metric("Coverage", f"{s['coverage_pct']:.2f} %")
        c3.metric("Water", f"{s['lakes']:,} lakes" if s["has_water"] else "none")

        st.markdown(
            f"""
- **Samples:** {s['samples'][0]:,} × {s['samples'][1]:,}
- **Source:** {s['source']}{' (demo / synthetic)' if s['demo'] else ''} ·
  **CRS:** EPSG:{s['epsg']}
- **Extent (UTM):** {fmt_extent(lat)}
- **Region:** {s['width_m']:,.0f} × {s['height_m']:,.0f} m
- **Fetched:** {fmt_when(s['created_utc'])}
- **Path:** `{s['root']}`
""")
        if s["has_water"]:
            st.caption(
                f"Water: {s['rivers']:,} elvenett segments + {s['main_rivers']:,} "
                f"hovedelv segments + {s['lakes']:,} lake polygons, stored as raw "
                f"NVE GeoJSON and parsed fresh on every process run. "
                f"{ds.manifest.get('water', {}).get('attribution', '')}")
        else:
            st.warning("This dataset was fetched **without water**. Rivers and lakes "
                       "cannot be added without re-fetching the region.")

        if s["coverage_pct"] < 99.9:
            st.warning(
                f"{100 - s['coverage_pct']:.2f}% of the lattice came back as nodata. "
                f"That is usually the region reaching past Kartverket's coverage "
                f"(a border, or open sea) — check the preview before processing.")


def dataset_map(datasets, *, height: int = 300, highlight=None):
    """A small folium map of where the datasets are. Read-only — the draw tool
    lives on the fetch page, and giving this one a toolbar would only invite
    drawing a rectangle that does nothing."""
    import folium
    from streamlit_folium import st_folium

    items = [datasets] if isinstance(datasets, ds_mod.Dataset) else list(datasets)
    if not items:
        return

    m = folium.Map(location=[64.0, 12.0], zoom_start=4, tiles="OpenStreetMap")
    bounds = []
    for d in items:
        lat = d.lattice
        ring = lat.outline_latlon()
        bounds.extend(ring)
        is_hi = highlight is None or d.root == getattr(highlight, "root", None)
        folium.Polygon(
            ring,
            color="#e8590c" if is_hi else "#868e96",
            weight=2 if is_hi else 1,
            fill=True,
            fill_opacity=0.18 if is_hi else 0.05,
            fill_color="#e8590c" if is_hi else "#868e96",
            tooltip=f"{d.name} — {lat.area_km2:,.0f} km² @ {lat.spacing_m:g} m",
        ).add_to(m)
    if bounds:
        lats = [p[0] for p in bounds]
        lons = [p[1] for p in bounds]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(30, 30))

    return st_folium(m, key=f"ds_map_{len(items)}", height=height, width=None,
                     returned_objects=[])


# --------------------------------------------------------------------------- #
# Attribution                                                                  #
# --------------------------------------------------------------------------- #

def attribution_footer() -> None:
    from kvterrain import core, water as kvwater
    st.divider()
    st.caption(f"Height data {core.ATTRIBUTION}. Water data {kvwater.WATER_ATTRIBUTION} "
               f"— {kvwater.WATER_LICENSE}. Attribute both in anything you ship.")
