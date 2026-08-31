"""
kvterrain — Streamlit UI
========================

One application, two tools, because the two halves of this pipeline have nothing
to say to each other except through a dataset on disk:

    Fetch    draw a rectangle, pull Kartverket heights + NVE water, store it
    Process  pick a stored dataset, tune the pipeline, build the Unity export

Fetching costs minutes of network and never changes for a given region; the
post-processing is what you actually iterate on. Keeping them in one process but
on separate pages means you fetch Lierne once and then carve it twenty different
ways without downloading it again.

Run:
    streamlit run app.py

Height data © Kartverket (CC BY 4.0). Water data © NVE (Elvenett / Innsjødatabase).
"""
import streamlit as st

st.set_page_config(page_title="kvterrain — Kartverket → Unity terrain",
                   page_icon="⛰️", layout="wide")

from ui import fetch_page, process_page      # noqa: E402 — must follow set_page_config

PAGES = [
    st.Page(fetch_page.render, title="Fetch", icon=":material/download:",
            url_path="fetch"),
    st.Page(process_page.render, title="Process", icon=":material/terrain:",
            url_path="process", default=True),
]

st.navigation(PAGES).run()
