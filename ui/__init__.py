"""Streamlit UI for kvterrain, split the same way the pipeline is.

`fetch_page` talks to the network and writes datasets; `process_page` reads a
dataset and builds the export. Neither knows how the other works — the only
thing they share is `kvterrain.dataset`.

The engine (`kvterrain/`) stays headless and importable without Streamlit; this
package is the only thing that imports it.
"""
