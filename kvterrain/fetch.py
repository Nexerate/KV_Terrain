"""
kvterrain.fetch
===============

Stage one of two: **get the data, store it, stop.**

Everything that talks to a remote service lives here, and nothing that makes a
decision about the data does. A fetch assembles the Kartverket height lattice and
pulls the NVE vectors that intersect it, writes both into a `kvterrain.dataset`
directory, and ends. No void repair, no shoreline snap, no level estimation, no
rasterising, no carving, no decimation, no packing — those are
`kvterrain.process`, and they are the part you iterate on.

What belongs to this stage
--------------------------

A parameter belongs here if changing it means new bytes have to come off the
network:

* the rectangle, the UTM zone, and the sample spacing (they define the lattice);
* DTM vs DOM (a different ImageServer);
* `max_fetch_px` (how the region is chunked into exportImage requests);
* whether to pull the `hovedelv` layer at all (a second, separate NVE query).

`tile_cells` is the one judgement call. It is recorded, because `plan_grid` needs
one to pad the region out to whole power-of-two tiles — but the padded lattice it
produces supports every smaller power-of-two tiling too, so process time may
re-tile freely. See `dataset.Lattice.valid_tile_cells`.

Progress
--------

`progress(fraction, label)` — a float in [0, 1] and something to show a human.
This is a different protocol from the `(done, total, msg)` callback `core` and
`water` use internally, because a fetch is two dissimilar phases (a chunked
raster download, then three paged vector queries) and only a single fraction
makes sense on one bar. The adapters are below.

Data © Kartverket (CC BY 4.0) and © NVE (Elvenett / Innsjødatabase).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import core, dataset as ds_mod

# How the one progress bar is split between the two phases. Heights dominate
# wall-clock on any region big enough to care about.
HEIGHT_SHARE = 0.72

Progress = Callable[[float, str], None]


# --------------------------------------------------------------------------- #
# Demo fetcher                                                                 #
# --------------------------------------------------------------------------- #

def synthetic_height_fetcher(u, e, sx, sy, nx, ny, sp) -> np.ndarray:
    """
    A deterministic hill + ripple surface with the `Fetcher` signature, for demo
    mode and offline tests. Returned north-up, like the real ImageServer.

    Single copy on purpose: this used to exist verbatim in both `cli.py` and
    `app.py`, which meant an offline dry run could pass in one and fail in the
    other.
    """
    xs = sx + np.arange(nx) * sp
    ys = sy + np.arange(ny) * sp
    XX, YY = np.meshgrid(xs, ys)
    surf = (300.0
            + 400.0 * np.exp(-(((XX - XX.mean()) / 4000.0) ** 2
                               + ((YY - YY.mean()) / 4000.0) ** 2))
            + 60.0 * np.sin(XX / 900.0) * np.cos(YY / 700.0))
    return surf.astype(np.float32)[::-1, :]


# --------------------------------------------------------------------------- #
# Result                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class FetchResult:
    dataset: ds_mod.Dataset
    seconds_total: float
    seconds_heights: float
    seconds_water: float

    @property
    def path(self) -> str:
        return self.dataset.root


# --------------------------------------------------------------------------- #
# The fetch                                                                    #
# --------------------------------------------------------------------------- #

def run_fetch(
    plan: core.GridPlan,
    *,
    name: Optional[str] = None,
    root: Optional[str] = None,
    dest: Optional[str] = None,
    source_kind: str = "DTM",
    fetcher: Optional[core.Fetcher] = None,
    server_url: Optional[str] = None,
    max_fetch_px: int = core.DEFAULT_MAX_FETCH_PX,
    include_water: bool = True,
    include_main_rivers: bool = True,
    compression: str = ds_mod.DEFAULT_COMPRESSION,
    demo: bool = False,
    session=None,
    progress: Optional[Progress] = None,
    request_bbox_lonlat: Optional[tuple] = None,
    make_preview: bool = True,
    auto_name: bool = True,
) -> FetchResult:
    """
    Fetch `plan`'s region and write it as a dataset directory.

    `dest` names the directory outright; otherwise one is created under `root`
    (default `dataset.DEFAULT_ROOT`) from `name`, de-duplicated so a second fetch
    of the same place never overwrites the first.
    """
    from . import water as kvwater, preview as kvpreview

    t_start = time.time()

    def emit(frac: float, label: str) -> None:
        if progress:
            progress(min(max(float(frac), 0.0), 1.0), label)

    if fetcher is None:
        if demo:
            fetcher = synthetic_height_fetcher
        else:
            fetcher = (lambda u, e, sx, sy, nx, ny, sp: core.export_image_fetch(
                u, e, sx, sy, nx, ny, sp, source_kind=source_kind, session=session))
    if server_url is None:
        server_url = "(demo)" if demo else core.IMAGESERVER[source_kind]

    # ---- phase 1: the height lattice --------------------------------------- #
    emit(0.0, "planning")

    def height_progress(done, total, msg):
        emit(HEIGHT_SHARE * done / max(int(total), 1),
             f"heights: {msg}")

    t0 = time.time()
    heights = core.assemble_region(plan, fetcher, server_url,
                                   max_fetch_px=max_fetch_px,
                                   progress=height_progress)
    seconds_heights = time.time() - t0
    emit(HEIGHT_SHARE, "heights assembled")

    # ---- phase 2: the NVE vectors ------------------------------------------ #
    rivers = main_rivers = lakes = []
    seconds_water = 0.0
    if include_water:
        t0 = time.time()
        n_layers = 3 if include_main_rivers else 2
        state = {"i": 0, "label": "water"}

        def on_layer(i, n, label):
            state["i"], state["label"] = i, label
            emit(HEIGHT_SHARE + (1.0 - HEIGHT_SHARE) * i / max(n, 1), f"{label}: querying")

        def water_progress(done, total, msg):
            # A paged ArcGIS query cannot know its own total, so the bar advances
            # by LAYER and the running feature count goes in the label. Pretending
            # to know a percentage inside a layer would just make the bar lie.
            base = HEIGHT_SHARE + (1.0 - HEIGHT_SHARE) * state["i"] / max(n_layers, 1)
            emit(base, msg)

        if demo:
            rivers, main_rivers, lakes = kvwater.features_to_geojson(
                kvwater.synthetic_water_features(plan))
            if not include_main_rivers:
                main_rivers = []
        else:
            rivers, main_rivers, lakes = kvwater.fetch_water_geojson(
                plan, include_main_rivers=include_main_rivers, session=session,
                progress=water_progress, on_layer=on_layer)
        seconds_water = time.time() - t0
    emit(0.97, "writing dataset")

    # ---- write -------------------------------------------------------------- #
    lattice = ds_mod.Lattice.from_plan(plan)

    if not name and auto_name:
        # Suggested, never imposed: a fetch of a region straddling two
        # municipalities gets the one under its centre, which is a fine label and
        # a poor description. The UI shows it in an editable field.
        name = "Demo" if demo else ds_mod.suggest_name(lattice, session=session)
    name = name or "dataset"

    if dest is None:
        root = root or ds_mod.DEFAULT_ROOT
        os.makedirs(root, exist_ok=True)
        dest = ds_mod.unique_dir(root, ds_mod.slugify(name))

    preview_png = None
    if make_preview:
        try:
            preview_png = kvpreview.dataset_thumbnail(
                heights, lattice, rivers=rivers, lakes=lakes)
        except Exception:      # noqa: BLE001 — a thumbnail is never worth losing a fetch
            preview_png = None

    d = ds_mod.write(
        dest, lattice, heights,
        name=name,
        source_kind=source_kind,
        server_url=server_url,
        demo=demo,
        rivers=rivers, main_rivers=main_rivers, lakes=lakes,
        include_water=include_water,
        include_main_rivers=include_main_rivers,
        compression=compression,
        preview_png=preview_png,
        request={
            "bbox_lonlat": list(request_bbox_lonlat) if request_bbox_lonlat else None,
            "spacing_m": plan.spacing_m,
            "tile_cells": plan.tile_cells,
            "epsg": plan.epsg,
            "max_fetch_px": int(max_fetch_px),
            "source": source_kind,
            "include_water": bool(include_water),
            "include_main_rivers": bool(include_main_rivers),
            "leaf_tiles_at_fetch": [plan.leaf_tiles_x, plan.leaf_tiles_y],
            "num_levels_at_fetch": plan.num_levels,
        },
        extra={"timing": {
            "seconds_heights": round(seconds_heights, 2),
            "seconds_water": round(seconds_water, 2),
        }},
    )
    emit(1.0, "done")
    return FetchResult(dataset=d,
                       seconds_total=time.time() - t_start,
                       seconds_heights=seconds_heights,
                       seconds_water=seconds_water)
