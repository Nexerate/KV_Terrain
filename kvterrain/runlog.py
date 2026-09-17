"""
kvterrain.runlog
================

A per-dataset log of every process run: what you set, and what came out.

The point of the fetch/process split is that you run stage two over and over
with different settings. Two runs later you no longer remember whether the
shoreline looked better at slope 0.8 or 1.0, and the export itself only carries
the settings of the LAST run — `process` overwrites in place. This is the thing
that remembers.

It lives beside the DATASET, not the export, because the history is "what have I
tried on this region" and it has to outlive an export being overwritten (or
deleted, or moved to Unity). One JSON object per line, appended, never rewritten:

    data/datasets/<slug>/runs.jsonl

Each record pairs the settings with the OUTCOMES that settings produced — how
many lakes ended up with no resolvable level, how far the flow directions
disagreed, the vertical range, how long it took. Settings alone would only tell
you what you did; the pairing is what tells you whether it worked.

This is a log, not a format anything depends on. A record that fails to parse is
skipped rather than raised on: losing history must never break a build.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Optional

RUNS_NAME = "runs.jsonl"

# Records are capped so a long tuning session cannot grow without bound. The cap
# is generous; it exists to stop a runaway script, not to prune real history.
MAX_RECORDS = 2000


def path_for(dataset_root: str) -> str:
    return os.path.join(dataset_root, RUNS_NAME)


def summarise_run(*, manifest: dict, tile_cells: int, include_water: bool,
                  water_opts: Optional[dict], nodata_fill_m: float,
                  height_min, height_max, out_dir: str, seconds: float,
                  stage_seconds: Optional[dict] = None) -> dict:
    """Flatten one run into the record that gets logged."""
    atlas = manifest.get("atlas", {}) or {}
    wm = manifest.get("water_surface", {}) or {}
    ll = wm.get("lake_levels", {}) or {}
    wv = manifest.get("water_vector", {}) or {}
    val = wv.get("validation", {}) or {}
    wo = dict(water_opts or {})

    return {
        "when_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(float(seconds), 2),
        "out_dir": out_dir,
        # ---- settings ---------------------------------------------------- #
        "tile_cells": int(tile_cells),
        "water": bool(include_water),
        "nodata_fill_m": nodata_fill_m,
        "hmin_fixed": height_min,
        "hmax_fixed": height_max,
        "lake_max_depth_m": wo.get("lake_max_depth_m"),
        "lake_shore_slope": wo.get("lake_shore_slope"),
        "lake_min_depth_m": wo.get("lake_min_depth_m"),
        "lake_snap_px": wo.get("lake_snap_px"),
        "fill_lake_holes": wo.get("fill_lake_holes"),
        "estimate_lake_levels": wo.get("estimate_lake_levels"),
        "lake_perimeter_cap": wo.get("lake_perimeter_cap"),
        "river_width_scale": wo.get("width_scale"),
        "river_depth_scale": wo.get("river_depth_scale"),
        "river_bank_tolerance_m": wo.get("river_bank_tolerance_m"),
        "river_vertex_stride_m": wo.get("river_vertex_stride_m"),
        "ocean_level_m": wo.get("ocean_level_m"),
        "lake_level_max_above_shore_m": wo.get("lake_level_max_above_shore_m"),
        "lakes_corrected": ll.get("levels_corrected"),
        "downhill_river_bed": wo.get("downhill_river_bed"),
        "lake_edge_is_shore": wo.get("lake_edge_is_shore"),
        "hierarchy": wo.get("hierarchy"),
        # ---- outcomes ------------------------------------------------------ #
        # Settings alone tell you what you did. These tell you whether it worked.
        "tiles": manifest.get("num_levels") and atlas.get("height_bytes")
        and _tiles_from(manifest),
        "height_min_m": manifest.get("height_min_m"),
        "height_max_m": manifest.get("height_max_m"),
        "atlas_bytes": atlas.get("height_bytes"),
        "lakes": wm.get("lake_count"),
        "lakes_from_nve": ll.get("levels_from_nve"),
        "lakes_estimated": ll.get("levels_estimated"),
        "lakes_unresolved": ll.get("levels_unresolved"),
        "lakes_capped": ll.get("levels_capped_to_perimeter"),
        "shore_snap_samples": (wm.get("lake_bathymetry", {})
                               .get("shoreline_snap", {}).get("samples_added")),
        "river_segments": (wv.get("rivers") or {}).get("segments"),
        "flowdir_disagreement_pct": val.get("flowdir_disagreement_pct"),
        "descent_rising_pct": val.get("descent_rising_pct"),
        "bed_pixel_rising_pct": ((wm.get("river_bathymetry") or {})
                                 .get("downhill_bed") or {}).get("pixel_rising_pct"),
        "hierarchy_leaves": (manifest.get("hierarchy") or {}).get("leaf_count"),
        "hierarchy_nodes": (manifest.get("hierarchy") or {}).get("node_count"),
        "audit_lakes_ok": (((manifest.get("water_audit") or {}).get("summary") or {})
                           .get("lakes_by_status") or {}).get("ok"),
        "audit_lakes_held_by_outflow": (((manifest.get("water_audit") or {})
                                         .get("summary") or {}).get("lakes_by_status")
                                        or {}).get("held_by_outflow"),
        "audit_lakes_above_spill": (((manifest.get("water_audit") or {})
                                     .get("summary") or {}).get("lakes_by_status")
                                    or {}).get("level_above_spill"),
        "audit_river_pits": (((manifest.get("water_audit") or {}).get("summary") or {})
                             .get("river_findings_by_status") or {}).get("pit"),
        "stage_seconds": stage_seconds or {},
    }


def _tiles_from(manifest: dict) -> Optional[int]:
    atlas = manifest.get("atlas", {}) or {}
    tb = atlas.get("tile_bytes")
    hb = atlas.get("height_bytes")
    return int(hb // tb) if tb and hb else None


def append(dataset_root: str, record: dict) -> None:
    """Append one record. Never raises — a lost log entry must not fail a build."""
    try:
        os.makedirs(dataset_root, exist_ok=True)
        with open(path_for(dataset_root), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def read(dataset_root: str, *, limit: int = MAX_RECORDS) -> list:
    """Every logged run for this dataset, newest last. Bad lines are skipped."""
    path = path_for(dataset_root)
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out[-limit:]


# Columns worth putting in front of a human, in the order they answer questions:
# when, what it cost, the settings most often tuned, then whether it worked.
DISPLAY_COLUMNS = (
    "when_utc", "seconds", "tile_cells",
    "lake_max_depth_m", "lake_shore_slope", "lake_min_depth_m", "lake_snap_px",
    "river_width_scale", "river_depth_scale", "river_bank_tolerance_m",
    "lakes", "lakes_unresolved", "lakes_estimated",
    "flowdir_disagreement_pct", "height_min_m", "height_max_m",
    "downhill_river_bed", "hierarchy", "bed_pixel_rising_pct",
    "lakes_corrected", "audit_lakes_ok", "audit_lakes_held_by_outflow",
    "audit_lakes_above_spill", "audit_river_pits",
)
