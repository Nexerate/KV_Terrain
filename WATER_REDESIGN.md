# Water Redesign — Exporter Design Document

> **STATUS: NOT IMPLEMENTED. DESIGN ONLY (2026-09-17).**
>
> Nothing below exists in code. The exporter that runs today is the pipeline in
> `kvterrain/process.py` and documented in `readme.md`, producing `heights.atlas`,
> `surface.atlas`, `water_id.atlas`, `rivers.bin`, `lakes.json` and `junctions.json`. Stage names,
> files and formats below are PROPOSALS.
>
> The runtime half of this design lives in the Terraform repository: `WATER_REDESIGN.md` (branch
> `water/redesign`). Read its §1–§4 first. This document only covers what the exporter must do.

---

## 1. Context in one paragraph

The runtime is moving from "solve hydrology for the whole world at runtime from rasters, then
reconcile with authored water" to "the authored water graph is the truth, and each terrain edit
locally repairs it". Lakes become **implicit** (a depression label per cell plus a level per
depression node). Rivers stay **explicit** polylines in a graph with discharge. Everything that is
expensive and global (the depression hierarchy, flow directions, catchment areas, hypsometry) is
computed **once, here, offline**, and the runtime only rebuilds it inside edited regions.

**Terrain is the source of truth.** The exporter's job is to make the carved terrain and the
authored water agree well enough that the depression hierarchy built on that terrain reproduces
the authored water, and to report every place where it does not.

---

## 2. What already exists and stays

The current pipeline already does much of the hard, data-specific work. Keep it:

- **Fetch / dataset split.** The new stages are all `process` stages and need no new network data,
  with one possible exception (culverts, §7).
- **Void repair** (`bathymetry.fill_lake_surface`) before anything reads the bed.
- **Shoreline snap** (`water.snap_lakes_to_flat_water`).
- **Lake level resolution** (`water.apply_estimated_levels`): NVE `hoyde` verbatim, else the DTM
  interior estimate with the perimeter cap.
- **Lake carve** (`bathymetry.carve_lake_beds`): ramp from the waterline, max depth.
- **River surface from the uncarved centreline** (`watersurface.river_surface_moh`) and the **river
  carve** (`bathymetry.carve_river_beds`).
- **The river polyline graph** (`rivernet`): densified vertices, upstream/downstream links, lake
  spans, junctions, bed `z` and surface `level`, and the descent / flow-direction / connectivity
  validators.

A correction to the runtime-side discussion: rivers are **already** exported as a linked polyline
graph (`rivers.bin`). What the export lacks is lake **geometry** (lakes.json carries records and a
bbox; identity lives only in `water_id.atlas`), plus everything hierarchy-related. Under the new
model lake geometry is still not needed as output, because a lake is a hierarchy node, but keeping
the NVE polygon available (e.g. as optional GeoJSON) helps debugging and matching.

---

## 3. New pipeline

Current order (`process.run_process`): rasterise → void repair → shore snap → snapshot → lake
levels → river surface → carve rivers → snapshot → carve lakes → surface pyramid → water id →
river network → pack.

Proposed order. Steps marked **NEW** do not exist; everything else is today's step, possibly
reshaped:

1. Rasterise authored water (still needed as a working mask for carving and matching).
2. Void repair.
3. Shore snap.
4. Snapshot the uncarved bed.
5. Lake levels.
6. River surface from the uncarved centreline.
7. **NEW: monotone river bed.** Enforce a downstream running minimum on each densified polyline's
   bed profile, through junctions and lake spans, **before** carving. See §4.1.
8. Carve rivers against the monotone profile.
9. Carve lakes.
10. **NEW: embankment treatment** (optional, §7).
11. **NEW: Priority-Flood + depression hierarchy** on the final carved terrain. §4.2.
12. **NEW: flat-resolved flow directions and flow accumulation.** §4.3.
13. **NEW: match authored water to hierarchy nodes, and audit.** §4.4.
14. **NEW: hypsometry per node.** §4.5.
15. **NEW: discharge and width per river segment.** §4.6.
16. River network (extended with `Q`, width, node links).
17. Pack: heights, **label atlas**, **flow direction atlas**, **hierarchy table**, river graph,
    lakes, audit report. During the transition also the current `surface.atlas` and
    `water_id.atlas`.

**The ordering rule that matters:** the hierarchy (11) must be built on exactly the heights that
are packed (17). Any step that touches heights after 11 invalidates it.

---

## 4. The new stages

### 4.1 Monotone river bed

The runtime currently enforces descent itself: `rivernet` reports non-monotonic descent and
leaves the running minimum to the runtime burn. Under the redesign there is no runtime burn, so
the enforcement moves here.

- **Keep densification.** The earlier PAVA fit failed because sparse Elvenett chords cut across
  banks and the fit pooled surface levels metres above the bed (see `rivernet` module docstring).
  That was a fitting-on-chords problem. A running minimum over a **densified** bed profile is what
  the runtime burn already does successfully.
- Apply it to the **bed** `z` along each segment, carrying the minimum across `downstream` links so
  a junction never steps up.
- Carve to that profile. Any remaining pit along a mapped channel is then an audit finding, not a
  hierarchy node the runtime has to reason about.
- Keep the surface `level` rule as today (levelled across the channel, descending along it), and
  clamp it to never rise downstream either.

### 4.2 Priority-Flood and the depression hierarchy

Build Barnes' depression hierarchy (Barnes, Callaghan & Wickert 2020) on the carved lattice.

Per node:

| Field | Meaning |
|---|---|
| `id`, `parent` | Tree structure. Leaves are single pits; the root is the ocean/outside |
| `floor_m` | Lowest cell |
| `spill_m` | Elevation of the lowest saddle out of the node |
| `spill_cell` | Where it overflows |
| `overflow_to` | The node on the other side of the saddle |
| `cell_count`, `area_m2` | Own cells |
| `catchment_m2` | Total draining area (from §4.3) |
| `lake_id` | Authored lake matched to it, or none (§4.4) |
| `authored_level_m` | From lake levels, when matched |
| `hypso` | Offset into the hypsometry blob (§4.5) |

Plus a **per-cell leaf label**.

**Ocean.** Today's `waterid.ocean_mask_from_edges` flood-fills from the map edges; the hierarchy
needs the same idea as its root: seed Priority-Flood from edge cells at or below ocean level. Decide
explicitly how **inland depressions cropped by the export boundary** are treated; the runtime
already has a recorded defect where a boundary-cropped lake cannot fill.

**Scale.** 40 km at 5 m spacing is roughly 8193² ≈ 67M cells. Pure Python with `heapq` is not
viable. Options, in order of preference:

1. Wrap Barnes' open-source C++ implementation (DepressionHierarchy / Fill-Spill-Merge) with
   pybind11. Check its licence first.
2. RichDEM's Python bindings for Priority-Flood and flow accumulation, with the hierarchy built on
   top.
3. A Numba implementation.

Offline minutes are acceptable; tens of minutes on a laptop are probably not.

**Noise pits.** The hierarchy will contain enormous numbers of tiny depressions. Keep them all in
the tree (the runtime walk needs their spill points to escape pits) but do not compute hypsometry
for them, and mark nodes below a depth/area threshold as never-displayed.

### 4.3 Flow directions and accumulation

- D8 directions on the carved terrain, with flats resolved (Garbrecht & Martz 1997, or Barnes'
  improved flat resolution). A flat resolve was declined for the **runtime** because it was costly
  and the pools already produce the partition; offline it is cheap and the runtime walks need a
  direction on flats.
- Directions inside a depression point toward its floor; the walk jumps via `spill_cell` from
  there.
- Flow accumulation gives catchment area per cell, hence `catchment_m2` per node and per river
  segment.
- Pack directions as a 1-byte-per-sample atlas on the same lattice and pyramid scheme as heights.
  Coarse levels should take the direction of the child with the largest accumulation, not a
  majority vote.

### 4.4 Matching authored water, and the audit

**Lakes.** For each authored lake, find the node whose cells best cover its rasterised mask at its
authored level. Check:

- **Level above spill** (`authored_level_m > spill_m + tol`): the lake should be leaking. Either the
  level is wrong, the carve broke the rim, or there is an outlet the DEM does not see (a dam, a
  weir). Report it with its location.
- **Level well below spill** with a matched inflow river: plausible for a regulated lake. Report
  but do not flag.
- **No node found** (a lake on a slope in the DEM): report.

**Rivers.** For each segment, check that the steepest-descent path from its upstream end stays
inside the carved corridor. Report where it leaves.

**Unexplained depressions.** Nodes that are deep and large enough to display, with no authored lake
and a catchment large enough to fill under the export rainfall. Sorted by visual severity. In
Norway these will largely be road and rail embankments over culverts (§7).

The audit is a first-class product. Write it as JSON (with coordinates) and show it in the Process
page, like the existing lake-level breakdown and network validation.

### 4.5 Hypsometry

For each node large enough to hold a visible lake: a cumulative area-by-elevation curve from
`floor_m` to `spill_m`, including its descendants' cells. This lets the runtime solve
`L = min(spill, Area⁻¹(Q / E))` with a lookup.

- Fixed bin count per node (e.g. 64), elevation-spaced. Store as one flat blob plus offsets.
- The area above the shoreline comes from LiDAR. The area below an authored lake's surface comes
  from the **invented carved bowl**. That does not affect levels at or above the authored level,
  but it does affect what a breached, drained lake looks like. Record it per node
  (`bed_is_carved`).

### 4.6 Discharge and width

Elvenett has no width or discharge; today width is a heuristic on Strahler order.

- **`Q` per segment = catchment area at its downstream end × export rainfall.** Record the rainfall
  value in the manifest; the runtime needs it to decide whether its own rainfall knob scales
  authored rivers.
- **Width = f(Q)**, e.g. `w = a · Q^0.5` (Leopold & Maddock). **This function must be identical in
  exporter and runtime.** Write its constants into the manifest and have the runtime read them
  rather than duplicating them.
- Keep Strahler order as metadata and as a fallback; compare the two width models on a real export
  before switching.

---

## 5. Export format additions (sketch)

| File | Contents |
|---|---|
| `labels.atlas` | u32 leaf node id per sample; coarse levels store the covering ancestor |
| `flowdir.atlas` | u8 D8 direction per sample, flats resolved |
| `hierarchy.bin` | Node table (§4.2) + hypsometry blob (§4.5) |
| `rivers.bin` v2 | v1 + per-segment `Q`, width, `node_id`; bed profile monotone |
| `lakes.json` | + `node_id`, `spill_m`, audit flags |
| `water_audit.json` | §4.4 findings with coordinates |
| `manifest.json` | + hierarchy/label/flowdir descriptors, export rainfall, width model constants |

Row order, corner-centred sampling and shared edges follow the existing conventions. Label and
direction atlases must be sliced from one assembled array like everything else, so tile edges
agree.

---

## 6. Testing

- **Synthetic worlds** (demo mode already exists): a bowl with a known spill level, two bowls
  merging at a known saddle, a river through a pit, a flat, an embankment with a culvert. Assert
  node counts, spill levels, labels and overflow targets exactly.
- **Hierarchy vs brute force:** on small random heightfields, compare the hierarchy's spill levels
  to a naive flood-fill per depression.
- **Round trip:** the flattened level table on unedited terrain must reproduce the current
  `surface.atlas` for lakes within tolerance. This is the "authored stays authored" check, and the
  same kind of baseline the runtime already uses (FLAT-authored 85.5% within 1 unit on the Norway
  export).
- **The in-place update prototype** (runtime doc §6) should use hierarchies produced by this code as
  its reference, so exporter and runtime cannot disagree about what a hierarchy is.

---

## 7. Open questions

- **Culverts and embankments.** Options: breach narrow embankments automatically (Lindsay 2016
  least-cost breaching, capped by length and depth); fetch culvert locations if a source exists
  (NVDB may carry them; unverified); or leave them in and rely on the audit plus a runtime minimum
  display depth. Probably the most visible artifact in a Norwegian export, so decide with a real
  export in hand.
- **Bridges** in DTM vs DOM. DTM should already remove them; verify on a real export.
- **Regulated lakes** whose authored level sits below the DTM's flown surface or above the spill.
- **Hierarchy library choice** (§4.2) and its licence.
- **Hypsometry threshold:** which nodes get curves.
- **Transition:** how long both old and new products are exported side by side.
- **Lake polygons:** export NVE geometry as optional GeoJSON for debugging and matching, or not.
