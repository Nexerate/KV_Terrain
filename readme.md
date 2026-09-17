# kvterrain — Kartverket height (+ NVE water) → Unity quad-tree terrain

A standalone offline tool that takes a rectangle drawn on a map of Norway,
fetches Kartverket elevation data, and builds a **quad-tree pyramid of 16-bit
height tiles**, packed into one dense `heights.atlas`, plus a `manifest.json` for
a Unity terrain system. It never talks to Unity — it only produces the data files
your engine will consume later.

It also fetches NVE's national river network and lake database (on by default,
`--no-water` to skip) and writes two more atlases, tile-for-tile and
sample-for-sample aligned with the heights: **`surface.atlas`** (the water's
elevation in metres above sea level) and **`water_id.atlas`** (water class plus
authored lake identity). Beside them go the river network as polylines
(`rivers.bin`), per-lake records (`lakes.json`) and river/lake junctions
(`junctions.json`). Depth is `max(0, water_surface − terrain)` — no rim
reconstruction, no seed-fill.

> **Planned, not implemented:** a redesign of the water exports around a
> depression hierarchy is described in [`WATER_REDESIGN.md`](WATER_REDESIGN.md).
> Everything in this readme describes the tool as it runs today.

Height data © Kartverket, licensed **CC BY 4.0**. Water data © **NVE** (Elvenett /
Innsjødatabase). Attribute "© Kartverket" and "© NVE" in anything you ship.

---

## Two tools, one application

Fetching a region takes minutes of network and never changes once you have it.
Post-processing it takes minutes more and is the part you actually iterate on —
lake carving, river rasterisation, level estimation, packing. Doing both in one
pass meant re-downloading twenty square kilometres of LiDAR every time you moved
a slider, so the tool comes apart in the middle:

```
        FETCH                      dataset                    PROCESS
  ┌──────────────────┐      ┌───────────────────┐      ┌──────────────────┐
  │ rectangle        │      │ heights.hgt.gz    │      │ rasterise water  │
  │ spacing, DTM/DOM │─────▶│ (raw float32,     │─────▶│ repair, snap     │
  │ Kartverket       │      │  as fetched)      │      │ resolve levels   │
  │ NVE elvenett     │      │ water/*.geojson.gz│      │ carve, pyramid   │
  │ NVE innsjø       │      │ dataset.json      │      │ pack the atlases │
  │                  │      │ preview.png       │      │                  │
  └──────────────────┘      │ runs.jsonl        │      └──────────────────┘
     network, minutes       └───────────────────┘        no network, re-run
                             data/datasets/<name>/         as often as you like
```

A **dataset** is a fetch frozen on disk: the assembled height lattice exactly as
the servers returned it, plus the raw NVE vectors that intersect it. Nothing in
it has been carved, rasterised, repaired or packed. Fetch Lierne once, then carve
it twenty different ways for free.

The **exported format is unchanged** — same `manifest.json`, same atlases, same
`rivers.bin`, byte for byte. The dataset format in between is internal and ours
to change; see [The dataset format](#the-dataset-format-intermediate).

The Streamlit app is one process with three pages — **Fetch**, **Process**, and
**Exports** (open a finished export and check it holds together) — and the CLI
has the matching `fetch` / `process` / `validate-*` subcommands. `build` still
does both stages in one pass for when you genuinely only want the region once.

---

## What it produces

```
out/
  manifest.json     header: lattice, pyramid, packing range, atlas names, water metadata
  heights.atlas     carved terrain, every tile of every level, one dense blob
  surface.atlas     water surface (moh), same layout            (water only)
  water_id.atlas    water class + authored lake id, same layout (water only)
  rivers.bin        river network polylines                     (water only)
  lakes.json        one record per authored lake                (water only)
  junctions.json    where rivers enter and leave lakes          (water only)
```

### The dense atlas format (`dense_v1`)

Each `.atlas` file is **every tile of every pyramid level concatenated**, with no
header and no offset table: the offsets follow from `manifest.json` alone. The
helpers in `kvterrain/core.py` (`tiles_at_level`, `atlas_tile_offset`, …) ARE the
format contract, and the runtime must match them exactly.

```
tile_samples        = tile_cells + 1                       (default 129)
tileBytes           = tile_samples * tile_samples * 2
tilesX(L), tilesY(L) = max(1, leaf_tiles >> L)
levelByteBase(L)    = tileBytes * Σ_{l<L} tilesX(l) * tilesY(l)
tileOffset(L, x, y) = levelByteBase(L) + (y * tilesX(L) + x) * tileBytes
```

Levels ascend (L0 finest first); tiles within a level are row-major. All three
atlases use **2 bytes per sample**, so a tile sits at the **identical byte offset**
in each of them: three file handles, one offset computation. Offsets exceed 32
bits on real exports, so use 64-bit arithmetic. Every atlas's size is checked
against `atlas_total_bytes` when it is written.

- **`heights.atlas`** — little-endian **uint16**, `tile_samples²` per tile, rows
  **north → south**. Packed against one **global** `[height_min_m,
  height_max_m]` range from the manifest, so every tile at every level shares one
  vertical scale. These are the **carved** heights (see [The carves](#the-carves)).
- **`surface.atlas`** — little-endian **uint16** on the **same** range, mapped to
  `[0, 65534]`, with **65535 = no water**. See [Water surface](#water-surface-surfaceatlas).
- **`water_id.atlas`** — little-endian **uint16** code per sample. See
  [Water class and lake identity](#water-class-and-lake-identity-water_idatlas).

All atlases share one lattice, one row order and one shared-edge guarantee, so
sample `(i,j)` of tile `(L,x,y)` is the same world point in every one of them.

### The conventions that matter

- **Corner-centered / "pixel-is-a-point".** A tile of `tile_cells` cells carries
  `tile_cells + 1` samples. Sample `(i,j)` _is_ the height at a fixed world
  point. Adjacent tiles **share their boundary samples** — the right column of
  one tile is bit-identical to the left column of the next — so a meshed terrain
  has no cracks. Matches a 128×128-quad patch using a 129×129 height texture.
- **One assembled array, then sliced.** The whole region is fetched into a single
  continuous sample grid; tiles are cut out of it. Shared edges are identical by
  construction.
- **Centered `[1 2 1]/4` decimation for parents** (height), _not_ a 2×2 box
  average, so a parent sample lands exactly on the world point of child sample
  `2i`. Interior parent samples are exact; the outermost row/column of the whole
  region is an edge-clamp approximation — extend your rectangle slightly if you
  need the rim exact.
- **Single UTM zone** (EUREF89 / EPSG:25832 / 25833 / 25835, picked by longitude
  unless overridden). Fine for city- or mountain-range-sized areas.

---

## Install & run

```bash
pip install -r requirements.txt
```

### The UI

```bash
streamlit run app.py
```

Three pages in the left nav.

**Fetch.** Draw a rectangle with the toolbar (top-left of the map). A **dashed
orange outline** appears around it: that is the area actually fetched, which is
bigger than what you drew because the grid is padded out to whole power-of-two
tiles. Use the toolbar's **edit tool to drag or resize** the rectangle — the
outline and the plan follow it — so you can slide the region off a border or a
coverage gap instead of deleting and redrawing by eye. The panel beside the map
previews the snapped grid, the extent in UTM, how much padding was added on each
axis, the size of the height lattice and the fetch-request count before you
commit. Leave the name blank and the dataset is named after the municipality
under its centre (Kartverket's `kommuneinfo` API). **Demo mode** builds from a
synthetic surface and a synthetic river + lake with no network.

**Process.** Pick a dataset, tune the pipeline, run it. Its run history for that
dataset is right there, so you can see what you tried last time and what it did. The picker shows each
fetch's stored thumbnail, area, resolution, height range, coverage and size, and
what NVE vectors came with it. After a run you get, as pictures:

| Panel | What it answers |
| --- | --- |
| Terrain | what the bed looks like after every carve |
| Water classes | where the rasteriser decided river and lake are |
| **Depth** (`surface − terrain`) | **what the runtime will actually render** — a lake that came out as a dry pan shows up here and nowhere else |
| Water surface | how the level runs along a river and across a lake |
| Ground changed (`fetched − processed`) | signed: blue removed by the carves, orange added by the lake void repair and the shoreline snap |

…plus a **detail inspector** at native resolution (the overview panels are
strided down, and a shoreline judged from a 900-px thumbnail of a 4097-px lattice
is not judged), the lake-level breakdown, the advisory network validation, and
stage timings.

**Exports.** Open a finished export and check it — see
[Reading an export back](#reading-an-export-back).

### Headless

```bash
# stage one: fetch a region into data/datasets/<name>/
python -m kvterrain.cli fetch --bbox 8.30 61.28 8.55 61.40 \
    --spacing 5 --source DTM

# tighter on disk, still lossless (the default); --compress mm halves it again
python -m kvterrain.cli fetch --bbox 8.30 61.28 8.55 61.40 \
    --spacing 5 --compress lossless

# what have I already got?
python -m kvterrain.cli datasets

# stage two: build the export — run this as often as you like
python -m kvterrain.cli process --dataset lierne --out ./out \
    --lake-max-depth 20 --lake-shore-slope 1.0 --river-width-scale 1.5

# re-tile the SAME fetched samples, no network
python -m kvterrain.cli process --dataset lierne --out ./out --tile-cells 256

# both at once, storing nothing in between (the old single-pass build)
python -m kvterrain.cli build --bbox 8.30 61.28 8.55 61.40 \
    --spacing 5 --out ./out --source DTM --water

# offline synthetic dry-run of the whole thing (no network)
python -m kvterrain.cli fetch --bbox 8.30 61.28 8.55 61.40 --spacing 5 --demo
python -m kvterrain.cli process --dataset demo --out ./out
```

`--dataset` takes a directory path or the slug/name of one under the dataset
root. `kvterrain datasets --verbose` prints each dataset's full summary,
including the `tile_cells` values its lattice can be re-tiled to.

**Which flags go where.** A flag belongs to `fetch` if changing it means new
bytes have to come off the network — the rectangle, `--spacing`, `--epsg`,
`--source`, `--max-px`, `--compress`, and `--no-main-rivers` (a second NVE
query). Everything
else is a `process` flag and is free to re-run: `--river-width-scale`,
`--river-depth-scale`, `--river-bank-tolerance`, `--lake-max-depth` (default
**20**, real metres, how far below the water surface a lake bed is carved),
`--lake-shore-slope` (default **1.0**, how fast it gets there, in metres of depth
per metre of shore), `--lake-min-depth`, `--lake-snap-px`, `--ocean-level`,
`--river-vertex-stride-m`, `--tile-cells`, `--hmin` / `--hmax`, and the
`--(no-)estimate-lake-levels` / `--(no-)lake-perimeter-cap` pair. `build` accepts
both sets.

`--tile-cells` is the one that straddles the line: `fetch` needs one to pad the
region out to whole power-of-two tiles, but the padded lattice supports every
smaller power-of-two tiling too, so `process --tile-cells` re-slices the same
samples without refetching.

---

## The dataset format (intermediate)

Ours to change; not what your engine consumes.

```
data/datasets/<slug>/
  dataset.json                 lattice, provenance, stats, what's inside
  heights.hgt.gz               float32 (samples_y, samples_x), north-up, NaN = nodata
  runs.jsonl                   every process run made on this dataset
  preview.png                  stored thumbnail for the picker
  water/rivers.geojson.gz      NVE elvenett  — RAW GeoJSON features
  water/main_rivers.geojson.gz NVE hovedelv  — RAW GeoJSON features
  water/lakes.geojson.gz       NVE Innsjødatabase — RAW GeoJSON features
```

### How the lattice is stored

Plain gzip on float32 elevations only saves ~17%: the high bytes of neighbouring
samples are nearly identical, but the mantissa bytes are noise, and interleaving
them hands zlib no runs to find. A **byte shuffle** fixes that — group byte-plane
0 of every sample, then plane 1, and so on, so each plane is internally smooth.

Measured on real Kartverket DTM10:

| mode | size | error | notes |
| --- | --- | --- | --- |
| `none` | 100% | — | plain `.npy`, memory-mappable |
| `lossless` **(default)** | **60%** | none | byte shuffle + gzip |
| `mm` | 47% | ≤ 0.6 mm | rounds first — 200× finer than the DTM's own accuracy |
| `cm` | 32% | ≤ 5 mm | rounds first — at the DTM's own accuracy |

Decompression is ~60 ms on a full 4097² lattice, which is nothing next to a
process run. `--compress` on `fetch` picks the mode; `dataset.json` records it
and the layout, so a reader never has to guess.

### Five decisions worth knowing:

* **Heights are stored uncooked** — exactly what `core.assemble_region` returned,
  before void repair, before the shoreline snap, before any carve. All of those
  are post-processing decisions, and freezing a repaired array would bake today's
  repair into every future run. `.npy` rather than raw `.f32` so the file is
  self-describing and `np.load(mmap_mode="r")` works.

* **Water vectors are stored RAW, not parsed.** `water.features_from_geojson`
  does fuzzy attribute resolution against NVE field names that are not
  contractually stable — exactly the kind of thing you fix without wanting to
  refetch. Keeping the raw features means a better field map applies
  retroactively to datasets already on disk. Parsing costs seconds; the fetch
  costs minutes.

* **The lattice is the contract, not the tile grid.** A dataset records
  `origin / spacing / samples`, not `leaf_tiles` / `num_levels`, so the tiling is
  a process-time choice (`dataset.Lattice.valid_tile_cells`).

* **Nothing here is a Unity product.** The final format is produced only by
  `kvterrain.process` and is unchanged.

* **`runs.jsonl` remembers what you tried.** `process` overwrites its output in
  place, so the export only ever carries the LAST run's settings. The log lives
  beside the dataset — one JSON object per line, appended — and pairs each run's
  settings with what they produced: unresolved lakes, vertical range, flow-direction
  disagreement, per-stage timings. Settings alone tell you what you did; the
  pairing tells you whether it worked. Written by `process_dataset`, so the CLI
  logs too, not just the UI.

`process` records which dataset an export came from under `generator.dataset` in
the exported `manifest.json` — an additive provenance block (name, slug, fetch
date; no absolute path, which would bake one machine's layout into a file that
ships onward). Every pre-existing key is untouched.

### Where the time goes

`process` records per-stage timings on every run (see `runs.jsonl`), which is
worth looking at before optimising anything. A 1 678 km² / 5 m Lierne export,
87 131 river features:

| stage | before | after |
| --- | ---: | ---: |
| building the river polyline network | 1 075.4 s | **20.1 s** |
| rasterising rivers & lakes | 36.6 s | 37.8 s |
| everything else | 73.5 s | 81.8 s |
| **total** | **1 185.5 s** | **139.7 s** |

The river-network pass was 90.7% of the run. It was not an algorithmic problem —
the pass is linear in features — but a single `np.asarray(water_surface,
dtype=np.float64)` **inside** the per-feature loop. On an 8193² export that
rebuilt the entire surface grid as float64 (537 MB), allocated it, converted it,
read a few hundred values out of it and threw it away — once per feature, 87 131
times. Selecting the vertices first and converting those is identical arithmetic
on a few hundred elements instead of 67 million; every exported byte is
unchanged.

Two things worth taking from it. The cost scaled with *grid area × feature
count*, so it stayed invisible on small test regions (Grong, 513×1025, was
0.26 ms/feature; Lierne, 8193², was 12 ms/feature) — and per-stage timings are
what made it visible at all. And a stage that runs for eighteen minutes behind a
label that never changes is indistinguishable from a hang, which is why the
longest loop now reports its own progress and every stage says which step it is.

---

## Reading an export back

The **Exports** page — and `kvterrain/exports.py` behind it — opens a finished
export from the outside: what is in it, which fetch and which settings produced
it, what it looks like, and whether it holds together.

The previews are assembled tile by tile through `core.atlas_tile_offset`, the
same arithmetic a consumer has to implement. That makes the page a test as much
as a viewer: a picture that comes out right is evidence the header and the bytes
agree, which a preview drawn from the in-memory arrays could never be.

Three checks, shared with the CLI so the two cannot drift:

| check | CLI | needs network |
| --- | --- | --- |
| dense atlases: exact size, sampled offsets land on full tiles | `validate-atlas` | no |
| `rivers.bin` walks cleanly, links resolve, junctions name real lakes, `water_id` is dense | `validate-water` | no |
| packed heights vs Kartverket's point API | `validate` | **yes** |

One caveat on the last one, and it is a big one on a water-heavy export: the
packed heights are the **carved** bed, while the point API returns untouched DTM.
A sample landing in a lake disagrees by roughly the carve depth, and that is the
pipeline working. Each sample is therefore tagged with the water class it hit,
and the **dry-only median** is the number that actually measures alignment — on
a 10 m Grong export that is 0.26 m, while the all-sample median is dragged to
0.50 m by one 20.00 m lake sample that is exactly the 20 m carve.

```bash
python -m kvterrain.cli validate-atlas  --out ./exports/lierne
python -m kvterrain.cli validate-water  --out ./exports/lierne
python -m kvterrain.cli validate        --out ./exports/lierne --n 12
```

The ground-truth check calls `ws.geonorge.no/hoydedata/v1/punkt`. If it errors,
check the live Swagger page and adjust `ost`/`nord`/`koordsys` in
`exports.check_against_point_api`.

---

## Source & limits (height)

- Primary fetch: **ArcGIS ImageServer `exportImage`** at
  `https://hoydedata.no/arcgis/rest/services/{DTM,DOM}/ImageServer`, requesting
  `format=tiff, pixelType=F32` — true 32-bit float heights. The request bbox is
  offset by half a texel so pixel centers land on the sample grid (this is what
  makes the output corner-centered).
- **Per-request cap is 15000 px/side**; the tool tiles fetches at ≤4096 by default
  and stitches them seamlessly.
- **DTM1 / DOM1 are 1 m** national coverage — don't set `--spacing` below ~1 m or
  you're just paying to interpolate. 5 m is a sensible default for terrain.
- No API key; open data. Be polite with concurrency.

---

## Water (rivers & lakes)

Unless `--no-water` is given, the tool fetches two NVE datasets for the same
rectangle, in the same UTM zone, stores them raw in the dataset, and at process
time rasterises them onto the **identical** corner-centered sample grid as the
heights:

- **Rivers** — NVE **Elvenett (ELVIS)**, the national river-network database: a
  connected set of polylines _with defined flow direction_. Two layers are read:
  `elvenett` (every stream) and `hovedelv` (main rivers, used to upgrade size).
  Elvenett geometry is strictly **2D** — it has no elevation.
- **Lakes** — NVE **Innsjødatabase** (~243k lakes). The lake polygon layer carries
  a **`hoyde`** attribute: the lake's **surface elevation in metres above sea
  level**. This is the authoritative water level and is used directly.

Water products:

| File | What it is |
| --- | --- |
| `surface.atlas` | water surface elevation per sample, lakes and rivers in one field |
| `water_id.atlas` | water class (dry / river / ocean / lake) plus authored lake identity per sample |
| `rivers.bin` | the river network as densified polylines with links, bed and surface per vertex |
| `lakes.json` | one record per authored lake: NVE ids, name, area, level and its provenance, bbox |
| `junctions.json` | points where a river segment enters (`Inflow`) or leaves (`Outflow`) a lake |

The manifest describes them under `water_surface` (encoding, sources, width
model, lake levels, carve parameters), `water_id` (encoding) and `water_vector`
(the polyline products and their validation report).

### Water class and lake identity (`water_id.atlas`)

One little-endian **uint16** code per sample, same layout as the height atlas:

| code | meaning |
| :---: | --- |
| `0` | dry |
| `1` | river |
| `2` | ocean |
| `3–15` | reserved |
| `16 + n` | lake with local id `n` — the `lake_id` in `lakes.json` |

Class and identity share one channel because the lake id already encodes "is
lake", and 2 bytes per sample keeps the atlas on the same offsets as the other
two. **Ocean** is flood-filled inward from the map edges at `--ocean-level`, not
thresholded, so inland ground below sea level (including carved lake bowls) is
never labelled sea. Precedence at any overlap is lake > river > ocean > dry.
Coarse levels are **categorical**, never averaged: the same precedence over the
3×3 children, and among lakes the id held by the most children.

`lakes.json` emits no polygon geometry; `water_id.atlas` is the geometry
reference (`water_id == 16 + lake_id`).

### Water surface (`surface.atlas`)

Every water sample — lake or river — carries its **surface elevation (moh)**,
packed as uint16 on the **same** `[height_min_m, height_max_m]` range as the
height atlas, with **65535 = no water**. The runtime reads one field for all
water:

```python
import json, numpy as np
from kvterrain import core

m = json.load(open("out/manifest.json"))
TS, (ltx, lty), n = m["tile_samples"], m["leaf_tiles"], m["num_levels"]
off = core.atlas_tile_offset(ltx, lty, n, TS, level=0, x=3, y=2)

def tile(name):
    return np.fromfile(f"out/{name}", dtype="<u2", count=TS * TS,
                       offset=off).reshape(TS, TS)          # row 0 = north

hmin, hmax = m["height_min_m"], m["height_max_m"]
code = tile(m["atlas"]["surface_file"])
is_water = code != 65535
surface_moh = hmin + (code / 65534.0) * (hmax - hmin)     # where is_water
terrain_moh = hmin + (tile(m["atlas"]["height_file"]) / 65535.0) * (hmax - hmin)
depth = np.where(is_water, np.maximum(0, surface_moh - terrain_moh), 0)
```

Where the surface comes from:

- **Lakes:** the lake's resolved level, stamped on every one of its pixels (flat),
  and stored per lake in `lakes.json` as `authored_level_m`.
  **NVE `hoyde` is used exactly as published** wherever it exists — it is the
  lake's authored level, and the DTM gets no vote on it. (It is worth being
  explicit: the DTM is one flight's snapshot of where the water was that day, so on
  a regulated lake it can sit metres below the level the place actually has.
  Second-guessing `hoyde` against it drags recognisable lakes down and pulls their
  shorelines inland.)
  `hoyde` very often does not exist, though (25% of lake records in a Lierne
  export, 11% in a Krøderen one). Those lakes get a level read off the **LiDAR
  water surface inside the polygon**: the mean of its 1st–25th percentile band —
  low enough that land inside an over-wide polygon cannot float the lake, high
  enough that one stray low cell cannot sink it — and then **capped by a perimeter
  scan**, a low percentile of the land ring just outside the polygon, so an
  estimated lake cannot end up standing above the terrain that encircles it.
  `--no-estimate-lake-levels` turns the estimate off (those lakes are then left
  uncarved rather than guessed at); `--no-lake-perimeter-cap` turns off the cap.
  `lakes.json` carries the level used as `authored_level_m` and its provenance as
  `level_source` (`nve_hoyde` or `dtm_interior_low`); the raw NVE field stays
  visible, and null, as `hoyde_moh`.
- **Rivers:** Elvenett has no elevation, so the surface is the **terrain height
  under the channel's own centreline**, sampled from the leaf DTM before any carve
  and spread flat across the samples that centreline seeded. So a river surface is
  **level across a channel and descends along it**, like water — it is not each
  sample's own ground, which would paint water onto whatever the rasterised buffer
  covers and send a thin film climbing every cliff a channel runs past. A sample
  more than `--river-bank-tolerance` (default 2 m) above that level is a bank: it
  is neither carved nor flooded, so the waterline lands where the ground crosses
  the level rather than where the buffer ends. The water column itself comes from a
  **trench carved under the river** (see below). Where a channel meets a lake the
  surface is **raised** to the lake's level — never lowered to it, or the last texel
  or two of river would clip to zero depth and render as a dry gap.

### The carves

The DTM records a lake's **surface** flat at the water level (LiDAR gets no bed
return), and a river's channel is likewise only as deep as the DTM resolved it.
Neither leaves room for water in a depth renderer, so the tool cuts the bed down
and leaves the water surface where the ground is:

- **Lakes** descend on a **straight ramp** at `--lake-shore-slope` metres of depth
  per metre of shore (default **1.0**) until they reach `--lake-max-depth` (default
  20 real metres), and are flat from there inward. At the defaults that is 20 m of
  depth over 20 m of shore — four texels at 5 m spacing, giving 0 / 5 / 10 / 15 / 20.
  `--lake-ramp-radius` sets the ramp length in metres directly if you would rather
  not think in slopes; `--lake-min-depth` (default 2 m) is the floor every body
  reaches at its deepest sample, so a pond too small to ramp still holds water.

  A lake wider than twice the ramp reaches full depth in the middle, which at the
  default slope is most lakes. That is deliberate. An earlier revision stretched the
  ramp to 150 m so that lake *size* decided depth and small lakes stayed shallow —
  and the cost was that near-shore water was too shallow to render, so lakes came out
  as rounded blobs sitting inside their own shorelines with every bay and headland
  gone (measured at a 0.5 m visibility threshold: Kroktjønna rendered 55% of its own
  polygon, median depth 0.61 m). Depth now follows distance from shore and nothing
  else. Lower `--lake-max-depth` for shallower lakes, `--lake-shore-slope` for
  gentler sides.
- **Rivers** get a trench under the channel, deepest along the middle and feathering
  out at the banks, scaled by stream order (1.5 m for a headwater stream to 11 m for
  an order-8 trunk, × `--river-depth-scale`) and by the modelled channel width. It
  is cut in full only within `--river-bank-tolerance` of the water level and tapers
  to nothing at twice that, so terrain that will never hold water — a bank, a cliff
  face the buffer lapped onto — is left alone rather than notched.

Neither carve is surveyed bathymetry; both exist to give the renderer enough depth
to read opaque. (If your engine compresses height, e.g. 1:5, size them so they still
clear your opaque threshold after compression: 20 m → 4 units at 1:5.)

**Shoreline snap.** An NVE outline and a Kartverket LiDAR block are independent
products, so the polygon boundary lands *near* — not on — the flat plane the flight
recorded for that lake, and it is then rasterised centre-in-polygon on top of that.
The leftover ring, one or two texels wide, IS the lake's own water surface and would
be classified as dry land: it keeps its flown height while the lake beside it is
carved to the authored level, so it renders as a raised rim tracing the true
shoreline just outside the water. `--lake-snap-px` (default 2) lets a lake claim
those samples back — only samples within a couple of texels whose height is within
0.35 m of that lake's own flown surface, so real bank stops the growth immediately.
On the export that showed it, Kroktjønna went from 855 such samples rendered as land
to 50, and its mask grew 9.6% to match the real shoreline.

**Islands** — holes in a lake polygon — are rasterised **as lake**, so the water
surface runs underneath them continuously, but their DTM height is left alone and
they act as shore for the depth ramp. `depth = max(0, level − terrain)` then hides
the surface again wherever the island stands above the water. Without this, the
half-texel misalignment between a hole boundary and the sample lattice leaves a
one-texel dry moat around every island. `--keep-lake-holes` restores the old
behaviour.

### Design decisions worth knowing

- **Surface, not depth, is baked.** Depth is a property of water + terrain, so it
  breaks under terrain edits; surface (moh) is a property of the water alone. Bake
  the surface and the runtime re-derives depth from whatever terrain is current —
  so an edited or streamed-over channel stays consistent with no reseed.
- **Flow direction from geometry.** ELVIS polylines are digitised downstream, so a
  segment's bearing comes from its vertex order — no dependence on attribute names.
- **River width & depth are modelled.** ELVIS has no width or discharge, so channel
  width (pixels seeded) and carve depth are heuristics keyed on Strahler order
  (`manifest.water_surface.width_model`, scaled by `--river-width-scale`). A river's
  rendered depth is a modeled constant; its surface _is_ the real terrain.
- **Depth is carved, never raised.** Both water types put the surface on the ground
  and cut the bed beneath it. Raising a surface above the terrain instead — which
  is how river depth used to be produced — guarantees a water column at the price of
  water standing proud of the landscape, which is exactly what it looks like.
- **A water surface is a level, not a copy of the ground.** Lakes get one level per
  lake; rivers get one level per channel cross-section, from the centreline. Any
  rule that gives each _sample_ its own surface height paints water onto terrain —
  the same artefact whether the surface floats above the ground or sits on it, and
  most visible where the ground is steepest.
- **Priority is lake > river > ocean > dry** at any overlap, at every level.
- **Two different pyramids.** `water_id.atlas` uses a **categorical** downsample
  (highest-priority class in the 3×3 neighbourhood wins, so thin rivers survive
  to the coarsest level). `surface.atlas` uses a **water-only** downsample: a
  parent is water if **any** child is, and its surface is the **min over water
  children** (the conservative spill level). This is what keeps thin rivers and
  small lakes visible at the coarse fallback resolution instead of averaging into
  their banks.
- **Rivers are also kept as polylines.** Rasterising a river destroys its flow
  direction, its connectivity and its links to lakes, so `rivers.bin` keeps them:
  densified vertices with bed `z` and surface `level`, upstream/downstream links
  and lake spans. `level` is read verbatim from the surface raster, so the two
  agree by construction.
- **Same shared-edge guarantee.** All atlases are sliced from one assembled array
  per product, so adjacent edges are bit-identical and mutually aligned.

`manifest.water_surface` records the surface encoding, the `65535` sentinel, the
runtime depth formula, the width model, attribution and the lake level and carve
reports. `manifest.water_id` records the class/identity encoding.
`manifest.water_vector` names the polyline products and carries their validation.

### Water source & limits

- Fetch: **ArcGIS REST `/query?f=geojson`** on
  `kart.nve.no/enterprise/rest/services/Elvenett1` (layers `2` elvenett, `1`
  hovedelv) and `.../Innsjodatabase2` (layer `5`). NVE migrated all their map
  services to this cloud host in Dec 2025 — the old `nve.geodataonline.no` no
  longer resolves (the domain was retired; see NVE's
  [migration notice](https://www.nve.no/kart/nytt-om-gis-api/nye-url-er-for-alle-nve-s-karttjenester/)).
  If the endpoints move again, browse `kart.nve.no/enterprise/rest/services` and
  update `RIVER_SERVICE` / `LAKE_SERVICE` at the top of `kvterrain/water.py`.
  The server reprojects to the plan's UTM zone via `outSR`; results page at up to
  2000 features.
- Attribute field names (stream order, lake number/name/area, **`hoyde`**) are
  **not contractually stable**. Every attribute read is optional with a graceful
  fallback; candidate names live in constants at the top of `kvterrain/water.py`
  (`RIVER_ORDER_FIELDS`, `LAKE_*_FIELDS`, `LAKE_HOYDE_FIELDS`) — adjust there if a
  query returns empty attributes. Flow direction and geometry don't depend on
  attributes and will still be correct; a lake missing `hoyde` falls back to a DTM
  shoreline estimate.
- No API key; open data. Be polite with concurrency.

---

## Unity import notes

The atlases are for a runtime that reads tiles by byte offset, not for Unity's
terrain importer. A single tile is nonetheless exactly the raw 16-bit layout
Unity's terrain RAW import expects, so one can be cut out with
`core.atlas_tile_offset` and imported by hand:

- **Bit depth:** 16-bit. **Byte order:** Windows / little-endian.
- **Resolution:** `tile_samples` (129 by default).
- **Flip:** rows are written **north→south**; use Unity's RAW vertical-flip toggle
  (or the manifest's `"row_order": "north_to_south"` if you load bytes yourself).
- **Vertical scale:** map packed `[0,65535]` → `[height_min_m, height_max_m]`. In a
  `TerrainData`, set height `size.y = height_max_m − height_min_m` and offset the
  object by `height_min_m`.
- **Horizontal placement:** there is no per-tile table. Tile `(x, y)` at level `L`
  covers `tile_cells × leaf_spacing_m × 2^L` metres per side, measured from
  `origin_utm` (the south-west corner): `x` counts east and `y` counts **north**,
  so `y = 0` is the southernmost row of tiles even though samples inside a tile run
  north → south (see `north_up_tile_slice` in `kvterrain/core.py`).

Because tiles share edges, neighbouring patches line up seamlessly at matching
levels (or morph CDLOD-style at the shared edge).

**Water** (`surface.atlas`, `water_id.atlas`) is data for the runtime water
system, not Unity terrain layers. It shares the height atlas's layout, row order
and offsets, so the same placement logic applies.

- `water_id.atlas` is categorical — sample it **nearest-neighbour**, never
  bilinear.
- `surface.atlas` is a scalar height field — it can be filtered like height, but
  **do not let filtering blend the `65535` sentinel** into real water. Gate on the
  sentinel (or on `water_id != 0`) first.
- Runtime depth for any water sample is `max(0, surface_moh − terrain_moh)`. If
  your engine applies a vertical scale to terrain height (e.g. 1:5 compression),
  apply the **identical** scale to the unpacked surface before subtracting — the
  surface is packed in the same moh space as the heights precisely so the two are
  comparable.
- `hoyde` is a reference level and the DTM is a single-day capture, so a lake's
  waterline won't trace the DTM contour to the centimetre — expect a thin shore
  ring that may read slightly wet or dry. `max(0, …)` handles it; it's data, not a
  bug. Measured over a Lierne block, `hoyde` sits above the LiDAR shoreline on
  about 11% of shore samples, by up to 0.92 m. That is **deliberately not
  corrected**: the alternative — pulling `hoyde` down to the DTM — is metres wrong
  on any regulated lake, where the flight caught the reservoir drawn down. A
  sub-metre shore ring is the cheaper error, and the first texel of the shore ramp
  feathers it.

---

## Files

**The split**

- `kvterrain/fetch.py` — **stage one**. The only module that talks to the height
  and water services on its own account. Assembles the lattice, pulls the NVE
  vectors, writes a dataset, stops. Also holds the single copy of the synthetic
  demo surface (it used to exist verbatim in both `cli.py` and `app.py`).
- `kvterrain/dataset.py` — the intermediate format: `Lattice` (the sample grid,
  and which tilings it supports), `Dataset` (read one), `write` (write one),
  `list_datasets`, plus the municipality name suggestion.
- `kvterrain/process.py` — **stage two**. The whole post-fetch sequence —
  rasterise, repair, snap, resolve levels, river surface, carve, pyramid, water
  id, polylines, pack — with staged progress and per-stage timings. This is
  `core.run_export`'s old second half, unmoved and unchanged in behaviour.
- `kvterrain/exports.py` — the only module that opens a finished export from the
  OUTSIDE: reassembles a pyramid level from an atlas by computed byte offset,
  unpacks it, and holds the three checks (`check_atlas`, `check_water`,
  `check_against_point_api`) that both the CLI and the Exports page run.
- `kvterrain/runlog.py` — the per-dataset `runs.jsonl`: each process run's
  settings paired with the outcomes they produced.

**The engine**

- `kvterrain/core.py` — headless height engine (plan, fetch, assemble, pyramid,
  slice, pack) + the shared `north_up_tile_slice` used by every tile packer, and
  the dense-atlas offset arithmetic that IS the format contract. `run_export` is
  now a thin fetch-then-process wrapper for one-shot builds. No UI or Unity deps.
- `kvterrain/water.py` — water engine: fetch NVE rivers/lakes (`fetch_water_geojson`
  returns them raw, for the dataset; `fetch_water_features` parses), rasterise
  onto the height grid, lake identity and levels, shoreline snap. Pulls in
  rasterio/shapely; imported lazily.
- `kvterrain/bathymetry.py` — resolves lake levels (published `hoyde`, else the
  LiDAR interior capped by a perimeter scan) and carves both beds: lakes deepening
  inward on a straight shore ramp, rivers as a trench under the channel that stops
  at the waterline.
- `kvterrain/watersurface.py` — computes the unified per-pixel water surface
  (lakes = `hoyde`, rivers = the levelled channel ground) and packs `surface.atlas`.
- `kvterrain/waterid.py` — per-pixel class + authored lake identity, packed into
  `water_id.atlas`.
- `kvterrain/rivernet.py` — the polyline network: connectivity, junctions,
  validation, `rivers.bin` / `lakes.json` / `junctions.json`.
- `kvterrain/preview.py` — hillshade, hypsometric, depth, signed-delta and water
  ramps; the dataset thumbnail. Presentation only, numpy + Pillow, no matplotlib.
- `kvterrain/cli.py` — `fetch`, `datasets`, `process`, `build`, `validate`,
  `validate-atlas`, `validate-water`, `describe-services`. The three `validate*`
  commands print reports that `exports.py` produces, so the CLI and the UI check
  exactly the same things.

**The UI**

- `app.py` — the router: one Streamlit process, three pages.
- `ui/fetch_page.py` — draw a rectangle, fetch it, store it.
- `ui/process_page.py` — pick a dataset, tune, run, look at what it did.
- `ui/exports_page.py` — open a finished export, preview it from its own bytes,
  run the checks.
- `ui/widgets.py` — shared formatting, dataset cards, map overlays, image panels.

## Known caveats

- Region **outer rim** height decimation is an edge-clamp approximation (interior
  is exact). Extend the rectangle slightly if you need the rim exact.
- Non-square regions produce a small **forest** of root tiles rather than one root;
  the pyramid stops when the smaller axis reaches a single tile.
- **Water attribute names** on the NVE services are best-effort. If a build reports
  0 lakes/rivers, all rivers come out the same size, or lakes lack a surface, the
  layer's field names likely changed — check the live REST layer and update the
  candidate lists in `kvterrain/water.py`.
- **Modelled river width & depth** are heuristics, not surveyed values. Tune
  `--river-width-scale` / the `width_by_order` table; width decides how many
  pixels a river seeds, how wide its level is flattened, and how wide the trench feathers.
- **Lake depth follows distance from shore**, not lake size: at the default 1:1
  slope every lake deeper than four texels from its bank is at `--lake-max-depth`.
  If you want small lakes shallower, lower that; `--lake-min-depth` is only the
  floor for bodies too small to ramp at all.
- **River surfaces are not forced to descend.** A river's level is the uncarved
  ground under its centreline, measured, never fitted (see `channel_level`), so DTM
  noise can make it step UP going downstream. `rivers.bin` reports this rather
  than fixing it: on the Lierne export 10.5% of open-channel vertex steps rise, in
  65 395 of 87 126 segments, worst 4.8 m (`water_vector.validation.descent_*` in
  the manifest). Enforcing descent is left to the runtime's running minimum when
  it burns the polylines.
- **Confluences get no special treatment, and need none for the surface.** Every
  river sample takes the level of the NEAREST centreline sample across all
  centrelines, so tributaries meeting a trunk share one level field rather than
  per-segment surfaces that could disagree. The segment graph in `rivers.bin`
  (upstream/downstream links, the highest-order candidate taken as the trunk) is
  built but not used to shape the surface.
- **Lake junctions are raised, not pinned.** A river sample touching a lake is
  lifted to the lake's level where the lake stands higher and otherwise keeps its
  own level, so a channel arriving over a bank stays wet up to the shoreline.
- **Coarse rivers fade.** A ~5 m channel in a coarse cell has its bed averaged with
  its banks, so `surface − coarse_terrain` shrinks and distant thin rivers render
  faint even though the surface field keeps the cell marked as water. Carving the
  channel instead of raising its surface made this start one level earlier: over a
  Lierne block, river samples still holding >0.25 m of depth went 100% / 74% / 35%
  (L0/L1/L2) under the old raise and 100% / 40% / 8% under the trench. `--river-depth-scale`
  is the knob if distant rivers matter more than close-up ones; the structural fix
  would be to re-apply the channel carve to each pyramid level, which is not done
  because it widens the trench to a whole coarse cell and pops as you approach.
- `hovedelv` is slightly generalised; where it deviates from `elvenett` you can get
  a faint double line. `--no-main-rivers` avoids it at the cost of the size signal.
- The NVE layer ids and attribute names can change without notice. After any
  fetch that looks wrong, run `validate` for height and `validate-water` for the
  water products, and use `describe-services` to check the live layers and fields.
