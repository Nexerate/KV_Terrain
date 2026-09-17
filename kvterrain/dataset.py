"""
kvterrain.dataset
=================

The intermediate format that splits this tool in two.

Fetching a region takes minutes and hits three remote services; post-processing
it takes minutes more and is the part you actually iterate on (lake carving,
river rasterisation, level estimation, packing). Before this module the two were
one pass, so tuning a carve meant re-downloading the same twenty square
kilometres of Kartverket LiDAR and the same fifty thousand NVE segments.

A **dataset** is that fetch, frozen on disk: the assembled height lattice exactly
as the servers returned it, plus the raw NVE vectors that intersect it. Nothing
in it has been carved, rasterised, decimated, packed or interpreted. It is the
input to `kvterrain.process`, and it can be reprocessed any number of times with
different settings at no network cost.

Layout
------

    data/datasets/<slug>/
      dataset.json               this manifest
      heights.npy                float32 (samples_y, samples_x), north-up
      preview.png                thumbnail for the picker (hillshade + water)
      water/rivers.geojson.gz    NVE elvenett  — raw GeoJSON features
      water/main_rivers.geojson.gz  NVE hovedelv — raw GeoJSON features
      water/lakes.geojson.gz     NVE Innsjødatabase — raw GeoJSON features

Design decisions worth knowing
------------------------------

* **The heights are stored uncooked.** float32, NaN for nodata, north-up rows,
  on the corner-centered sample lattice `core` fetches into — i.e. the exact
  array `core.assemble_region` returns, before void repair, before the shoreline
  snap, before any carve. Every one of those passes is a post-processing
  decision, and freezing a repaired array would bake today's repair into every
  future run. `.npy` (not raw `.f32`) so the file is self-describing and
  `np.load(mmap_mode="r")` works.

* **The water vectors are stored RAW, as GeoJSON, not as parsed WaterFeatures.**
  `water.features_from_geojson` does fuzzy attribute resolution against NVE's
  not-contractually-stable field names, and that resolution is exactly the kind
  of thing we improve without wanting to refetch. Raw features mean a better
  field map applies retroactively to datasets already on disk. Gzipped because
  ELVIS GeoJSON is mostly repeated key names — typically 8-12x.

* **The lattice, not the tile grid, is the contract.** A dataset records
  `origin/spacing/samples`, not `leaf_tiles`/`num_levels`. `plan_grid` always
  pads the leaf cell count out to a power of two, so the SAME lattice supports
  every power-of-two `tile_cells` up to its own size (128 cells can be sliced as
  4x32, 2x64 or 1x128 tiles). `tile_cells` is therefore a post-processing
  choice, recorded here only as the default the fetch was planned with. See
  `Lattice.valid_tile_cells`.

* **Nothing here is a Unity product.** The final format (`heights.atlas`,
  `surface.atlas`, `water_id.atlas`, `rivers.bin`, `manifest.json`) is unchanged
  and still produced only by `kvterrain.process`. This format is ours to change.

Data © Kartverket (CC BY 4.0) and © NVE (Elvenett / Innsjødatabase).
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import core

DATASET_FORMAT = "kvterrain-dataset/1"

MANIFEST_NAME = "dataset.json"
HEIGHTS_NAME = "heights.npy"          # uncompressed; still read, no longer written
HEIGHTS_GZ_NAME = "heights.hgt.gz"    # byte-shuffled + gzip (the default)
PREVIEW_NAME = "preview.png"
WATER_DIR = "water"
RIVERS_NAME = "water/rivers.geojson.gz"
MAIN_RIVERS_NAME = "water/main_rivers.geojson.gz"
LAKES_NAME = "water/lakes.geojson.gz"

# Where datasets live by default: project-relative, so a checkout is
# self-contained and the picker always has one place to look.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data", "datasets")

# Kartverket's municipality lookup — same Geonorge service family as the height
# point API, so a name suggestion costs one call to a host we already depend on.
KOMMUNE_API = "https://ws.geonorge.no/kommuneinfo/v1/punkt"


# --------------------------------------------------------------------------- #
# Height storage                                                               #
#                                                                              #
# A 4097² lattice is 67 MB of float32, and the whole point of a dataset is that #
# you keep several. Plain gzip on float32 elevations only saves ~17%: the high  #
# bytes of neighbouring samples are nearly identical but the mantissa bytes are #
# noise, and interleaving them hands zlib no runs to find.                      #
#                                                                              #
# A BYTE SHUFFLE fixes that. Group byte-plane 0 of every sample, then plane 1,  #
# and so on, so each plane is internally smooth. Measured on real Kartverket    #
# DTM10: 83% -> 60% of raw, and decompression is ~60 ms on a full lattice, i.e. #
# lost in the noise next to a process run. Fully lossless, so it is the default.#
#                                                                              #
# The quantised modes go further by rounding to a fixed step before shuffling.  #
# They are LOSSY and off by default — but note Kartverket's DTM is accurate to  #
# ~10-20 cm, so storing float32's ~1e-5 m of mantissa is storing noise. `mm`    #
# rounds 200x finer than the source data is accurate and still saves half.      #
# --------------------------------------------------------------------------- #

NODATA_I32 = -2147483648          # INT32_MIN, the sentinel for NaN when quantised

COMPRESSION_MODES = {
    # name       quantum_m  human label
    "none":     (None,      "uncompressed float32 (.npy, memory-mappable)"),
    "lossless": (None,      "byte-shuffled float32 + gzip — exact"),
    "mm":       (0.001,     "rounded to 1 mm + gzip — 200x finer than the DTM's "
                            "own accuracy"),
    "cm":       (0.01,      "rounded to 1 cm + gzip — at the DTM's own accuracy"),
}
DEFAULT_COMPRESSION = "lossless"


def _shuffle(buf: bytes, itemsize: int) -> bytes:
    """Interleaved samples -> byte planes. Its own inverse is `_unshuffle`."""
    return np.frombuffer(buf, np.uint8).reshape(-1, itemsize).T.copy().tobytes()


def _unshuffle(buf: bytes, itemsize: int) -> bytes:
    return np.frombuffer(buf, np.uint8).reshape(itemsize, -1).T.copy().tobytes()


def write_heights(dest: str, heights: np.ndarray,
                  compression: str = DEFAULT_COMPRESSION) -> dict:
    """Write the lattice and return the `heights` manifest block's codec info."""
    if compression not in COMPRESSION_MODES:
        raise ValueError(f"unknown compression {compression!r}; "
                         f"expected one of {sorted(COMPRESSION_MODES)}")
    heights = np.ascontiguousarray(heights, dtype=np.float32)

    if compression == "none":
        np.save(os.path.join(dest, HEIGHTS_NAME), heights, allow_pickle=False)
        return {"file": HEIGHTS_NAME, "codec": "npy", "dtype": "float32",
                "quantum_m": None, "lossless": True}

    quantum = COMPRESSION_MODES[compression][0]
    if quantum is None:
        payload, itemsize, dtype = heights.tobytes(), 4, "float32"
    else:
        q = np.rint(heights / quantum).astype(np.int32)
        q[~np.isfinite(heights)] = NODATA_I32
        payload, itemsize, dtype = q.tobytes(), 4, "int32"

    with gzip.open(os.path.join(dest, HEIGHTS_GZ_NAME), "wb", compresslevel=6) as f:
        f.write(_shuffle(payload, itemsize))

    return {"file": HEIGHTS_GZ_NAME, "codec": "shuffle+gzip", "dtype": dtype,
            "quantum_m": quantum, "itemsize": itemsize,
            "lossless": quantum is None,
            "layout": ("gzip of a byte-shuffled C-order array: all byte-plane 0 of "
                       "every sample, then all byte-plane 1, and so on. Undo the "
                       "shuffle, then read as `dtype`; when `quantum_m` is set, "
                       f"multiply by it and treat {NODATA_I32} as nodata.")}


def read_heights(root: str, block: dict, shape: tuple) -> np.ndarray:
    """Read a lattice back as float32 with NaN for nodata, whatever the codec."""
    path = os.path.join(root, block.get("file", HEIGHTS_NAME))
    if block.get("codec", "npy") == "npy" or path.endswith(".npy"):
        return np.load(path, allow_pickle=False)

    itemsize = int(block.get("itemsize", 4))
    with gzip.open(path, "rb") as f:
        flat = _unshuffle(f.read(), itemsize)

    quantum = block.get("quantum_m")
    if quantum is None:
        return np.frombuffer(flat, dtype=np.float32).reshape(shape).copy()

    q = np.frombuffer(flat, dtype=np.int32).reshape(shape)
    out = q.astype(np.float32) * np.float32(quantum)
    out[q == NODATA_I32] = np.nan
    return out


# --------------------------------------------------------------------------- #
# The lattice                                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Lattice:
    """
    The sample grid a dataset was fetched onto: a corner-centered
    ("pixel-is-a-point") lattice of `samples_x` x `samples_y` points, spaced
    `spacing_m` apart, with sample (0, 0) at the SW corner `(origin_x, origin_y)`
    in EPSG:`epsg`. Rows in the stored array run north -> south, so array row 0
    is the NORTHMOST sample line — the same convention `core` uses end to end.
    """
    epsg: int
    spacing_m: float
    origin_x: float
    origin_y: float
    samples_x: int
    samples_y: int

    # -- derived geometry --------------------------------------------------- #
    @property
    def leaf_cells_x(self) -> int:
        return self.samples_x - 1

    @property
    def leaf_cells_y(self) -> int:
        return self.samples_y - 1

    @property
    def width_m(self) -> float:
        return self.leaf_cells_x * self.spacing_m

    @property
    def height_m(self) -> float:
        return self.leaf_cells_y * self.spacing_m

    @property
    def area_km2(self) -> float:
        return self.width_m * self.height_m / 1e6

    @property
    def bbox_utm(self) -> tuple[float, float, float, float]:
        return (self.origin_x, self.origin_y,
                self.origin_x + self.width_m, self.origin_y + self.height_m)

    def bbox_lonlat(self) -> tuple[float, float, float, float]:
        """The extent as (lon_min, lat_min, lon_max, lat_max), for map overlays."""
        from pyproj import Transformer
        tr = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        x0, y0, x1, y1 = self.bbox_utm
        xs, ys = tr.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        return (min(xs), min(ys), max(xs), max(ys))

    def centre_utm(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_utm
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def outline_latlon(self) -> list[tuple[float, float]]:
        """Closed (lat, lon) ring of the extent, ready for folium.Polygon."""
        from pyproj import Transformer
        tr = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        x0, y0, x1, y1 = self.bbox_utm
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        return [(lat, lon) for lon, lat in (tr.transform(x, y) for x, y in ring)]

    # -- tiling ------------------------------------------------------------- #
    def valid_tile_cells(self, choices=(64, 128, 256, 512)) -> list[int]:
        """
        Every `tile_cells` this lattice can be sliced into.

        `plan_grid` pads the leaf cell count out to `tile_cells * 2^k`, and both
        factors are powers of two, so the fetched cell count is itself a power of
        two. That means the tiling is a free choice after the fact: any
        power-of-two tile size that divides both axes into a power-of-two number
        of tiles yields the same dense pyramid the fetch would have produced.
        """
        out = []
        for tc in choices:
            try:
                self.check_tile_cells(int(tc))
            except ValueError:
                continue
            out.append(int(tc))
        return out

    def check_tile_cells(self, tile_cells: int) -> tuple[int, int, int]:
        """(leaf_tiles_x, leaf_tiles_y, num_levels), or ValueError with a reason."""
        tc = int(tile_cells)
        if tc <= 0 or tc & (tc - 1):
            raise ValueError(f"tile_cells must be a power of two, got {tc}")
        lcx, lcy = self.leaf_cells_x, self.leaf_cells_y
        if lcx % tc or lcy % tc:
            raise ValueError(
                f"tile_cells {tc} does not divide this dataset's {lcx}x{lcy} leaf "
                f"cells — the tile grid would be ragged and the atlas offsets "
                f"would misalign")
        ltx, lty = lcx // tc, lcy // tc
        for n, axis in ((ltx, "x"), (lty, "y")):
            if n & (n - 1):
                raise ValueError(
                    f"tile_cells {tc} gives {n} tiles on {axis}, which is not a "
                    f"power of two — the pyramid could not halve cleanly")
        return ltx, lty, int(math.log2(min(ltx, lty))) + 1

    def plan(self, tile_cells: int) -> core.GridPlan:
        """Rebuild the `core.GridPlan` this lattice implies for a given tiling."""
        ltx, lty, levels = self.check_tile_cells(tile_cells)
        return core.GridPlan(
            epsg=int(self.epsg),
            spacing_m=float(self.spacing_m),
            origin_x=float(self.origin_x),
            origin_y=float(self.origin_y),
            tile_cells=int(tile_cells),
            leaf_tiles_x=int(ltx),
            leaf_tiles_y=int(lty),
            num_levels=int(levels),
        )

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "epsg": int(self.epsg),
            "spacing_m": float(self.spacing_m),
            "origin_x": float(self.origin_x),
            "origin_y": float(self.origin_y),
            "samples_x": int(self.samples_x),
            "samples_y": int(self.samples_y),
            "row_order": "north_to_south",
            "sampling": "corner_centered",
            "bbox_utm": [float(v) for v in self.bbox_utm],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Lattice":
        return cls(
            epsg=int(d["epsg"]),
            spacing_m=float(d["spacing_m"]),
            origin_x=float(d["origin_x"]),
            origin_y=float(d["origin_y"]),
            samples_x=int(d["samples_x"]),
            samples_y=int(d["samples_y"]),
        )

    @classmethod
    def from_plan(cls, plan: core.GridPlan) -> "Lattice":
        return cls(
            epsg=plan.epsg,
            spacing_m=plan.spacing_m,
            origin_x=plan.origin_x,
            origin_y=plan.origin_y,
            samples_x=plan.samples_x,
            samples_y=plan.samples_y,
        )


# --------------------------------------------------------------------------- #
# Naming                                                                       #
# --------------------------------------------------------------------------- #

def slugify(name: str) -> str:
    """Folder-safe, lowercase, ASCII. 'Indre Østfold' -> 'indre-ostfold'."""
    s = (name or "").strip()
    # Norwegian letters do not decompose usefully under NFKD (ø has no base
    # letter + mark form), so transliterate the three explicitly first.
    for a, b in (("ø", "o"), ("Ø", "O"), ("æ", "ae"), ("Æ", "AE"),
                 ("å", "aa"), ("Å", "AA")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "dataset"


def suggest_name(lattice: Lattice, *, session=None, timeout: int = 15) -> str:
    """
    A human name for a fetch, from Kartverket's municipality lookup at the
    region's centre point ("Lierne", "Indre Østfold").

    Purely a convenience — the caller always gets to override it, and any failure
    (offline, sea, service moved) falls back to a coordinate name that is still
    unique and still tells you where you were.
    """
    cx, cy = lattice.centre_utm()
    try:
        import requests
        sess = session or requests.Session()
        r = sess.get(KOMMUNE_API,
                     params={"nord": cy, "ost": cx, "koordsys": lattice.epsg},
                     timeout=timeout)
        r.raise_for_status()
        name = (r.json() or {}).get("kommunenavn")
        if name:
            return str(name)
    except Exception:      # noqa: BLE001 — a name is never worth failing a fetch over
        pass
    return f"N{cy / 1000:.0f}-E{cx / 1000:.0f}"


def unique_dir(root: str, slug: str) -> str:
    """`root/slug`, or `root/slug-2`, `-3`… so a repeat fetch never overwrites."""
    base = os.path.join(root, slug)
    if not os.path.exists(base):
        return base
    for n in range(2, 1000):
        cand = f"{base}-{n}"
        if not os.path.exists(cand):
            return cand
    raise RuntimeError(f"could not find a free directory name for {slug!r} in {root}")


# --------------------------------------------------------------------------- #
# Reading a dataset                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    """A fetched region on disk. Cheap to construct — nothing is read until asked."""
    root: str
    manifest: dict = field(repr=False)

    # -- identity ----------------------------------------------------------- #
    @property
    def name(self) -> str:
        return self.manifest.get("name") or os.path.basename(self.root)

    @property
    def slug(self) -> str:
        return self.manifest.get("slug") or os.path.basename(self.root)

    @property
    def created_utc(self) -> str:
        return self.manifest.get("created_utc", "")

    @property
    def source_kind(self) -> str:
        return self.manifest.get("source", {}).get("kind", "DTM")

    @property
    def is_demo(self) -> bool:
        return bool(self.manifest.get("source", {}).get("demo"))

    @property
    def lattice(self) -> Lattice:
        return Lattice.from_dict(self.manifest["lattice"])

    @property
    def default_tile_cells(self) -> int:
        tc = int(self.manifest.get("request", {}).get("tile_cells", 128))
        valid = self.lattice.valid_tile_cells()
        return tc if tc in valid else (valid[-1] if valid else tc)

    @property
    def has_water(self) -> bool:
        return bool(self.manifest.get("water", {}).get("present"))

    @property
    def bytes_total(self) -> int:
        return int(self.manifest.get("bytes_total", 0)) or dir_bytes(self.root)

    def plan(self, tile_cells: Optional[int] = None) -> core.GridPlan:
        return self.lattice.plan(tile_cells or self.default_tile_cells)

    # -- payloads ----------------------------------------------------------- #
    def heights_path(self) -> str:
        return os.path.join(self.root, self.manifest["heights"].get("file",
                                                                    HEIGHTS_NAME))

    @property
    def compression(self) -> str:
        """The mode this dataset's lattice was stored with, for display."""
        block = self.manifest.get("heights", {})
        if block.get("codec", "npy") == "npy":
            return "none"
        q = block.get("quantum_m")
        return "lossless" if q is None else ("mm" if q == 0.001 else
                                             "cm" if q == 0.01 else f"{q} m")

    def heights(self) -> np.ndarray:
        """
        The assembled lattice as fetched: float32, NaN where the server had no
        data, north-up rows — decoded from whatever codec it was stored with.

        Always a private, writable array. Every post-processing pass writes into
        the bed, and handing out a shared (or read-only, or memory-mapped) buffer
        would mean the second `process` run of a dataset saw the first one's carve.
        """
        lat = self.lattice
        shape = (lat.samples_y, lat.samples_x)
        arr = read_heights(self.root, self.manifest.get("heights", {}), shape)
        if arr.shape != shape:
            raise ValueError(
                f"{self.heights_path()} is {arr.shape}, but dataset.json says the "
                f"lattice is {shape} — the dataset is inconsistent")
        return arr

    def water_geojson(self) -> tuple[list, list, list]:
        """(elvenett, hovedelv, lakes) raw GeoJSON feature lists."""
        if not self.has_water:
            return [], [], []
        return (read_gz_json(os.path.join(self.root, RIVERS_NAME)),
                read_gz_json(os.path.join(self.root, MAIN_RIVERS_NAME)),
                read_gz_json(os.path.join(self.root, LAKES_NAME)))

    def water_features(self):
        """
        Parse the stored vectors into `water.WaterFeatures`.

        Parsing happens on every process run rather than at fetch time on
        purpose: the attribute resolution in `features_from_geojson` is a moving
        target against NVE's field names, and re-running it means a fix reaches
        datasets already on disk. It costs seconds; the fetch costs minutes.
        """
        if not self.has_water:
            return None
        from . import water
        rivers, main_rivers, lakes = self.water_geojson()
        return water.features_from_geojson(rivers, main_rivers, lakes)

    def preview_path(self) -> Optional[str]:
        p = os.path.join(self.root, PREVIEW_NAME)
        return p if os.path.exists(p) else None

    # -- summary for the picker --------------------------------------------- #
    def summary(self) -> dict:
        """Flat facts for a dataset card — no large reads."""
        lat = self.lattice
        h = self.manifest.get("heights", {})
        w = self.manifest.get("water", {})
        return {
            "name": self.name,
            "slug": self.slug,
            "root": self.root,
            "created_utc": self.created_utc,
            "source": self.source_kind,
            "demo": self.is_demo,
            "epsg": lat.epsg,
            "spacing_m": lat.spacing_m,
            "samples": (lat.samples_x, lat.samples_y),
            "width_m": lat.width_m,
            "height_m": lat.height_m,
            "area_km2": lat.area_km2,
            "bbox_utm": lat.bbox_utm,
            "bytes_total": self.bytes_total,
            "compression": self.compression,
            "height_min_m": h.get("min_m"),
            "height_max_m": h.get("max_m"),
            "coverage_pct": h.get("coverage_pct"),
            "has_water": self.has_water,
            "rivers": w.get("rivers", {}).get("count", 0),
            "main_rivers": w.get("main_rivers", {}).get("count", 0),
            "lakes": w.get("lakes", {}).get("count", 0),
            "tile_cells": self.default_tile_cells,
            "valid_tile_cells": lat.valid_tile_cells(),
        }


def load(root: str) -> Dataset:
    with open(os.path.join(root, MANIFEST_NAME)) as f:
        man = json.load(f)
    fmt = man.get("format", "")
    if not str(fmt).startswith("kvterrain-dataset/"):
        raise ValueError(f"{root} is not a kvterrain dataset (format={fmt!r})")
    return Dataset(root=os.path.abspath(root), manifest=man)


def list_datasets(root: Optional[str] = None) -> list[Dataset]:
    """Every readable dataset under `root`, newest fetch first."""
    root = root or DEFAULT_ROOT
    if not os.path.isdir(root):
        return []
    out = []
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isfile(os.path.join(d, MANIFEST_NAME)):
            continue
        try:
            out.append(load(d))
        except Exception:      # noqa: BLE001 — a broken dataset must not hide the rest
            continue
    out.sort(key=lambda ds: ds.created_utc, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Writing a dataset                                                            #
# --------------------------------------------------------------------------- #

def write_gz_json(path: str, obj) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # `separators` matters here: NVE GeoJSON is mostly punctuation and repeated
    # keys, and dropping the default spaces is a few percent before gzip even runs.
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(data)
    return os.path.getsize(path)


def read_gz_json(path: str):
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def dir_bytes(path: str) -> int:
    total = 0
    for base, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(base, fn))
            except OSError:
                pass
    return total


def height_stats(heights: np.ndarray) -> dict:
    finite = np.isfinite(heights)
    n = int(heights.size)
    n_ok = int(finite.sum())
    vals = heights[finite]
    return {
        "min_m": float(vals.min()) if n_ok else None,
        "max_m": float(vals.max()) if n_ok else None,
        "mean_m": float(vals.mean()) if n_ok else None,
        "nodata_samples": n - n_ok,
        "coverage_pct": (100.0 * n_ok / n) if n else 0.0,
    }


def write(
    dest: str,
    lattice: Lattice,
    heights: np.ndarray,
    *,
    name: str,
    source_kind: str = "DTM",
    server_url: Optional[str] = None,
    demo: bool = False,
    rivers: Optional[list] = None,
    main_rivers: Optional[list] = None,
    lakes: Optional[list] = None,
    include_water: bool = True,
    include_main_rivers: bool = True,
    compression: str = DEFAULT_COMPRESSION,
    request: Optional[dict] = None,
    preview_png: Optional[bytes] = None,
    extra: Optional[dict] = None,
) -> Dataset:
    """
    Write one dataset directory and return it.

    `heights` must already be on `lattice` — north-up, float32, NaN for nodata,
    i.e. straight out of `core.assemble_region` with nothing else done to it.
    """
    from . import water as _water

    if heights.shape != (lattice.samples_y, lattice.samples_x):
        raise ValueError(
            f"heights {heights.shape} does not match lattice "
            f"{(lattice.samples_y, lattice.samples_x)}")

    os.makedirs(dest, exist_ok=True)
    heights = np.ascontiguousarray(heights, dtype=np.float32)
    codec = write_heights(dest, heights, compression)
    height_bytes = os.path.getsize(os.path.join(dest, codec["file"]))

    water_block: dict = {"present": False}
    if include_water:
        r_bytes = write_gz_json(os.path.join(dest, RIVERS_NAME), rivers or [])
        m_bytes = write_gz_json(os.path.join(dest, MAIN_RIVERS_NAME), main_rivers or [])
        l_bytes = write_gz_json(os.path.join(dest, LAKES_NAME), lakes or [])
        water_block = {
            "present": True,
            "attribution": _water.WATER_ATTRIBUTION,
            "license": _water.WATER_LICENSE,
            "include_main_rivers": bool(include_main_rivers),
            "format": "raw_geojson_features_gz",
            "note": ("Raw NVE GeoJSON, unparsed. `dataset.water_features()` runs "
                     "`water.features_from_geojson` at process time so improvements "
                     "to attribute resolution reach datasets already on disk."),
            "rivers": {"file": RIVERS_NAME, "layer": "elvenett",
                       "count": len(rivers or []), "bytes": r_bytes},
            "main_rivers": {"file": MAIN_RIVERS_NAME, "layer": "hovedelv",
                            "count": len(main_rivers or []), "bytes": m_bytes},
            "lakes": {"file": LAKES_NAME, "layer": "Innsjodatabase",
                      "count": len(lakes or []), "bytes": l_bytes},
        }

    if preview_png:
        with open(os.path.join(dest, PREVIEW_NAME), "wb") as f:
            f.write(preview_png)

    man = {
        "format": DATASET_FORMAT,
        "name": name,
        "slug": os.path.basename(dest),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "kind": source_kind,
            "server": server_url,
            "demo": bool(demo),
            "attribution": core.ATTRIBUTION,
        },
        "request": dict(request or {}),
        "lattice": lattice.to_dict(),
        "heights": {
            **codec,
            "shape": [int(lattice.samples_y), int(lattice.samples_x)],
            "nodata": "nan",
            "row_order": "north_to_south",
            "units": "metres above sea level",
            "stage": ("as fetched — no void repair, no shoreline snap, no carve. "
                      "Every one of those is a post-processing decision."),
            "bytes": height_bytes,
            "uncompressed_bytes": int(heights.nbytes),
            **height_stats(heights),
        },
        "water": water_block,
        "preview": {"file": PREVIEW_NAME} if preview_png else None,
    }
    if extra:
        man.update(extra)
    man["bytes_total"] = dir_bytes(dest)

    with open(os.path.join(dest, MANIFEST_NAME), "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)

    return Dataset(root=os.path.abspath(dest), manifest=man)
