"""
kvterrain.exports
=================

Read a finished export back: what is in it, what it looks like, and whether it
is intact.

Every other module in this package *writes* the export. This one is the only one
that opens it again from the outside, which is exactly why it is worth having —
it exercises the same arithmetic a consumer has to implement (the dense-atlas
offset formula, the u16 unpacking, the `rivers.bin` layout) against the bytes
that were actually written, rather than against the arrays that were in memory
at the time. A preview drawn by this module is proof the offsets are right;
a preview drawn from `PackResult` is not.

The checks here are the ones `kvterrain.cli validate-atlas` and `validate-water`
have always run. They live in this module now, returning reports, and the CLI
prints them — so the UI and the CLI cannot drift into checking different things.

Nothing here writes to an export. Reading is all it does.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import core

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "exports")

MANIFEST_NAME = "manifest.json"

# The three atlases, in the order a reader most wants them.
ATLAS_KINDS = ("height", "surface", "water_id")


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class Export:
    """A finished export directory. Nothing is read until asked."""
    root: str
    manifest: dict = field(repr=False)

    # -- header facts ------------------------------------------------------- #
    @property
    def name(self) -> str:
        return os.path.basename(self.root)

    @property
    def epsg(self) -> int:
        return int(str(self.manifest["crs"]).split(":")[1])

    @property
    def tile_cells(self) -> int:
        return int(self.manifest["tile_cells"])

    @property
    def tile_samples(self) -> int:
        return int(self.manifest["tile_samples"])

    @property
    def leaf_tiles(self) -> tuple[int, int]:
        lx, ly = self.manifest["leaf_tiles"]
        return int(lx), int(ly)

    @property
    def num_levels(self) -> int:
        return int(self.manifest["num_levels"])

    @property
    def height_range(self) -> tuple[float, float]:
        return (float(self.manifest["height_min_m"]),
                float(self.manifest["height_max_m"]))

    @property
    def has_water(self) -> bool:
        return bool(self.manifest.get("atlas", {}).get("surface_file"))

    @property
    def generated_utc(self) -> str:
        return self.manifest.get("generator", {}).get("generated_utc", "")

    @property
    def dataset_slug(self) -> Optional[str]:
        """Which fetch this was built from, when `process` wrote the provenance."""
        return (self.manifest.get("generator", {}).get("dataset") or {}).get("slug")

    def atlas_path(self, kind: str) -> Optional[str]:
        key = {"height": "height_file", "surface": "surface_file",
               "water_id": "water_id_file"}[kind]
        fn = self.manifest.get("atlas", {}).get(key)
        if not fn:
            return None
        p = os.path.join(self.root, fn)
        return p if os.path.exists(p) else None

    def files(self) -> list:
        """(name, bytes) for everything in the export, largest first."""
        out = []
        for base, _, names in os.walk(self.root):
            for fn in names:
                fp = os.path.join(base, fn)
                try:
                    out.append((os.path.relpath(fp, self.root), os.path.getsize(fp)))
                except OSError:
                    pass
        return sorted(out, key=lambda t: -t[1])

    def bytes_total(self) -> int:
        return sum(b for _, b in self.files())

    def bbox_utm(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.manifest["region_bbox_utm"])

    def outline_latlon(self) -> list:
        from pyproj import Transformer
        tr = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        x0, y0, x1, y1 = self.bbox_utm()
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        return [(lat, lon) for lon, lat in (tr.transform(x, y) for x, y in ring)]

    def summary(self) -> dict:
        lx, ly = self.leaf_tiles
        x0, y0, x1, y1 = self.bbox_utm()
        hmin, hmax = self.height_range
        return {
            "name": self.name,
            "root": self.root,
            "generated_utc": self.generated_utc,
            "dataset": self.dataset_slug,
            "epsg": self.epsg,
            "spacing_m": float(self.manifest["leaf_spacing_m"]),
            "tile_cells": self.tile_cells,
            "leaf_tiles": (lx, ly),
            "num_levels": self.num_levels,
            "tiles_total": _total_tiles(lx, ly, self.num_levels),
            "height_min_m": hmin,
            "height_max_m": hmax,
            "width_m": x1 - x0,
            "height_m": y1 - y0,
            "area_km2": (x1 - x0) * (y1 - y0) / 1e6,
            "bytes_total": self.bytes_total(),
            "has_water": self.has_water,
            "lakes": self.manifest.get("water_surface", {}).get("lake_count", 0),
            "river_segments": (self.manifest.get("water_vector", {})
                               .get("rivers", {}).get("segments", 0)),
            "source": self.manifest.get("source", "?"),
        }

    # -- reading the atlases ------------------------------------------------ #
    def level_shape(self, level: int) -> tuple[int, int]:
        lx, ly = self.leaf_tiles
        tx, ty = core.tiles_at_level(lx, ly, level)
        tc = self.tile_cells
        return (ty * tc + 1, tx * tc + 1)

    def read_level_u16(self, kind: str, level: int) -> np.ndarray:
        """
        Reassemble a whole pyramid level from an atlas, by computed byte offset.

        This is the consumer's job done in reverse, and it is the point of the
        module: every tile is located with `core.atlas_tile_offset` and nothing
        else, so if the header and the file disagree the picture comes out wrong
        rather than quietly plausible.
        """
        path = self.atlas_path(kind)
        if path is None:
            raise FileNotFoundError(f"export has no {kind} atlas")
        lx, ly = self.leaf_tiles
        ts, tc = self.tile_samples, self.tile_cells
        tiles_x, tiles_y = core.tiles_at_level(lx, ly, level)
        SY, SX = self.level_shape(level)
        out = np.zeros((SY, SX), dtype="<u2")
        tile_bytes = core.atlas_tile_bytes(ts)

        with open(path, "rb") as fh:
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    off = core.atlas_tile_offset(lx, ly, self.num_levels, ts,
                                                 level, tx, ty)
                    fh.seek(off)
                    buf = fh.read(tile_bytes)
                    if len(buf) != tile_bytes:
                        raise ValueError(
                            f"{kind} atlas: short read for L{level} tile {tx},{ty} "
                            f"at offset {off} — the file does not match the header")
                    tile = np.frombuffer(buf, dtype="<u2").reshape(ts, ts)
                    r0, c0, _ = core.north_up_tile_slice(SY, tc, tx, ty)
                    out[r0:r0 + ts, c0:c0 + ts] = tile
        return out

    def preview_level(self, max_px: int = 1100) -> int:
        """The finest level whose assembled sample grid still fits in `max_px`."""
        for lvl in range(self.num_levels):
            SY, SX = self.level_shape(lvl)
            if max(SY, SX) <= max_px:
                return lvl
        return self.num_levels - 1

    def read_heights(self, level: int) -> np.ndarray:
        """Metres above sea level, unpacked from the R16 atlas."""
        hmin, hmax = self.height_range
        packed = self.read_level_u16("height", level)
        return (hmin + (packed.astype(np.float32) / 65535.0) * (hmax - hmin))

    def read_surface(self, level: int) -> np.ndarray:
        """Water-surface m.o.h., NaN off water."""
        from . import watersurface
        hmin, hmax = self.height_range
        return watersurface.unpack_surface_u16(
            self.read_level_u16("surface", level), hmin, hmax)

    def read_water_id(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """(class, lake_id). class: 0 dry, 1 lake, 2 river, 3 ocean."""
        from . import waterid
        return waterid.decode_water_id(self.read_level_u16("water_id", level))


def _total_tiles(lx: int, ly: int, levels: int) -> int:
    n = 0
    for lvl in range(levels):
        tx, ty = core.tiles_at_level(lx, ly, lvl)
        n += tx * ty
    return n


def load(root: str) -> Export:
    with open(os.path.join(root, MANIFEST_NAME)) as f:
        man = json.load(f)
    if not str(man.get("format", "")).startswith("kvterrain"):
        raise ValueError(f"{root} is not a kvterrain export "
                         f"(format={man.get('format')!r})")
    return Export(root=os.path.abspath(root), manifest=man)


def list_exports(root: Optional[str] = None) -> list:
    """Every readable export under `root`, newest first. Also matches an export
    written directly INTO `root` rather than into a subdirectory of it."""
    root = root or DEFAULT_ROOT
    if not os.path.isdir(root):
        return []
    found = []
    if os.path.isfile(os.path.join(root, MANIFEST_NAME)):
        try:
            found.append(load(root))
        except Exception:      # noqa: BLE001
            pass
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if os.path.isfile(os.path.join(d, MANIFEST_NAME)):
            try:
                found.append(load(d))
            except Exception:      # noqa: BLE001 — one bad export must not hide the rest
                continue
    found.sort(key=lambda e: e.generated_utc, reverse=True)
    return found


# --------------------------------------------------------------------------- #
# Checks                                                                       #
#                                                                              #
# These return reports instead of printing, so the CLI and the UI run the same  #
# code and cannot end up checking different things.                            #
# --------------------------------------------------------------------------- #

def check_atlas(exp: Export, *, samples_per_level: int = 8) -> dict:
    """
    Verify each atlas against the manifest header: exact file size, and that a
    sample of computed tile offsets each yield a full tileBytes block inside the
    file. A size mismatch means the tile grid was not dense, which would silently
    misalign every tile past the gap for a reader using pure arithmetic.
    """
    lx, ly = exp.leaf_tiles
    ts = exp.tile_samples
    tile_bytes = core.atlas_tile_bytes(ts)
    expect = core.atlas_total_bytes(lx, ly, exp.num_levels, ts)

    rng = np.random.default_rng(0)
    report = {"expected_bytes": expect, "tile_bytes": tile_bytes,
              "atlases": [], "ok": True}

    for kind in ATLAS_KINDS:
        path = exp.atlas_path(kind)
        if path is None:
            if kind == "height":
                report["atlases"].append(
                    {"kind": kind, "ok": False, "error": "missing height atlas"})
                report["ok"] = False
            continue

        entry = {"kind": kind, "file": os.path.basename(path)}
        actual = os.path.getsize(path)
        entry["bytes"] = actual
        entry["size_ok"] = (actual == expect)
        entry["offsets_checked"] = 0
        entry["short_reads"] = []

        if entry["size_ok"]:
            with open(path, "rb") as fh:
                for lvl in range(exp.num_levels):
                    tx, ty = core.tiles_at_level(lx, ly, lvl)
                    coords = [(x, y) for y in range(ty) for x in range(tx)]
                    pick = rng.choice(len(coords),
                                      size=min(samples_per_level, len(coords)),
                                      replace=False)
                    for idx in pick:
                        x, y = coords[int(idx)]
                        off = core.atlas_tile_offset(lx, ly, exp.num_levels, ts,
                                                     lvl, x, y)
                        fh.seek(off)
                        if len(fh.read(tile_bytes)) != tile_bytes:
                            entry["short_reads"].append(
                                {"level": lvl, "x": x, "y": y, "offset": off})
                        entry["offsets_checked"] += 1

        entry["ok"] = entry["size_ok"] and not entry["short_reads"]
        report["ok"] &= entry["ok"]
        report["atlases"].append(entry)

    return report


def check_water(exp: Export) -> dict:
    """
    Re-read the exported water products and check them against the manifest:
    `rivers.bin` parses cleanly and its counts match, every segment link resolves,
    every junction names a lake that exists, and the water_id atlas is dense.

    The advisory checks (descent, flow direction) are replayed from the stored
    report rather than recomputed — the tool never corrects them, so the numbers
    only need surfacing.
    """
    from . import rivernet

    wv = exp.manifest.get("water_vector")
    if not wv:
        return {"ok": True, "skipped": "no water_vector block — built without water"}

    report = {"ok": True, "errors": [], "notes": [], "rivers": {}, "lakes": {},
              "junctions": {}, "validation": wv.get("validation", {})}

    # ---- rivers.bin -------------------------------------------------------- #
    path = os.path.join(exp.root, wv["rivers"]["file"])
    seen_ids, downstream_refs = set(), []
    total_v = 0
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(48)
            (magic, version, epsg, ox, oy, stride, nseg, nvert, _res) = struct.unpack(
                "<8sIIddfIII", hdr)
            if magic != rivernet.BIN_MAGIC:
                raise ValueError(f"bad magic {magic!r}")
            report["rivers"] = {"file": wv["rivers"]["file"], "version": version,
                                "epsg": epsg, "stride_m": stride,
                                "segments": nseg, "vertices": nvert}
            # Per-segment record, in write_rivers_bin's exact order. Walking it
            # in full (rather than seeking past it) IS the check: a wrong field
            # width desynchronises the stream and the walk runs off the end,
            # which is precisely the corruption a length-only check would miss.
            for _ in range(nseg):
                (sid, strekn, vatn, order, down) = struct.unpack("<IqqHi", fh.read(26))
                (nup,) = struct.unpack("<H", fh.read(2))
                fh.read(4 * nup)                       # upstream ids, u32 each
                for _field in range(2):                # elvid, vassdragsnr
                    (slen,) = struct.unpack("<B", fh.read(1))
                    fh.read(slen)
                (nspans,) = struct.unpack("<H", fh.read(2))
                fh.read(nspans * 10)                   # lake spans: u32,u32,u16
                fh.read(16)                            # blake2b-128 geometry hash
                (nv,) = struct.unpack("<I", fh.read(4))
                fh.read(nv * 16)                       # 4 x f32 per vertex
                seen_ids.add(sid)
                total_v += nv
                if down >= 0:
                    downstream_refs.append(down)
            trailing = fh.read()
            if trailing:
                report["errors"].append(
                    f"rivers.bin has {len(trailing)} bytes after the last segment")
        report["rivers"]["vertices_walked"] = total_v
        if total_v != nvert:
            report["errors"].append(
                f"rivers.bin header says {nvert} vertices, walked {total_v}")
        if nseg != wv["rivers"]["segments"]:
            report["errors"].append(
                f"rivers.bin has {nseg} segments, manifest says "
                f"{wv['rivers']['segments']}")
        dangling = [d for d in downstream_refs if d not in seen_ids]
        report["rivers"]["dangling_downstream"] = len(dangling)
        if dangling:
            report["errors"].append(
                f"{len(dangling)} downstream links point at no segment")
    except Exception as e:      # noqa: BLE001 — reported, not raised
        report["errors"].append(f"rivers.bin: {e}")

    # ---- lakes.json / junctions.json --------------------------------------- #
    try:
        from . import water as kvwater
        with open(os.path.join(exp.root, wv["lakes"]["file"])) as f:
            lakes = json.load(f)
        rows = lakes.get("lakes", lakes)
        lake_ids = {int(l["lake_id"]) for l in rows if "lake_id" in l}
        # The runtime pin reads `authored_level_m`, which carries an ESTIMATED
        # level too — so this must not test `hoyde_moh`, or every estimated lake
        # would read as unpinnable.
        no_level = [l["lake_id"] for l in rows if l.get("authored_level_m") is None]
        est = [l for l in rows
               if l.get("level_source") == kvwater.LEVEL_SOURCE_ESTIMATED]
        report["lakes"] = {"count": len(rows), "with_ids": len(lake_ids),
                           "estimated_level": len(est),
                           "without_level": len(no_level)}
        if no_level:
            # Not an error: an unlevelled lake is left uncarved on purpose, so it
            # stays flat ground rather than becoming an empty bowl. It just cannot
            # be pinned AuthoredWins, which is worth knowing before you ship it.
            report.setdefault("notes", []).append(
                f"{len(no_level)} lakes have no level from either source. They "
                f"cannot be pinned AuthoredWins, and are left uncarved (flat "
                f"ground, not an empty bowl). Re-run with lake level estimation "
                f"on to fill them in.")
    except Exception as e:      # noqa: BLE001
        report["errors"].append(f"lakes.json: {e}")
        lake_ids = set()

    try:
        with open(os.path.join(exp.root, wv["junctions"]["file"])) as f:
            junc = json.load(f)
        rows = junc.get("junctions", junc)
        missing = [j for j in rows
                   if "lake_id" in j and int(j["lake_id"]) not in lake_ids]
        report["junctions"] = {"count": len(rows), "naming_unknown_lake": len(missing)}
        if missing:
            report["errors"].append(
                f"{len(missing)} junctions name a lake that is not in lakes.json")
    except Exception as e:      # noqa: BLE001
        report["errors"].append(f"junctions.json: {e}")

    # ---- water_id atlas ----------------------------------------------------- #
    wid_path = exp.atlas_path("water_id")
    if wid_path:
        lx, ly = exp.leaf_tiles
        expect = core.atlas_total_bytes(lx, ly, exp.num_levels, exp.tile_samples)
        actual = os.path.getsize(wid_path)
        report["water_id"] = {"bytes": actual, "expected_bytes": expect,
                              "ok": actual == expect}
        if actual != expect:
            report["errors"].append(
                f"water_id atlas is {actual} bytes, expected {expect}")
    elif exp.manifest.get("water_id"):
        report["errors"].append("manifest names a water_id atlas that is not there")

    report["ok"] = not report["errors"]
    return report


def check_against_point_api(exp: Export, *, n: int = 12, session=None,
                            seed: int = 0, progress=None) -> dict:
    """
    Spot-check packed leaf heights against Kartverket's open point API.

    The only check here that needs the network, and the only one that can catch a
    georeferencing error: everything else verifies the export is self-consistent,
    while this asks whether the height at a world coordinate is the height
    Kartverket says is there. Errors include R16 quantisation, which is reported
    alongside so a small residual is not mistaken for a misalignment.

    CAVEAT, and it is a big one on a water-heavy export: the packed heights are
    the CARVED bed, while the point API returns the untouched DTM. Any sample
    that lands in a lake or a river channel will disagree by roughly the carve
    depth, and that is the pipeline working, not a misalignment. Read the median
    over many samples rather than any single row, and treat a handful of large
    errors as "those landed in water" until you have checked where they are.
    `water_class` on each sample says which ones did.
    """
    import requests

    man = exp.manifest
    hmin, hmax = exp.height_range
    tc, ts = exp.tile_cells, exp.tile_samples
    spacing = float(man["leaf_spacing_m"])
    ox, oy = man["origin_utm"]
    lx, ly = exp.leaf_tiles

    path = exp.atlas_path("height")
    if path is None:
        return {"ok": False, "errors": ["no height atlas"], "samples": []}

    rng = np.random.default_rng(seed)
    coords = [(x, y) for y in range(ly) for x in range(lx)]      # level-0 grid
    pick = rng.choice(len(coords), min(n, len(coords)), replace=False)

    sess = session or requests.Session()
    tile_bytes = core.atlas_tile_bytes(ts)
    out = {"quantisation_m": (hmax - hmin) / 65535.0, "samples": [],
           "errors": [], "ok": True}

    with open(path, "rb") as fh:
        for k, idx in enumerate(pick):
            x, y = coords[int(idx)]
            off = core.atlas_tile_offset(lx, ly, exp.num_levels, ts, 0, x, y)
            fh.seek(off)
            a16 = np.frombuffer(fh.read(tile_bytes), dtype="<u2").reshape(ts, ts)

            i = int(rng.integers(0, tc + 1))
            j = int(rng.integers(0, tc + 1))
            h_packed = hmin + (a16[tc - j, i] / 65535.0) * (hmax - hmin)
            X = ox + (x * tc + i) * spacing
            Y = oy + (y * tc + j) * spacing
            try:
                r = sess.get(core.POINT_API,
                             params={"ost": X, "nord": Y, "koordsys": exp.epsg},
                             timeout=30)
                h_api = (r.json().get("punkter", [{}])[0] or {}).get("z")
            except Exception as e:      # noqa: BLE001 — reported, not raised
                out["errors"].append(f"point API call failed: {e}")
                out["ok"] = False
                break
            if h_api is None:
                continue
            out["samples"].append({"x": X, "y": Y, "packed_m": float(h_packed),
                                   "api_m": float(h_api),
                                   "abs_error_m": abs(float(h_api) - float(h_packed)),
                                   "water_class": _class_at(exp, X, Y)})
            if progress:
                progress((k + 1) / len(pick), f"point {k + 1}/{len(pick)}")

    errs = [s["abs_error_m"] for s in out["samples"]]
    out["median_abs_error_m"] = float(np.median(errs)) if errs else None
    out["max_abs_error_m"] = float(np.max(errs)) if errs else None
    return out


_CLASS_NAMES = ("dry", "lake", "river", "ocean")


def _class_at(exp: Export, X: float, Y: float) -> str:
    """The water class of the leaf sample nearest (X, Y), or 'unknown' with no
    water_id atlas. Used to explain a large ground-truth error: a sample in a
    carved channel is SUPPOSED to sit below the untouched DTM."""
    path = exp.atlas_path("water_id")
    if path is None:
        return "unknown"
    man = exp.manifest
    spacing = float(man["leaf_spacing_m"])
    ox, oy = man["origin_utm"]
    tc, ts = exp.tile_cells, exp.tile_samples
    lx, ly = exp.leaf_tiles

    i = int(round((X - ox) / spacing))
    j = int(round((Y - oy) / spacing))
    if not (0 <= i <= lx * tc and 0 <= j <= ly * tc):
        return "outside"
    tx, ty = min(i // tc, lx - 1), min(j // tc, ly - 1)
    off = core.atlas_tile_offset(lx, ly, exp.num_levels, ts, 0, tx, ty)
    try:
        with open(path, "rb") as fh:
            fh.seek(off)
            tile = np.frombuffer(fh.read(core.atlas_tile_bytes(ts)),
                                 dtype="<u2").reshape(ts, ts)
    except Exception:      # noqa: BLE001
        return "unknown"
    from . import waterid
    cls, _ = waterid.decode_water_id(tile[tc - (j - ty * tc), i - tx * tc])
    return _CLASS_NAMES[int(cls)]
