"""
kvterrain.shoreline
===================

Where a lake's water meets the scanned land: the terrain above the water stays as
scanned, the bed below it is ours.

The NVE outline and the LiDAR block are independent products, and they disagree at
every shore. On Lierne (2026-09-30) a quarter of the median lake's outline samples
are scanned LAND standing above the lake's level; the lake bed used to cut them down
to the waterline, so the shore took the outline's pixel staircase instead of the
land's own contour. And every outline sample was held on the waterline, a line of
zero depth all round each lake that the water shader draws as dry ground; a lake two
or three samples wide was nothing else.

Each lake sample is sorted by what the scan says it is, relative to the lake's
authored level L and its own scanned water PLANE (LiDAR returns a lake's surface
flat, as the height it stood at when flown; `Pz`, the median over the lake's
interior, within `plane_tol_m`):

  * LAND, scan >= L and not the plane: above the water, so untouched. Where it joins
    the land around the lake within `shore_band_m` of the outline — the strip where
    outline and scan disagree — it stops being lake (`TYPE_LAND`, no lake id): the
    shoreline is the land's contour at the level, not the outline's staircase.
    Dry scan in the strip not joined to the shore is an island (`lake_island`)
    where it rises `island_min_m` above the level somewhere, and water otherwise:
    vegetation on a boggy lake, where the laser hit reeds and sedge (owner,
    2026-09-30: a bog lake full of dry holes looks more wrong than a shallow one).
    BEYOND the strip nothing is read from the scan: it is all plane. Kept as
    scanned, interior noise — waves, flight-strip steps, filled data gaps — stood
    as shoals 0.35 m under water in 40 m deep lakes and as flat, straight-edged
    islands hundreds of metres from shore (the owner's first export, 2026-09-30).
  * GROUND, scan < L and not the plane: real ground the scan saw dry (the lake stood
    lower when flown), now under water. Untouched, flooded.
  * PLANE: the scanned water. Only here does the lake bed model build a bed, and it
    is anchored at min(Pz, L - edge_drop_m) instead of at L: its rim meets the
    scanned ground at the plane's own height, and is never shallower than
    `edge_drop_m` under the water, so there is no line of zero depth.

The ring on the waterline did one job that still has to be done: it held each lake
at its level in the depression hierarchy. A lake sample that touches anything lower
than the level that is not the lake itself — ground outside the outline below the
level, a river, another lake — or the map edge, would let the lake drain there. Only
those samples are SEALED on the waterline now (`seal_shores`, after the bed is
built); the river mouths then open their own throats (`rivermouth`). Everywhere else
the lake is held by the scanned land around it, which stands above the level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           find_objects, label, maximum_filter, minimum_filter)

K8 = np.ones((3, 3), bool)


@dataclass
class ShoreParams:
    """Every tunable of the shoreline split. Recorded verbatim in the manifest."""
    enabled: bool = True
    # A sample within this of the lake's scanned plane is the plane (scanned water).
    plane_tol_m: float = 0.15
    # The modelled bed's rim is at least this far below the level.
    edge_drop_m: float = 0.3
    # The plane is measured this many samples in from the outline.
    plane_inset_px: int = 2
    # A piece of plane smaller than this, below the level, is flooded ground.
    min_plane_samples: int = 10
    # A sample whose 3x3 neighbourhood spans no more than this is FLAT: scanned
    # water, whatever its height. A big lake's water is not one plane: flight
    # strips and wind put patches of it most of a metre off the lake's median.
    flat_tol_m: float = 0.05
    # Only this strip along the outline — where outline and scan disagree — is
    # read from the scan (land, ground, islands). Further in everything is plane.
    shore_band_m: float = 15.0
    # In the strip, dry scan not joined to the shore is an island only where it
    # stands this far above the level somewhere (a rock); lower bumps are
    # vegetation on a boggy lake (reeds, sedge, floating mats), and are water.
    island_min_m: float = 1.0
    # Scanned land outside the outline but below the level joins the lake when
    # the low patch is enclosed within this distance of the outline and no larger
    # than the lake; otherwise the lake is sealed against it.
    max_extension_m: float = 50.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shores:
    plane: np.ndarray          # bool: the samples the lake bed model builds
    anchor: np.ndarray         # f4: the modelled bed's rim height on plane samples
    report: dict


def _extend_over_low_ground(wg, ground, lid, L, sl, *, spacing_m, p, island):
    """Join to lake `lid` the scanned land outside its outline that lies below its
    level, where that low patch is enclosed: it reaches neither the map edge nor
    further than `max_extension_m` from the outline, and is no larger than the
    lake. Returns how many samples joined."""
    from . import water as kvwater

    H, W = ground.shape
    pad = int(np.ceil(p.max_extension_m / spacing_m)) + 2
    r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
    c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
    t = wg.type[r0:r1, c0:c1]
    ids = wg.lake_id[r0:r1, c0:c1]
    lake = (t == kvwater.TYPE_LAKE) & (ids == lid)
    s = ground[r0:r1, c0:c1]
    low = (t == kvwater.TYPE_LAND) & np.isfinite(s) & (s < L) & ~island[r0:r1, c0:c1]
    if not low.any():
        return 0
    lbl, n = label(low, structure=K8)
    touching = np.unique(lbl[binary_dilation(lake, K8) & low])
    touching = touching[touching > 0]
    if touching.size == 0:
        return 0
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~lake) * spacing_m
    on_edge = np.zeros_like(low)
    on_edge[0, :] = on_edge[-1, :] = True
    on_edge[:, 0] = on_edge[:, -1] = True
    budget = int(lake.sum())
    joined = np.zeros_like(low)
    for k in touching:
        comp = lbl == k
        if (comp & on_edge).any():            # the map edge, or beyond the window
            continue
        if float(dist[comp].max()) > p.max_extension_m or comp.sum() > budget:
            continue
        joined |= comp
    rr, cc = np.nonzero(joined)
    wg.type[rr + r0, cc + c0] = kvwater.TYPE_LAKE
    wg.lake_id[rr + r0, cc + c0] = lid
    return int(joined.sum())


def split_shores(wg, ground: np.ndarray, *, spacing_m: float = 5.0,
                 params: Optional[ShoreParams] = None) -> Shores:
    """
    Sort every lake sample into land, ground and plane (module docstring), retyping
    the land that joins the shore as land and the rest as islands. `wg` (a
    `water.WaterGrid` with levels resolved) is modified IN PLACE; `ground` is the
    repaired, uncarved DTM. Lakes without a level are left alone.
    """
    from . import water as kvwater

    p = params or ShoreParams()
    H, W = ground.shape
    plane = np.zeros((H, W), bool)
    anchor = np.full((H, W), np.nan, np.float32)
    rep = {"params": p.to_dict(), "lakes": 0, "land_samples_to_shore": 0,
           "land_samples_to_island": 0, "ground_samples": 0, "plane_samples": 0,
           "plane_minus_level_m": None}
    if not p.enabled:
        return Shores(plane=(wg.type == kvwater.TYPE_LAKE), anchor=wg.lake_surface_moh(),
                      report=rep)

    island = (wg.lake_island if wg.lake_island is not None
              else np.zeros((H, W), bool))
    if wg.lake_island is None:
        wg.lake_island = island
    lakes_only = np.where(wg.type == kvwater.TYPE_LAKE, wg.lake_id, 0)
    slices = find_objects(lakes_only)
    del lakes_only
    offsets = []
    for lid0, sl in enumerate(slices):
        lid = lid0 + 1
        if sl is None:
            continue
        L = wg.lake_level((wg.lake_table or {}).get(lid, {}) or {})
        if L is None or not np.isfinite(L):
            continue
        extended = _extend_over_low_ground(wg, ground, lid, float(L), sl,
                                           spacing_m=spacing_m, p=p, island=island)
        rep["ground_samples_joined"] = rep.get("ground_samples_joined", 0) + extended
        if extended:
            pad = int(np.ceil(p.max_extension_m / spacing_m)) + 2
            sl = (slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)),
                  slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)))
        r0, r1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, H)
        c0, c1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, W)
        t = wg.type[r0:r1, c0:c1]
        ids = wg.lake_id[r0:r1, c0:c1]
        isl = island[r0:r1, c0:c1]
        s = ground[r0:r1, c0:c1].astype(np.float64)
        m = (t == kvwater.TYPE_LAKE) & (ids == lid) & ~isl & np.isfinite(s)
        if not m.any():
            continue
        inner = m & binary_erosion(m, K8, iterations=p.plane_inset_px,
                                   border_value=0)
        pz = float(np.median(s[inner] if inner.sum() >= 10 else s[m]))
        sf = np.where(np.isfinite(s), s, pz)
        flat = (maximum_filter(sf, size=3) - minimum_filter(sf, size=3)) <= p.flat_tol_m
        # ...and so is a sample at the height of flat water beside it: the edge
        # of a patch, where two strips of the scan meet in a step.
        beside = flat.copy()
        fz = np.pad(np.where(flat, sf, np.nan), 1, constant_values=np.nan)
        hh, ww = sf.shape
        with np.errstate(invalid="ignore"):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = fz[1 + dr:1 + dr + hh, 1 + dc:1 + dc + ww]
                    beside |= np.abs(sf - nb) <= p.plane_tol_m
        pl = m & ((np.abs(s - pz) <= p.plane_tol_m) | beside)
        # Scraps of plane cut off from the rest by the shore (a flat bit of bog at
        # the water's height) are no basin of their own. Below the level they are
        # flooded ground; above it they stay plane, or the lake would show dry
        # specks.
        if pz < L:
            lbl_p, n_p = label(pl, structure=K8)
            if n_p > 1:
                sizes = np.bincount(lbl_p.ravel())
                sizes[0] = 0
                pl &= sizes[lbl_p] >= p.min_plane_samples
        # Only the edge strip is read from the scan. Further in, a lake's scan is
        # its water — waves, flight-strip steps, filled data gaps — and anything
        # there that is neither flat nor at the plane's height is noise on it, not
        # ground: kept as scanned it stood as shoals and flat islands out in the
        # middle of deep lakes (Lierne, 2026-09-30). Real islands are the
        # outline's holes. So the interior is all plane.
        from_edge = distance_transform_edt(m | isl) * spacing_m
        band = from_edge <= p.shore_band_m
        pl |= m & ~band
        land = m & ~pl & (s >= L)
        gnd = m & ~pl & (s < L)
        # Land joining the shore: connected, through land, to a sample that is not
        # this lake (or to the map edge).
        outside = ~((t == kvwater.TYPE_LAKE) & (ids == lid))
        lbl, n = label(land | outside, structure=K8)
        shore_lbl = np.unique(lbl[outside])
        shore_lbl = shore_lbl[shore_lbl > 0]
        to_shore = land & np.isin(lbl, shore_lbl)
        edge = np.zeros_like(m)
        edge[0 if r0 == 0 else slice(0, 0), :] = True
        edge[-1 if r1 == H else slice(0, 0), :] = True
        edge[:, 0 if c0 == 0 else slice(0, 0)] = True
        edge[:, -1 if c1 == W else slice(0, 0)] = True
        if (land & edge).any():
            lbl2, _ = label(land, structure=K8)
            on_edge = np.unique(lbl2[land & edge])
            to_shore |= land & np.isin(lbl2, on_edge[on_edge > 0])
        # The rest of the dry scan in the strip: an island where it rises clearly out
        # of the water somewhere (a rock off the shore), else vegetation on the
        # water, which is water.
        rest = land & ~to_shore
        to_island = np.zeros_like(rest)
        wet = np.zeros_like(rest)
        if rest.any():
            lbl3, n3 = label(rest, structure=K8)
            peak = np.full(n3 + 1, -np.inf)
            np.maximum.at(peak, lbl3[rest], s[rest])
            high = peak >= L + p.island_min_m
            high[0] = False
            to_island = rest & high[lbl3]
            wet = rest & ~to_island
            pl |= wet

        rr, cc = np.nonzero(to_shore)
        wg.type[rr + r0, cc + c0] = kvwater.TYPE_LAND
        wg.lake_id[rr + r0, cc + c0] = 0
        rr, cc = np.nonzero(to_island)
        island[rr + r0, cc + c0] = True
        rr, cc = np.nonzero(pl)
        plane[rr + r0, cc + c0] = True
        anchor[rr + r0, cc + c0] = np.float32(min(pz, L - p.edge_drop_m))

        rep["lakes"] += 1
        rep["land_samples_to_shore"] += int(to_shore.sum())
        rep["land_samples_to_island"] += int(to_island.sum())
        rep["low_dry_samples_to_water"] = rep.get("low_dry_samples_to_water", 0) + int(wet.sum())
        rep["ground_samples"] += int(gnd.sum())
        rep["plane_samples"] += int(pl.sum())
        offsets.append(pz - L)
    if offsets:
        o = np.asarray(offsets)
        rep["plane_minus_level_m"] = {
            "p10": round(float(np.percentile(o, 10)), 2),
            "p50": round(float(np.median(o)), 2),
            "p90": round(float(np.percentile(o, 90)), 2)}
    return Shores(plane=plane, anchor=anchor, report=rep)


def seal_shores(bed: np.ndarray, wg, lake_surface: np.ndarray, *,
                edge_is_shore: bool = True) -> int:
    """
    Hold on the waterline every lake sample that touches (8-connected) something
    lower than its lake's level that is not its own lake — ground below the level,
    a river, another lake — or the map edge; otherwise the lake would drain there.
    Raises `bed` IN PLACE to the level at those samples (never lowers) and returns
    how many were sealed.
    """
    from . import water as kvwater

    H, W = bed.shape
    lake = (wg.type == kvwater.TYPE_LAKE) & np.isfinite(lake_surface)
    if wg.lake_island is not None:
        lake &= ~wg.lake_island
    ids = np.where(lake, wg.lake_id, 0).astype(np.int64)
    island = (wg.lake_island if wg.lake_island is not None
              else np.zeros((H, W), bool))
    seal = np.zeros((H, W), bool)
    pad_b = np.pad(bed, 1, mode="edge")
    pad_id = np.pad(ids, 1, constant_values=-1)       # the map edge: not the lake
    pad_isl = np.pad(island, 1, constant_values=False)
    pad_isl_id = np.pad(np.where(island, wg.lake_id, 0).astype(np.int64), 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = pad_b[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            nid = pad_id[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            nisl = pad_isl[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            nisl_id = pad_isl_id[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            other = (nid != ids) & ~(nisl & (nisl_id == ids))
            low = nb < lake_surface
            if edge_is_shore:
                low |= nid == -1
            seal |= lake & other & low
    seal &= bed < lake_surface
    bed[seal] = lake_surface[seal].astype(bed.dtype)
    return int(seal.sum())
