"""
Confirm what the DTM actually returns inside lakes.
Usage: python diagnose_lake_fill.py /path/to/export_dir
Reads the leaf level, finds lake samples via the .water sidecar, and reports the
distribution of *height* values sitting under lake pixels.
"""
import sys, os, json, numpy as np

WATER = np.dtype([("type","u1"),("weight","u1"),("flow","u1"),("lake_id","<u2")])

def main(root):
    man = json.load(open(os.path.join(root, "manifest.json")))
    TS = man["tile_samples"]; hmin = man["height_min_m"]; hmax = man["height_max_m"]
    leaf = next(l for l in man["levels"] if l["level"] == 0)
    vals = []
    lake_vals = []
    for t in leaf["tiles"]:
        r16 = os.path.join(root, t["file"])
        wtr = r16[:-4] + ".water"
        if not (os.path.exists(r16) and os.path.exists(wtr)):
            continue
        a = np.fromfile(r16, dtype="<u2").reshape(TS, TS).astype(np.float64)
        h = hmin + (a/65535.0)*(hmax-hmin)
        w = np.fromfile(wtr, dtype=WATER).reshape(TS, TS)
        lake = w["type"] == 2
        if lake.any():
            lake_vals.append(h[lake])
        vals.append(h.ravel())
    if not lake_vals:
        print("No lake samples found in leaf tiles."); return
    lv = np.concatenate(lake_vals)
    allv = np.concatenate(vals)
    print(f"height_min_m = {hmin:.2f}   height_max_m = {hmax:.2f}")
    print(f"lake samples: {lv.size}")
    print(f"  min/median/max under lakes : {lv.min():.2f} / {np.median(lv):.2f} / {lv.max():.2f}")
    # modal value under lakes (rounded) — a void sentinel shows up as a huge spike
    r = np.round(lv, 1)
    u, c = np.unique(r, return_counts=True)
    top = u[np.argsort(c)[::-1][:5]]
    frac = c[np.argsort(c)[::-1][:5]] / lv.size
    print(f"  top lake values (val: share) : " +
          ", ".join(f"{v:.1f}:{f*100:.0f}%" for v, f in zip(top, frac)))
    at_min = np.mean(np.isclose(lv, hmin, atol=(hmax-hmin)/65535*2))
    print(f"  share of lake samples pinned at height_min: {at_min*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_lake_fill.py /path/to/export_dir")
        sys.exit(1)
    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)
    if not os.path.exists(os.path.join(root, "manifest.json")):
        print(f"No manifest.json found in {root} — point this at the folder "
              f"that directly contains manifest.json and the L0/ tiles "
              f"(e.g. terrain_export or terrain_export/out).")
        sys.exit(1)
    main(root)