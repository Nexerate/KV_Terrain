# Runtime spec: draw mapped rivers at their visual width

**For:** the Unity runtime. The affected files are:
- `WaterHierarchy/RiverQuadGather.cs`
- `WaterHierarchy/RiverSegmentIndex*.cs`
- the river scatter job
- `RiverNetworkLoadSystem.cs`
- the mapped-river stamps in `DerivedRiverSystem.cs`

**From:** kvterrain, item 4 (N50 river widths), 2026-09-30.

**Size:**
- one new optional input file;
- per-vertex radius in the mapped-river stamp.

`rivers.bin` does not change: still v2, byte for byte the same layout.

## Why

A mapped river's width has so far come from its modelled discharge through the
shared function (`width_model`). The runtime recomputes it and stamps
`StampRadiusUnity(WidthM(q))` along each segment. That function is right for
streams and wrong for trunk rivers. On Lierne, the rivers that Kartverket's N50
maps as water areas are 15–57 m wide (median 28 m). The discharge model made them
5–15 m (median 12 m): a uniform ribbon inside a channel twice its width.

The exporter now gives every river vertex a **visual width**:
- **N50 polygon rivers:** the polygon's width across the river at each vertex,
  smoothed along the river.
- **N50 centrelines with a width class:** the discharge width held to the class
  (at most 3 m, or 3–15 m).
- **Elsewhere:** the discharge width.

It rasterises, carves and levels each river to that width. Without the runtime
change, the wide parts of the carved channel sit outside the stamp and render
without water.

## The file: `river_widths.bin`

The manifest lists it at `water_vector.river_widths` when present. That block
repeats the layout below and carries the file name. The file is absent from
exports made before this change, and from exports made with
`--no-river-n50-widths` whose rivers have no visual width.

Layout (little-endian):
- **Header:** `8s magic "KVRWIDT1"`, `u32 version = 1`, `u32 segment_count`,
  `u32 vertex_count`.
- **Per segment**, in `rivers.bin` segment order: `u32 seg_id`,
  `u32 vertex_count`, then `vertex_count` × `f32 width_m`.
- **Widths:** full channel width in real metres at each vertex. The vertices are
  exactly `rivers.bin`'s, in the same order.
- **Segments without a visual width** repeat their hydraulic `width_m`.

**Validation at load.** Refuse the file, log a warning and fall back to the
hydraulic width if any of these fail:
- magic and version;
- `segment_count` and `vertex_count` equal `rivers.bin`'s;
- each segment's `seg_id` and `vertex_count` equal the segment at the same
  position in `rivers.bin`;
- every width is finite and > 0.

Store the widths as one `NativeArray<float> VertexWidthM`, indexed like the vertex
array (`VertexStart + i`).

**`WidthM` in `rivers.bin` stays the hydraulic width.** It is still what the
load-time check compares with `WidthM(DischargeM3S)`. It is still the width for
every discharge-to-width conversion (`DischargeForWidth`, re-routing, derived
rivers). Only drawing and stamping a MAPPED river change.

## The rule

Everywhere the runtime turns a mapped segment into a stamp or draw radius, use the
visual width at the vertex. Keep the ratio a flow edit gives it:

```
k(seg)       = WidthM(EffectiveQ[seg]) / WidthM(seg.DischargeM3S)   // 1 without flow edits
width_v      = VertexWidthM[seg.VertexStart + v] * k(seg)
radius_v     = StampRadiusUnity(width_v, WorldScale)                  // same floor + touch reach
```

The rule keeps today's edge cases:
- If `WidthM(seg.DischargeM3S)` is 0, or the segment has no discharge, use `k = 1`.
- A reach whose `EffectiveQ` drops to 0 is still hidden exactly as today.

The sites to change:

1. **`RiverSegmentIndex.SegmentRadius`** and **`SegmentBounds`**: use the
   segment's largest `radius_v` with `k = 1`. Bounds only need to contain the
   stamp. If flow edits can raise `k` above 1, expand by `k` where the bounds are
   recomputed, or keep a conservative margin. Pick whichever matches the current
   handling of `WidthM(q2)` growing past the indexed radius.
2. **The scatter** (`RiverQuadGather` builds jobs; `RiverScatterJob` draws them):
   - Replace the job's single `Radius` with a per-vertex radius.
   - Easiest: the job carries `RadiusScale = k`, and the scatter reads
     `VertexWidthM[VertexStart + i]`, converts it with `StampRadiusUnity`, and
     interpolates linearly between the two vertices of each sub-segment.
   - `Coverage` uses the largest radius in the job.
   - Suppressed-run jobs (`AnySuppressed`) take the same per-vertex radii.
3. **Mapped-river stamps in `DerivedRiverSystem`:** `MappedStampRadius`,
   `BuildReaches` and similar sites use `s.WidthM` or `WidthM(q)` for a mapped
   segment. There, use the visual width at the vertex being stamped, times `k`.
   Where a single radius per segment is structurally required, use the segment's
   largest `radius_v`: over-stamping is harmless, because a stamped sample whose
   ground stands above the level clips dry (`StampRadiusUnity`'s doc comment).
4. **Derived rivers** (re-routed flow with no mapped segment) keep `WidthM(q)`.
   They have no N50 width. Where a derived reach rejoins a wide mapped river, the
   width steps at the join. That is accepted: a re-routed river is new water, and
   the mapped river's width is Kartverket's.

## What the export guarantees

- **Visual width vs hydraulic width.** The visual width is at least the hydraulic
  width wherever N50 has a river polygon or class 3. It is capped at 3 m on class-2
  streams, which only ever narrows a stream's hydraulic width when that is over
  3 m. Elsewhere the two are equal.
- **The carved channel.** It follows the visual width. The exported river mask for
  a polygon-derived stretch is clipped to N50's river area plus one sample, so it
  never extends past the mapped banks by more than a sample.
- **A circle of the vertex width stays inside N50's area** (kvterrain at or after
  2026-09-30). Two exporter changes make a per-vertex circle stamp safe without
  clipping in the runtime:
  - Where the width comes from an N50 river area, the vertex in `rivers.bin` is
    moved to the middle of the area across the river. N50's lines ran a median
    3 m, and up to 75 m, off-centre, so the stamp spilled past the near bank by
    that much.
  - Each width is then capped at twice the vertex's distance to the area's nearest
    bank plus one sample. That removes the rest: bends, islands and irregular
    reaches up to 290 m wide.
  On Lierne no vertex stamp now reaches more than 2.5 m past an N50 bank, where it
  was up to 68 m. The capped width can underfill a very irregular area; the carved
  channel and `surface.atlas` still cover the whole area there.
- **Stamping past the carved channel** onto dry banks is harmless in the runtime,
  as above.
- **`surface.atlas` and `water_id.atlas`** cover the visual width, so heads, lake
  tie-ins and river classes are consistent with it.

## How to verify

1. **Load:** load a Lierne export made from kvterrain at or after this change.
   The log reports `river_widths.bin` accepted, with counts equal to `rivers.bin`.
2. **Scene:** the trunk rivers in Lierne fill their channels bank to bank.
   - Namsen near 399600 E, 7186100 N (EPSG:25833) should be 30–90 m wide rather
     than a 12 m ribbon.
   - Small streams are unchanged.
3. **Flow edits:**
   - Dig a diversion off a wide river. The remaining reach narrows by the ratio
     `k`, and a reach that loses all its flow still hides.
   - The derived river keeps its discharge width.
4. **Fallback:** delete `river_widths.bin` from an export. The runtime logs the
   fallback and draws exactly as before.
