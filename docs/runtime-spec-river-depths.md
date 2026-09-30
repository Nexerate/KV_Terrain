# Runtime spec: river depth from discharge

**For:** the Unity runtime, mainly `WaterHierarchy/DerivedRiverSystem.cs`. Reading the
manifest also touches `ManifestScan.cs` and `HeightTileBootstrapSystem.cs`.
**From:** kvterrain, 2026-09-30 (`--river-depth-model discharge`, the default).
**Size:** optional. Exports load and behave sensibly without any change (see
"Compatibility").

## What changed in the export

The exporter's river channels were carved to a table by stream order: 1.5 m for a
headwater stream up to 11 m for an order-8 trunk. That made a 6 m wide river 5.5 m
deep. The depth now comes from the segment's modelled discharge, the depth half
of the hydraulic geometry whose width half the runtime already shares:

```
depth_m = clamp(c * Q^f, min_m, max_m) * scale
c = 0.3, f = 0.4, min_m = 0.5, max_m = 4.0, scale = --river-depth-scale (1.0)
```

- `f = 0.4` is the runtime's own `DEPTH_EXPONENT`.
- On Lierne nearly every river is 0.5–0.7 m deep. The floor keeps small streams
  reading as water.

The manifest carries the model twice, with the same content:
- `water_vector.rivers.depth_model`
- `water_surface.depth_model`

## Compatibility (no runtime change needed)

`water_surface.river_carve_depth_by_order_m`, which `RiverWidthModel.DepthByOrderM`
reads, is still written. It now holds the median modelled depth of each order's
segments: 0.5 m for orders 1–7 and 0.7 m for order 8 on Lierne. So `DepthM(order)`
in `DerivedRiverSystem` already returns depths that match the carved channels.
Without this, reroutes and springs would have used depths 3–15 times too large.

## Recommended change

Where a mapped segment's depth is read by order, read it from discharge when
`depth_model` is present. Parse it next to `width_model`, then:

```
DepthM(q) = clamp(c * max(q, 0)^f, min_m, max_m) * scale    // evaluate in double
```

The call sites:
- **Overflow source, around line 1277:**
  `depthM = widths.HasDepths ? widths.DepthM(qs.Order) : DepthFor(radius)`
  becomes `DepthM(q)` with the same `q` the width uses there (`NaturalQ`).
- **`Depth(in RiverSegment s, float added)`, around line 2000.** It currently
  combines `DepthM(s.Order)` with the added flow via
  `pow(pow(D, 1/DEPTH_EXPONENT) + added, DEPTH_EXPONENT)`. With the model this is
  simply `DepthM(CurrentQ(seg, s) + addedQ)`, the same form as the width beside it.

This makes a river's depth follow its flow the way its width already does. A
reach that gains water from an edit deepens, and one that loses it gets
shallower. The order table becomes a fallback for exports without `depth_model`.

## How to verify

1. Load a Lierne export from kvterrain at or after this change. `HasDepths` is
   true, and `DepthM(order)` returns 0.5–0.7 m.
2. Reroute a river into a dry valley. The derived channel is carved about 0.5–1 m
   deep, not 2–10 m.
3. With the recommended change, dig a diversion off a river. The remaining reach's
   depth drops with its flow, and so does the diversion's.
