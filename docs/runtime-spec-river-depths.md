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
c = 2.0, f = 0.3, min_m = 0.5, max_m = 5.0, scale = --river-depth-scale (1.0)
```

- On Lierne a headwater stream stays at the 0.5 m floor, a lake inlet or outlet
  (0.1–0.5 m³/s) is 1–1.6 m deep and the largest rivers (about 13 m³/s) about 4 m.
- `f = 0.3` is not the runtime's `DEPTH_EXPONENT` (0.4). The textbook
  `0.3 * Q^0.4` was tried first and left nearly every river 0.5–0.8 m deep, since
  modelled discharge on a 40 km tile stays small. Read `c`, `f`, `min_m`, `max_m`
  from the manifest rather than hard-coding them.

The manifest carries the model twice, with the same content:
- `water_vector.rivers.depth_model`
- `water_surface.depth_model`

## Compatibility (no runtime change needed)

`water_surface.river_carve_depth_by_order_m`, which `RiverWidthModel.DepthByOrderM`
reads, is still written. It now holds the median modelled depth of each order's
segments. On Lierne that is 0.5, 0.6, 0.9, 1.4, 1.6, 2.7, 2.5 and 3.8 m for orders
1–8. So `DepthM(order)`
in `DerivedRiverSystem` already returns depths that match the carved channels.
Without this, reroutes and springs would have used depths 2–3 times too large.

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

Note the runtime's added-flow formula assumes `DEPTH_EXPONENT = 0.4`. With the
model it should use the manifest's `f` instead, or better, evaluate `DepthM(q)`
directly as above.

This makes a river's depth follow its flow the way its width already does. A
reach that gains water from an edit deepens, and one that loses it gets
shallower. The order table becomes a fallback for exports without `depth_model`.

## How to verify

1. Load a Lierne export from kvterrain at or after this change. `HasDepths` is
   true, and `DepthM(order)` returns 0.5–3.8 m, rising with order.
2. Reroute a river into a dry valley. The derived channel is carved about 0.5–2 m
   deep, not 2–10 m.
3. With the recommended change, dig a diversion off a river. The remaining reach's
   depth drops with its flow, and so does the diversion's.
