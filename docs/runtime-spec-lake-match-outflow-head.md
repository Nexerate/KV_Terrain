# Runtime spec: match lakes at spill + outflow head

**For:** the Unity runtime, `Assets/Terraform/WaterHierarchy/DepressionLakeMatch.cs`
**From:** kvterrain `2273841` ("open lake outlets onto riffles held by the outflow head"), 2026-09-30
**Size:** one rule, applied at three places in `MatchOne`. No file format changes.

## Why

Until now every lake's spill was its shore ring, which sat exactly on the authored
level. Outflow heads existed, but almost no lake depended on one. On Lierne, 2,171
lakes were `ok` and 0 were `held_by_outflow`.

The exporter now opens lake outlets so rivers connect to their lakes. Each outlet
becomes a **riffle**: the river's first samples are raised to a crest below the
lake's level, and the river surface over it (in `surface.atlas`) is the lake's
level. So the lake's spill is a river sample (`WaterClass == 1`), and its outflow
head is `level − crest`. This is the WATER_REDESIGN model: a lake stands above its
outlet by the depth of the water leaving over it.

`OwnLevelM` and the level table already hold such a node to `spill + head`.
Lake matching does not. `MatchOne` floods each candidate node only to
`min(level, spill)`. For a lake held by its head, that pool is just the part of
the lake deeper than the riffle. A lake shallower than its outlet river (bogs,
ponds) lies entirely above its bare spill, so the pool is empty and the lake
matches nothing.

The exporter's audit has the same flaw, and it is fixed there in the same commit
(`kvterrain/wateraudit.py`, `audit_lakes`, `hold_code`). On Lierne without that
fix, 284 lakes had no node; with it, 11 do (7 before the outlet change).

## The rule

Everywhere `MatchOne` uses a node's bare `SpillCode` as the height its pool may
fill to, use the height the node **holds** instead:

```
holdCode(u) = (double)t.Nodes[u].SpillCode + headM(u) / stepM

headM(u) = (t.Nodes[u].Flags & FLAG_EDITED_ESCAPE) != 0
         ? t.Nodes[u].OutflowHeadM
         : DepressionEditor.OutflowHead(in t.Nodes[u], l)
```

`headM` is the **physical** head: river surface minus spill, read at the spill and
outlet cells, only where they are river samples. `MatchOne` already uses this same
expression after the walk. Do **not** use the stored `OutflowHeadM` for an
unedited node. It can carry an earlier match's `NeededHead` raise, and the lake
would then match itself on it.

The three sites:

1. **Chain building** (the `for (uint u = leaf; …)` loop before `hopeless`):
   ```
   double e = Math.Min(levelCode, holdCode(u)) + tolCode;
   …
   if (holdCode(u) > levelCode + tolCode) break;
   ```
2. **The walk** (`while (v != ROOT && v != NONE)`):
   ```
   double eff = Math.Min(levelCode, holdCode(v)) + tolCode;
   …
   if (holdCode(v) > levelCode + tolCode) break;
   ```
3. **Monotonicity.** `eff` can now fall going up the chain. A child held by a river
   head can have a parent that spills over land with head 0, so its hold is lower.
   The existing `monotone` flag detects this, but check two things:
   - that `GrowingPool`'s non-monotone path (`m_Pending`, re-testing waiting
     samples) stays exact;
   - that `hopeless` is still only used when `monotone`.

   The comments saying "eff never falls, since spills do not fall going up" (near
   `THE POOL ONLY GROWS UP THE WALK` and in `GrowingPool`) are no longer true in
   general and need rewording.

Nothing else in `MatchOne` changes. The status test after the walk
(`spillM + tolM < LevelM <= spillM + headM + tolM` gives `NeededHead`) is already
correct.

### Arithmetic, for bit parity with the exporter

```
exporter: step_m    = (height_max - height_min) / 65535           (float64)
          head_code = float64(outflow_head_m[v]) / step_m          (head is float32)
          hold      = float64(spill_code[v]) + head_code
          eff       = min(level_code, hold) + tol_code
runtime:  stepM = (hmax - hmin) / 65535.0
          holdCode = (double)SpillCode + (double)headF / stepM      (headF = OutflowHead(), a float)
```

The exporter's head is `set_outflow_heads`: the float64 difference, stored as
float32. `OutflowHead` computes the same difference and returns it as a float, so
both sides see the same float32 value. Keep the division (not a multiplication by
a reciprocal) so that `holdCode` matches the exporter bit for bit.

## What does not change

- **`OutflowHead`** still reads heads only at river samples. The riffle is a river
  sample, and `surface.atlas` carries the lake's level there.
- **`OwnLevelM`, the level table, rendering.** These already use `spill + head` and
  the authored level.
- **Formats.** `hierarchy.bin`, `surface.atlas`, `water_id.atlas`, `rivers.bin`
  and `lakes.json` keep their layouts. Only their contents change:
  - Heights: lake beds get delta and funnel channels, and riffle river samples
    are raised.
  - `surface.atlas`: riffle samples are at the lake's level.
  - `rivers.bin`: `z` and `level` at riffle vertices.
  - `hierarchy.bin`: about 800 lake nodes now spill at a riffle, with
    `outflow_head_m > 0`, and are raised to `level − spill` where needed.

## Parity item that predates this change

Since kvterrain `0be3ac8` (2026-09-29), the exporter's walk has **two** starting
leaves:
- It groups the lake's leaves by their first ancestor that holds above
  `level + tol`. It then starts from the leaf holding most of the mask within the
  group holding most of the mask.
- It also starts from the overall majority leaf, as before.

It also skips re-flooding a node when both it and its parent hold below
`level − tol`. Both rules now use `hold`, not the bare spill. The runtime starts
from the majority leaf only.

On beds with relief these can pick different nodes. `0be3ac8` added the group
start because on Lierne 12 lakes lost their node when the first bed with relief
shipped. If the harness compares runtime matches against the exporter's
`water_audit.json`, expect differences on those lakes. Decide whether to port the
group start (see `audit_lakes` in `kvterrain/wateraudit.py`).

## How to verify

1. **Unit test** (mirrors `test_a_lake_shallower_than_its_outlet_river_is_matched_through_its_head`
   in kvterrain `tests/test_lake_levels.py`):
   - Build a 15 × 15 lattice at 20 m, with a lake bed at 13 m in `[4:11, 4:11]`.
   - Add a scour hole at 11.5 m at (7, 10) and a riffle at 12 m at (7, 11), both
     river samples.
   - Continue the river at 11 m to the east edge, with its surface at 14 m.
   - Set the authored level to 13.5 m.
   - Expected: a match with IoU > 0.95, spill 12 m, head 2 m, `NeededHead` 1.5 m.
     Before the change this gives no match.
2. **Harness on a fresh Lierne export** (from kvterrain `2273841` or later):
   - Compare the exporter/runtime lake match: nodes that agree, nodes that differ,
     and levels that differ.
   - Check that incremental and full matches stay identical under the random-edit
     sweep (`SWEEP_LAKES=1`).
   - Check that `LAKE_CHECK` still shows no mismatch between the growing pool and
     the full flood. The non-monotone path is now exercised much more often.
3. **In the scene:**
   - A lake held only by its head, such as a bog lake with a deep outlet river,
     renders at its authored level.
   - Digging through its riffle (a breach) lowers it to
     `min(authored, new spill + head)`.
   - Cutting its inflow and letting it drain leaves it standing at the riffle
     crest.

## Exporter numbers (Lierne, 2026-09-30)

| | before the mouth work | now |
|---|---|---|
| `ok` | 2,171 | 1,371 |
| `held_by_outflow` | 0 | 789 |
| `no_node` | 7 | 11 |
| `level_above_spill` | 2 | 9 |
| inlets open | 0 of 1,573 | 1,488 |
| outlets on a riffle | 0 of 1,159 | 827 (median 1.4 m deep) |
| river pits in the audit | 578 | 223 |

The remaining 332 outlets keep their sill. Each of those lakes has a hollow in its
bed, which a real share of the lake drains into, whose rim lies within 10 cm of
the level, so there is no room for a riffle.
