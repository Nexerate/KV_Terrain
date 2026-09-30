# Runtime spec: match lakes at spill + outflow head

**For:** the Unity runtime, `Assets/Terraform/WaterHierarchy/DepressionLakeMatch.cs`
**From:** kvterrain `2273841` ("open lake outlets onto riffles held by the outflow head"), 2026-09-30
**Size:** two rules in `MatchOne`: fill to spill + head (three places), and do not
pick a basin that spills into its own lake (see "Second rule"). No file format
changes.

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

## Second rule: a basin that spills into its own lake is not the lake

Added with kvterrain's shoreline change (the commit after `4c64c40`).

**Why.** Lake edges no longer sit exactly on the level. A narrow neck of a lake now
lies about 30 cm under water, so the depression hierarchy sees the lake as two basins
that join at the neck, below the level. The walk evaluates both:
- the basin on one side of the neck, and
- its parent, which holds the whole lake.

The parent's pool often also covers an inflow trench the lake backs up into, which
lowers its IoU. The one-sided basin can then win the IoU contest. It is reported as
`level_above_spill`, although its water only spills into the other half of the same
lake. On Lierne this caused about 30 false findings.

**The rule.** During the walk, a node `v` whose saddle lies inside its own lake is
not eligible as the lake's node, provided some other node on the walk is eligible.
A node's saddle lies inside the lake when all of these hold:
- `holdCode(v) < levelCode − tolCode`;
- its `SpillCell` is one of this lake's samples (lake class, this lake id, not an
  island);
- its `OutletCell` is also one of this lake's samples.

Keep evaluating such a node, and keep climbing past it. Only pick it if the walk
finds no other candidate. The exporter keeps two bests, `best` and `fallback`, and
uses `best if best is not None else fallback`. See `audit_lakes` in
`kvterrain/wateraudit.py`. A cell is "one of this lake's samples" exactly when the
exporter's audit mask says so: water class lake, the lake's id, and not an island.

**Effect on Lierne**, together with the shoreline change:

| Audit status | Count |
|---|---|
| `ok` | 1,227 |
| `held_by_outflow` | 945 |
| `no_node` | 4 |
| `level_above_spill` | 4 |

## Third rule: no node holds above an authored lake inside it

**From:** kvterrain, 2026-09-30, after the runtime's river audit.

The exporter now caps `outflow_head_m` on every node that is **not** an authored
lake, so that `spill_m + outflow_head_m` never exceeds the lowest
`authored_level_m` of an authored lake in its subtree. Such nodes carry the new
flag `FLAG_HEAD_CAPPED = 1 << 7` (bit 6 stays the runtime's `FLAG_EDITED_ESCAPE`).

Why: the head is read at the saddle's river samples, and for these nodes that
sample was not the river carrying water out. It was an inflow's carved trench on
the lake's rim, or a stray river sample beside the lake carrying a nearby river's
higher surface. The level table fills a subtree to the highest level on its
ancestor path, so 392 lakes were drawn above their shores. The runtime's
`inflow_heads_check.py` reports 0 cases on an export with this change (195 before).

Runtime change needed: wherever a head is recomputed for a node that is not an
authored lake (`DepressionEditor.OutflowHead` on rebuilt nodes), apply the same cap:

```
lowest(u) = min of
    authored_level_m over authored lake NODES strictly inside u's subtree, and
    authored_level_m (lakes.json) over lakes with a sample (water_id >= 16) that
    u or a descendant labels (labels.atlas)
if (!authored(u) && lowest(u) is finite)
    head = min(head, max(lowest(u) - spill_m(u), 0))
```

The second term (added after the runtime's re-audit) catches a basin that holds
some of a lake's samples without being the lake node's ancestor: a basin beside
the lake, or a small one at its outlet. It left 24.9k lake samples drawn high.
`hierarchy.lake_sample_levels` and `lowest_authored_level_below` are the reference.

Without it, an edit that rebuilds such a node brings the old head back. Unedited
nodes need nothing: the level table reads the stored head.

`validate-hierarchy` recomputes heads from `surface.atlas` and applies the same cap
before comparing.

## Fourth rule (runtime only): cap the whole hold

A head cap cannot help where a node spills on bare ground **above** an authored
lake inside it with no head at all. The scan's rim there stands higher than
NVE's level (Kroktjønna, Gristjønnin, Småtjønnin and others: 24.7k lake samples).
The authored level is taken as right, so the runtime caps the hold itself:

```
hold(u) = spill_m(u) + head(u)
if (!authored(u) && lowest(u) is finite)
    hold(u) = min(hold(u), lowest(u))
```

The exporter does not open those rims. Most have no mapped outlet river at the
spill, so opening them would carve channels into terrain where nothing is mapped.
Nothing in the export changes for this rule. `outflow_head_m` stays the physical
head (after the third rule), so `validate-hierarchy` needs no change.

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
