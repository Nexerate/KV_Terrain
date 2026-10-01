# Runtime spec: side channels follow the river they leave

**For:** the Unity runtime. The affected files are:
- `RiverNetworkLoadSystem.cs` (load the new file)
- `WaterHierarchy/RiverFlowEdits.cs` (the flow pass)
- `WaterHierarchy/RiverEditSuppression.cs` (its own Kahn pass)
- any harness check that treats a segment without links as a headwater or an orphan

**From:** kvterrain, side channels phase 1, 2026-10-01.

**Size:**
- one new optional input file;
- one extra loop after the flow pass.

`rivers.bin` does not change: still v2, the same layout. `river_widths.bin` does
not change either; it simply has more segments.

## Why

NVE maps one line per river. Where a river splits around islands, or fans into a
lake over several mouths, the line follows one channel. N50's river area holds
all of them. The runtime only draws water along lines, so the other channels
rendered as bare ground. At Ågardselva the river narrowed to a third of its width
about 230 m above Skjeldbreivatnet, where two of its three mouths left it. On
Lierne, 18% of N50's river area inside the region was undrawn, and 89% of that had
no line through it at all.

The exporter now traces a line down the middle of each undrawn piece that
connects:
- **a side arm**, which leaves a drawn river and rejoins it further down;
- **a delta channel**, which leaves a drawn river and ends in the lake that river
  runs into.

On Lierne that is 92 side arms and 2 delta channels, 25 km of line; the undrawn
share of the N50 river area falls from 18% to 13%. Pieces with only one
contact (bulges, backwaters, pools) are not traced yet.

Each side channel is an ordinary `rivers.bin` segment, appended after the mapped
ones, with its own widths, levels, discharge and carved bed. It has **no upstream
and no downstream links**. Linking it would break two things:
- The flow pass assumes one downstream per segment.
- A side arm linked back into its river would count the same water twice where it
  rejoins: `local = Q − ΣQ_up` would go negative and clamp.

Without links, a runtime that ignores the new file treats each side channel as a
headwater carrying its exported flow. That draws correctly. The only thing lost is
that a side channel does not dry up when its river is dammed. **The owner wants it
to dry up** (2026-10-01); that is what this spec adds.

## The file: `river_branches.bin`

The manifest lists it at `water_vector.river_branches` when present. That block
repeats the layout below and carries the file name. The file is absent when the
export has no side channels, including exports made before this change.

Layout (little-endian):
- **Header:** `8s magic "KVRBRAN1"`, `u32 version = 1`, `u32 branch_count`.
- **Per branch**, 28 bytes:

  | Field | Type | Meaning |
  |---|---|---|
  | `seg_id` | u32 | the side channel's `rivers.bin` segment id (= index) |
  | `parent_seg_id` | u32 | the mapped segment it leaves |
  | `parent_vertex` | u32 | the vertex of the parent it leaves at |
  | `share` | f32 | its share of the parent's flow, 0 < share < 1 |
  | `rejoin_seg_id` | i32 | the segment it rejoins, −1 for a delta channel |
  | `rejoin_vertex` | u32 | the vertex it rejoins at, 0 for a delta channel |
  | `lake_id` | u16 | the local lake id a delta channel ends in, 0 for a side arm |
  | `pad` | u16 | 0 |

- **Order:** a parent always appears before any branch that names it. Today
  every parent is a mapped segment.

**Validation at load.** Refuse the file, log a warning and treat every side
channel as an ordinary headwater if any of these fail:
- magic and version;
- every `seg_id`, `parent_seg_id` and non-negative `rejoin_seg_id` is a valid
  segment index;
- `parent_vertex` and `rejoin_vertex` are within their segments' vertex counts;
- `share` is finite and in (0, 1);
- the side channel's segment has `UpstreamCount == 0` and `DownstreamId == -1`.

## The rule

After the flow pass has set `EffectiveQ` for every segment, in file order:

```
EffectiveQ[b.seg_id] = b.share * EffectiveQ[b.parent_seg_id]
```

- A side channel feeds nothing downstream, so nothing else needs recomputing.
- A dammed or dried parent narrows or dries its side channels with it. One
  whose `EffectiveQ` drops to 0 is hidden like any other reach.
- Count a side channel in `dried` only when its parent's `EffectiveQ` is 0 and
  its own exported discharge was above 0. That keeps the dried count about
  rivers, not about how they are split.
- **Breaking a side channel itself:** its captured flow is
  `share × EffectiveQ[parent]`, attributed up the hierarchy from its break node
  as today. Its flow is not taken out of the parent, so damming a side arm does
  not add flow to the river beside it. That is deliberate: the exporter never
  subtracted it either, and the river's own exported discharge is the measured
  quantity.
- **`RiverEditSuppression`** runs its own Kahn pass. Apply the same override
  after it, if it decides visibility from flow.

The width rule from the widths spec still holds:
`k = WidthM(EffectiveQ[b]) / WidthM(b.DischargeM3S)`. A side channel narrows with
its parent and disappears with it.

## What the export guarantees

- **Joins.** A side channel's first vertex is exactly at its parent's
  `parent_vertex`, the same x/y. A side arm's last vertex is exactly at
  `rejoin_vertex`.
- **Levels.** A side channel's level runs from its parent's level at
  `parent_vertex`, falling evenly by arc length:
  - a side arm falls to the rejoin segment's level at `rejoin_vertex`;
  - a delta channel falls to the lake's level at its first lake-span vertex, and
    stays at the lake level after that.
  
  So the scatter's "nearest first, max on a tie" meets the same level at both
  ends: no step at a split, a rejoin or a mouth. The `w ≥ ground` and
  `surface.atlas` properties are the same as for mapped rivers.
- **Delta lake spans.** A delta channel has exactly one lake span, which runs to
  its last vertex. It reaches about 15 m into the lake, so its mouth is an inflow
  like any other.
- **Footprint.** A side channel's width at each vertex is twice its distance to
  the nearer bank of the mapped piece it runs through. Its stamp stays inside
  N50's river area, as for mapped rivers. Near a split it overlaps the parent's
  stamp, which is harmless.
- **Discharge.** A side channel's discharge is `share × the parent's exported
  discharge`, stored f32. Its `WidthM` and depth come from that discharge by the
  usual models. `share = w_b² / (w_t² + Σ w²)`, where `w_b` is the side channel's
  median visual width, `w_t` the river's alongside, and the sum runs over every
  side channel beside the same reach of that river, this one included. A reach
  runs from where a channel leaves to where it rejoins, or to the river's lake;
  channels whose reaches overlap share. So the shares at one split always sum
  to less than 1 (kvterrain after d9f93af; before, Ågardselva's two mouths each
  took their share against the river alone and summed to 1.40).

## Harness checks

Checks that walk links will see side channels as unlinked segments:
- **Connectivity:** count them separately, not as orphans.
- **Junction level jumps:** compare a side channel's first vertex with
  `parent_seg_id[parent_vertex]`. Compare a side arm's last vertex with
  `rejoin_seg_id[rejoin_vertex]`. Expect 0.
- **"Dry partway" and similar walks:** start a side channel's walk at its parent.

## How to verify

1. **Load:** load a Lierne export made from kvterrain at or after this change.
   The log reports `river_branches.bin` accepted with 94 branches.
2. **Ågardselva into Skjeldbreivatnet** (around E421050 N7153650): three mouths
   are drawn, not one.
3. **Dam the parent** upstream of the split: all three mouths narrow together
   to what still joins the parent below the dam (the runtime narrows a reach to
   its local inflow rather than drying it). Undo the edit: they return.
4. **Dam one side channel only:** that channel is hidden below the dam and the
   river beside it is unchanged.
