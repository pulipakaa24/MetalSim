# Heightfield contacts in MetalSim's MuJoCo Warp fork: three defects, the fix, and its effect

2026-10-31. Scope: make heightfield contacts of primitive geoms (sphere, capsule, box, cylinder, ellipsoid) in
[my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) match CPU
[MuJoCo](https://github.com/google-deepmind/mujoco), check the mesh path (Isaac's G1 feet) against the same standard,
and settle the margin convention. Found through the [`g1_bumps` film](../renders/G1_BUMPS.md)
([Unitree's G1](https://github.com/unitreerobotics/unitree_rl_gym), 5 mm sphere feet, on
[Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s random rough tile): CPU MuJoCo 11 of 16 robots fall, my fork far
fewer (level 0.10 of `runs/film/g1_bumps/measure/cpu_levels.json` and `gpu_levels.json`). **measured** = run here
(MacBook Pro 14-inch M4 Max, Mac16,6; GPU work through `scripts/gpu_run.sh`); **reported** = from the cited source;
**estimated** = our judgement.

Code: branch [`metalsim-hfield-contacts`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim-hfield-contacts) of
my fork on
[`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) (the pinned
`metalsim` head); full diff
[`8eb51fe...93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/compare/8eb51fe637437108185f5a4750b7f5f64ef26624...93e5ce18e76854c430ce669314af00ddceb92147).
**Landed 2026-10-31:** my fork's
`metalsim` was fast-forwarded to it
([`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147)), then to the
follow-up of section 12
([`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f)), which
MetalSim pins (the commit README.md's `pip install` line names); the six commits are `0018`-`0023` of
[`patches/mujoco_warp/`](../../patches/mujoco_warp/) (on
[`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624)), the
follow-up's eight `0024-0031` (on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147)), the whole
branch `patches/mujoco_warp/metalsim-787a5b3.diff` (on upstream MuJoCo Warp v3.14.0
[`88af9cc`](https://github.com/google-deepmind/mujoco_warp/commit/88af9cc682d42e2e925daa0bae25f53277999f7b)). The
probes, harnesses, logs and summaries cited below are committed in `runs/readme_media/records/hfield-contacts/`: the
bare `probe/`, `falls/`, `gpu/`, `cc*` and `sel_proto.py` paths of sections 2-11 are relative to it (the scripts still
name the scratch directory they ran in). The per-placement contact arrays (6.4 MB of `.npz`) are not published.
Sources:
[MJW] [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): my fork at
[`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) and upstream
`main` at [`d41a47c0`](https://github.com/google-deepmind/mujoco_warp/commit/d41a47c0837fc71abb44d7d4799bf85ca19ff7ab)
(2026-10-31);
[C] [MuJoCo C 3.14.0](https://github.com/google-deepmind/mujoco/tree/3.14.0) sources
([`src/engine/`](https://github.com/google-deepmind/mujoco/tree/3.14.0/src/engine)), the version installed here;
[C-main] https://github.com/google-deepmind/mujoco (main, read 2026-10-31), commit
[8626d3c](https://github.com/google-deepmind/mujoco/commit/8626d3c35eceb8b36dbc5182852b25ffde7a6465) "Fix margin
and gap double-counting in heightfield collisions" (2026-10-06).

## 1. Summary

MuJoCo C and MuJoCo Warp collide a convex geom with a heightfield the same way: every triangle of the grid cells
under the geom is extruded down to the heightfield base into a triangular prism, and each prism is a convex body for
GJK/EPA; one contact per penetrating prism [C `mjc_ConvexHField`, `engine_collision_convex.c:1170-1328`]. MuJoCo Warp
then keeps at most 4 contacts per pair. Two defects shared with upstream MuJoCo Warp (the second also in MuJoCo C)
and one in MetalSim's own mesh path made the contacts of
[my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) wrong:

1. **Side-wall override** (upstream MuJoCo Warp [#928](https://github.com/google-deepmind/mujoco_warp/pull/928),
   [`f2f79579`](https://github.com/google-deepmind/mujoco_warp/commit/f2f795796fc433adf8e235f01fae3747585ae5db), still
   on upstream `main`): when GJK's closest feature of a prism is a side wall, `_inflate` (and `_epa_witness` for deep
   contacts) replaced the result with a vertical contact against the prism's top triangle and `dist = -|x1 - x2|`, a
   penetration whatever the real separation. A 5 mm sphere 8.6 mm clear of a prism came back 25.6 mm deep with normal
   (0, 0, 1). The kernel also kept separated prisms as candidates, and its 4-contact selection dropped every candidate
   within 1 mm of the deepest by position alone, so a sphere in a crease lost its second face.
2. **Prism support index** (upstream MuJoCo Warp, and MuJoCo C 3.14 and main): EPA on discrete pairs (box, mesh)
   stops when a new support point repeats a vertex-index pair already in the polytope. The prism's support function
   gave all six vertices the same index (MuJoCo Warp: -2 / -3 by the sign of dir.z; MuJoCo C: never set, -1), so EPA
   stopped early: a box 9.25 mm into a prism came back 2.57 mm (Warp) or 3.44 mm (C) deep along wrong axes.
3. **Mesh plane path** (my fork's
   [`f2716b4`](https://github.com/pulipakaa24/mujoco_warp/commit/f2716b4e7873ed40b8405c0840238610349c7e7e)): a mesh of
   at most 256 vertices was tested vertex by vertex against each triangle's plane inside its column; any contact whose
   deepest point is a terrain vertex or edge inside the mesh was missed. Isaac's G1 foot on the random rough tile: no
   contact in 178 of 1000 placements that touch.

After the fix every heightfield contact on the tile matches the exact geometry: per-triangle closest features for
spheres and capsules, the separating-axis penetration per prism for boxes and the G1 foot mesh; normals within
0.02 deg (p99), depth errors 0.0 mm (p95); contact sets 99.9-100 % equal to the exact ones within the 4-contact cap
(pinned fork: 53-89 %; MuJoCo C 3.14: 69-99.5 %, section 8.1). Unitree's G1 on Isaac's bumps now falls as often as on
CPU MuJoCo within the run-to-run spread (75.8 % of 240 robots vs 68.3-68.8 %; pinned fork 18.8 %). Cost: about 15 % of
Isaac's G1 rough-task env step under random actions (its feet are meshes and now take GJK/EPA), 1.2-1.3x per PPO
iteration in the 2.3.2 rough configs and, with Isaac Lab 3.0's geom gap, 1.7x after the follow-up of section 12
(2.05x and dropped contacts before it); nothing for primitives. The heightfield margin now follows MuJoCo C main
(section 7). Every rough-terrain result of Isaac's G1 on MuJoCo Warp
([PARITY §1.5](../PARITY.md#15-learning-on-the-g1-task)) needed a re-run (section 9). Re-run on the repaired kernel
(section 13): the 2.3.2 rough preset's three seeds end within the seed spread of the old ones (+18.0 ± 2.4 at 1500,
level 6.11, against +19.4 ± 3.1, 6.07; Isaac +14.1, 5.90) but climb the curriculum later (level 5.48 ± 0.09 at 1000
against 5.83 ± 0.21); `rough_il3` ends at +12.0, level 5.93 (was +8.4, 5.96; Isaac +14.5, 5.80; one seed, initial reset
added); transfer falls barely change. The Isaac-side and timing rows are pending.

## 2. Baseline, re-measured on the pinned fork ([`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624))

The `metalsim` head already sends primitives to upstream's GJK/EPA prism path
(`HFIELD_PLANE_CONTACTS_PRIMITIVES = False`) and meshes of at most 256 vertices to the plane path. The g1-bumps column
"MetalSim fork: 81.9 %, 1/16 falls" had measured a stale checkout
([`07a51a6`](https://github.com/pulipakaa24/mujoco_warp/commit/07a51a63bce527d2d26b363a32c52765df4412fc)); the pinned
fork is the column "upstream MJW". **measured**:

- Contact check (`runs/film/g1_bumps/measure/contact_check.py`, 3000 spheres of 5 mm pressed 0.5-3 mm into the tile,
  [Warp](https://github.com/NVIDIA/warp)'s CPU device): contact count equal to the exact set 88.8 % (126 fewer, 211
  more), 6.11 % of contacts more than 10 deg from the nearest true normal (p95 21.8 deg), deepest-contact depth error
  p95 8.0 mm. CPU MuJoCo: 99.6 % (0 fewer, 11 more: in all 11 a duplicate of one feature reached from two prisms, same
  point within 0.1 mm and normal within 0.05 deg), 0 normals off. As the brief.
- G1 falls on Isaac's tile (seed 0, 16 robots, 8 s): GPU runs of the pinned fork gave 2, 5, 4, 3, 1 and 3 of 16 in
  six runs: MuJoCo Warp on Metal is not run-to-run repeatable here (contact order from atomics, chaotic gait), so
  the earlier 6/16 (the stale checkout with every geom on GJK/EPA, level 0.10 of
  `runs/film/g1_bumps/measure/gpu-upstreamhf_levels.json`) is inside that spread, and a single
  run is not a measurement. Section 8.3 uses five tile seeds and repeats.

## 3. Defect 1: the heightfield side-wall override, separated prisms, and the 1 mm selection rule

Probe `runs/readme_media/records/hfield-contacts/probe/gjk_point_prism.py 15` (placement 15: exact depth 1.85 mm,
fork 25.6 mm with normal (0, 0, 1)) runs MuJoCo Warp's `gjk` and `ccd` on the sphere against each of the 18 prisms
around it. **measured**: GJK's point-to-prism distance is right (prism 9: 3.147 mm, float64 3.147 mm; prism 8: 13.567
mm vs 13.574 mm), but `ccd` returns -25.6 mm for prism 8, which is 8.6 mm clear of the sphere. The source:
`collision_gjk._inflate` [MJW `collision_gjk.py:2455-2485` at
[`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624)] has a
heightfield branch (upstream [#928](https://github.com/google-deepmind/mujoco_warp/pull/928), "Improve
sphere/capsule <-> hfield collisions",
[`f2f79579`](https://github.com/google-deepmind/mujoco_warp/commit/f2f795796fc433adf8e235f01fae3747585ae5db),
2025-12-17): when the GJK simplex holds both top and bottom prism vertices (the closest feature is a side wall) it takes
the geom's support point, projects it vertically onto the top triangle (or a vertex height) and returns
`dist = -|x1 - x2|`, always a penetration. `_epa_witness` [`collision_gjk.py:996-1027`] does the same for deep contacts
(and calls `support` with a position as the direction). MuJoCo C has no heightfield case in its GJK/EPA [C
`engine_collision_gjk.c`]: the prism is a plain polytope there. Both overrides are still on upstream `main` at
[`d41a47c0`](https://github.com/google-deepmind/mujoco_warp/commit/d41a47c0837fc71abb44d7d4799bf85ca19ff7ab).

Two kernel details turned the phantom contacts into wrong contact sets [MJW `collision_convex.py` `ccd_hfield_kernel`]:
the pinned fork keeps a prism whenever `ccd` reports `ncontact` (it reports 1 for separated geoms too; upstream main
has since added `dist >= margin + gap` as a skip), and contacts 1-3 are chosen by spread (furthest from contact 0,
furthest from that line, ...) but the selection stops as soon as the furthest remaining candidate is within 1 mm of
contact 0 (`MIN_DIST_TO_NEXT_CONTACT`, upstream [#847](https://github.com/google-deepmind/mujoco_warp/pull/847)) by
position only. In a crease a 5 mm sphere touches two faces at points 0.4 mm apart (p5 of the tile's two-contact sets;
`sel_proto.py`), so the second face was lost.

## 4. Defect 2: the prism's support vertex index (MuJoCo Warp and MuJoCo C)

With defect 1 fixed, spheres and capsules matched the exact sets, but boxes still disagreed with MuJoCo C in 5 % of
contact normals. Probe `probe/sat_box.py` computes the exact penetration of the box into each prism by the separating
axis test (face normals of both, cross products of all edge directions); `probe/prism_mesh_c.py` gives the same prism
to MuJoCo C as a convex mesh (plain `mjc_Convex`, multiccd off). **measured**, box placement 257: SAT 9.250 and 3.321
mm; C mesh-box 9.250 and 3.321 mm (agree); C heightfield 3.436 and 1.111 mm; MuJoCo Warp heightfield 2.572 and
1.111 mm. A box corner sits 8.97 mm below its triangle's plane, so the heightfield results are wrong, and not from
the iteration cap (C gives the same with `ccd_iterations` 35, 200, 1000).

Mechanism: EPA for discrete geoms (boxes, meshes, heightfields, no margin) ends when the new support point repeats an
(index1, index2) vertex pair already in the polytope [C `engine_collision_gjk.c:1409-1418`; MJW `collision_gjk.py`
`_epa` 1411]. The heightfield support function never gives the six prism vertices distinct indices: MuJoCo C's
`mjc_prism_support` [`engine_collision_convex.c:459-477`] does not set `vertindex` (it stays -1 from
`mjc_initCCDObj`, line 732), and MuJoCo Warp's [`collision_gjk.py:205`] sets -2 or -3 by the sign of dir.z. Every new
prism vertex therefore looks like a repeat as soon as the box vertex index repeats, and EPA returns the face it had.
Only discrete pairs are affected (box, mesh); spheres, capsules, cylinders and ellipsoids are not. **reported**:
MuJoCo C main still leaves `vertindex` unset in `mjc_prism_support` [C-main, read 2026-10-31].

## 5. Defect 3: the mesh plane path (MetalSim's own)

Commit [`f2716b4`](https://github.com/pulipakaa24/mujoco_warp/commit/f2716b4e7873ed40b8405c0840238610349c7e7e) of
[my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) (PARITY §1.6) replaced GJK/EPA for
meshes of at most 256 vertices with a per-triangle plane test: per prism, the deepest mesh vertex below the triangle's
plane whose footprint lies in the triangle. It was written because GJK/EPA returned inverted normals and 2 m
penetrations for the G1 feet. **estimated**: the `_epa_witness` override of defect 1 (a support point taken in the
direction of a position vector) is the likely source, but not confirmed: §1.6's minimal repro (a box mesh pressed 3 cm
into a flat heightfield, base 1.9 m; `probe/mesh_repro.py`, four yaw angles) on the pinned fork's GJK/EPA path no longer
shows a 2 m penetration (depths 1.2-3.0 cm; 4 of 16 normals sideways; the fixed fork 8 of 16, all at the box's edge
prisms, like MuJoCo C's own 1.2-1.3 cm side contacts there). The plane test cannot see a terrain vertex or edge that
pokes into a mesh face. **measured** on Isaac's G1 foot collider (8-vertex box mesh, 0.203 x 0.065 x 0.019 m) at 1000
random poses pressed 0.5-3 mm into the tile: no contact at all in 178, 9.6 % of normals more than 10 deg off, depth
error p95 1.1 mm (max 10 mm), against the per-prism SAT.

## 6. The fix (fork commits on `metalsim-hfield-contacts`)

Every change but the penetration test is behind a module flag that restores the previous form.

| change | file | flag (default) |
|---|---|---|
| prism = plain convex polytope: no side-wall override in `_inflate` / `_epa_witness` (C's behaviour) | `collision_gjk.py` | `HFIELD_PRISM_POLYTOPE` (True) |
| prism support returns its vertex index 0-5, so discrete EPA's repeated-vertex test works | `collision_gjk.py` `support` | same flag |
| keep a prism only when GJK/EPA reports penetration of the margin-inflated shapes (C keeps `mjc_ccd < 0`) | `collision_convex.py` `ccd_hfield_kernel` | (always) |
| contacts 1-3: upstream's spread rules, skipping only duplicates of a selected contact (within 1e-4 m and 0.05 deg: one feature reached from two prisms); upstream stopped at the first candidate within 1 mm | same | `HFIELD_SELECT_DEDUP` (True), `HFIELD_DUP_DIST`, `HFIELD_DUP_COS` |
| prisms built relative to the other geom (one rounding per subgrid): float32 witness noise is ~1e-5 m 100 m from the centre of a 200 m field, ~0.3 deg on a 5 mm sphere's normal, above the dedup tolerance | same | `HFIELD_LOCAL_ORIGIN` (True) |
| margin and gap: C main's convention (section 7) | same, `_hfield_filter` | `HFIELD_TRUE_DISTANCE` (True) |
| meshes on GJK/EPA instead of the plane path; Warp's deterministic mode keeps the plane path for meshes (the GJK/EPA path is not repeatable there) | same, `convex_narrowphase` | `HFIELD_PLANE_CONTACTS` (now False) |

The 4-contact cap per heightfield pair (upstream [#847](https://github.com/google-deepmind/mujoco_warp/pull/847)) is
unchanged. It is the only remaining difference from the exact sets on the tile: 13 of 3000 spheres touch 5 or 6 distinct
triangle features.

## 7. Margin convention

MuJoCo C 3.14 (the installed version) passes margin + gap to `mjc_ConvexHField`, raises each prism's top by it and
inflates the geom by half of it, then calls `mjc_penetration` with margin 0 [C `engine_collision_convex.c:1150-1166,
1272, 1298`]; its prism height test compares the raised top with the geom's un-inflated bottom. **measured**
(`probe/margin2.py`, sphere / box / capsule / box mesh over a flat heightfield, margin 0.01 on both geoms, so 0.02 for
the pair): C reports `dist = g - 1.5 (margin + gap)` for a gap `g` and keeps the contact while `g < margin + gap`,
every shape; the pinned fork matched it for primitives and was 0.5 margin off on the mesh plane path. Every other
pair in C and in MuJoCo Warp reports `dist = g`. **reported**: MuJoCo C main fixed this on 2026-10-06
([8626d3c](https://github.com/google-deepmind/mujoco/commit/8626d3c35eceb8b36dbc5182852b25ffde7a6465), "Fix margin and
gap double-counting in heightfield collisions"): the prism is inflated like the geom (support offset by half the
margin), `mjc_penetration` gets the margin, the prism height test expands the geom's bounds by it, and its test
`MjcConvexTest.HFieldMarginAndGap` (terrain margin 0.02, gap 0.01, sphere 15 mm above) expects `dist` 0.015 and the
contact point halfway. [My fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) now follows C
main (`HFIELD_TRUE_DISTANCE`; the plane path too), so a heightfield behaves like a plane under a margin. It matches C
3.14 only at margin = gap = 0, which is every heightfield pair in MetalSim (Isaac G1 and Unitree G1 models: all margins
and gaps 0). False restores C 3.14's form.

## 8. Acceptance

### 8.1 Contacts against the exact geometry (static, Warp's CPU device)

Harness `runs/readme_media/records/hfield-contacts/cc2.py` (extends `contact_check.py`; outputs `cc/<shape>/`, whose
`compare.json` summaries are committed):
Isaac's random rough tile (seed 0, bumps 2-10 cm every 0.1 m); 3000 spheres (the g1_bumps placements) and 1000
random poses for each other shape, pressed 0.5-3 mm into the surface by bisection on MuJoCo C's deepest contact.
Reference: **exact** for spheres and capsules = every triangle whose closest point to the centre / segment is within
the radius (one contact each, duplicates by point merged); for the box and the G1 foot mesh = the separating-axis
penetration of the geom into each prism (MuJoCo's model, computed exactly; duplicates = equal depth and normal);
for cylinders and ellipsoids (no exact reference written) MuJoCo C, which is exact for these non-discrete shapes.
Cells: contact count equal to the reference's distinct contacts (capped at 4, MuJoCo Warp's per-pair limit) / worlds
with no contact where the reference has one / contacts more than 10 deg from every reference normal / normal error
p99 (deg) / deepest-depth error p95 (max), mm. **measured**:

| shape (size) | reference | pinned fork [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) | fixed fork | MuJoCo C 3.14 |
|---|---|---|---|---|
| sphere r 5 mm (Unitree G1 sole) | exact | 88.8 % / 0 / 6.11 % / 147.8 / 8.00 (70.0) | 99.97 % / 0 / 0 % / 0.02 / 0.00 (0.0) | 99.5 % / 0 / 0 % / 0.00 / 0.00 (0.0) |
| capsule r 2 cm, half-length 5 cm | exact | 82.6 % / 0 / 0.35 % / 0.05 / 0.00 (38.7) | 100.0 % / 0 / 0 % / 0.01 / 0.00 (0.0) | 74.5 % / 0 / 0 % / 0.00 / 0.00 (0.0) |
| box 10 x 6 x 4 cm | exact (SAT) | 76.6 % / 0 / 16.9 % / 139.8 / 7.27 (157.1) | 99.9 % / 0 / 0 % / 0.01 / 0.00 (0.0) | 69.4 % / 0 / 16.3 % / 59.9 / 6.09 (17.9) |
| Isaac G1 foot mesh (8 vertices) | exact (SAT) | 52.9 % / 178 / 9.5 % / 41.6 / 1.11 (10.2) | 99.9 % / 0 / 0 % / 0.02 / 0.00 (0.4) | 70.7 % / 0 / 9.9 % / 61.8 / 5.43 (32.0) |
| cylinder r 3 cm, half-height 2 cm | MuJoCo C | 83.7 % / 0 / 0.22 % / 1.02 / 0.00 (46.9) | 97.6 % / 0 / 0 % / 1.25 / 0.00 (0.0) | (reference) |
| ellipsoid 4 x 3 x 2 cm | MuJoCo C | 77.6 % / 0 / 0 % / 0.74 / 0.00 (0.0) | 97.6 % / 0 / 0 % / 0.85 / 0.00 (0.0) | (reference) |

Notes. MuJoCo C has no 4-contact cap and writes one contact per prism, so a feature reached from two prisms (a
shared edge, a terrain vertex in six prisms) is written twice or more; that, and its EPA defect for boxes and meshes
(section 4), is why its count column is lower. On the brief's own harness (`contact_check.py`, raw count equal to the
exact set, no cap) the fixed fork scores 99.5 % (14 spheres: 13 with 5 or 6 true contacts above the cap, one grazing
contact of 0.000 mm depth) against C's 99.6 % (11 spheres with duplicates); its one "normal > 10 deg" is the
harness's own artefact (`np.sign(-0.0)` zeroes my fork's horizontal normal (-1, -0, -0), which equals C's for that
sphere, a prism side wall). The pinned fork's mesh column is the plane path (its default for meshes). Near-horizontal
normals (|n_z| < 0.3), the side-wall contacts of the prism model, are 1.1 % (box) and 1.4 % (foot) of exact contacts,
1.2 / 1.6 % for the fixed fork and 2.9 / 2.5 % for C.

### 8.2 Tests

| suite | pinned fork | fixed fork |
|---|---|---|
| upstream `collision_driver_test.py -k hfield` (CPU) | 3 passed | 3 passed + 5 new (`test_hfield_sphere_crease`, `_no_phantom`, `_box_epa_depth`, `_mesh_on_peak`, `_margin_gap`); 4 of the new ones fail on the pinned fork |
| fork collision suites, CPU (`collision_driver_test`, `collision_gjk_test`, `collision_primitive_core_test`) | - | 115 passed, 2 skipped (before the 5 new tests) |
| fork collision suites, Metal | `-k hfield` 3 passed | 120 passed, 2 skipped (the 5 new tests included) |
| MetalSim `tests/test_terrain.py` (Metal) | 4 contact tests passed | 21 passed |

`test_terrain.py::test_hfield_mesh_contacts_match_mujoco_c` (G1 on Isaac's rough layout under a PD hold, 0.5 s):
fixed fork min normal z 0.912, max penetration 5.2 mm (C 4.6 mm), pelvis heights within 1 mm of C; pinned 0.913,
4.2 mm, within 3 mm.

### 8.3 Unitree G1 on Isaac's bumps (the g1_bumps measurement)

`runs/film/g1_bumps/measure/measure.py` through `falls/measure_seeds.py`: Unitree's G1 with its own pretrained policy,
16 robots per run, 8 s, command 0.5 m/s, Isaac's random rough tile (noise 0.02-0.10 m) with tile seeds 0-4; fall =
pelvis below 0.4 m. CPU MuJoCo is deterministic (one run per seed); MuJoCo Warp on Metal is not, so each seed is
repeated. **measured** (`falls/summary.json`):

| engine | runs per seed | falls per seed 0 / 1 / 2 / 3 / 4 (each run) | falls per 80-robot run | total |
|---|---|---|---|---|
| CPU MuJoCo 3.14 (Unitree's loop; the film's engine) | 1 | 11 / 10 / 14 / 12 / 8 | 55 | 55 / 80 = 68.8 % |
| CPU MuJoCo, start joints + N(0, 1e-6 rad) | 3 | 12, 11, 10 / 13, 12, 11 / 13, 11, 11 / 11, 11, 10 / 10, 13, 5 | 59, 58, 47 | 164 / 240 = 68.3 % |
| MuJoCo Warp, pinned fork [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) (Metal) | 3-6 | 2, 5, 4, 3, 1, 3 / 3, 3, 6 / 1, 6, 5 / 3, 1, 4 / 1, 1, 2 | 10, 12, 20 | 54 / 288 = 18.8 % |
| MuJoCo Warp, fixed fork (Metal) | 3 | 10, 11, 13 / 16, 14, 14 / 13, 12, 14 / 12, 13, 15 / 7, 8, 10 | 58, 58, 66 | 182 / 240 = 75.8 % |

The fixed fork falls as often as CPU MuJoCo within the run-to-run spread (per 80-robot run 60.7 +- 4.6 vs 54.8 +- 5.4,
mean +- sd; seed 0, the film's tile: 10-13 vs 10-12); the pinned fork kept 81 % of the robots walking. CPU MuJoCo is
deterministic, so its single run per seed has no error bar; a 1e-6 rad perturbation of the start already moves a seed
by up to 8 falls (seed 4: 5 to 13). On flat ground the two engines already agree
(`runs/film/g1_unitree/sim2sim_check/compare.json`: within 0.1 mm of Unitree's CPU run over 8 s; no heightfield, so
unaffected here). The remaining gap (about 6 falls per 80 robots,
about one standard deviation) is not resolved further here; candidates are float32 vs float64 and MuJoCo C writing
duplicate contacts at shared edges (each adds stiffness) where my fork writes one.

### 8.4 Relative cost

Isaac's G1 rough task (`python -m metalsim.learn.g1_velocity 4096 rough`: full env step with random actions,
synchronized env-steps/s over 110 steps; `gpu/batch_b.log`, `gpu/batch_c.log`), runs interleaved in one queue job
each. **measured on the 14-inch MacBook Pro (Mac16,6) on battery (41 -> 32 %, powermode 2): not a timing machine; the
ratios are indicative only, the absolute numbers are not performance figures.**

| fork | env-steps/s (runs) | relative |
|---|---|---|
| pinned [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) (meshes on the plane path) | 62,257 / 62,346 | 1.00 |
| fixed, with `HFIELD_PLANE_CONTACTS = True` (meshes on the plane path) | 62,279 / 62,407 | 1.00 |
| fixed, default (meshes on GJK/EPA) | 52,073 / 52,820 / 53,234 / 53,499 | 0.85 |

The primitive-path, selection, local-origin and margin changes cost nothing measurable; the whole cost, about 15 % of
the env step, is the G1's feet (8-vertex meshes) on per-prism GJK/EPA instead of the plane test. That is the env step
under random actions at margin = gap = 0; per PPO iteration and with a geom gap the ratio is larger (section 12).
Primitive geoms were already on GJK/EPA, so scenes like the Unitree G1 (sphere feet) pay nothing extra (the pinned
fork had the same path, with wrong results).

### 8.5 Stability under random actions (PARITY §1.6's protocol)

`gpu/random_actions.py`: Isaac's G1 rough task, 1024 envs from reset, 400 control steps of unit-std Gaussian x 0.5
actions (the task's own action map), no resets; heightfield contacts sampled every 10 steps. MuJoCo C runs the
task's MjModel (`mj_step`, as `test_hfield_mesh_contacts_match_mujoco_c`) on worlds 0-31 from the same start states
with the same actions. **measured**:

| engine | max \|qvel\| (rad/s) | worlds non-finite | heightfield normals \|n_z\| < 0.3 | min n_z | depth p99 / max (mm) |
|---|---|---|---|---|---|
| pinned fork (plane path) | 33.4 | 0 | 0 % | 0.662 | 13.0 / 30.3 |
| fixed fork, `HFIELD_PLANE_CONTACTS = True` | 49.2 | 0 | 0 % | 0.662 | 13.0 / 37.6 |
| fixed fork, default (two runs) | 147.8, 124.1 | 0 | 3.6 % | -0.000 | 12.0 / 30.8, 42.3 |
| MuJoCo C 3.14 (32 worlds) | 134,657 | 0 | 30.7 % | -0.775 | 45.4 / 67.8 |

No run blows up. Robots driven by random actions fall and lie with feet and the third collider 1-4 cm into the
terrain; at that depth MuJoCo's prism model gives near-horizontal contacts through the internal prism walls (a prism
whose horizontal overlap with the geom is smaller than the depth), 3.6 % of contacts in the fixed fork, and the
fastest joint reaches 124-148 rad/s against 49 with the plane path (PARITY §1.6 recorded 56 rad/s on flat ground).
MuJoCo C, with the same contact model plus its EPA defect, does far worse here (indicative only: it steps the compiled
MjModel, while BatchSim runs MuJoCo Warp with the task's solver options on top). This is the main residual risk of the
change for training on rough terrain: the side-wall contacts are MuJoCo's semantics, not a fork defect, but they are
not real surfaces (section 10).

## 9. Effect on PARITY §1.5 (rough-terrain training with Isaac's G1)

Isaac's G1 collides with the terrain through three 8-vertex box meshes (both feet and one more; every other geom
is excluded), so on the heightfield cells (random rough, slopes; the stair and box cells use exact boxes,
with the heightfield lowered under them) its contacts went through the plane path, which the static check
shows wrong on random rough (no contact in 17.8 % of touching foot poses, 9.5 % of normals > 10 deg off).
[My fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) now sends them through GJK/EPA.
**Affected**: every rough-terrain number in PARITY §1.4-1.6 measured on MuJoCo Warp (training curves and final returns
of the three seeds, terrain levels, the transfer tables in both directions, the wall-cell transfer counts, the rough
throughput rows, `test_hfield_mesh_contacts_match_mujoco_c`'s figures), the Isaac Lab 3.0 like-for-like rough runs
(`docs/research/isaaclab3_like_for_like_2026-09-25.md` §6.8) and any film or evaluation of Isaac's G1 on the rough task.
Not affected: flat-terrain results, Newton-backend results, the box-cell contacts, the terrain geometry and height scan.
**estimated** size of the effect (before the re-runs of section 13): unknown until re-run; the foot's contacts on
random-rough and slope cells change in roughly one placement in five (missing contacts made the feet sink into bumps,
which the policy may have learned to exploit or tolerate).

Re-runs needed (after the pin bump), in order of cost:
1. `tests/test_terrain.py` on the bumped pin (done here on the branch: 21 passed) and the rough rows of §1.4
   (physics only / step / step + inference at 4096 envs; section 8.4 gives the relative cost). Tests: done again
   in section 13; §1.4 rows: pending (the timing machine, the 16-inch MacBook Pro (M4 Max)).
2. Transfer of Isaac's rough checkpoints into MetalSim (`runs/g1_rough_transfer.jsonl` protocol; 7 / 24 falls before)
   and the wall-cell transfer (14 / 96 before): no training needed, shows the size of the change first. Isaac's
   checkpoints are not on this machine; section 13 measures the change on our old checkpoint and on NVIDIA's
   pretrained rough policy instead.
3. Rough training, three seeds x 1500 iterations on the 2.3.2 preset (+16.4 / +22.6 / +19.1 at 1500 before; Isaac
   +14.1), then our checkpoints transferred to Isaac. Training and the MetalSim-side transfers: done (section 13);
   in Isaac: pending (the NVIDIA L4 GPU).
4. The Isaac Lab 3.0 like-for-like rough run (`rough_il3`, +8.4 / level 5.96 before vs Isaac Lab 3.0 on
   [Newton](https://github.com/newton-physics/newton) +14.5). Done on
   [`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f) (section 13).
5. Films that put Isaac's G1 on rough terrain on MuJoCo Warp, and the g1_bumps cell's engine decision (MuJoCo Warp
   now falls like CPU MuJoCo there, so the cell could run on MetalSim's own physics).

## 10. Upstream

**estimated**, worth filing (none filed from here):
- MuJoCo Warp: the heightfield side-wall override in `_inflate` / `_epa_witness`
  ([#928](https://github.com/google-deepmind/mujoco_warp/pull/928)) returns penetrations for separated prisms; the prism
  support's vertex index (-2 / -3) defeats discrete EPA's termination test; the 4-contact selection's position-only 1 mm
  stop drops a sphere's second face in a crease. The commits on
  [my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) are a candidate patch (each behind
  its flag).
- MuJoCo C: `mjc_prism_support` does not set `vertindex`, so box / mesh against a heightfield stop EPA early
  (3.44 mm reported for a 9.25 mm penetration; main still has it). A one-line fix: set `obj->vertindex = ibest`.
- Both: the prism model gives contacts through internal prism walls when a geom is deeper than its horizontal
  overlap with an edge prism (section 8.5). Upstream [#928](https://github.com/google-deepmind/mujoco_warp/pull/928)
  tried to replace them by vertical contacts, unconditionally (defect 1). A correct version would apply only to
  penetrating prisms and only to walls shared with a neighbour (not the heightfield's outer border), measured against
  the exact surface; not attempted here, because it departs from MuJoCo C's semantics and the brief asked for parity.

## 11. Reproduce

All from the repository root with `.venv/bin/python`; MuJoCo Warp copy chosen by `PYTHONPATH` (a checkout of branch
[`metalsim-hfield-contacts`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim-hfield-contacts) of
[my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim), or `mjw_base` =
`git archive 8eb51fe`). Scripts in `runs/readme_media/records/hfield-contacts/`; they name the scratch directory they
ran from: copy them back to that path (or edit it) to rerun.

- Static: `cc2.py place|c|exact|exactprism <shape> <out>/cc/<shape>`,
  `cc_all.sh <tag> <copy>`, `cc2.py compare <shape> <dir>`, `cc_show.py <dir>/compare.json`; spheres also
  `runs/film/g1_bumps/measure/contact_check.py plane|compare <dir>`.
- Probes: `probe/gjk_point_prism.py 15` (defect 1), `probe/sat_box.py`, `probe/prism_mesh_c.py` (defect 2),
  `probe/mesh_repro.py`, `probe/margin2.py`, `probe/margin_main_test.py` (section 7).
- Falls: `falls/measure_seeds.py cpu|gpu <tag> <seeds>`, `falls/repeat.sh <copy> <tag> <n> <seeds>`, `falls/summary.py`.
- GPU batches: `gpu/batch_a.sh` (tests, falls), `gpu/batch_b.sh` (Metal tests, timing, `gpu/random_actions.py`),
  `gpu/batch_c.sh` (random actions with the MuJoCo C reference, cost split); logs next to them.

## 12. Addendum (2026-10-31, evening): EPA horizon overflow and the cost of a geom gap

The rough-terrain re-runs on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147)
(their smoke runs, `runs/rough_rerun_2026-10-31/smoke/` and `smoke_v1/`) found two problems this note's fix had exposed:
Isaac Lab 3.0's
like-for-like config (`rough_il3`) flagged MuJoCo Warp's `EPA_HORIZON` capacity overflow in a few of the 4096 worlds
within the first iterations (3 in each 8-iteration smoke run; the pair's contact is dropped for that substep), and its
PPO iteration took about 2.1x as long as on the plane-path pin
[`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624), against the
-15 % of section 8.4. Both come from the G1's mesh feet now being on per-prism GJK/EPA (section 6) together with
`rough_il3`'s geom gap (Isaac Lab 3.0's 1 cm per geom; MuJoCo Warp sums a pair's gaps, so `margin + gap` = 0.02 m).
Branch [`metalsim-hfield-epa`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim-hfield-epa) of
[my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) (on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147); pushed, and my
fork's `metalsim` fast-forwarded to it):
[`439b808`](https://github.com/pulipakaa24/mujoco_warp/commit/439b8083ae0ff39f58937a083cfc2d3c56ce022f),
[`f85879c`](https://github.com/pulipakaa24/mujoco_warp/commit/f85879cd44bd6439d4fa7bace80dd67702184885),
[`1eb40d5`](https://github.com/pulipakaa24/mujoco_warp/commit/1eb40d55956d073e4fa544ca0f0eb0d456681791), then, after an
independent review,
[`d7a90fb`](https://github.com/pulipakaa24/mujoco_warp/commit/d7a90fb635b90f3207e79ef222c3be12997d29bc),
[`3d8f48c`](https://github.com/pulipakaa24/mujoco_warp/commit/3d8f48ca77fa41c980aa9595dd90071c6eb43e6c),
[`ba9cfb7`](https://github.com/pulipakaa24/mujoco_warp/commit/ba9cfb7d2079278dfc56f8dd613fe07773360cac),
[`489f421`](https://github.com/pulipakaa24/mujoco_warp/commit/489f421dd0ee8a243831dd0452d353bb1bbbfd5b),
[`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f), which MetalSim
pins (`0024`-`0031` of [`patches/mujoco_warp/`](../../patches/mujoco_warp/)); diff
[`93e5ce1...787a5b3`](https://github.com/pulipakaa24/mujoco_warp/compare/93e5ce18e76854c430ce669314af00ddceb92147...787a5b3a37acc2551ffc8a2c1af1e9781d76723f).
Sources: [MJW] as above;
[C-main] `engine_collision_gjk.c` read 2026-10-31 (`horizon` / `horizonRec`, `epa`, `mjc_ccd`); MuJoCo issue
[google-deepmind/mujoco#3646](https://github.com/google-deepmind/mujoco/issues/3646) (C's horizon buffer, fixed in
[`6d662aed`](https://github.com/google-deepmind/mujoco/commit/6d662aedda693a1225a03a26c6b3ddd5e3cc4fbe) by a bound check
and `24 + max(0, ccd_iterations - 35)` edges); MuJoCo Warp PR
[google-deepmind/mujoco_warp#1845](https://github.com/google-deepmind/mujoco_warp/pull/1845) (open: the same
disconnected-horizon failure on aligned mesh plates). Records: the probes and Metal validation jobs (V1-V4) in
`runs/readme_media/records/hfield-epa/` (`probe/`, `val/`, `val2/`, `val4/`), the CPU checks after the review in
`runs/readme_media/records/epa-fixes/` (`logs/`, `probe/`); the captured states and contact sets (18-147 MB per
directory) are not published.

**Mechanism (measured).** A substep hook on Metal stored the qpos of the first `EPA_HORIZON` event per world during
12 `rough_il3` iterations on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) (3 events,
iterations 7-10). Replayed on Warp's CPU device, all three overflow again, and so do 20 copies of each with the root
position jittered by 1e-6 m. In each, a foot mesh against one prism ran EPA for 33-39 iterations on the *inflated*
shapes, with upper and lower depth bounds converging at 19.2-19.9 mm, i.e. the uninflated shapes 0.1-0.8 mm apart: EPA
crawling over the rounded part of the inflated Minkowski difference, where the support point lies (to rounding) on the
planes of many small faces. Two things in my fork made that fatal:
1. **Horizon scan** [MJW `_epa`, since upstream's first EPA port
   [`66c2ebce`](https://github.com/google-deepmind/mujoco_warp/commit/66c2ebce1cafb174dafb9d3e30d035f27a91537c)]: the
   visible faces are found by testing every face on its own. MuJoCo C recurses through face adjacency from the closest
   face, so its deleted faces are always connected [C-main `horizonRec`]. With the support point on several faces'
   planes, rounding can delete faces that are not connected to the closest one; the horizon is then not one loop (two
   regions touching at a vertex, or a ring), the faces attached to it do not close the polytope, and the edge list
   (`_add_edge` toggles an edge in and out, so the buffer must hold the intermediate set, not just the final horizon)
   overflows its fixed 24 entries. Static G1 foot and torso placements on the training terrain: 31-64 non-loop horizons
   per 40,000 placements (none overflowing at 24; the training states did).
2. **Margin handling** [MJW `gjk_phase`]: with a margin or gap, every prism within `margin + gap` of the foot went
   through EPA on the inflated (smooth, hence non-discrete) shapes, iterating to the 1e-6 m tolerance. MuJoCo C's
   `mjc_ccd` first runs GJK on the uninflated shapes and, when they are apart, returns that distance less the margins
   without EPA ("special handling for sphere, capsule, and margin (shrink to underlying geom)"); upstream MuJoCo Warp
   does the same since 2026-10-23 for every pair except heightfields (`main` at
   [`23efc9a`](https://github.com/google-deepmind/mujoco_warp/commit/23efc9aa41a29db65ab3c7f6bfbb485feedae905),
   `has_non_hfield_margin`), because upstream's heightfield path uses the older margin convention that my fork replaced
   (section 7).

A third, rarer drop: when EPA's face buffer (6 + 5 x ccd_iterations) fills, `_attach_face` fails and the contact is
dropped without any flag; MuJoCo C stops with its closest face instead [C-main `epa`, "out of memory for faces"]
(1 in 40,000 static foot placements).

**Fix.**
- [`439b808`](https://github.com/pulipakaa24/mujoco_warp/commit/439b8083ae0ff39f58937a083cfc2d3c56ce022f) EPA: the
  original scan runs first; when its horizon is not one simple loop it is undone (each face it deleted carries a
  per-iteration mark in `face_norm2`, which nothing reads for deleted faces) and the deleted region grows from the
  closest face by passes over the face list, adding a visible face only when it shares an edge with the region, MuJoCo
  C's region. A scan whose horizon is one loop is kept as it is, so results are bitwise unchanged there. The horizon
  buffer holds every edge of EPA's largest polytope, `3 (5 + ccd_iterations) - 6` (114 at 35). If it still fills
  (`EPA_HORIZON`) or the grown region is still not a disk, EPA returns no contact, as MuJoCo C's `epa` when its horizon
  runs out of edges
  ([`d7a90fb`](https://github.com/pulipakaa24/mujoco_warp/commit/d7a90fb635b90f3207e79ef222c3be12997d29bc); C has no
  test for a region that is not a disk); neither occurs at the derived size (0 in 351,278 worlds of the static sets and
  training-state replays, instrumented copy). A full face buffer stops EPA with its closest face, as in C.
- [`f85879c`](https://github.com/pulipakaa24/mujoco_warp/commit/f85879cd44bd6439d4fa7bace80dd67702184885)
  `collision_gjk.HFIELD_MARGIN_SHRINK` (default True): heightfield prism pairs with a margin or gap take MuJoCo C's
  route: GJK on the uninflated shapes; EPA (on the inflated shapes, as C) only when they overlap. No effect at margin =
  gap = 0. Setting it False in my fork's `collision_gjk.py` (read at import) restores the previous form; the shrink is
  off whenever `HFIELD_PRISM_POLYTOPE` is False
  ([`3d8f48c`](https://github.com/pulipakaa24/mujoco_warp/commit/3d8f48ca77fa41c980aa9595dd90071c6eb43e6c)).

**Results (measured).**

| check | [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) | fix |
|---|---|---|
| captured `rough_il3` overflow states, CPU replay (3 states x 21) | 63 / 63 overflow | 0 / 63 (also 0 with the EPA fix alone) |
| `rough_il3` smoke, 4096 envs x 8 it, `EPA_HORIZON` worlds (two runs) | 3, 3 | 0, 0 |
| 12-iteration `rough_il3` run on Metal with the capture hook, first-flag events | 3 | 0 |
| rough-transfer protocol (full: levels 3 and 6, protocols A and B), overflow flags at exit | `LS_ITERATIONS` only | `LS_ITERATIONS` only (falls 1 of 144 vs 2 of 144) |
| heightfield contact checks of section 8.1 (3000 spheres; 1000 each capsule, box, cylinder, ellipsoid, G1 foot; gap 0) | | bitwise identical |
| static G1 foot / torso placements on the training terrain, gap 0.01 per geom (5 x 40,000) and the g1_bumps tile (2 x 40,000), EPA fix alone | | bitwise identical in all but 2-7 worlds per set, every one a non-loop horizon; one dropped contact recovered, one face-buffer drop recovered |
| G1 foot on box stairs (boxes_local-like, 20,000 placements incl. flat, grid-aligned soles) | | bitwise identical |
| fork suites `collision_driver_test`, `collision_gjk_test`, `collision_primitive_core_test` (with the new tests) | | CPU and Metal: 127 passed, 2 skipped ([`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f); 122 at [`1eb40d5`](https://github.com/pulipakaa24/mujoco_warp/commit/1eb40d55956d073e4fa544ca0f0eb0d456681791)) |
| MetalSim `tests/test_terrain.py` (Metal) | 21 passed | 21 passed, the heightfield-mesh test's figures unchanged (min normal z 0.912, max penetration 5.2 mm) |

The fix column was measured at
[`1eb40d5`](https://github.com/pulipakaa24/mujoco_warp/commit/1eb40d55956d073e4fa544ca0f0eb0d456681791).
[`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f) changes only the
fallbacks above, the shrink's condition and the tests. At the defaults it gives bitwise-identical contacts and overflow
flags on the CPU sets above (the section 8.1 checks, the static placements, box stairs, and 49,278 replayed `rough_il3`
training states: 359,278 worlds); the suites and `tests/test_terrain.py` were re-run on it; the Metal smoke, capture and
transfer rows were not (`runs/readme_media/records/epa-fixes/logs/`: `bitwise_old_vs_new.log`,
`suites_cpu_787a5b3.log`, `gpu_check.log`).

With the shrink, prisms apart within the gap get GJK's exact distance (EPA on the inflated shapes had it to about
2e-7 m, with normals of neighbouring prisms 0.3 deg apart, which the duplicate test of section 6 then kept as extra
contacts): on 40,000 foot placements with the gap the contact count falls from 144,701 to 144,199 and the deepest
contact per placement matches the exact float64 signed distance as before (3000 placements checked, p99 error
0.005 mm). New tests:
`test_hfield_mesh_gap_distance` (EPA on the inflated shapes returned 4.000172 mm for a 4 mm gap), the horizon-size
test, `test_horizon_is_loop` (synthetic horizons: one loop, figure-8s, two loops, open paths),
`test_epa_nonloop_horizon` (three box-stack poses and one ellipsoid pose whose scan is not one loop, deepest contact
against MuJoCo C to 2e-8 m / 1e-6 m; the old scan gave the same contact on them, so it pins the restart, not a change
of result), and `test_box_box_horizon` updated (its configuration now needs 5 horizon edges instead of 8; with 3 or 4
it overflows and gives no contact, as MuJoCo C). In the three captured states the dropped contacts were inactive
gap-zone contacts (MuJoCo Warp: detected below `margin + gap`, active below `margin`), and the four selected contacts
per pair are the same with the fix; the `rough_il3` run's dynamics were affected only where a dropped candidate would
have been among the four (not the case in the three captured states).

**Cost (measured on the 14-inch MacBook Pro on AC power, powermode 2: not a timing machine, ratios only).** The
rough-rerun smoke commands (`runs/rough_rerun_2026-10-31/train_wrap.py`, 4096 envs, 8 iterations, `--initial_reset 1`),
time per PPO iteration over iterations 4-8, forks interleaved in one queue job (the fix timed at
[`1eb40d5`](https://github.com/pulipakaa24/mujoco_warp/commit/1eb40d55956d073e4fa544ca0f0eb0d456681791);
[`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f) was not
re-timed):

| config | [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) (plane path) | [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) | fix | [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) / [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) | fix / [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) |
|---|---|---|---|---|---|
| `rough_il3` (2 rounds) | 1.870 / 1.874 s | 3.838 / 3.848 s | 3.145 / 3.151 s | 2.05 | 1.68 |
| 2.3.2 seed-0 config, heightfield everywhere, no gap (2 / 6 / 6 runs, medians) | 1.910 s | 2.483 s | 2.509 s | 1.30 | 1.31 |
| 2.3.2 seed-1 config, boxes_local + heightfield, no gap (1 / 4 / 4 runs, medians) | 2.129 s | 2.541 s | 2.569 s | 1.19 | 1.21 |

Section 8.4's -15 % is the env step under random actions (on battery); a PPO iteration (rollout with the learning
policy, plus the update) shows larger ratios: 1.30x / 1.19x the plane path in the 2.3.2 configs on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147), 1.31x / 1.21x
with this fix. The difference between the two protocols was not isolated. Where `rough_il3`'s cost goes (training states
after iteration 12, 512 worlds, every prism the kernel processes classified by its exact float64 signed distance,
`runs/readme_media/records/hfield-epa/probe/profile_counts.py`): with the 2 cm pair gap, 266 / 259 of the 512 left /
right feet have prisms within reach (125 / 115 without the gap, same states); each such pair processes 13.4 prisms (GJK
each), of which 1.25 penetrate and 5.3-5.5 are apart within the gap. EPA calls per foot pair: 6.5-6.8 on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) (all on the
inflated shapes), 1.24-1.26 with the fix (-81 %). What remains of the 1.68x is the gap's semantics (twice the foot
pairs, each with ~13 GJK calls, producing the inactive gap-zone contacts MuJoCo Warp's `gap` asks for) on per-prism
GJK/EPA instead of the plane test. Not adopted (measured on CPU): EPA on the uninflated shapes for penetrating prisms
too, plus the margins (another -36 % of CPU collision time on 40,000 foot placements, but one placement 0.95 mm off
the exact depth, and it departs from MuJoCo C's algorithm).

**Consequences.** The 2.3.2-config runs (no gap) never ran EPA on inflated shapes and showed no overflow (smoke and
the 1500-iteration seed-0 run); the fix changes their contacts only where a scan's horizon was not one loop or EPA's
face buffer filled, and adds about 1 % to their cost (medians of 4-6 interleaved runs; single runs spread by up to 7 %;
CPU collision time +1 % on 40,000 foot placements). `rough_il3` on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) dropped inactive
contacts and paid 2.05x; with the fix its gap-zone contacts are exact and it pays 1.68x the plane path, so its re-run
belongs on the fix's commit. Still open: my fork's subgrid takes the mesh's tight horizontal bounds while MuJoCo C main
grows them by the margin (prisms beside a geom within the gap are not candidates; gap-zone contacts only, at MetalSim's
margin 0); MuJoCo Warp's four-contact selection counts inactive gap-zone contacts against the cap; upstream issues not
filed (MuJoCo Warp: horizon scan, face-buffer drop, heightfield margin shrink;
[#1845](https://github.com/google-deepmind/mujoco_warp/pull/1845) covers the first).

## 13. Re-runs on the repaired kernel (2026-10-31, night)

The re-runs of section 9 that need neither the L4 nor the timing machine, run on the 14-inch MacBook Pro (Mac16,6)
through the GPU queue, every job on AC power (battery 75-85 %, charging, powermode 2). Job directory
`runs/rough_rerun_2026-10-31/`: the pipeline `run_all.sh` and `run_il3fix.sh` (their header names the original commands;
the flags map those settings onto today's CLI), `job.sh`, `jobs/`, `status.jsonl`, `logs/<job>.out` (each with a
provenance header), training logs `logs/<job>.log`, `transfer/`, and the exports for Isaac
([`isaac_export/metalsim_ckpts_rough/`](/runs/rough_rerun_2026-10-31/isaac_export/metalsim_ckpts_rough): the nine
`rerun_s{0,1,2}_it{500,1000,1500}.pt`, 1.3 MB each, with their `SHA256`); its training checkpoints (`ckpt/`) are not
published. Analysis: `runs/readme_media/records/rough-parity/` (`analyze_training.py`,
`analyze_transfer.py` and their outputs `training.md`, `transfer.md`, `curriculum.txt`, `anomalies.txt`). "Old" is the
record PARITY cites (plane-path kernel, 2026-09-25/26 code), "new" these runs. **measured** unless marked.

| job | MuJoCo Warp loaded (job header) | outcome |
|---|---|---|
| tests | [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) | `tests/test_terrain.py` 21 passed (heightfield-mesh test: min normal z 0.912, max penetration 5.2 mm vs MuJoCo C 4.6 mm, pelvis within 1 mm, as section 8.2); `tests/test_solver_presets.py tests/test_g1_task_terms.py` 38 passed, 4 skipped |
| transfers, the same checkpoints on both kernels | [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) (git archive, plane path) and [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) | all exit 0; overflow flags at exit: the line-search cap `LS_ITERATIONS` only, no capacity flag |
| 2.3.2 rough preset, seeds 0 / 1 / 2, 1500 iterations each; seed 0's checkpoints through both transfer protocols | [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) | all exit 0; 0 blow-ups, 0 dropped rows, no capacity flag, `EPA_HORIZON` 0; solver caps, as in the old seed-0 run: `LS_ITERATIONS` at every log point, `ITERATIONS` at 198-237 of 1500 (old seed 0: 82) |
| exports of seeds 0-2 for Isaac | [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) (seeds 0, 1), [`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f) (seed 2) | the shared checkout was synced to the pin at 21:48:28, 15 s after seed 2's training ended and 8 min before its export; the export reads only the checkpoint and MuJoCo C's model (`metalsim/parity/export_policy.py:2-3`), so MuJoCo Warp's version does not enter |
| `rough_il3` seed 0, 1500 iterations | [`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f) (git archive) | exit 0; 67 blow-ups (below), no capacity flag, `EPA_HORIZON` 0 |

The 2.3.2 seeds and the transfers ran on
[`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147), not on the pin:
without a margin or gap EPA never runs on inflated shapes, and
[`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f) changes such
contacts only at non-loop horizons and full face buffers (section 12), so these results stand for the pin (**estimated**
from section 12's static and replay sets; not re-run on
[`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f)).

Deviations from the originals and limits of the comparison (the flags are in `runs/rough_rerun_2026-10-31/run_all.sh`
and `run_il3fix.sh`):
- Every training run starts from the task reset (`--initial_reset 1`). The originals started each env's first episode
  at `qpos0` (PARITY §1.5, start-state defect), which on the repaired kernel overflows `njmax` on the box cells (`NEFC`
  at iteration 1 of a 4096-env sizing run). The first ~12 iterations differ by design, and `rough_il3`'s early
  transient (823 blow-ups at iterations 10-14, caused by that start) cannot recur.
- `EPA_HORIZON` would have been reported instead of aborting the run (`runs/rough_rerun_2026-10-31/train_wrap.py`);
  it never occurred.
- Seed 1's feet-slide velocity point ("origin") is inferred from commit times, not proven.
- Old logs against new runs is not a clean A/B: the code also changed at float level since 2026-09-25 (factorization,
  fused launches, earlier pin moves), Metal training is not run-to-run deterministic (isaaclab3 note §6.4b), and no
  plane-path training control was run. The transfer jobs carry their own A/B: the same checkpoints on both kernels on
  today's code.
- The transfers read the base velocity at the pelvis COM (today's default); the 2026-09-25 record predates that option
  (frame origin, from the commit times). On the plane path today's protocol reproduces the record for our old
  checkpoint to within 0.07 m (level 3) and 0.20 m (level 6) of travel per env, which bounds the effect.
- `rough_il3` repeats the corrected run of isaaclab3 note §6.8 (`il3fix_rough_s0`, drive effort clamp on the actuator)
  with the initial reset added. It is not the final run with both trainer fixes (PARITY §1.8: +13.7 / level 5.99),
  whose log is not on this machine.

**2.3.2 rough preset, three seeds** (old `runs/g1_rough_ppowarp_fixed.log`, `_seed1.log`, `_seed2.log`, seed 2 cut
to its second start; Isaac `runs/parity/isaac/rough/isaac_train_g1_rough_terms.txt`; MetalSim values at the
iteration, as PARITY's table; mean ± sample sd over the three seeds):

| iteration | return, old: s0 / s1 / s2; mean ± sd | return, new | terrain level, old | terrain level, new | Isaac: return / level |
|---|---|---|---|---|---|
| 300 | +4.0 / −0.4 / −0.1; +1.2 ± 2.5 | +0.8 / −1.8 / −2.6; −1.2 ± 1.8 | 1.61 / 1.29 / 0.67; 1.19 ± 0.48 | 1.18 / 1.05 / 0.58; 0.94 ± 0.31 | +2.1 / 2.19 |
| 500 | +8.4 / +4.2 / +5.0; +5.9 ± 2.3 | +5.4 / +3.7 / +2.9; +4.0 ± 1.3 | 4.41 / 3.41 / 3.05; 3.62 ± 0.70 | 3.53 / 3.25 / 2.46; 3.08 ± 0.56 | +5.8 / 4.83 |
| 1000 | +11.3 / +11.0 / +14.3; +12.2 ± 1.8 | +9.7 / +11.4 / +10.3; +10.5 ± 0.9 | 6.01 / 5.60 / 5.88; 5.83 ± 0.21 | 5.51 / 5.56 / 5.38; **5.48 ± 0.09** | +5.4 / 5.96 |
| 1500 | +16.4 / +19.1 / +22.6; +19.4 ± 3.1 | +16.0 / +20.6 / +17.4; **+18.0 ± 2.4** | 6.10 / 6.03 / 6.08; 6.07 ± 0.04 | 6.13 / 6.11 / 6.11; **6.11 ± 0.01** | +14.1 / 5.90 |

- **At 1500 the result holds within the seed spread**: return −1.4 (inside either side's sd), level +0.05 at the
  curriculum's ceiling of about 6.1; every seed stays above Isaac's +14.1 / 5.90. Episode lengths are 991-1000 at
  1000 and 1500 on both kernels.
- **The curriculum climbs later on the repaired kernel.** Every seed's level is lower at 300, 500 and 1000; at 1000 the
  mean is 0.35 levels lower, more than either side's sd (Welch p = 0.09 with three seeds: suggestive, not shown).
  First iteration after 100 at level 5.5: 983 / 956 / 1115 against 598 / 940 / 754; at 6.0: 1425 / 1237 / 1398
  against 996 / 1342 / 1082; at 5.9, Isaac's level from iteration 1000 on: 1338 / 1197 / 1337 against 910 / 1257 /
  1004. Seed 0 (heightfield on every cell, so the fix acts on all of them) and seed 2 (box cells
  exact, the fix only on random rough and slopes) lag most, seed 1 least: three seeds cannot separate a terrain
  effect from the seed spread.
- Physics health, from the training monitor (`metalsim/learn/monitor.py:113-120`: deepest contact over all worlds at
  each log point when deeper than 2 cm, as the kernel under test reports it): seed 0, iterations 1000-1500, median 5.4
  cm and at most 8.4 cm against 8.0 and 21.8 cm on the plane path; seeds 1 and 2, whose stair and box cells collide
  with exact boxes, unchanged (median 6.7 / 6.4 cm against 6.8 / 6.4 cm). No blow-ups in either version; touch forces
  above 50× the robot's weight only on the box-cell seeds (282 / 306 log points against 602 / 440).

**`rough_il3`, seed 0** (old `runs/il3/il3fix_rough_s0.log` and `runs/il3/curves_il3fix_rough.md`; new through
`scripts/diagnostics/g1_learning_curves.py il3 rough <log> --at 1000,1499`; Isaac Lab 3.0 Newton / MuJoCo Warp
`runs/parity3/isaac/train/train_rough_newton_mjwarp_terms.txt`; MetalSim ±5-iteration means in brackets):

| at iteration 1499 unless stated | Isaac Newton | old: plane path, `qpos0` start | new: repaired kernel, task reset |
|---|---|---|---|
| return | +14.5 | +8.4 (+8.2) | **+12.0 (+11.5)** |
| return at 1000 | +8.9 | +4.4 (+4.0) | +7.1 (+6.5) |
| terrain level | 5.80 | 5.96 | 5.93 |
| linear / yaw tracking | 0.811 / 0.808 | 0.745 / 0.681 | 0.780 / 0.730 |
| joint-acceleration term (× Isaac's) | −0.0264 | −0.0600 (2.3×) | −0.0531 (2.0×) |
| falls, share of episode ends | 2.9 % | 6.9 % | 3.3 % |
| blow-ups (non-finite episodes) | none logged | 1,010: 823 at iterations 10-14, 187 from iteration 751 | 67, from iteration 1012 |

The return gap to Isaac narrows from 6.1 to 2.5 points, falls come down to Isaac's rate and half the yaw-tracking gap
closes. One seed on each side and a different start: the 3.6-point difference is about one seed sd of the 2.3.2 runs
(2.4-3.1 at 1500), so it is not separable from seed spread or from the start-state change (**estimated**). The late
blow-ups fall from 187 to 67; isaaclab3 note §6.10 traced late ones to the drive's effort clamp on the actuator, which
both runs keep; these 67 are not classified.

**Transfer** (`scripts/diagnostics/g1_rough_transfer.py`: levels 3 and 6, command 0.5 m/s, 400 control steps, mean
action; fall = pelvis less than 0.3 m above the local ground at the end; mean travel x per cell over envs and levels,
in m: pyramid stairs / inverted stairs / boxes / random rough). Protocol A, heightfield collision and grid scan on every
cell, one env per cell and level (old record `runs/g1_rough_transfer.jsonl`):

| checkpoint | kernel | falls | mean travel (m) |
|---|---|---|---|
| our seed 0, iteration 1500 (2026-09-25) | record | 0 / 8 | 4.09 / 2.72 / 3.64 / 3.74 |
| | plane path, today's code | 0 / 8 | 4.12 / 2.72 / 3.73 / 3.72 |
| | repaired | 0 / 8 | 2.84 / 2.70 / 3.10 / 2.94 |
| NVIDIA's pretrained rough policy (iteration 2999), in place of Isaac's missing checkpoints | plane path | 0 / 8 | 4.40 / 2.59 / 2.96 / 4.17 |
| | repaired | 0 / 8 | 3.30 / 3.74 / 3.47 / 3.84 |
| our seed 0 (2026-09-25), iterations 500 / 1000 / 1500 | record | 0 / 24 | 3.86 / 1.76 / 2.80 / 2.94; 2.42 / 2.40 / 2.72 / 2.91; 4.09 / 2.72 / 3.64 / 3.74 |
| seed 0 re-trained, iterations 500 / 1000 / 1500 | repaired | 0 / 24 | 0.30 / 1.63 / −0.93 / −0.32; 3.78 / 1.22 / 3.23 / 3.17; 3.86 / 2.61 / 3.27 / 3.12 |

Protocol B, `boxes_local` collision and exact scan (stair and box cells collide with exact boxes; the heightfield remains
under random rough), 8 starts per cell and level (old record `runs/terrain_walls/transfer_reps.jsonl`, rows with
`boxes_local` and `exact`):

| checkpoint | kernel | wall-cell falls (inverted stairs + boxes) | mean travel (m) |
|---|---|---|---|
| our seed 0, iteration 1500 (2026-09-25) | record | 1 / 32 | 2.28 / 3.79 / 3.71 / 3.72 |
| | plane path, today's code | 0 / 32 | 2.34 / 4.11 / 3.76 / 3.71 |
| | repaired | 1 / 32 | 2.18 / 4.01 / 3.91 / 2.97 |
| NVIDIA's pretrained rough policy | plane path | 0 / 32 | 3.20 / 3.83 / 2.99 / 3.94 |
| | repaired | 1 / 32 | 3.18 / 3.72 / 2.94 / 3.17 |
| our seed 0 (2026-09-25), iterations 500 / 1000 / 1500 | record | 4 / 96 | 2.23 / 2.75 / 3.25 / 3.22 |
| seed 0 re-trained, iterations 500 / 1000 / 1500 | repaired | 0 / 96 | 2.60 / 2.57 / 2.45 / 1.97 |

No fall on pyramid stairs or random rough in any B row. Reading (small samples: report counts, not rates):
- Falls barely move: none in protocol A on either kernel; in B one wall-cell fall per checkpoint on the repaired
  kernel and none on the plane path. The re-trained seed 0 falls 0 / 24 in A (record 0 / 24) and 0 / 96 on the wall
  cells in B (record 4 / 96).
- Run-to-run spread: the same protocol run twice on
  [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) (the
  pipeline's smoke and full runs, level 3) differs by up to 0.51 m of travel per env, on the plane path by 0.03 m; the
  full protocol on
  [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) and on the EPA
  fix (gap-0 contacts the same but for non-loop horizons) differs by up to 0.33 m per cell mean
  (job V2, `runs/readme_media/records/hfield-epa/probe/val_check.sh`, output `val/transfer/` there, against
  `runs/rough_rerun_2026-10-31/transfer/`). Changes beyond that: on heightfield cells the repaired contacts
  shorten the travel of policies trained elsewhere, our old checkpoint by 1.28 m on pyramid stairs and 0.77 m on random
  rough in A and 0.74 m on random rough in B, NVIDIA's by 1.10 m on pyramid stairs in A and 0.77 m on random rough in
  B (its other A cells: random rough −0.33 m, boxes +0.51 m, inverted stairs +1.15 m). On B's box-built cells, which
  the fix does not touch, the change is at most 0.16 m per cell, as expected.
- The re-trained iteration 1500 on the repaired kernel (A) travels 3.86 m on pyramid stairs against the old
  checkpoint's 2.84 m there (4.12 m on its own kernel); on the other three cells the two are within 0.2 m. The
  re-trained iteration 500 hardly moves forward (−0.9 to +1.6 m per cell; the 2026-09-25 one walked
  1.8-3.9 m), in line with its lower training level at 500 (3.53 against 4.41).

**Not re-run here** (PARITY keeps these marked, with what each waits for):
- Isaac's own checkpoints in MetalSim (A: 7 / 24 falls; B: 47 / 96 on the heightfield, 14 / 96 on the exact surface):
  `model_500/1000/1499.pt` of Isaac's 2026-09-25 rough run are not on this machine (only `params/`); they may still be
  on the L4.
- Our checkpoints played in Isaac (12 / 24): needs the L4; the re-trained seeds' exports (iterations 500 / 1000 /
  1500) were written by `runs/rough_rerun_2026-10-31/jobs/export.sh` (`metalsim/parity/export_policy.py`) from the
  training checkpoints; neither is published (nine exports of 1.3 MB).
- The rough throughput rows of PARITY §1.4 and every training throughput: the timing machine (16-inch) only; the
  14-inch figures in these logs are not performance figures. Section 12's per-iteration ratios remain the indicative
  cost.
- The per-term means at iteration 1000 of the 2.3.2 run (PARITY §1.5) and the 150-iteration smoke run.
- The 2.3.2 rough run under the elliptic default (+13.7 / 6.10) and the final `rough_il3` run with both trainer fixes
  (+13.7 / 5.99): not part of this pipeline.
