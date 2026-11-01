# Heightfield contacts in MetalSim's MuJoCo Warp fork: three defects, the fix, and its effect

2026-10-31. Scope: make heightfield contacts of primitive geoms (sphere, capsule, box, cylinder, ellipsoid) in
MetalSim's MuJoCo Warp fork match CPU MuJoCo, check the mesh path (Isaac's G1 feet) against the same standard, and
settle the margin convention. Found through the `g1_bumps` film (Unitree's G1, 5 mm sphere feet, on Isaac Lab's
random rough tile): CPU MuJoCo 11 of 16 robots fall, the fork far fewer (`runs/readme_media/handoff/g1-bumps.md`).
**measured** = run here (MacBook Pro 14-inch M4 Max, Mac16,6; GPU work through `scripts/gpu_run.sh`); **reported** =
from the cited source; **estimated** = our judgement.

Code: fork branch `metalsim-hfield-contacts` (worktree `runs/readme_media/scratch/hfield-contacts/mjw`) on `8eb51fe`
(the pinned `metalsim` head); full diff `runs/readme_media/handoff/hfield-contacts.patch`. **Landed 2026-10-31:** the
fork's `metalsim` was fast-forwarded to it (`93e5ce1`), then to the follow-up of section 12 (`787a5b3`), which
MetalSim pins; the six commits are `patches/mujoco_warp/0018-0023` (on `8eb51fe`), the follow-up's eight `0024-0031`
(on `93e5ce1`), the whole branch `patches/mujoco_warp/metalsim-787a5b3.diff` (on v3.14.0 `88af9cc`). The scratch paths cited below (`runs/readme_media/scratch/hfield-contacts/`) are local to the
machine that ran them and are not in the repository. Sources:
[MJW] MuJoCo Warp, the fork at `8eb51fe` and upstream `gdm/main` `d41a47c0` (2026-10-31);
[C] MuJoCo C 3.14.0 sources (`runs/readme_media/scratch/hfield-contacts/src_c/`), the version installed here;
[C-main] https://github.com/google-deepmind/mujoco (main, read 2026-10-31), commit
[8626d3c](https://github.com/google-deepmind/mujoco/commit/8626d3c35eceb8b36dbc5182852b25ffde7a6465) "Fix margin
and gap double-counting in heightfield collisions" (2026-10-06).

## 1. Summary

MuJoCo C and MuJoCo Warp collide a convex geom with a heightfield the same way: every triangle of the grid cells
under the geom is extruded down to the heightfield base into a triangular prism, and each prism is a convex body for
GJK/EPA; one contact per penetrating prism [C `mjc_ConvexHField`, `engine_collision_convex.c:1170-1328`]. MuJoCo Warp
then keeps at most 4 contacts per pair. Two defects shared with upstream MuJoCo Warp (the second also in MuJoCo C)
and one in MetalSim's own mesh path made the fork's contacts wrong:

1. **Side-wall override** (upstream MuJoCo Warp #928, `f2f79579`, still on `gdm/main`): when GJK's closest feature
   of a prism is a side wall, `_inflate` (and `_epa_witness` for deep contacts) replaced the result with a vertical
   contact against the prism's top triangle and `dist = -|x1 - x2|`, a penetration whatever the real separation.
   A 5 mm sphere 8.6 mm clear of a prism came back 25.6 mm deep with normal (0, 0, 1). The kernel also kept
   separated prisms as candidates, and its 4-contact selection dropped every candidate within 1 mm of the deepest by
   position alone, so a sphere in a crease lost its second face.
2. **Prism support index** (upstream MuJoCo Warp, and MuJoCo C 3.14 and main): EPA on discrete pairs (box, mesh)
   stops when a new support point repeats a vertex-index pair already in the polytope. The prism's support function
   gave all six vertices the same index (MuJoCo Warp: -2 / -3 by the sign of dir.z; MuJoCo C: never set, -1), so EPA
   stopped early: a box 9.25 mm into a prism came back 2.57 mm (Warp) or 3.44 mm (C) deep along wrong axes.
3. **Mesh plane path** (MetalSim's `f2716b4`): a mesh of at most 256 vertices was tested vertex by vertex against
   each triangle's plane inside its column; any contact whose deepest point is a terrain vertex or edge inside the
   mesh was missed. Isaac's G1 foot on the random rough tile: no contact in 178 of 1000 placements that touch.

After the fix every heightfield contact on the tile matches the exact geometry: per-triangle closest features for
spheres and capsules, the separating-axis penetration per prism for boxes and the G1 foot mesh; normals within
0.02 deg (p99), depth errors 0.0 mm (p95); contact sets 99.9-100 % equal to the exact ones within the 4-contact cap
(pinned fork: 53-89 %; MuJoCo C 3.14: 69-99.5 %, section 8.1). Unitree's G1 on Isaac's bumps now falls as often as on
CPU MuJoCo within the run-to-run spread (75.8 % of 240 robots vs 68.3-68.8 %; pinned fork 18.8 %). Cost: about 15 % of
Isaac's G1 rough-task env step under random actions (its feet are meshes and now take GJK/EPA), 1.2-1.3x per PPO
iteration in the 2.3.2 rough configs and, with Isaac Lab 3.0's geom gap, 1.7x after the follow-up of section 12
(2.05x and dropped contacts before it); nothing for primitives. The heightfield margin now follows MuJoCo C main
(section 7). Every rough-terrain result of Isaac's G1 on MuJoCo Warp (PARITY §1.5) needs a re-run (section 9).

## 2. Baseline, re-measured on the pinned fork (`8eb51fe`)

The `metalsim` head already sends primitives to upstream's GJK/EPA prism path (`b630530`,
`HFIELD_PLANE_CONTACTS_PRIMITIVES = False`) and meshes of at most 256 vertices to the plane path. The g1-bumps
column "MetalSim fork: 81.9 %, 1/16 falls" had measured a stale checkout (`07a51a6`); the pinned fork is the column
"upstream MJW". **measured**:

- Contact check (`runs/film/g1_bumps/measure/contact_check.py`, 3000 spheres of 5 mm pressed 0.5-3 mm into the tile,
  Warp's CPU device): contact count equal to the exact set 88.8 % (126 fewer, 211 more), 6.11 % of contacts more than
  10 deg from the nearest true normal (p95 21.8 deg), deepest-contact depth error p95 8.0 mm. CPU MuJoCo: 99.6 %
  (0 fewer, 11 more: in all 11 a duplicate of one feature reached from two prisms, same point within 0.1 mm and
  normal within 0.05 deg), 0 normals off. As the brief.
- G1 falls on Isaac's tile (seed 0, 16 robots, 8 s): GPU runs of the pinned fork gave 2, 5, 4, 3, 1 and 3 of 16 in
  six runs: MuJoCo Warp on Metal is not run-to-run repeatable here (contact order from atomics, chaotic gait), so
  the g1-bumps hand-off's 6/16 (the stale checkout with every geom on GJK/EPA) is inside that spread, and a single
  run is not a measurement. Section 8.3 uses five tile seeds and repeats.

## 3. Defect 1: the heightfield side-wall override, separated prisms, and the 1 mm selection rule

Probe `runs/readme_media/scratch/hfield-contacts/probe/gjk_point_prism.py 15` (placement 15: exact depth 1.85 mm,
fork 25.6 mm with normal (0, 0, 1)) runs MuJoCo Warp's `gjk` and `ccd` on the sphere against each of the 18 prisms
around it. **measured**: GJK's point-to-prism distance is right (prism 9: 3.147 mm, float64 3.147 mm; prism 8: 13.567
mm vs 13.574 mm), but `ccd` returns -25.6 mm for prism 8, which is 8.6 mm clear of the sphere. The source:
`collision_gjk._inflate` [MJW `collision_gjk.py:2455-2485` at `8eb51fe`] has a heightfield branch (upstream #928,
"Improve sphere/capsule <-> hfield collisions", `f2f79579`, 2025-12-17): when the GJK simplex holds both top and bottom
prism vertices (the closest feature is a side wall) it takes the geom's support point, projects it vertically onto
the top triangle (or a vertex height) and returns `dist = -|x1 - x2|`, always a penetration. `_epa_witness`
[`collision_gjk.py:996-1027`] does the same for deep contacts (and calls `support` with a position as the
direction). MuJoCo C has no heightfield case in its GJK/EPA [C `engine_collision_gjk.c`]: the prism is a plain
polytope there. Both overrides are still on upstream `gdm/main` `d41a47c0`.

Two kernel details turned the phantom contacts into wrong contact sets [MJW `collision_convex.py` `ccd_hfield_kernel`]:
the pinned fork keeps a prism whenever `ccd` reports `ncontact` (it reports 1 for separated geoms too; upstream main
has since added `dist >= margin + gap` as a skip), and contacts 1-3 are chosen by spread (furthest from contact 0,
furthest from that line, ...) but the selection stops as soon as the furthest remaining candidate is within 1 mm of
contact 0 (`MIN_DIST_TO_NEXT_CONTACT`, upstream #847) by position only. In a crease a 5 mm sphere touches two faces at
points 0.4 mm apart (p5 of the tile's two-contact sets; `sel_proto.py`), so the second face was lost.

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

MetalSim's `f2716b4` (PARITY §1.6) replaced GJK/EPA for meshes of at most 256 vertices with a per-triangle plane test:
per prism, the deepest mesh vertex below the triangle's plane whose footprint lies in the triangle. It was written
because GJK/EPA returned inverted normals and 2 m penetrations for the G1 feet. **estimated**: the `_epa_witness`
override of defect 1 (a support point taken in the direction of a position vector) is the likely source, but not
confirmed: §1.6's minimal repro (a box mesh pressed 3 cm into a flat heightfield, base 1.9 m;
`probe/mesh_repro.py`, four yaw angles) on the pinned fork's GJK/EPA path no longer shows a 2 m penetration
(depths 1.2-3.0 cm; 4 of 16 normals sideways; the fixed fork 8 of 16, all at the box's edge prisms, like MuJoCo C's own
1.2-1.3 cm side contacts there). The plane test cannot see a terrain vertex or edge that pokes into a mesh face.
**measured** on Isaac's G1 foot collider (8-vertex
box mesh, 0.203 x 0.065 x 0.019 m) at 1000 random poses pressed 0.5-3 mm into the tile: no contact at all in 178,
9.6 % of normals more than 10 deg off, depth error p95 1.1 mm (max 10 mm), against the per-prism SAT.

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

The 4-contact cap per heightfield pair (upstream #847) is unchanged. It is the only remaining difference from the
exact sets on the tile: 13 of 3000 spheres touch 5 or 6 distinct triangle features.

## 7. Margin convention

MuJoCo C 3.14 (the installed version) passes margin + gap to `mjc_ConvexHField`, raises each prism's top by it and
inflates the geom by half of it, then calls `mjc_penetration` with margin 0 [C `engine_collision_convex.c:1150-1166,
1272, 1298`]; its prism height test compares the raised top with the geom's un-inflated bottom. **measured**
(`probe/margin2.py`, sphere / box / capsule / box mesh over a flat heightfield, margin 0.01 on both geoms, so 0.02 for
the pair): C reports `dist = g - 1.5 (margin + gap)` for a gap `g` and keeps the contact while `g < margin + gap`,
every shape; the pinned fork matched it for primitives and was 0.5 margin off on the mesh plane path. Every other
pair in C and in MuJoCo Warp reports `dist = g`. **reported**: MuJoCo C main fixed this on 2026-10-06 (8626d3c, "Fix
margin and gap double-counting in heightfield collisions"): the prism is inflated like the geom (support offset by
half the margin), `mjc_penetration` gets the margin, the prism height test expands the geom's bounds by it, and its
test `MjcConvexTest.HFieldMarginAndGap` (terrain margin 0.02, gap 0.01, sphere 15 mm above) expects `dist` 0.015
and the contact point halfway. The fork now follows C main (`HFIELD_TRUE_DISTANCE`; the plane path too), so a
heightfield behaves like a plane under a margin. It matches C 3.14 only at margin = gap = 0, which is every
heightfield pair in MetalSim (Isaac G1 and Unitree G1 models: all margins and gaps 0). False restores C 3.14's form.

## 8. Acceptance

### 8.1 Contacts against the exact geometry (static, Warp's CPU device)

Harness `runs/readme_media/scratch/hfield-contacts/cc2.py` (extends `contact_check.py`; outputs `cc/<shape>/`):
Isaac's random rough tile (seed 0, bumps 2-10 cm every 0.1 m); 3000 spheres (the g1_bumps placements) and 1000
random poses for each other shape, pressed 0.5-3 mm into the surface by bisection on MuJoCo C's deepest contact.
Reference: **exact** for spheres and capsules = every triangle whose closest point to the centre / segment is within
the radius (one contact each, duplicates by point merged); for the box and the G1 foot mesh = the separating-axis
penetration of the geom into each prism (MuJoCo's model, computed exactly; duplicates = equal depth and normal);
for cylinders and ellipsoids (no exact reference written) MuJoCo C, which is exact for these non-discrete shapes.
Cells: contact count equal to the reference's distinct contacts (capped at 4, MuJoCo Warp's per-pair limit) / worlds
with no contact where the reference has one / contacts more than 10 deg from every reference normal / normal error
p99 (deg) / deepest-depth error p95 (max), mm. **measured**:

| shape (size) | reference | pinned fork `8eb51fe` | fixed fork | MuJoCo C 3.14 |
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
harness's own artefact (`np.sign(-0.0)` zeroes the fork's horizontal normal (-1, -0, -0), which equals C's for that
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
| MuJoCo Warp, pinned fork `8eb51fe` (Metal) | 3-6 | 2, 5, 4, 3, 1, 3 / 3, 3, 6 / 1, 6, 5 / 3, 1, 4 / 1, 1, 2 | 10, 12, 20 | 54 / 288 = 18.8 % |
| MuJoCo Warp, fixed fork (Metal) | 3 | 10, 11, 13 / 16, 14, 14 / 13, 12, 14 / 12, 13, 15 / 7, 8, 10 | 58, 58, 66 | 182 / 240 = 75.8 % |

The fixed fork falls as often as CPU MuJoCo within the run-to-run spread (per 80-robot run 60.7 +- 4.6 vs 54.8 +- 5.4,
mean +- sd; seed 0, the film's tile: 10-13 vs 10-12); the pinned fork kept 81 % of the robots walking. CPU MuJoCo is
deterministic, so its single run per seed has no error bar; a 1e-6 rad perturbation of the start already moves a seed
by up to 8 falls (seed 4: 5 to 13). On flat ground the two engines already agree (the g1-unitree hand-off: within 0.1
mm of Unitree's CPU run over 8 s; no heightfield, so unaffected here). The remaining gap (about 6 falls per 80 robots,
about one standard deviation) is not resolved further here; candidates are float32 vs float64 and MuJoCo C writing
duplicate contacts at shared edges (each adds stiffness) where the fork writes one.

### 8.4 Relative cost

Isaac's G1 rough task (`python -m metalsim.learn.g1_velocity 4096 rough`: full env step with random actions,
synchronized env-steps/s over 110 steps; `gpu/batch_b.log`, `gpu/batch_c.log`), runs interleaved in one queue job
each. **measured on the 14-inch MacBook Pro (Mac16,6) on battery (41 -> 32 %, powermode 2): not a timing machine; the
ratios are indicative only, the absolute numbers are not performance figures.**

| fork | env-steps/s (runs) | relative |
|---|---|---|
| pinned `8eb51fe` (meshes on the plane path) | 62,257 / 62,346 | 1.00 |
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
is excluded), so on the heightfield cells (random rough, slopes; the stair and box cells use exact boxes since
`0cfb188`, with the heightfield lowered under them) its contacts went through the plane path, which the static check
shows wrong on random rough (no contact in 17.8 % of touching foot poses, 9.5 % of normals > 10 deg off). The fork now
sends them through GJK/EPA. **Affected**: every rough-terrain number in PARITY §1.4-1.6 measured on MuJoCo Warp
(training curves and final returns of the three seeds, terrain levels, the transfer tables in both directions, the
wall-cell transfer counts, the rough throughput rows, `test_hfield_mesh_contacts_match_mujoco_c`'s figures), the
Isaac Lab 3.0 like-for-like rough runs (`docs/research/isaaclab3_like_for_like_2026-09-25.md` §6.8) and any film
or evaluation of Isaac's G1 on the rough task. Not affected: flat-terrain results, Newton-backend results, the
box-cell contacts, the terrain geometry and height scan. **estimated** size of the effect: unknown until re-run;
the foot's contacts on random-rough and slope cells change in roughly one placement in five (missing contacts made
the feet sink into bumps, which the policy may have learned to exploit or tolerate).

Re-runs needed (after the pin bump), in order of cost:
1. `tests/test_terrain.py` on the bumped pin (done here on the branch: 21 passed) and the rough rows of §1.4
   (physics only / step / step + inference at 4096 envs; section 8.4 gives the relative cost).
2. Transfer of Isaac's rough checkpoints into MetalSim (`runs/g1_rough_transfer.jsonl` protocol; 7 / 24 falls before)
   and the wall-cell transfer (14 / 96 before): no training needed, shows the size of the change first.
3. Rough training, three seeds x 1500 iterations on the 2.3.2 preset (+16.4 / +22.6 / +19.1 at 1500 before; Isaac
   +14.1), then our checkpoints transferred to Isaac.
4. The Isaac Lab 3.0 like-for-like rough run (`rough_il3`, +8.4 / level 5.96 before vs Isaac Newton +14.5).
5. Films that put Isaac's G1 on rough terrain on MuJoCo Warp, and the g1_bumps cell's engine decision (MuJoCo Warp
   now falls like CPU MuJoCo there, so the cell could run on MetalSim's own physics).

## 10. Upstream

**estimated**, worth filing (none filed from here):
- MuJoCo Warp: the heightfield side-wall override in `_inflate` / `_epa_witness` (#928) returns penetrations for
  separated prisms; the prism support's vertex index (-2 / -3) defeats discrete EPA's termination test; the
  4-contact selection's position-only 1 mm stop drops a sphere's second face in a crease. The fork's commits are a
  candidate patch (each behind its flag).
- MuJoCo C: `mjc_prism_support` does not set `vertindex`, so box / mesh against a heightfield stop EPA early
  (3.44 mm reported for a 9.25 mm penetration; main still has it). A one-line fix: set `obj->vertindex = ibest`.
- Both: the prism model gives contacts through internal prism walls when a geom is deeper than its horizontal
  overlap with an edge prism (section 8.5). Upstream #928 tried to replace them by vertical contacts, unconditionally
  (defect 1). A correct version would apply only to penetrating prisms and only to walls shared with a neighbour
  (not the heightfield's outer border), measured against the exact surface; not attempted here, because it departs
  from MuJoCo C's semantics and the brief asked for parity.

## 11. Reproduce

All from the repository root with `.venv/bin/python`; MuJoCo Warp copy chosen by `PYTHONPATH` (the fork worktree
`runs/readme_media/scratch/hfield-contacts/mjw`, or `mjw_base` = `git archive 8eb51fe`).

- Static: `cc2.py place|c|exact|exactprism <shape> runs/readme_media/scratch/hfield-contacts/cc/<shape>`,
  `cc_all.sh <tag> <copy>`, `cc2.py compare <shape> <dir>`, `cc_show.py <dir>/compare.json`; spheres also
  `runs/film/g1_bumps/measure/contact_check.py plane|compare <dir>`.
- Probes: `probe/gjk_point_prism.py 15` (defect 1), `probe/sat_box.py`, `probe/prism_mesh_c.py` (defect 2),
  `probe/mesh_repro.py`, `probe/margin2.py`, `probe/margin_main_test.py` (section 7).
- Falls: `falls/measure_seeds.py cpu|gpu <tag> <seeds>`, `falls/repeat.sh <copy> <tag> <n> <seeds>`, `falls/summary.py`.
- GPU batches: `gpu/batch_a.sh` (tests, falls), `gpu/batch_b.sh` (Metal tests, timing, `gpu/random_actions.py`),
  `gpu/batch_c.sh` (random actions with the MuJoCo C reference, cost split); logs next to them.
