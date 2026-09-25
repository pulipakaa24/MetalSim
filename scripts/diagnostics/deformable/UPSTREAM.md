# MuJoCo Warp flex fixes proposed upstream (not filed yet)

**Pre-filing step (required):** the MuJoCo Warp project's CLA check rejects commits co-authored by AI assistants
(`AGENTS.md` in the repository). Before filing, rewrite the `metalsim-flex` commits of the fork (and the flex commits
in the merge / `patches/mujoco_warp/0008-0016`) without the `Co-Authored-By: Claude ...` trailer, on a fresh branch cut
for the upstream PRs. MetalSim's own commits keep the trailer. Not rewritten yet (the fork branches are pinned by
the patch series and the merged heads).

Target: google-deepmind/mujoco_warp (base v3.14.0, `88af9cc`). Fixes live on github.com/pulipakaa24/mujoco_warp branch
`metalsim-flex` (commits listed per item); each is behind a module flag in `mujoco_warp/_src/collision_flex.py` so the
upstream behaviour is one assignment away. Reference: MuJoCo C 3.14.0 (pip) and MuJoCo C `main` (2026-09-23, built
locally) where they differ.

Reproducers: `upstream_repro.py` (plain MuJoCo Warp API, no MetalSim; run it with the unmodified and the fixed
MuJoCo Warp) and `row_order_repro.py`. Each case steps MuJoCo C (float64) into contact, hands that exact state to
MuJoCo Warp, and compares the contact set (geom, flex, element, vertex) after `forward` and the velocities after one
`step`. A 1e-4 m random perturbation of the initial state avoids exact ties. Numbers below are **measured** on the
Warp CPU device (`runs/deformable/upstream_repro_cpu.jsonl`); float32 against float64, so ~1e-5 m/s is agreement.

| # | issue | reproducer case | before: ncon W/C, sets equal, one-step max Δqvel | after | fix commit(s) |
|---|---|---|---|---|---|
| 1 | element-geom contacts of any flex other than the first carry the global element id | `second_flex` (control `first_flex`) | 2/2, **no**, 3.2e-2 m/s (control: 1/1, yes, 2.5e-6) | 2/2, yes, 4.6e-6 | a277152 |
| 2 | flex edge-equality rows ordered by an atomic counter | `row_order_repro.py` on `metal:0` | efc_pos in order off by **8.4e-2** (sorted 1.2e-8); `FlexConstraintTest.test_constraint_parity2/3` fail on Metal | in-order = sorted; 237/237 flex tests on Metal | 9c6e09f |
| 3 | geom-flex contacts not capped at mjMAXCONPAIR per pair | `plane_cap` | **100**/50, no, 3.7e-2 | 50/50, yes, 2.7e-6 | b2e9ea5, bbe19bb |
| 4 | box-triangle keeps at most 2 contacts per element | `box_cloth_3x3` (no cap involved) | **16**/24, no, 1.6e-2 | 24/24, yes, 3.2e-7 | b2e9ea5 |
| 3+4 | both, cloth draped on a box | `box_cloth` | **265**/50, no, 6.3e-2 | 50/50, yes, 1.3e-5 | – |
| 5 | 1D flexes collide as vertex spheres, not capsule segments | `cable_box` | **6**/12, no, 3.6e-2 | 12/12, yes, 2.7e-6 | a277152 |
| 6 | every tetrahedron of a volume collides, not only active (surface) layers | `soft_cube` | **54**/34, no, 9.6e-8 | 34/34, yes, 9.6e-8 | 6234396 |

## 1. Wrong element for flexes after the first (bug)

`_flex_narrowphase_elem_detect` writes the global element index (`elemid`, a `wp.tid()` over `nflexelem`) into
`contact.elem`, while the contact Jacobian (`constraint.py`, `_efc_contact_*_flex`: `flex_elemdataadr[f] + e * (dim+1)`)
and MuJoCo C use the flex-local index. For flex 0 the two coincide, so every upstream test passes; for any later flex
the contact force is applied to another element's vertices (or out of range). In a larger scene a 10×10 cloth that
is the second flex fell through a box (0.76 m vertex error after 0.8 s vs MuJoCo C; with the fix one-step
3.8e-5 m/s). Fix: pass `local_elemid` (three call sites). No flag (plain bug fix).

## 2. Nondeterministic flex equality row order

`_equality_flex` reserves each edge's row with `wp.atomic_add(nefc_out, worldid, 1)`, so the row order is the
device's thread schedule. On CUDA it happens to be edge order; on Apple Metal rows 4–7 and 8–11 of the 3×3 cloth come
out swapped and `FlexConstraintTest.test_constraint_parity2/3` fail (the test compares rows in order; values match
MuJoCo C to 1.2e-8 after sorting). Beyond the test, the row order feeds the Newton solver's block layout, so results
could differ run to run on any GPU. Fix: a `_equality_flex_reserve` kernel reserves one block per world (sum of
active flex edge counts) and every edge writes at its (equality, edge) offset, MuJoCo C's order. Reproducer needs a
device whose thread order differs from edge order (Metal here); on CPU both orders coincide.

## 3. Per-pair contact cap (mjMAXCONPAIR) for geom-flex contacts

MuJoCo C filters every geom-flex (per body-flex pair), flex-flex and self-collision pair to `mjMAXCONPAIR` = 50
contacts (`filterFlexContacts` in 3.14, `filterPreContacts` on main); MuJoCo Warp applied its FPS only to flex-flex
and self groups and its test `test_plane_cloth_no_fps_limit` asserted the uncapped 100 contacts. Changes:

* groups per (world, body, flex) (MuJoCo C's midphase pair), geom-flex groups included (`ENABLE_GEOM_FLEX_FPS`);
* coincident candidates (the same vertex contact from adjacent triangles) stay eligible, as in C (the FPS stopped at
  distance 0: a draped cloth kept 9 instead of 50);
* ties go to the first candidate in C's order: plane-vertex contacts by vertex, element contacts by element (for
  3.14: by the flex BVH's depth-first leaf order, new `Model.flex_elemorder`), float32 near-ties within 1e-8 m depth /
  1e-6 relative distance (`FPS_DEPTH_TIE`, `FPS_DIST_REL_TIE`);
* `FLEX_FPS_MODE`: `"c314"` reproduces 3.14 exactly (it swaps each chosen contact to the front of the array but not its
  `selected`/`min_dist` bookkeeping, so its choice is not a plain FPS: a plain FPS matched 0 of 83 capped plane sets,
  the emulation 83/83), `"main"` is C main's plain FPS, `"parallel"` the upstream block-parallel FPS; default by the
  installed mujoco version. The serial modes use one thread per group in one launch (the parallel one needs 3
  launches per selected contact without conditional graphs) and allocate no FPS scratch (~0.7 GB at 4096 worlds).

Scene-level effect (cloth + cable on a plane, 400 steps): capped sets equal in 83/83 checked steps; settled mean
height bias 2.3 mm → 0.03 mm.

## 4. Box-triangle contacts

`collision_primitive_core.box_triangle` returns the first two of MuJoCo C's `mjraw_BoxTriangle` contacts and skips
its margin tests; C returns every triangle vertex inside the (radius+margin-inflated) box and every box corner
touching the triangle, up to mjMAXCONPAIR per element (at most 11). Fix: generate all, in C's order, with C's tests
(`BOX_TRIANGLE_ALL_CONTACTS`). A draped 10×10 cloth: raw candidates 286 → 384 = MuJoCo C's 384 (uncapped C build).

## 5. Cable (dim 1) segments as capsules

MuJoCo C collides 1D flex elements as capsules (`mj_collideGeomElem` → `makeCapsule` → `mjraw_SphereCapsule`,
`mjraw_CapsuleCapsule`, `mjraw_CapsuleBox`, else `mjc_ConvexElem`); MuJoCo Warp collided the vertices as spheres
(`test_sphere_rope_collision` asserted Warp's 1 contact against C's 2). Fix (`CABLE_CAPSULE_ELEMENTS`): dim-1 elements
in the element narrowphase with the raw primitives (normal geom → flex, as C), segments as zero-radius capsules
inflated by the flex radius for CCD geoms; plus inverse-distance vertex weights for dim-1 element contacts in the
contact Jacobian (`_flex_contact_bodies_weights` had no dim-1 branch: the rows were empty).

## 6. Only active layers of volumes collide with geoms

MuJoCo C builds each flex's BVH from its active elements (`flex_elemlayer < flex_activelayers`, the surface layer of
a tetrahedral volume by default), so interior tetrahedra never generate geom contacts; MuJoCo Warp tested every
element. Fix (`FLEX_ACTIVE_LAYERS_ONLY`) with the existing `_elem_active`. In the reproducer the interior contacts
duplicate surface ones, so the one-step dynamics are unchanged but the contacts (and their constraint rows) grow by 59 %;
over a 0.4 s drop of the same cube the capped contact sets went from 0/73 to 45/73 equal and the one-step error
median from 3.1e-4 to 6.5e-5 m/s (`docs/research/deformables_2026-09-25.md` §2).

## Also on the branch, lower priority for upstream

* Mesh-flex contact normal: upstream snaps to the nearest mesh face within 5 mm; C uses the penetration direction.
  Now the EPA/GJK direction when the witnesses are > 0.1 mm apart and it is > 0.01 rad from the face normal
  (`MESH_FLEX_FACE_NORMAL`, `MESH_FLEX_NORMAL_MIN_SEP`); contact point at the midpoint of the radius-inflated
  penetration (`FLEX_RADIUS_CONTACT_POS`). Mesh scenes still differ from C (EPA witness points, float32), so this one
  should go in as an issue with data rather than a finished fix.
* Device-side sorts/scans for the flex filter and SAP (`FLEX_DEVICE_SORT`, 361f11f): only relevant where Warp's
  utilities are host implementations (Apple Metal fork); upstream CUDA unaffected (default off on CUDA).
* Warp (innate-inc fork, Metal backend): `segmented_sort_pairs` validated segments with a synchronous `.numpy()` and
  so could not be captured in a graph (pulipakaa24/warp `metalsim-flex` 9dcb140). Belongs to the Warp Metal fork, not
  mujoco_warp.

## Rigid-body neutrality

On the branch's merge into `metalsim`, Go1, Panda and Isaac's G1 run bitwise identical to the unmerged `metalsim`
over 200 steps × 8 worlds on the CPU device; the Metal run and the MetalSim rigid tests are in the final report
(`runs/deformable/merge_rigid_ab.log`). Upstream test suite on the branch: 1450 passed, 2 failed (both pre-existing
and unrelated: `io_test::test_put_data_nefc_zero_dense`, `collision_driver_test::test_hfield_maxconpair` — the latter
from MetalSim's heightfield patch, fails identically without the flex changes).
