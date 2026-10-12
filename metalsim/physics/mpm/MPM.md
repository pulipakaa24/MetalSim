# MPM on Metal (2026-10-09)

Isaac Lab 3.0-EA's only MPM is Newton 1.5.2's `SolverImplicitMPM` (an implicit Drucker-Prager rheology + frictional
contact solve per step, after Daviet & Bertails-Descoubes 2016 and NVIDIA's "Mixed MPM" 2026); PhysX dropped its MPM in
5.4 and Isaac Lab does not use PhysX's PBD particles. metalsim therefore runs **that solver** on Metal, made to mean
on the M4 Max what it means on Isaac's L4, and accelerates it only with changes that reproduce it bit for bit.
Research behind every choice: `docs/research/mpm_2026-10-09.md` (notes in `docs/research/mpm_2026-10-09_notes/`).

## 1. Layers

| layer | where | what |
|---|---|---|
| oracle | Newton 1.5.2 tag + `newton_compat.py` (stock: `scripts/setup_newton152.sh` with `METALSIM_NEWTON152_REF=v1.5.2`, see section 10) | Newton's solver unchanged, with the semantic fixes below |
| Newton fork | [pulipakaa24/newton](https://github.com/pulipakaa24/newton) branch `metalsim-1.5.2` off v1.5.2, pinned by `scripts/setup_newton152.sh` into `.venv-newton152` (`METALSIM_CHANGES.md` there lists every commit) | Metal launch configuration and device-side paths that CUDA already had, scale bounds, repeatable sums, rigid collision at scale; upstreamable |
| Warp fork | [pulipakaa24/warp](https://github.com/pulipakaa24/warp) branch `metalsim`, pinned by `scripts/setup_warp.sh` (one checkout shared by every environment) | Metal GPU primitives, deterministic mode, launch bounds, argument space, replay batching, fast_math forms |
| metalsim accelerations | `newton_compat.py` (`chunk_graph`), `simd_gs.py`, `fused_gs.py` | opt-in, each tested bitwise against the oracle |
| scenes / recorder / comparison | `scenes.py`, `metalsim/parity/isaac_side/record_mpm.py`, `compare.py` | the same two files run on the L4 and on the Mac |

### Semantic fixes (newton_compat.install)

1. **float32 accumulation.** `warp.fem.integrate` accumulates in float64 by default; Metal has no float64. Default
   becomes float32. On CUDA the same switch records the precision-control reference (R0 `_f32`).
2. **CUDA's stopping rule everywhere.** With conditional CUDA graphs (Isaac on the L4) the rheology loop checks
   convergence every 5 iterations (counting from 1, stop when converged or past `max_iterations`); without them
   (Metal, CPU) it runs `max_iterations // solve_granularity` host batches of 25 (Gauss-Seidel) or 50 (Jacobi). Isaac
   Lab's Franka-Pour (Jacobi, 24 iterations) therefore runs 25 iterations on CUDA and **zero** off CUDA (an upstream
   Newton defect). `loop="cuda"` runs the graph body from the host with Newton's own `update_condition` kernel, so every
   device stops at the same iteration; it also logs iterations and residuals per solve (`SOLVE_LOG`), because a speed
   number is only comparable at the residual it reached.
3. **Metal form of Newton's `fast_math` kernels** (`metal_fast_math`, section 8, 1b): `"off"` (default: correctly
   rounded division and sqrt), `"cuda"` (CUDA's `--use_fast_math` results bit for bit, Warp fork
   `metalsim` from `91b8a859`; Warp's own default is `"off"` too), `"relaxed"` (Metal's fast divide / sqrt).

### First Newton fork commits (topic branch `metalsim-mpm-1.5.2`, now the start of `metalsim-1.5.2`)

| commit | change | effect |
|---|---|---|
| `6f6075d1` | colored Gauss-Seidel kernels launch 2 blocks per GPU core on Metal (were 1 block of 64 threads per color off CUDA) | granular strain solve 390 -> 54 ms; bitwise |
| `00c24ef0` | dense-grid bounds on the device on Metal (host path ignored `particle_flags`, so the grid differed from CUDA's when particles are inactive) | parity fix; bitwise with all particles active |
| `a3fa1140` | batched (separate-worlds) residual norms and system scaling use CUDA's 256-lane tiles on Metal | faster; CUDA's reduction structure; bitwise on a 2-world column |

## 2. Verified so far (Metal, measured)

* **Determinism (T0):** two Metal runs are bitwise identical (column a = 2, 180 frames; `tests/test_mpm.py`).
* **Every acceleration is bitwise equal to the oracle:** cores, chunk graph, SIMD sweep, fused solve, device bounds,
  batched tiles (column, 180 frames; granular example 15 frames for cores + SIMD + chunk graph; `tests/test_mpm.py`
  8/8 with the branch, 7/8 + 1 skip without).
* **Metal vs Warp's CPU device (same float32 algorithm, a = 2, 3 s; measured on the earlier 13-wide lattice):** runout (L_f - L_i)/L_i 2.215 vs 2.205 (0.5 %),
  final height 0.5 %, deposit profile RMS 3 mm, centre of mass 0.12 mm, mean iterations 87 vs 93. Particle paths
  differ by ~1 mm RMS (chaotic granular flow), which is why the suite compares observables after the first frames.
* **Literature sanity:** a = 2 runout 2.21 lies between Lajeunesse 2005 (a = 2.0) and Lube 2005 (1.2a = 2.4).
* **Coverage:** column, plate (RFT), and Newton's granular, multi_material, viscous, twoway_coupling (454k particles)
  and anymal (486k particles, MuJoCo Warp robot) examples run on Metal; snow_ball and beam_twist need the Warp fix
  `abd59275` (branch `metalsim-fem-threadgroup`, `upstream/warp-innate-tgfix`: a `wp.static(x) > 1` condition kept an
  unused 75 KB `tile_sum` in `warp.fem`'s bilinear dispatch kernel).
* **Newton behaviour to account for:** with `separate_worlds`, two identical worlds give different trajectories
  (Gauss-Seidel color order differs per world), so per-environment comparisons must pair the same environment index.
  Isaac Lab's grid emitter works in float32: `ceil(3 * 0.2 / 0.05) = 12` particles across a 0.2 m column (13 in
  float64); `scenes.py` reproduces its positions bit for bit (section 4).

## 3. Performance (granular example, 226,981 particles, sparse grid, ~234 Gauss-Seidel iterations per step)

| configuration | frame | strain solve | how measured |
|---|---|---|---|
| oracle (Newton 1.5.2 as shipped, on Metal) | 435 ms | 390 ms | section timers, GPU idle |
| + all-core color launches (branch) | 87 ms | 54 ms | section timers |
| + chunk graph | **82.9 ms** | 44.8 ms | section timers, utilisation 0 % at start |

Interleaved A/B on one trajectory (all variants bitwise equal; robust to other processes' GPU load): all-core launches
2.9x over the oracle; chunk graph 1.4-1.8x over eager all-core launches; SIMD sweep 1.0x eager / +10 % with the graph;
fused single-threadgroup solve 0.9x (one GPU core cannot carry ~1,100 strain nodes per color).

Where the remaining time goes (graph replay, GPU side): a color pass ~20 us (8 per iteration), contact 25 us, residual
readback 0.23 ms every 5 iterations; grid allocation 13 ms (NanoVDB built on the host for the sparse grid) and
scratchpad 9 ms per step. Dispatch floor measured at 3-4.5 us per dependent launch (graph), 7.8-8.9 us eager.

Not yet measured: the L4 (Isaac) numbers for the same scenes, which every speed claim must be set against.

## 4. Isaac references (L4, recorded 2026-10-10; `runs/parity/isaaclab3_mpm/`)

First pass (85 runs, all exit 0), Isaac Lab 3.0-EA's Newton 1.5.2 / Warp 1.16.0 on the L4:
* **The L4 is deterministic:** host-run loop twice, bitwise equal (a = 2 column, 180 frames; granular, 60 frames).
* **metalsim's CUDA-stopping-rule loop is bitwise equal to Newton's conditional-graph loop on CUDA** (R0_a2_host vs
  R0_a2_stock): the semantic fix reproduces what Isaac runs, not just its iteration counts. This holds for Isaac Lab
  too: with a dense grid its `NewtonMPMManager` steps eagerly ("does not support CUDA graph capture"), but
  `_run_solver_loop` captures and launches its own conditional graph on CUDA either way.
* **float32-accumulation floor on CUDA** (the only intended numerical difference from Isaac): a = 2 column at 3 s,
  position RMS 66 um, max 0.41 mm, runout 2.20370 vs 2.20369, deposit profile RMS 3 mm; granular at 1 s, position RMS
  0.57 mm (max 5 cm on a few flung grains), centre of mass 9 um, KE 4.5e-6.
* **Lattice defect found by R2:** Isaac Lab's `_emit_grid` takes float32 corners, so a 0.2 m column at 3 ppc / 0.05 m
  voxels has 12 particles across, not the 13 `scenes.py` produced in float64. Fixed (`scenes.py` positions now equal
  Isaac Lab's bit for bit, a = 2 and 5); the column and plate runs (R0 column, R1, R7) are being re-recorded; the first
  pass is kept in `mpm_lattice13/`. Newton examples (R3, R4, R6) and R2 itself were unaffected.

Second pass (Isaac Lab lattice, 12 across; R0 column, R1, R2, R7 re-recorded, all exit 0):
* **R2: Isaac Lab's own `MPMObject` / `MPMSolverCfg` path is bitwise equal to `scenes.py` + Newton direct** (a = 2
  and 5, 180 frames, every observable), so the Newton-direct references are what Isaac Lab users get.
* R0 again: deterministic, host-run loop bitwise equal to the conditional graph; float32 floor on this lattice is
  tiny (position RMS 0.19 um at 3 s, runout 2.1792126 vs 2.1792140): how far float32 noise grows is scene-dependent
  (66 um on the 13-wide lattice), so each scene carries its own R0 floor.
* R1 on the L4 (normalized runout (L_f-L_i)/L_i; 10 jittered seeds, mean +- sd):

  | a | lattice | seeds | H_f/L_i | mean iterations / at cap |
  |---|---|---|---|---|
  | 0.5 | 0.423 | 0.440 +- 0.011 | 0.485 | 25 / 7 % |
  | 1 | 1.064 | 1.040 +- 0.015 | 0.954 | 38 / 6 % |
  | 2 | 2.179 | 2.187 +- 0.042 | 1.171 | 77 / 17 % |
  | 3 | 3.086 | 3.175 +- 0.029 | 1.268 | 93 / 24 % |
  | 5 | 4.590 | 4.793 +- 0.041 | 1.469 | 180 / 39 % |
  | 8 | 6.281 | 6.473 +- 0.034 | 1.855 | 175 / 39 % |

  Fitted exponents (seed means): 1.16 for a <= 2, 0.73 for a >= 3; experiments give 1 and 2/3 (Lajeunesse 2005,
  quasi-2D channel, closest to this slab; Lube 2005's prefactors are higher). Metal must match these per a, not the
  literature.
* L4 throughput, host-run loop (timings, not yet against matched Metal runs): granular 54.5-55.8 ms/frame median,
  247.75 iterations mean.


`metalsim/parity/isaac_side/isaaclab3/stage_mpm.sh` records, with Isaac Lab 3.0-EA's own Newton 1.5.2 on the L4:
R0 noise floors (host-run graph loop twice, Newton's own conditional-graph loop, float32 accumulation); R1 column
collapse a in {0.5, 1, 2, 3, 5, 8} on the Isaac Lab lattice plus 10 jittered seeds each; R2 the same column through
Isaac Lab's `MPMObject` / `MPMSolverCfg` (`record_mpm_il3.py`); R3 Newton's material examples; R4 two-way coupling;
R6 ANYmal on sand; R7 RFT plate intrusion and drag. Copy `record_mpm.py`, `record_mpm_il3.py`, `scenes.py`,
`newton_compat.py` to `~/parity3_scripts/mpm/` and the stage script to `~/parity3_scripts/`.

Pass criteria (proposals, to be re-based on R0): T0 bitwise determinism; T1 short-horizon per-particle RMS within
3x R0's float32-vs-float64 floor, centre of mass and KE within 1 %; T2 runout and final height within 10 % and the
combined 2-sigma over seeds at every a, fitted exponent within 0.05; plate force curves within 10 %; T3 robot tasks
within 10 %. Speed is compared only at matched settings with iterations and residuals reported.

## 5. Metal vs the L4 (2026-10-10)

`python -m metalsim.physics.mpm.parity_report runs/parity/isaaclab3_mpm/mpm runs/mpm/metal` computes every verdict.

### Why the first Metal recordings diverged from the L4, and the fixes

Before any fix Metal departed from the L4 by 6 um after one frame (a = 2 column), ~2,000x the L4's own
float32-vs-float64 floor. `probes/dump_first_solve.py` dumps every array of the first strain solve on both devices:
grid, strain matrices and coloring were identical; the inputs that differed were the collider friction / normals at
nodes exactly equidistant from two collider faces, and last-bit rounding everywhere. Two device-semantics gaps in our
Warp fork, both measured (`probes/fp_semantics.py`, `fp_contract_patterns.py`, `mesh_query_ties.py`):

1. **FMA contraction.** Warp's `fuse_fp` module option (documented default: on) compiles CUDA with `--fmad=true`, so
   `a*b+c` is one rounding; the fork's Metal path ignored it (strict IEEE, two roundings). Fix: branch
   `metalsim-fp-contract` (`upstream/warp-innate-fpcontract`): `#pragma clang fp contract(fast)` in the Metal
   prelude when `fuse_fp` is on. Metal then reproduces CUDA bit for bit on every tested expression shape (a*b+c,
   a*b+-c*d, three-term sums and squares, (a-b)*(c-d)+e, c-a*b). This changes rounding in every Metal Warp kernel,
   so it is a separate branch: merging it means re-running the rigid-body parity suite.
2. **BVH builder.** CUDA builds mesh BVHs with LBVH, the fork's Metal path falls back to host SAH; at exact ties the
   closest face differs (CUDA LBVH vs CUDA SAH alone: 4,962 / 7,126 / 3,793 tied points differ in three box cases).
   Being ported (Metal LBVH identical to CUDA's).

Remaining non-bitwise source on CUDA: kernels Newton marks `fast_math` use approximate `/` and `sqrt` on CUDA
(71 % / 83 % correctly rounded); Metal keeps them exact. Measured on the L4 itself (`record_mpm.py --precise-cuda`):
exact math moves the L4 by 2.6 nm (a = 2, frame 1) / 0.05 um (granular, frame 6), the size of the float32 floor,
so the T1 floor is the larger of the L4's float64-vs-float32 and fast-vs-exact differences.

### Results with the FMA fix (LBVH pending)

* **T1, a = 2 column:** Metal vs L4 (both float32) 28 nm at frame 10, 165 nm at 3 s; the L4's own floor 34 / 189 nm.
  PASS (ratio 0.83 / 0.87), identical iteration counts (76.58 mean).
* **T2, column collapse, all six aspect ratios: PASS.** Seed-mean runout within 0.1 %, final height within 0.4 %
  (the seed spread is 1-4 %); fitted exponents Metal 1.1555 / 0.7282 vs L4 1.1559 / 0.7275.
* **Granular example (mesh collider):** 0.24 mm at frame 6 vs a 0.05 um floor: the collider-normal ties (BVH);
  re-run after the LBVH port, together with R3/R4/R6/R7.

### Results with FMA + LBVH (Warp branch `metalsim-mpm-parity`), against CUDA's own floors (2026-10-10)

Research: `reports/CUDA and Metal float parity.md`. Bitwise CUDA emulation is not the goal: CUDA fuses multiply-adds
in NVVM and again in ptxas (undocumented, and with Warp 1.16's default PTX output the driver's ptxas), and at exact
closest-face ties inside Newton's box-collider meshes the chosen face (90-degree normal flip) is decided by last-bit
rounding on CUDA too. Parity is judged against the L4's own spread under perturbations that leave the algorithm
unchanged (`stage_mpm_floors.sh`; all runs float32 accumulation, CUDA stopping rule): exact `/` and `sqrt`
(`--precise-cuda`), no FMA fusion (`--cuda-nofma`), toolkit vs driver ptxas (`--cuda-cubin`), SAH vs LBVH collider
trees (`--cuda-sah`). Data: `runs/parity/isaaclab3_mpm_floors/`, Metal `runs/mpm/metal_floors/`.

Position RMS vs the L4 baseline (early / middle / end frame); "worst floor" = largest L4 perturbation:

| scene | Metal | worst L4 floor | |
|---|---|---|---|
| column a = 2 | 5.6 nm / 0.16 um / 0.17 um | 58 um / 1.7 mm / 1.7 mm (no-FMA) | inside, at the exact-math floor |
| granular (1 s) | 0.76 / 8.4 / 17 mm | 1.27 / 17.5 / 43 mm | inside |
| two-way coupling | 5.5 nm / 5.1 um / 126 um | 8.3 nm / 18 um / 192 um | inside |
| ANYmal on sand | 4.7 / 91 / 336 um | 4.1 / 96 / 355 um | inside (early x1.1) |
| multi_material | 9.2 um / 17 mm / 29 mm | 2.8 um / 14 mm / 24 mm | x3.3 early, x1.25 later |
| plate drag (force) | force-curve RMS 5.9 % of peak, final 240 N | 5.7 %, final 253-259 N (base 265 N) | x1.03; final force 10 % low vs 5 % floor |

CUDA is as tie-sensitive as Metal: no-FMA or SAH trees move the granular example 4 cm in 1 s (Metal: 1.7 cm).
Toolkit vs driver ptxas made no difference on this driver. Open: multi_material and the plate sit at or just above
the largest of 3-4 floor samples; the next step is a floor distribution (more perturbed L4 runs) and Metal's
percentile in it, plus seed ensembles for the plate force.

## 7. Performance plan (research: `reports/Metal GPU primitives for MPM speed.md`)

Timing-class runs, ms/frame, Metal vs the L4 (eager / whole step captured): column a=2 14.5 vs 20.4; column a=8
52.5 vs 28.6; plate drag 67.0 vs 35.7; granular 88.7 vs 35.8/36.0; multi_material 90.1 vs 35.5; ANYmal 107.7 vs
58.9/24.2; two-way coupling 254.4 vs 73.0/31.0. Two-way profile: MPM step 224 ms, of which ~130 ms are CPU radix sorts
in `warp.fem` `compress_node_indices` and ~40 ms GPU drains before host utility calls (the fork runs
`radix_sort_pairs` / `runlength_encode` / `array_scan` on the host on Metal). Order: GPU sort/RLE/scan (in progress,
branch `metalsim-metal-sort`), GPU-driven solver loop (indirect execution ranges), residual folded into the solve,
GPU NanoVDB grid build, whole-step capture, Gauss-Seidel layout work after microbenchmarks.

### Done (2026-10-10), each bitwise against the previous Metal path

| change | where | effect |
|---|---|---|
| GPU radix sort / RLE / integer scan (stable reduce-then-scan, pass skipping) | Warp `b23d6b81` (`metalsim-metal-sort`) | two-way 242 -> 76 ms; 2M-key sort 9 -> 0.6 ms |
| GPU BSR assembly (`bsr_set_from_triplets`, transpose) | Warp `8412a5c8` (`metalsim-metal-bsr`) | 12 drains/step gone; two-way 78 -> 70.5, granular 70.9 -> 65.4 ms |
| GPU NanoVDB builder (`allocate_by_voxels`, `rebuild`) | Warp `93ed7138` (`metalsim-metal-volume`) | granular "Allocate grid" 12.4 -> 2.2 ms |
| GPU-decided solver stop (indirect execution range zeroed by a gate kernel; `newton_compat gpu_exit`) | metalsim | granular 72.8 -> 63.3 ms (interleaved A/B); same stopping iterations |
| solver loop inside an outer capture (unrolled gated chunks) -> the examples' whole-step capture works on Metal | metalsim | two-way captured, bitwise equal to eager |

Merged Warp tree: `metalsim-mpm-fast` (`upstream/warp-innate-fast`, = FMA + LBVH + ray fix + sort + BSR + volume).
Timing-class runs, 2 s, ms/frame, Metal (eager / captured) vs L4 best: column a=2 11.7 vs 20.4 (Metal 1.7x faster);
ANYmal 44.5 / 32.2 vs 24.2; plate drag 49.6 vs 35.7; granular 53.8 vs 35.8; column a=8 43.2 vs 28.6; multi_material
63.2 vs 35.5; two-way 87.2 / 61 vs 31.0 (two-way frame cost grows over the run: ~53 ms in the first second, 62 in the
second). The strain solve is latency-bound (one strain node per thread, ~17.5 us per color pass at 1,000-1,300
nodes) and near its floor at equal results.

Open: Metal runs of the coupled scenes (ANYmal, two-way) are not deterministic run to run (rigid side first, frame
4 in ANYmal) while the L4's are; under investigation. Pure-MPM scenes are bitwise deterministic and the merged tree
reproduces the pre-optimisation Metal recordings byte for byte.

## 7b. Isaac Lab Franka-Pour at scale (2026-10-11)

Scene: `probes/scale_pour.py --part mpm` / `probes/pour_bitwise.py` (2,048 worlds x 735 particles, one
`SolverImplicitMPM`, separate worlds, rebuildable sparse grid, voxel 0.015, Jacobi 24 (25 run), P0 strain, pic27
collider, Isaac's capacity rule 1,024 cells per world, ~78 used). Branches: Warp `metalsim-mpm-scale` (from
`metalsim-metal-ordered-atomics` c160f7e4), Newton `metalsim-mpm-scale` (from `metalsim-mpm-determinism` 53a903d7).

* **Isaac's configuration includes a `fem.TemporaryStore`** (`NewtonMPMManager._create_solver`, also for coupler
  entries). The probe did not pass one, so every step allocated ~6.6 GB of fresh capacity-sized buffers at 2,048
  worlds. Both probes now pass it (bitwise equal to the store-less run); `--no-tstore` restores the old behaviour.
* **Capacity (item 1).** Everything sized by the reservation now stops at the part in use, read on the GPU:
  `wp.launch(..., dim_limit=)` (Metal: indirect dispatch sized by a one-thread prep kernel, also in graphs and gated
  ranges); Newton's solve loop bounded by the partitions' active node counts; `warp.fem` partitions, restrictions,
  integration, BSR assembly sorts, NanoVDB node-grid sorts and zeroing bounded by device counts. Off with
  `WP_METAL_BOUNDED_LAUNCH=0` (A/B in one build). Configuration unchanged (1,024 cells per world).
* **Undersized capacity (item 2).** The segfault was the warm-start gather `prev[cell_particle_indices]` (and the
  matching scatter) through PIC null indices (-1) of particles left outside the grid after the overflow, run as a host
  loop on Metal (CUDA reads and writes out of bounds silently). Newton now gathers/scatters with kernels that skip
  null indices; the overflow is reported as on CUDA (device warning, `check_sparse_grid_rebuild_status` raises).
  The Metal GPU builder keeps the previous grid on overflow (host semantics) where CUDA builds a truncated one, so a
  run continued past an overflow is not comparable; the state goes non-finite at 2,048 worlds x 64 cells.
* **Whole-step capture (item 3).** Strided / indexed `wp.copy` runs as a GPU kernel on Metal (CUDA's
  `wp_array_copy_device` indexing); the full step captures (in-capture gated solve) and replays bitwise like eager.
* **Body forces (item 4).** Isaac's coupler harvest (`coupling_harvest_proxy_wrenches`, float atomics: 20 different
  results in 20 identical calls at 2,048 worlds on Metal) sums per proxy body over its world's collider nodes in 32
  lanes, fixed order (node order within a lane, then lane order), on Metal; same cost (2.6 ms incl. impulse
  collection). `example_mpm_twoway_coupling` uses the same scheme (256 lanes, filled node prefix), 0.45 ms vs the
  chunked 0.49 ms, repeatable.
* **Environment Nanogrids (item 5).** `fem.Nanogrid.from_environment_voxels` (plain and rebuildable, and the adaptive
  variant) is allowed on Metal; Warp's multi-environment Nanogrid tests run on Metal and pass.

ms per MPM step, `--mode gpuexit` (chunk graph + GPU-decided stop) and the whole step captured, median of 10-20 steps
after 0.6 s of pouring (timing class):

| worlds | before, no store | before, store (Isaac) | after, store | after, captured | 128-cell reservation, before |
|---|---|---|---|---|---|
| 1 | 15.5 | 18.7 (13.7 pipelined) | 11.0 | 4.3 | |
| 64 | 23.3 | 18.3 | 15.7 | 8.0 | |
| 256 | 56.8 | 46.5 | 24.8 | 15.3 | 26.9 |
| 1,024 | 222 | 167 | 78.9 | 45.6 | 77.3 |
| 2,048 | 460 | 345 | 160 | 88.5 | 157 |

GPU kernel time at 2,048 worlds (eager, `WP_METAL_PROFILE=1`): 279 -> 103 ms per step. Bitwise: the pour trajectory
(every particle array, hashed every 10 steps over 320 steps) equals the pre-change Metal path at 1/64/256/2,048
worlds, eager and captured.

## 8. Closing the gap to the L4 (2026-10-11)

Starting point (`reports/Metal MPM speed and scalable determinism.md` and the kernel profile in
`runs/mpm/profile/kernels/`): granular 51.7 ms/frame on Metal vs 35.8 on the L4, ~75 % of it Newton's colored
Gauss-Seidel kernel (1,820 dispatches per frame at 16.4 us each), 8 host waits per frame. Four changes, in the order
ranked there; Warp branch `metalsim-mpm-speed` (on `metalsim-metal-ordered-atomics` `c160f7e4`), metalsim commits on
this branch. Every step was measured as its own frozen tree, timing-class jobs (`speed_suite.sh`, 60 frames after 5,
two repetitions; `speed_table.py`), Newton `metalsim-mpm-determinism` unchanged.

| step | change | bitwise? |
|---|---|---|
| 1 | Warp `f52da34`: `fast_math` modules compile on Metal as NVRTC's `--use_fast_math` does on CUDA | no (rounding; judged against the L4's floors) |
| 2 | Warp `7409510`: Metal volume destroy without a device drain; metalsim `4cda884`: unlogged GPU-exit solves read nothing back | yes |
| 3 | Warp `08084e0`: kernel arguments and launch bounds read through `device const` instead of `constant` | yes |
| 4 | Warp `ea6dcfd`: graph replays share command buffers (`WP_METAL_ICB_BATCH` default 1024) | yes |
| (opt-in) | Warp `da6adce`: eager `capture_while` / `capture_if` gated on the GPU (`WP_METAL_EAGER_CONDITIONALS=1`) | yes |

ms/frame (median of two runs; host waits per frame, command buffers per frame):

| scene | before | 1 fast_math | 2 no waits | 3 device args | 4 replay batching | L4 best |
|---|---|---|---|---|---|---|
| granular | 51.7 (8, 59) | 44.4 | 43.0 (6, 58) | 37.2 | **34.0** (6, 11) | 35.8 |
| multi_material | 62.6 (8, 59) | 47.4 | 46.9 (6, 58) | 41.3 | **37.9** (6, 11) | 35.5 |
| twoway (eager) | 73.5 (12.9, 25) | 71.8 | 69.1 | 67.6 | 69.3 (11.9, 16) | 73.0 |
| twoway (captured) | 44.4 | 41.7 | 42.7 | 42.2 | 42.8 | 31.0 |
| anymal (eager) | 45.5 (18, 29) | 44.0 | 42.8 | 41.2 | 41.4 (17, 20) | - |
| anymal (captured) | 32.7 | 31.4 | 30.5 | 28.7 | 28.8 | 24.2 |
| plate drag | 49.7 | 41.8 | 41.1 | 34.5 | **31.2** | 35.7 |
| column a = 8 (frames 6-65) | 44.5 | 36.5 | 35.9 | 29.0 | **25.5** | 28.6 (whole run) |
| column a = 2 (frames 6-65) | 32.4 | 27.0 | 26.6 | 21.2 | **19.4** | 20.4 (whole run) |

Granular Gauss-Seidel color pass (`probes/gs_pass.py`, graph replay, sustained clock): 16.5 -> 12.6 (fast_math) ->
10.1 us (device args); one solver iteration (5 per chunk) 151 -> 119 -> 93 us.

### 1. fast_math (rounding-level)

NVRTC documents `--use_fast_math` as `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`; the CUDA programming
guide maps `x/y` to `__fdividef` (2 ulp for |y| in [2^-126, 2^126]) and `sqrtf` to 1 ulp. Warp passes it for modules
with `fast_math` (Newton's rheology kernels); the fork compiled every Metal module with `MTLMathModeSafe` and precise
functions. The MSL specification (2026-06-04, sections 1.6.3 and 8) gives three math modes: `fast` and `relaxed` both
allow reassociation (relaxed keeps INF/NaN), `safe` allows nothing; `fast::` / `precise::` select function variants;
fast `x / y` is 2.5 ulp, fast `sin`/`cos` 2^-13 absolute (CUDA's `__sinf`: 2^-21.4). Measured on the M4 Max
(`probes/fast_math_modes.swift`): the fast and relaxed modes fold `(x + 2^24) - 2^24` to `x` (reassociation, which
CUDA's flag does not allow), so neither is usable as is; `#pragma METAL fp math_mode(relaxed)` followed by
`#pragma clang fp reassociate(off)` keeps reassociation off and lowers every division to `a * rcp(b)`, bit for bit
`metal::fast::divide`, the same structure as PTX `div.approx.f32`; Apple GPUs flush float32 denormals in every mode
(as `--ftz=true`); `#pragma clang fp reciprocal` is not accepted. Chosen lowering: that pragma pair (every division of
the module, as on CUDA), `metal::fast::sqrt`, transcendentals kept precise (Metal's fast ones are worse than CUDA's),
contraction still from `fuse_fp`. One license beyond CUDA: relaxed mode ignores the sign of zero. On
`probes/fp_semantics.py`'s inputs: division correctly rounded 72.8 % (L4 fast_math 71.4 %), sqrt 74.0 % (L4 83.3 %);
multiply-add and dot products unchanged and still equal to the L4 bit for bit; non-fast_math modules unchanged.

Parity (Metal recordings `runs/mpm/speed_rec/{base,s1_fastmath}`, L4 `runs/parity/isaaclab3_mpm{,_floors}`;
`floors_report.py`, `parity_report.py`):

| check | before | with fast_math |
|---|---|---|
| T1 column a = 2 vs the L4 float32 floor (frame 10 / 180) | 0.83 / 0.87 x floor, PASS | 0.89 / 0.87 x floor, PASS; 76.58 mean iterations, = L4 |
| T1 granular (frame 6 / 60) | FAIL (4625 / 27.7 x; collider ties, see section 5) | FAIL (same) |
| T3 R3 granular, multi_material, snow_ball, viscous, beam_twist, R4 twoway, R6 anymal, R7 plate intrude / drag | all PASS | all PASS (multi_material com 0.27 -> 0.24 %, plate intrude force RMS 1.9 -> 1.2 %, plate drag 3.6 -> 4.1 %, twoway body 0.74 -> 1.16 %) |
| T2 column collapse, 6 aspect ratios x 10 seeds | all 6 PASS; exponents 1.1561 / 0.7282 (L4 1.1559 / 0.7275) | all 6 PASS; exponents 1.1561 / 0.7280 |
| floors: column a = 2, granular, anymal (Metal RMS / worst of 4 L4 perturbations) | inside | inside (unchanged to 3 digits except column) |
| floors: multi_material | 3.0 / 1.25 / 1.25 x (outside, as before) | 3.1 / 1.27 / 1.25 x (outside) |
| floors: twoway (first / mid / last frame) | 0.76 / 0.28 / 0.65 x, inside | 0.77 / 0.32 / **1.07 x** |
| floors: plate drag force curve | 1.03 x (outside) | **1.11 x** (final force 244 N vs L4 253-265) |

A variant with fast division but precise sqrt (`runs/mpm/speed_rec/s1b_divonly`, not committed) puts twoway (0.91 x)
and the plate (0.76 x) inside, but moves the column a = 2 run by 4.8 um at 3 s (25 x the L4 float32 floor, T1 FAIL;
mean iterations 76.83 vs the L4's 76.58), where the committed form stays at 0.87 x with the L4's iteration counts. The
column is the one scene without collider ties, so it is the cleaner test of "same arithmetic as CUDA"; the twoway and
plate exceedances are a single deterministic Metal sample against the largest of four L4 samples of a chaotic flow.
The next step is the one MPM.md section 5 already names: an L4 floor distribution (more perturbed runs).

### 1b. CUDA-faithful fast_math (Warp `11aa54c8`, on fork `metalsim` as `91b8a859`; Warp default `"off"` since `152808b7`; 2026-10-11)

Research and measurements: `research_notes/Metal MPM speed and scalable determinism/cuda_approx_math.md`.

**What the L4 computes** (Warp 1.16 fast_math kernels; PTX from the kernel cache, SASS via ptxas 12.8 + cuobjdump):
every `/` is `div.approx.ftz.f32` = `MUFU.RCP` + `FMUL.FTZ` (100 % of 2^24 pairs and a special-value grid; not
`div.full`), `wp.sqrt` is `sqrt.approx.ftz.f32` = `MUFU.SQRT`, `1/sqrt` stays `MUFU.SQRT` + `MUFU.RCP`. MUFU.RCP is
correctly rounded on 86.79 % of inputs (biased up), MUFU.SQRT on 83.13 % (biased down, -1 ulp on 15.5 %); both are
exponent-invariant (checked over the whole normal range). Benjamin Carleton's ptx-math / sass-math interpolator model
(MIT) reproduces both on the L4 for every input (2^23 + 2^24 exhaustive), and Warp's compositions (`length`,
`normalize`, `cw_div`, `x/z +- y/z`, constant divisors, ...) are exactly per-operation compositions of it.

**Metal emulation** (`warp/native/metal_cuda_fast_math.h`, `warp.config.metal_fast_math = "cuda"`): the interpolator in
32-bit integer arithmetic (packed 128-row tables, the truncated squarer as a carry-free nibble-spread product), division
as `fma(a, rcp, signed zero)` so the Metal compiler cannot contract the quotient into a following subtraction (it does
for a plain product, even under `fp contract(off)`). Bit-identical to the L4 on all inputs of the sweeps, on 2^22
random divisions and on the 16 Warp compositions (2^20 inputs each); `"relaxed"` agrees on 66-88 %, `"off"` on 58-87 %.
Per-op cost (dependent chains, 2^20 threads; FMA 1.28 ps): division 10.8 ps (precise 3.7, fast 1.3), sqrt 13.6 ps
(precise 2.7, fast 1.3). With `"off"` / `"relaxed"` the recordings are bitwise those of the integrated tree's
no-fast-math / fast-math runs (all six floor scenes), so non-fast_math kernels are unchanged.

**Parity** (floor ensemble: 16 perturbed L4 runs, `floors_dist.py`; runs `runs/mpm/fm_{off,cuda,relaxed}` of the
worktree; M / max floor at the worst picked frame; verdict at the picked frames):

| scene | off | cuda | relaxed |
|---|---|---|---|
| column a = 2 (M at frame 180; L4 floor p50 0.178 um) | 0.165 um, inside | **0.144 um**, inside (L4 iterations 76.58) | 0.163 um, inside |
| granular | 0.66, inside | 0.66, inside | 0.66, inside |
| multi_material | 1.15, outside | 1.20, outside | 1.13, outside |
| twoway | 0.83, inside | 0.94, inside | 1.01, outside |
| anymal (frame 50) | 0.99, inside | 1.01, outside | 1.01, outside |
| plate drag (force curve) | 0.99, inside, final 240 N (L4 244-282) | **0.95**, inside, 250 N | 1.08, outside, 244 N |

parity_report.py (R suite): every T3 row PASS in all three modes; T1 column 0.83 / 0.87 (off), 0.93 / 0.76 (cuda),
0.89 / 0.87 (relaxed) x floor; T1 granular FAIL in all (collider ties, section 5).

**Speed** (timing class, 60 frames after 5, modes interleaved per scene, one repetition; ms/frame):

| scene | off | cuda | relaxed |
|---|---|---|---|
| granular | 48.2 | 83.0 (1.72x) | 37.0 |
| multi_material | 53.5 | 118.6 (2.22x) | 42.2 |
| twoway (captured) | 45.5 | 53.2 (1.17x) | 42.9 |
| anymal (captured) | 36.5 | 47.1 (1.29x) | 34.3 |
| plate drag | 47.6 | 80.4 (1.69x) | 36.0 |
| column a = 2 | 31.0 | 43.1 (1.39x) | 21.7 |
| column a = 8 | 39.8 | 64.3 (1.61x) | 25.6 |

**Default: `"off"`.** The faithful mode is CUDA's arithmetic exactly (closest to the L4 on the column and the plate,
the L4's iteration counts), but it does not pass the ensemble at least as well as `"off"` (anymal at 1.01 x vs 0.99 x,
multi_material 1.20 vs 1.15; differences at the size of one chaotic sample) and it is 1.2-2.2x slower: exact MUFU
results cost about 3x a correctly rounded division on Apple GPUs, so it cannot recover fast_math's speed. The column
"25 x" of the earlier fast-divide / precise-sqrt variant is consistent with a stopping-rule change (mean iterations
76.83 vs the L4's 76.58: one convergence check landing on the other side of the tolerance moves the trajectory by far
more than rounding), not with a lowering defect: all three modes here keep 76.58 and stay inside the column floor; that
mixed variant itself was not re-recorded.
`install(metal_fast_math="cuda")` remains for bitwise studies; `"relaxed"` is the fastest (0.64-0.94x off) at a small
parity cost (twoway, plate, anymal just outside the floor).

### 2. Host waits

`probes/sync_trace.py` attributes every wait (Python stack of the native call that moved the counter). Granular's 8:
1 harness synchronize; 5 in the sparse-grid allocation (`allocate_by_voxels`, the node and edge grids, `nanogrid`
active counts: exact-size builds that read counts back, as CUDA does; removing them needs device-side counts and
launch sizes, the other agent's active-count launch work); 1 `iteration_and_condition.numpy()` after the GPU-exit
chunk replays (only needed for the log: removed when unlogged); 1 `wp_volume_destroy_device` (the previous step's
grid; the Metal path drained the device before a free that is already deferred: removed). Coupled scenes: MuJoCo
Warp's `capture_while` constraint loop reads the condition before every iteration (anymal 15.5 of 18 waits per frame,
twoway 8 of 12); the opt-in eager conditionals cut anymal 17 -> 10 and twoway 11.9 -> 5.1 waits per frame
(bitwise; frame time -0.6 / -0.3 ms). Bitwise: unlogged recordings of all six floor scenes after steps 2, 3 and 4 equal
the step-1 recordings byte for byte (`bitwise_report.py`, `runs/mpm/speed_rec/s{1..4}_*_nolog`), and the eager
conditionals reproduce anymal and twoway (`s5_eagercond_nolog`).

### 3. Fixed per-dispatch cost: argument address space

`probes/fixed_cost.py` (Warp kernels, graph replay) and `probes/arg_space.swift` / `dispatch_anatomy.swift` (raw
Metal): the fixed cost of Newton's Gauss-Seidel kernel came from its argument struct, not its code size (an unused
large body adds nothing, `icache_cost.swift`; private memory adds nothing, `stack_cost.swift`). Apple GPUs preload the
`constant`-space data a kernel reads at each dispatch: zero-work dispatch with 19 array arguments 7.1 us vs 2.7 with 4;
16 descriptors in raw Metal 2.64 us `constant` vs 1.21 `device`. Reading arguments and launch bounds through
`device const` (all Metal kernels): 19 arguments 2.4 us, empty kernel 1.67 -> 1.36 us GPU per dispatch. Bitwise
(same values loaded). Tried and rejected: keeping a node's matrix entries in registers from gather to scatter
(bitwise, but 9.8 -> 11.6 us per pass: a larger register footprint costs per dispatch too, `walk4_regs`).

### 4. Graph replay: command buffers

`probes/replay_cb.py`: one command buffer per replay (the fork's default) cost ~26 us of GPU time per boundary (a
55-dispatch replay 103 us vs 77 us batched; raw Metal `cb_boundary.swift`: ~7 us with event, fence or queue order).
The GPU-exit loop replays its chunk up to 50 times per solve. Replays now append to the open command buffer up to
1024 dispatches, committed at once when nothing else is in flight: granular 58 -> 11 command buffers per frame,
-3.3 ms. Remaining per-dispatch gap to raw Metal (1.36 vs 0.82 us): launch-bounds load and the tile arena's per-lane
init (~0.1 us each, `dispatch_anatomy.swift`).

Tests: Warp `test_metal.py` (26, 3 new), `test_metal_primitives.py`, `test_metal_sparse.py`,
`geometry/test_volume_metal.py`, `test_metal_deterministic.py` `test_array.py`, `tile/test_tile.py`, `test_fixedarray.py`, `test_grad.py` pass on the branch (`test_struct.py` has the same 2 errors as the base tree); `tests/test_mpm.py` 10 passed,
1 xfailed (the SIMD sweep, as before), including the new unlogged GPU-exit test.

## 8b. Integrated branches (2026-10-11)

Warp `metalsim-integrated` (`upstream/warp-innate-integrated`) = `metalsim-mpm-scale` + `metalsim-mpm-speed` +
`metalsim-metal-det-sort` merged on `c160f7e4`; Newton `metalsim-integrated` (`upstream/newton-integrated`) =
`metalsim-mpm-scale` + `metalsim-rigid-det-sort` on `53a903d7`. On top of the merges:

* Deterministic launches honour `wp.launch(..., dim_limit=)` in every kernel pass (the passes after the kernel read
  only what the threads that ran recorded).
* 64-bit atomics Metal cannot make atomic (`atomic_cas` / `atomic_exch` on int64 / uint64, signed 64-bit
  `atomic_min` / `atomic_max`) raise at launch instead of running as a plain read-modify-write; 64-bit
  `atomic_and` / `or` / `xor` and shared-tile int64 adds are exact. Newton's HashTable keeps its 64-bit CAS off the
  Metal build (keys with bit 63 set are refused on Metal); contact matching claims with an unsigned 64-bit minimum
  (same winner on every device).
* Kernel arguments are read from constant memory by default again: device-memory arguments (`08084e0`) make the Metal
  compiler service fail on MuJoCo Warp's `_tendon_dot` (the Franka panda cannot step) and slow the G1 physics step
  (47.5 -> 54.4 ms at 4,096 worlds). `newton_compat.install(device_args=True)` (the default) opts MPM back in, so the
  numbers of section 8 are unchanged.

Verification on the integrated trees: the six floor scenes reproduce the step-4 recordings of section 8 bit for bit
for the pure-MPM scenes (granular, multi_material, plate drag, column, logged and unlogged); ANYmal (from frame 2) and
two-way (from frame 52) differ, as expected from the new rigid determinism (exact accumulation instead of ordered
launches) and the fixed-order two-way wrench sums, and are themselves repeatable (two independent runs bitwise
equal). Franka-Pour (section 7b) with `fast_math` ignored equals the pre-change recordings at 1 / 64 / 2,048 worlds,
eager and captured; with `fast_math` honoured (section 8, step 1) it differs by rounding, as intended. Rigid repeat
checks (`scale_pour.py --part rigid --det on --pipeline-det --max-triangle-pairs 8388608 --repeat-check`) are bitwise
at 64 and 2,048 worlds, eager and captured. `tests/test_mpm.py`: 14 passed, 1 xfailed.

ms/frame (speed_suite.sh, two runs; integrated vs section 8 step 4): granular 33.9 / 34.0, multi_material 37.9 / 37.9,
twoway 67.5 / 69.3, twoway captured 41.1 / 42.8, ANYmal 43.4 / 41.4, ANYmal captured 31.1 / 28.8 (scalable rigid
determinism costs more than ordered launches at one world), plate drag 31.4 / 31.2, column a = 8 24.9 / 25.5,
column a = 2 18.5 / 19.4. Franka-Pour ms per step (gpuexit / captured) at 1, 64, 256, 1,024, 2,048 worlds:
8.7 / 3.8, 12.3 / 7.6, 21.4 / 15.3, 75.0 / 46.8, 148.9 / 91.5 (section 7b: 11.0 / 4.3, 15.7 / 8.0, 24.8 / 15.3,
78.9 / 45.6, 160 / 88.5).

## 8c. Integrated-2, rigid collision at scale, fast_math forms (2026-10-11)

What a fresh setup installs (section 10). Warp fork `metalsim` = `c1bb061b` (integrated-2) + `91b8a859` (CUDA-faithful
fast_math, section 8 1b) + `152808b7` (`warp.config.metal_fast_math` defaults to `"off"`). Newton fork
`metalsim-1.5.2` = the integrated Newton `eac15793` + `c1ab1cdc` + the four rigid-scale commits + `METALSIM_CHANGES.md`.

* **Integrated-2 (Warp `d4a834ee` .. `c1bb061b`, Newton `c1ab1cdc`).** Kernel-argument address space chosen per kernel
  (`warp.config.metal_device_args = "auto"`: device memory for short tile-free kernels, constant otherwise, with a
  constant twin when the Metal compiler cannot build the device form; Newton's Gauss-Seidel kernels opt in), graph
  replays larger than the batch never split across command buffers, eager batches committed early while the GPU
  idles, small deterministic launches resolved in one threadgroup. Bitwise. ms/frame (Warp's METALSIM_CHANGES,
  interleaved, fast_math honoured as in section 8): granular 33.98 -> 31.62, multi_material 37.61 -> 35.38, two-way
  67.35 -> 58.34, ANYmal 42.78 -> 36.80, column a = 2 18.94 -> 18.48; G1 MuJoCo Warp env step at 4,096 worlds 109.1 ->
  92.8 ms (flat). Runs: `runs/mpm/integrated2/`.
* **Rigid collision at scale (Newton `c0a98819`, `555d6af8`, `1298fbbb`, `ddb015b9`; written as `de978b55`,
  `7beedf43`, `0013572a`, `ced01724`).** Franka-Pour at 2,048 worlds on the M4 Max: the mesh-convex midphase per thread
  on Metal (65 -> 0.26 ms per collide, same triangles), mesh threadgroups per GPU core (collide 179 -> 67 ms; the L4
  takes 90 ms at the same trajectory point), a barrier race in Newton's mesh-SDF kernels fixed (it hung the GPU at
  2,048 worlds; results unchanged where it did not fire), deterministic mode sorting only the active contacts (sort
  12.4 -> 0.2 ms, deterministic collide 192.6 -> 68.3 ms). Research: `research_notes/Metal MPM speed and scalable
  determinism/rigid_scale_perf.md`; runs: `runs/rigid_perf/`. The same mesh-SDF `progress` reset pattern is in the
  Newton 1.7 lines (`metalsim`, `metalsim-hashtable-metal`); the barrier fix is not ported there yet.
* **fast_math.** MPM picks the form with `newton_compat.install(metal_fast_math=...)` (default `"off"`), and Warp's
  own default is `"off"` as well, so programs that did not ask for it (MuJoCo Warp ray casts, Newton ray / collision
  helpers in the rigid stack) compile their `fast_math` kernels exactly as before `f52da34`. Section 8's timings were
  taken with fast_math honoured (`"relaxed"`); with the default `"off"` the MPM scenes run at section 1b's "off"
  column. `--metal-fast-math relaxed` (record_mpm.py, speed_bench.py) restores section 8's configuration.

## 9. Running

```
# Metal recording (CUDA semantics + Newton fork + chunk graph + GPU exit), in the environment of section 10
.venv-newton152/bin/python metalsim/parity/isaac_side/record_mpm.py \
    --scene column --set aspect=2 --device metal:0 --chunk-graph --gpu-exit --out runs/mpm/metal/R1_a2
.venv-newton152/bin/python -m metalsim.physics.mpm.compare <run_a> <run_b>   # T0/T1 metrics, column observables, solver stats
scripts/gpu_run.sh test_mpm check 15 -- .venv-newton152/bin/python -m pytest tests/test_mpm.py
# ms/frame of one scene (timing class); A/B of other trees: speed_suite.sh OUT WARP_TREE NEWTON_TREE . ('-' = installed)
scripts/gpu_run.sh mpm_speed timing 5 -- .venv-newton152/bin/python metalsim/physics/mpm/probes/speed_bench.py \
    example:mpm_granular --frames 60 --warmup 5 --metal-fast-math off
# Franka-Pour at scale (needs MuJoCo Menagerie's panda: upstream/mujoco_menagerie, or PANDA_XML=...)
.venv-newton152/bin/python metalsim/physics/mpm/probes/scale_pour.py --part mpm --device metal:0 --worlds 64 --mode capture
```
Probes (`probes/`): `ab_timing.py` (interleaved A/B), `profile_sections.py` (Newton's section timers),
`loop_breakdown.py` (GPU cost of each loop component from recorded graphs), `dispatch_latency.py`, `p2g.swift`
(P2G microbenchmark from the research round).

## 10. Reproducing from a fresh clone

Everything above runs from a clone of MetalSim plus the three forks; nothing depends on local branches.

```
# after the install of README / docs/GUIDE.md (.venv, scripts/setup_warp.sh, the MuJoCo Warp fork):
scripts/setup_newton152.sh          # .venv-newton152: Newton fork metalsim-1.5.2 @35791a07, shared Warp fork @152808b7
scripts/gpu_run.sh test_mpm check 15 -- .venv-newton152/bin/python -m pytest tests/test_mpm.py   # 15 passed, 1 xfailed
```

| what | where it lives | how a fresh setup gets it |
|---|---|---|
| Warp changes | pulipakaa24/warp `metalsim`, pinned `152808b7` | `scripts/setup_warp.sh` (fallback: innate-inc/warp `ce15f6bb` + `patches/warp/metalsim-152808b7.diff`) |
| Newton changes (MPM, Franka-Pour rigid) | pulipakaa24/newton `metalsim-1.5.2`, pinned `35791a07` | `scripts/setup_newton152.sh` (fallback: v1.5.2 + `patches/newton/metalsim-1.5.2-35791a07.diff`) |
| Newton 1.7 HashTable fix for the rigid stack | pulipakaa24/newton `metalsim-hashtable-metal` `24fb5367` (`.venv`'s `newton` extra) and `metalsim` `fe7f5bd4` (`.venv-newtonfork`) | `pip install -e ".[newton]"`, `scripts/setup_newton.sh` |
| compat layer, scenes, probes, tests, Isaac-side recorders | this repository | the clone |
| logs, JSON, reports of every run | `runs/mpm/`, `runs/parity/isaaclab3_mpm*/`, `runs/rigid_perf/`, `runs/fma_rigid_parity/` | the clone |
| Metal particle recordings (`.npz`, ~6.6 GB) | not committed | `record_metal_suite.sh`, `record_variant_suite.sh OUT - - .` (one GPU job per run; deterministic, so a re-recording is bitwise the original on the same trees) |
| L4 recordings (`runs/parity/isaaclab3_mpm*`, ~31 GB incl. tarballs) | not committed | on the L4 VM: copy `record_mpm.py`, `record_mpm_il3.py`, `scenes.py`, `newton_compat.py` to `~/parity3_scripts/mpm/`, run `isaaclab3/stage_mpm.sh`, `stage_mpm_floors.sh`, `stage_mpm_floors2.sh`, `stage_mpm_timing.sh` (Isaac Lab 3.0-EA install: `isaaclab3/`) |
| kernel caches, L4 scale logs > 1 MB | not committed | rebuilt on first run / re-run `probes/scale_pour.py` on the L4 |
| Franka panda model | MuJoCo Menagerie (not vendored) | `git clone https://github.com/google-deepmind/mujoco_menagerie upstream/mujoco_menagerie` or `PANDA_XML` |
