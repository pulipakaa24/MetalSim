# MPM on Metal (2026-10-09; sections through 8j, 2026-10-13)

Sections are a dated log: statements such as "being ported" or "in progress" are the state on their date, and a
**Done** / **Resolved** note points to where the result landed. Summary of the current state: `docs/STATUS.md`
("Deferred work items") and `docs/GAPS.md` (MPM row).

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
(Measured since: L4 timing runs in sections 7 and 8; coupled scenes re-timed against them in section 8j.)

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
  Isaac Lab's bit for bit, a = 2 and 5); the column and plate runs (R0 column, R1, R7) are being re-recorded (**done**:
  second pass below); the first pass is kept in `mpm_lattice13/`. Newton examples (R3, R4, R6) and R2 itself were unaffected.

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
  247.75 iterations mean. (Matched Metal runs since: sections 7, 8 and 8j.)


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
   so it is a separate branch: merging it means re-running the rigid-body parity suite. (**Done:** merged into Warp
   `metalsim`; the rigid suite had the same known failures before and after, `runs/fma_rigid_parity/`.)
2. **BVH builder.** CUDA builds mesh BVHs with LBVH, the fork's Metal path falls back to host SAH; at exact ties the
   closest face differs (CUDA LBVH vs CUDA SAH alone: 4,962 / 7,126 / 3,793 tied points differ in three box cases).
   Being ported (Metal LBVH identical to CUDA's). (**Done:** results with FMA + LBVH below.)

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
  re-run after the LBVH port, together with R3/R4/R6/R7 (**done**, next subsection).

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
percentile in it, plus seed ensembles for the plate force. **Resolved (2026-10-12, section 8e):** against the
16-run L4 floor ensemble (plus 6 rigid-jitter runs for ANYmal) all six scenes are inside; multi_material 0.80,
plate drag 0.97 at the worst frame (`runs/mpm/closeout/floors_final.md`).

## 7. Performance plan (research: `reports/Metal GPU primitives for MPM speed.md`)

Timing-class runs, ms/frame, Metal vs the L4 (eager / whole step captured): column a=2 14.5 vs 20.4; column a=8
52.5 vs 28.6; plate drag 67.0 vs 35.7; granular 88.7 vs 35.8/36.0; multi_material 90.1 vs 35.5; ANYmal 107.7 vs
58.9/24.2; two-way coupling 254.4 vs 73.0/31.0. Two-way profile: MPM step 224 ms, of which ~130 ms are CPU radix sorts
in `warp.fem` `compress_node_indices` and ~40 ms GPU drains before host utility calls (the fork runs
`radix_sort_pairs` / `runlength_encode` / `array_scan` on the host on Metal). Order: GPU sort/RLE/scan (in progress,
branch `metalsim-metal-sort`; **done**, table below), GPU-driven solver loop (indirect execution ranges), residual folded into the solve,
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
4 in ANYmal) while the L4's are; under investigation. **Resolved (2026-10-11, section 8b):** with the scalable rigid
determinism (exact accumulation) and the fixed-order two-way wrench sums, two independent ANYmal and two-way runs are
bitwise equal; Franka-Pour's coupled solver repeats bitwise from section 8g (`rigid_deterministic=True`). Pure-MPM scenes are bitwise deterministic and the merged tree
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
(Done: sections 1b and 8e; all six scenes inside in `"off"`.)

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

**Parity** (floor ensemble: 16 perturbed L4 runs, `floors_dist.py`; runs `runs/mpm/fm_{off,cuda,relaxed}`, timings
`runs/mpm/speed_fm`, emulation probes `runs/mpm/approx` (logs and JSON; recordings not committed); M / max floor at the worst picked frame; verdict at the picked frames):

| scene | off | cuda | relaxed |
|---|---|---|---|
| column a = 2 (M at frame 180; L4 floor p50 0.178 um) | 0.165 um, inside | **0.144 um**, inside (L4 iterations 76.58) | 0.163 um, inside |
| granular | 0.66, inside | 0.66, inside | 0.66, inside |
| multi_material | 1.15, outside | 1.20, outside | 1.13, outside |
| twoway | 0.83, inside | 0.94, inside | 1.01, outside |
| anymal (frame 50) | 0.99, inside | 1.01, outside | 1.01, outside |
| plate drag (force curve) | 0.99, inside, final 240 N (L4 244-282) | **0.95**, inside, 250 N | 1.08, outside, 244 N |

Section 8e supersedes this table. multi_material and ANYmal were outside for three reasons: a CUDA-only warp.fem tiling
path, the Metal library's log / exp / sinh, and a rigid-side floor with a single sample. With the fixes, all six scenes
are inside in `"off"`.

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
  determinism/rigid_scale_perf.md`; runs: `runs/rigid_perf/`. Ported to the Newton 1.7 lines (`metalsim` `522e2d4c`,
  `1a398b2e`, `464c0aff`; `metalsim-hashtable-metal` the same commits): Franka-Pour collide on that line at 2,048
  worlds 646.8 -> 57.2 ms, deterministic contacts unchanged (`runs/rigid17/`, section 8d).
* **fast_math.** MPM picks the form with `newton_compat.install(metal_fast_math=...)` (default `"off"`), and Warp's
  own default is `"off"` as well, so programs that did not ask for it (MuJoCo Warp ray casts, Newton ray / collision
  helpers in the rigid stack) compile their `fast_math` kernels exactly as before `f52da34`. Section 8's timings were
  taken with fast_math honoured (`"relaxed"`); with the default `"off"` the MPM scenes run at section 1b's "off"
  column. `--metal-fast-math relaxed` (record_mpm.py, speed_bench.py) restores section 8's configuration.

## 8d. Rigid-side loose ends (2026-10-12)

* **64-bit atomics.** The Warp fork refuses 64-bit `atomic_cas` / `atomic_exch` on Metal at launch (section 8b), which
  left four Newton sites unable to run there: contact-reduction finalize / rollback (the mesh-mesh SDF reduce kernel and
  the two-spatial-depth reducers; 1.7 line), the XPBD restitution manifold table (1.7), Kamino's 63-bit metric argmax
  keys (both lines) and the HashTable (fixed before). Metal protocols, CPU / CUDA unchanged (research:
  `research_notes/Metal MPM speed and scalable determinism/metal_64bit_atomics.md`): finalize by a 64-bit atomic max
  plus a read-back with no rollback (slots only grow); claim the high word with a 32-bit CAS, publish with a 64-bit
  min (tagged keys); per-thread key scratch plus a per-world packed max and a finalize launch. Every claim loop is
  bounded (HashTable and XPBD manifolds: probes + 65,536 passes, then a counted failure). Newton's contact / collision /
  SDF / heightfield / XPBD / Kamino-metrics tests on `metalsim-hashtable-metal`: 98 failures -> 13 on cpu + metal:0,
  no new failure; the 13 are SolverMuJoCo tests that do not build against the MuJoCo Warp fork 3.14
  (`contact_force_fn` signature). 1.5.2 Kamino metrics: CPU 9/9, Metal 8/9 (test_06's dense-vs-sparse `f_ncp`, 2.1e-6
  relative vs rtol 1e-6, in a kernel these changes do not touch).
* **Rigid-scale ports to the 1.7 line**: mesh-SDF barrier fix, per-thread mesh-convex midphase, 32 mesh threadgroups
  per core, bounded deterministic sort. Franka-Pour collide (`runs/rigid17/tools/collide17.py`, same in-contact state,
  interleaved, Warp fork `152808b7`): 256 worlds 93.9 -> 10.1 ms (midphase 72.4 -> 0.34 ms, mesh-mesh 20.9 -> 7.5 ms);
  2,048 worlds 646.8 -> 57.2 ms (two runs each); deterministic mode at 256 worlds 91.4 -> 9.4 ms with the sorted
  contacts bitwise equal (same hash before and after the ports, and across repeated collides).
* **GPU hang (2026-10-12 02:32).** Newton's tests on the *unported* 1.7 tree (`fe7f5bd4`, none of the changes above)
  hung the GPU in `TestDeterministicPipeline.test_cylinder_scale_update_keeps_generic_stage_metal_0` (cylinder on a
  plane: no mesh, HashTable, reducer or manifold code); the watchdog killed the job after 150 s, the process stayed
  stuck in exit holding the GPU. The same test passed in three other runs (ported trees and the unported
  `metalsim-hashtable-metal`), and 20 isolated repetitions after the reboot all passed (0.6 s each): the cause is open
  (intermittent, below Newton's mesh / atomics code). The new claim loops ran alone first after the reboot
  (test_hashtable, test_solver_xpbd, test_contact_reduction_global: no stall).
  **Update (2026-10-13): most likely not this test.** The record (queue history excerpt
  `runs/rigid17/out/gpu_history_2026-10-12_0130-0300.txt`, the test job's watchdog log `runs/rigid17/out/s17_base.watch`):
  the test job was a shared `check` job (granted 02:29:02); its host stopped advancing at about
  02:31:42 (killed at 02:34:13 after 151 s without CPU progress). That night the particle demo's 25 M-grain sand scene
  ran three times as `anymal_sand_seq`, granted 01:51:09, 02:03:53 (killed 02:20:17) and 02:32:49. Each of the
  three GPU wedges of 2026-10-12 (02:32, ~10:20, 18:26) followed a run of that scene. Its acceleration structure had
  25.1 M instances, above Metal's standard 2^24 limit, built without `MTLAccelerationStructureUsageExtendedLimits`
  (undefined behaviour). On macOS 26 that build never finished (still at 100 % after 900 s), and the GPU work kept
  running after its process was killed. The test passed in its 3 other runs and in 20 isolated repetitions
  (`runs/rigid17/out/hang_retest_base.log`). So the 02:32 stall is best explained by that build (the one killed at
  02:20 still running without an owner, or the one starting at 02:32), not by the test. The evidence is strong but
  circumstantial: the wedge was not reproduced on purpose and the test was not run on a GPU known to be free of
  ownerless work. Fixed on the particle side since: grains are a two-level structure in chunks of at most
  2^24 - 1 (`as_limits.check` refuses larger ones without an extended-limits path). The 25 M scene now builds and
  renders on both Macs: the 14-inch (macOS 27) and the 16-inch (macOS 26.7.1, 654 ms build, 5.1 s 1080p frame).
  See `metalsim/render/PARTICLES.md` and `runs/sand25m_16in/`.
* **Newton-engine contact buffer** (`metalsim/physics/newton_backend.py`): sized per contact pair
  (`rigid_contact_capacity`); 64 G1 worlds on rough terrain went from 1,000 slots (about half the contacts dropped) to
  7,680; `tests/test_g1_newton_engine.py::test_no_rigid_contacts_dropped`.
* **MPM timings** (granular 35.1 / column 20.9 ms now vs 31.7 / 18.8 earlier): no regression. Interleaved A/B of the
  pinned build against integrated-2 agrees within 0.4 ms in every window; both moved together by 8-11 % between
  windows minutes apart (machine state; CPU load alone moves granular +14 %). `runs/mpm/regress_check/`.
## 8e. Parity close-out: multi_material and ANYmal (2026-10-12)

Research note with every measurement and source: `research_notes/mpm_parity_closeout/NOTE.md`; scripts in the same
directory; L4 data `runs/parity/isaaclab3_mpm_closeout/` and `runs/parity/isaaclab3_mpm_floors/mpm_floors3/`, Metal
`runs/mpm/closeout/` (recordings not committed).

**multi_material was a systematic difference (Metal took other algorithm paths), not chaos.** At frame 6, while the
snow is still in free fall, Metal was 150 nm from the L4 in every math mode. The L4's own arithmetic perturbations
were 35-43 nm, and the Metal modes were 36-42 nm from each other. The cause sat in the particle-to-grid transfer:

1. **warp.fem tiles its dispatched assembly on CUDA only.** For "long" quadratures, a node's contributions are
   summed by 32 lanes (strided sums, then the warp shuffle tree). The condition starts `not device.is_cuda`
   (`_generate_auxiliary_kernels`, the same in the L4's Warp 1.16), so Metal summed sequentially. After step 1, node
   mass and momentum differed on 72 % of the nodes by up to +-10 ulp. No-FMA and exact math on the L4 leave those
   arrays bitwise equal, so the floor ensemble could not cover the difference. The L4 tiles in five of the six floor
   scenes (not two-way).
2. **`log` / `exp` / `sinh` were the Metal library's.** On CUDA they are libdevice's. Over all 2^32 inputs, Metal's
   `log` equals CUDA's on about half of the finite inputs, and `exp` / `sinh` on 97 %. Newton uses them in the
   plastic-flow (`Jp`) and hardening laws. Snow (hardening 10, zero initial yield strength) is the most sensitive.
3. **svd3 / eig3 use `rsqrt.approx` (MUFU.RSQ) on CUDA and `1 / sqrt` elsewhere.** The two differ on 33 % of
   inputs. This does not affect the first multi_material frames; it is fixed for faithfulness.

Fixes (Warp fork `metalsim`):

* `3c4e251e`: Metal takes CUDA's tiling choice. Metal's SIMD-group `tile_sum` is the CUDA shuffle tree.
* `47a4031d`: `warp.config.metal_cuda_libm` (default on) adds `native/metal_cuda_libm.h`, the PTX of
  `__nv_logf` / `__nv_expf` / `__nv_sinhf` with MUFU.EX2 through sass-math's model, and adds MUFU.RSQ to
  `metal_cuda_fast_math.h`.
  * Measured equal to the L4 on all 2^32 inputs of each of the three functions.
  * MUFU.RSQ equals the L4-validated model on all of [1, 4) and across exponents.
  * When on, every module hash is salted once; when off, behaviour and hashes are those of before.

Effects:

* Node mass and velocity after step 1 now equal the L4's bit for bit.
* Snow `Jp` after step 1 now equals the L4's on all particles.
* The first iteration counts now equal the L4's.
* The early steps sit at the size of the L4's exact-math perturbation.
* The tiling decisions of the six scenes equal the L4's call for call.
* Neutrality: kernels without `log` / `exp` / `sinh` / `svd3` / `eig3` / fem dispatch compile to the same code.
  `WP_METAL_CUDA_LIBM=0` restores the previous hashes and results for everything except fem tiling.
  MuJoCo Warp hashes: see the rigid check below.

**ANYmal frames 4-8 were a floor-estimation gap.** In the first ten frames, only the L4's no-FMA runs move the robot
(three near-identical runs); the jitter and exact-math runs leave it at the base pose bitwise. The new L4 stage
`isaaclab3/stage_mpm_floors3.sh` (`record_mpm.py --jitter-body-ulp 1`, six seeds) perturbs the robot's initial joint
coordinates by 1 ulp:

* The robot moves by 0.6-1.0e-7 m at frame 2 (Metal: 0.9-1.0e-7).
* The sand moves by up to 7.85 um at frame 6 (Metal: 7.0 um).

**Metal vs the floor ensemble** (`floors_dist.py`, `metal_fast_math="off"`). The ensemble is the 16 L4 runs, plus the
6 rigid-jitter runs for ANYmal. The ratio is M / largest perturbed-L4 distance, at the worst frame:

| scene | before | after | verdict |
|---|---|---|---|
| multi_material | 1.15 | **0.80** (last frame 0.75) | inside, and inside p95 pairwise |
| ANYmal | 1.22 (1.00 with the rigid-jitter runs) | **0.94** (last frame 0.84) | inside |
| two-way | 0.83 | 0.83 | inside (no tiling, so practically unchanged) |
| granular | 0.66 | 0.66 | inside (collider-tie event at frame 6 dominates) |
| plate drag (force curve) | 0.99, 240 N | **0.97**, 250 N (L4 244-282 N) | inside |
| column a = 2, 3 s | 0.165 um | 0.172 um (L4 exact math 0.155 um, float32 floor ~0.19 um); 76.58 iterations = L4 | inside |

With `"cuda"` fast math:

* multi_material 0.81, ANYmal 0.94, two-way 0.95, plate 0.38 (final force 283 N, 1 N above the L4's range).
* Column a = 2: a convergence check at solve 76 takes 30 iterations against the L4's 25. Before that frame, Metal-cuda
  is 1.6e-7 from the L4, like `"off"` and like the L4's exact-math run. Afterwards the column grows to 10 um at 3 s.
  This is a near-threshold stop, the kind the L4's own no-FMA run takes at solve 11; only the fast_math kernels
  differ between the two modes. It does mean `"cuda"` now fails the strict column T1, and the default stays `"off"`.

Verification of the pushed Warp tree (`43ffc6dd`, rebuilt):

* The six floor recordings equal the development tree's bit for bit.
* Attribution (multi_material / ANYmal, worst-frame ratio):
  * fem tiling alone (`WP_METAL_CUDA_LIBM=0`): 0.88 / 1.00.
  * fem tiling with `metal_cuda_libm`: 0.80 / 0.94.
* Rigid parity harness (`runs/fma_rigid_parity/closeout_libm{off,on}{1,2}`, `classify_closeout_libm.txt`,
  `metrics_closeout_libm.txt`):
  * The option off and on both give the same five known failures and 273 passes.
  * 29 captured locals in 4 tests differ stably between the two settings: the Warp policy MLP (activations in the 7th
    digit) and one MuJoCo-C flex error (8.971145e-3 vs 8.971146e-3). The other differences are run-to-run noise
    (336 locals, 68 tests).
  * Bitwise hashes of MuJoCo Warp BatchSim cannot test neutrality: it is not run-to-run repeatable
    (`runs/mpm/closeout/rigid_neutral.log`).

Cost: no measurable change. multi_material ms/frame with `metal_cuda_libm` on vs off, three interleaved repetitions:
52.7 / 55.7 / 54.2 vs 54.5 / 54.4 / 55.1. Granular, column and ANYmal before/after tiling: within +-1 ms, two
repetitions each (`runs/mpm/closeout/speed/`; the device was not idle at the start of these timing jobs, so the
spread is ~+-3 ms).

## 8f. The Franka-Pour MPM step at 2,048 worlds (2026-10-12)

Research and every number: `research_notes/Metal MPM speed and scalable determinism/mpm_step_at_scale.md`; logs and
JSON: `runs/mpm/step_at_scale/`. Tools: `probes/section_kprof.py` (per-section, per-kernel GPU time, eager, every
dispatch attributed), `probes/pour_occupancy.py` (does the grid occupancy repeat?), `probes/first_launch.py` (first
launches of changed kernels with a host-side completion timeout).

Profile at 2,048 worlds (base: Warp `43ffc6dd`, Newton `c0427fc7`; ms of GPU time per step, about 108 in all): strain
solve 34.1 (rigidity products 8.8, transposes 9.1), scratchpad 16.1, grid rebuild 15.3, collider rasterization 14.4
(collider-matrix sort ~6), unconstrained velocity 8.9 (APIC generic integrate 5.5), strain matrix 6.5, collider
compliance 6.4.

Adopted, each bitwise (particle state and every `state.mpm` array hashed every 10 steps over 320 steps at 1 / 64 /
2,048 worlds, eager and captured):

| change | where | GPU ms per step at 2,048 worlds |
|---|---|---|
| rigidity operator assembled from its kept triplet slots only | Newton `27415f95` | collider compliance 6.4 -> 3.1 |
| rigidity products in the loop on non-empty rows only (empty rows are fixed points after the initial guess; from 8,192 rows) | Newton `80e79684` | 8.8 -> 2.6 |
| compact Metal transposes stop at the stored blocks | Warp `8631b6b0` | 9.1 -> 4.9 |
| node-index compression skips the ignored tail | Warp `fddebdbd` | 5.9 -> 1.4 |

Captured ms per MPM step, base vs all four, two interleaved rounds: 1 world 3.96 / 3.84 vs 4.00 / 3.99; 64 worlds
8.00 / 7.75 vs 7.87 / 7.46; 256 worlds 15.40 / 15.53 vs 14.19 / 14.16; 1,024 worlds 45.97 / 46.77 vs 41.38 / 41.07;
**2,048 worlds 91.16 / 90.50 vs 78.95 / 78.88**. Franka-Pour policy step (`franka_pour_ppo bench --mode step`, 2,048
envs, this Mac): 2.06 -> 1.81 s. `tests/test_mpm.py` 14 passed, 1 xfailed.

Evaluated and not adopted (all bitwise; `runs/mpm/step_at_scale/rejected/`): reusing the grid when the occupancy
repeats (it never does: 0 of 199 consecutive steps keep the (world, cell) set at 64 or 2,048 worlds, ~46 % of the
worlds change cells every step); the whole Jacobi iteration fused per world (one threadgroup per environment,
barriers between passes; faster at 64 worlds, 7.71 -> 7.05 ms, slower from 256 on, 92.1 -> 105.2 ms at 2,048); all
of a node's dofs in one thread for the generic APIC integrate (no change); a row-ordered presort instead of the radix
sort for fem's triplets (79.2 -> 85.8 ms).

Pins: Newton `metalsim-1.5.2` is pinned at `3fd1f052` (the two Newton commits). The Warp commits are on the fork's
`metalsim` (`3342e5d8`), on top of `41b7bc45`, which faults on a one-world captured Franka-Pour step (base tree too);
the Warp pin stayed at `43ffc6dd` until that was fixed (section 8g; pinned `85b92466`, then `f0df4cf8`). What a fresh setup gets now (Warp `43ffc6dd`, Newton `3fd1f052`) is verified on its own: bitwise against the base at 1 / 64 / 2,048 worlds, eager and captured; `tests/test_mpm.py` 14 passed, 1 xfailed; captured ms per step at 1 / 64 / 256 / 1,024 / 2,048 worlds 3.81 / 7.58 / 15.53 / 46.45 / 90.61 (base) vs 3.78 / 8.07 / 15.15 / 44.15 / 84.41, second round 3.94 / 7.76 / 15.35 / 46.00 / 90.54 vs 3.82 / 7.73 / 14.80 / 43.58 / 84.97.

`warp.config.metal_cuda_libm` (Warp `43ffc6dd`, default on) costs nothing measurable: on vs off, three interleaved
runs each, twice: granular 40.5 vs 40.7 ms/frame, multi_material 50.6 vs 50.7, Franka-Pour 2,048 worlds 90.6 vs 90.7
ms per captured step (`runs/mpm/step_at_scale/libm/`).

## 8g. Graph capture memory (2026-10-12; Warp `41b7bc45..85b92466`)

Isaac Lab's Franka-Pour policy step captured as one graph (4 sim steps: 16 MPM solves, 4 collides, 24 MuJoCo
substeps) replayed NaN arm state at 2,048 worlds on Warp `152808b7` / `43ffc6dd`. It was correct at 1,024 worlds and
with one graph per sim step.

* Cause: a Metal capture kept every capture-time allocation in its own buffer for the graph's lifetime, so a graph of
  several steps held every step's temporaries (MuJoCo Warp's per-solve contexts and the rest) at once. Allocated at
  2,048 worlds:

  | graph | allocated | result |
  |---|---|---|
  | one per policy step | 70.7 GB | NaN |
  | one per sim step | 44.3 GB | fine |
  | one per policy step, 1,024 worlds | 35.3 GB | fine |
  | two sim steps, 2,048 worlds | 53.1 GB | NaN once, finite once |

  The M4 Max's working set (`recommendedMaxWorkingSetSize`) is 55.7 GB. In the failing run the residency set held
  35.4 GB of the 70.7 GB, and the first replay made the arm state NaN in 2,004-2,048 worlds with no error.
  `WP_METAL_ICB=0` (per-dispatch replay) replays NaN too.
* `41b7bc45`: memory freed during a capture goes to a per-capture pool that later allocations of the same capture
  draw from (replays are serial, as with a CUDA graph's memory pool). Policy-step graph: 37.4 GB, finite.
* `41b7bc45` then faulted on a one-world captured MPM step (`kIOGPUCommandBufferCallbackErrorPageFault`). A solve's
  80-byte indirect-execution-range array (the host writes its fields at each range's end) had taken the buffer of a
  2,940-byte fem temporary freed earlier in the capture. Every replay, the temporary's recorded dispatches overwrote
  the ranges first, and garbage ranges ran.
  * Found by zeroing one reuse at a time on the GPU and logging the allocation stacks
    (`pour_bitwise.py record --worlds 1 --mode capture`).
  * Reuse was opt-in for the interim (`270660b1`).
* `85b92466` turns reuse back on with three changes:
  * Only buffers of 64 KiB or more are pooled or served. The arrays the host fills while recording are smaller.
  * A large range array that lands in reused memory raises at `metal_capture_range_begin`.
  * Deterministic scratch pools allocated during a capture record their zeroing.
  * The setting is `warp.config.metal_capture_reuse` / `WP_METAL_CAPTURE_REUSE`.
* `732b6e83`: buffers held only by graphs stay in the address table. Before, kernels that reach an array through a
  descriptor stored in device memory read zeros once the array was freed.

Verification of `85b92466`:

* Warp's Metal test files and `tests/test_mpm.py` (14 passed, 1 xfailed) pass.
* Pour recordings (`pour_bitwise.py`) at 1 / 64 / 2,048 worlds finish, eager and captured. Captured is bitwise equal
  with reuse on and off.
* With Newton `3fd1f052`, every recording at 1 / 64 / 2,048 worlds, eager and captured, is bitwise that of Warp
  `43ffc6dd`; `tests/test_mpm.py` passes with that pair too.
* The six floor recordings are bitwise those of `43ffc6dd` (`bitwise_dirs.py`).
* The policy-step graph at 2,048 worlds is finite.

Default `capture_unit` stays `"sim"` (Isaac's granularity); `"policy"` now works.

`f0df4cf8` (pinned after `85b92466`) changes only deterministic kernels. The phase-0 store guard's counter-target
check is now a function call instead of being inlined at every guarded store. Under global deterministic mode, the
inlined form crashed the Metal compiler service on Newton's narrow-phase kernel. Results are unchanged:
`tests/test_mpm.py` passes and the six floor recordings are bitwise those of `43ffc6dd`.

Run-to-run repeatability of Franka-Pour (newton_compat, same day): the scene differed between two runs from the first
uncaptured step, even with `rigid_deterministic=True`. `SolverCoupled` compacts each entry's contacts with an atomic
counter, so MuJoCo Warp got them in thread-arrival order. This is not a missing barrier in Warp's runtime: the
writer/reader, memset/atomic and bounded-launch probes show none, and the collision pipeline and the contact
conversion each repeat exactly on fixed inputs. `install(rigid_deterministic=True)` now builds the coupled-solver
modules in RUN_TO_RUN mode on Metal. The scene then repeats bitwise at 64 and 2,048 worlds, eager and captured, and
eager, one graph per sim step and one graph per policy step all agree. Cost per step is unchanged.
Test: `test_coupled_contact_filter_keeps_source_order`.

## 8h. Franka-Pour GPU memory at 2,048 worlds (2026-10-12; Warp `95f6f121`)

Tools (`scripts/diagnostics/franka_pour_memory/`, run with the `metalsim-franka-pour` branch's `metalsim` on the
path): `mem_attrib.py`
records every Metal allocation Warp makes (size, phase, owner from the Python stack), the native allocation tracker
(`wp_alloc_tracker`), `MTLDevice.currentAllocatedSize`, the process footprint and resident size, TemporaryStore
borrow/redeem traces, and what a captured graph keeps alive; `grid_demand.py` the NanoVDB node and collision demand
over a rollout; `franka_bitwise.py` per-step trajectory hashes. Logs: `runs/mpm/memory_2048/`.

Where the 37.6 GB (device allocations after the first policy step, Warp `f0df4cf8`, one graph per sim step) go:

| owner | GB |
|---|---|
| sparse-grid NanoVDB volumes (native): vertex grid 11.04 (272,256 lower nodes reserved, ~2.3k used), cell grid 1.38 | 12.4 |
| warp.fem `TemporaryStore` (pooled temporaries; 6.5 GB of simultaneous requests, mostly the pic27 collider restriction's 56.6M-slot tables) | 11.1 |
| other MPM arrays (fields, BSR matrices, restriction / partition tables, sort scratch), sized by the 1,024-cell reservation | 5.8 |
| collision pipeline (contacts sized for 7.07M rigid contacts, 8.4M triangle pairs; peaks seen 57k / 30k) | 2.8 |
| model / state (particle attributes 1.5M x ...) | 1.6 |
| MuJoCo Warp data (njmax 2,560 per world) | 1.1 |
| coupler, task | 0.5 |
| held by the captured graph (pool net of reuse 1.8, retained 0.4) | 2.2 |

Process: footprint 44.0 GB, resident 19.6 GB. Most of the allocation is capacity that is never touched (cell
reservation 1,024 per world, 80 in use; NanoVDB node capacities), which is why resident memory (and earlier RSS
readings) is far below the allocation; on CUDA every allocated byte is physical, which is why Isaac's own task
runs out of the L4's 24 GB at this size.

Reductions, each bitwise (Franka-Pour with `rigid_deterministic=True`, captured, per-step hashes of body, joint and
particle state: 60 steps at 64 envs, 30 at 2,048, equal to the base, whose two runs are equal; MPM pour scene
recordings at 1 / 64 / 2,048 worlds, eager and captured, equal; `tests/test_mpm.py`, warp.fem and sparse tests pass):

| change | where | device GB after the first policy step (2,048 worlds) | policy step |
|---|---|---|---|
| base | Warp `f0df4cf8`, Newton `3fd1f052` | 37.64 | 1.84 s |
| tight temporary pools (exact sizes, no array > 1.25x a request of >= 1 MiB) | Warp `95f6f121` (default on Metal) | 34.48 | MPM step 81.2 / 81.5 vs 81.5 / 81.4 ms |
| + measured NanoVDB node budgets (16 leaves, 4.5 lower nodes per world, 80+ upper; 4x the peak of 600 random-action policy steps) | Franka-Pour `mpm_grid_budget="measured"` (opt-in; written on local branch `franka-pour-memory`, merged with the Franka-Pour task in PR #6, `fe3c1f2f`) | 24.85 | 1.79 s |
| + cell reservation 256 per world (peak use 80) | Franka-Pour `mpm_cells_per_world=256` (opt-in) | 15.66 | 1.44 s |

The two task options are capacities: a grid within them is the same grid, and an overflow is reported (the PPO loop
now calls `check_status` once per iteration). They are opt-in because the budgets were measured with random
actions, not a trained policy (pouring spreads particles further), and the cell reservation departs from Isaac's
capacity rule. With both, the Metal allocation at 2,048 worlds is 15.7 GB, which would fit the L4's 24 GB on this
accounting (the L4 run itself was not repeated). Not changed: collision capacities (an overflow drops contacts
silently, and earlier traces of real Franka-Pour states saw 1.1-5 M triangle pairs per collide, far above the
random-action peak), MuJoCo Warp's `njmax` / `nconmax` (Isaac's), and the graph-held buffers (the graph's replays
read them).

## 8i. Franka-Pour as a full task: environment + PPO (2026-10-12)

Section 7b's probe mirrored the scene with scripted cup motion; `metalsim/learn/franka_pour*.py` now runs Isaac's task
end to end: Isaac's own assets (Franka USD, cups, table, the 20,000-row reset dataset), Isaac Lab's coupled solver
(`SolverCoupledProxy`: MuJoCo Warp arm entry + this section's `SolverImplicitMPM` as the media entry, proxy mass x
1000, staggered), Isaac's MDP and the task's rsl_rl PPO. Parity rows and tests: `docs/PARITY.md` §1b; reading notes:
`docs/research/franka_pour_2026-10-11.md`; usage: `docs/GUIDE.md` (Franka-Pour).

* `newton_compat.install(accumulate="float32", loop="cuda", chunk_graph, gpu_exit, metal_fast_math="off",
  device_args=None)`; rigid determinism configurable (`PourPhysicsConfig.rigid_deterministic`, default off as Isaac's
  `NewtonCfg.deterministic_mode`).
* Capture: one graph per `sim.step` (collide + 2 coupled steps: 6 MuJoCo + 4 MPM solves), as Isaac's
  `_simulate_physics_only`; the arm-target kernel runs between replays. A single graph per policy step (4 sim steps,
  `PourPhysicsConfig(capture_unit="policy")`) replayed NaN joint state at 2,048 worlds on Warp `152808b7` / `43ffc6dd`.
  It was correct at 1,024 worlds. Fixed in Warp `41b7bc45`: the capture held every step's temporaries (70.7 GB, past
  the 55.7 GB working set). Now 37.4 GB and finite at 2,048 worlds; see sections 8g-8h.
* Per policy step at 2,048 worlds (M4 Max): about 2 s (step only); the L4 cannot hold Isaac's own task at 2,048 envs
  (CUDA out of memory creating the graph, and out of memory eager), so the matched comparison is at 1,024 envs (§1b.4).
* Section 8h's opt-in memory budgets (`mpm_grid_budget="measured"`, `mpm_cells_per_world=256`) hold on the trained
  policy: a deterministic 2,048-episode evaluation of the 300-iteration policy gives success 0.470 vs 0.477 with
  Isaac's capacities (same rows; within the 0.011 standard error), the same pour metrics, and no grid overflow (`docs/PARITY.md` §1b.5, `runs/franka_pour/eval/`).
* Learning against Isaac (`docs/PARITY.md` §1b.5, 2026-10-13): 5 of our seeds at Isaac's matched 1,024-env
  configuration against Isaac's single run. Isaac's run is inside our prediction interval through iteration 49. At
  iterations 50-59 it leads by about 0.06 success, just outside the interval (verdict partial). Isaac seeds 1-4
  (`stage_pour_seeds.sh`) were waiting for L4 time; seeds 1 and 2 were started on the L4 later on 2026-10-13 (in
  flight, `docs/NOW.md`; no result recorded yet).
* Upstream observation (code reading, same in Isaac and here): the proxy wrench harvest uses only the last MPM
  substep's impulses and divides by the outer dt, so with the media entry's 2 substeps the reaction on the cups is about
  half the mean force (`solver_coupled_proxy.py:1547-1557`, `solver_implicit_mpm.py:1773-1812, 3266-3268`).

## 8j. Coupled scenes against the L4, re-timed (2026-10-13)

Logs, JSON and scripts: `runs/mpm/coupled_retime/`. The run used the 16-inch M4 Max (macOS 26.7.1), main `52469d93`,
Warp `95f6f121`, Newton `3fd1f052`, MuJoCo Warp 3.11.0 (`.venv-newton152`, `env_check.log`).
* Every run was its own exclusive `timing` job through `scripts/gpu_run.sh`. Utilisation was 0 % at the start of
  each job, with no process at 10 % or more.
* Three repetitions were interleaved: for each repetition, for each mode, for each scene, run the L4's protocol, then
  the `speed_suite.sh` protocol.
* The L4's protocol (`stage_mpm_timing.sh`) is `record_mpm.py`, 2 s, no per-solve log. The figure is the median
  ms/frame over the whole run.
* "Captured" is the example's own whole-step graph.

The L4 figures are those of section 7 (Isaac Lab 3.0-EA's Newton 1.5.2, Warp 1.16; the L4 log tarball is not
committed). An 8-frame check after 3 frames of warm-up is committed with the L4 kernel profiles
(`runs/mpm/profile/kernels_l4/timing_*.json`): two-way 68.9 / 29.7, ANYmal 56.6 / 23.2 (eager / captured).

ms/frame, median of the three per-run medians; `"off"` is the parity default:

| scene | L4 | Metal `"off"` (runs) | Metal / L4 | Metal `"relaxed"` | Metal / L4 | Metal frames 6-65 (`"off"`) |
|---|---|---|---|---|---|---|
| two-way, eager | 73.0 | 79.6 (79.6 / 78.8 / 79.6) | **1.09x** | 76.8 | 1.05x | 62.1 |
| two-way, captured | 31.0 | 54.4 (53.1 / 54.4 / 54.9) | **1.76x** | 52.6 | 1.70x | 36.9 |
| ANYmal, eager | 58.9 | 39.6 (39.0 / 39.6 / 41.3) | **0.67x** | 37.5 | 0.64x | 39.9 |
| ANYmal, captured | 24.2 | 31.2 (31.2 / 31.3 / 29.8) | **1.29x** | 27.8 | 1.15x | 29.7 |

The spread between repetitions is at most 4 %. The eager L4 figures include CUDA's per-launch host cost (23 us per
eager launch on the L4, 1 us per graph node; `kernels_l4/chain.json`), which is why Metal leads ANYmal eager. The
fair comparison is the captured rows, and there Metal is slower on both scenes.

**Where the time goes** (`profile.sh`; per-kernel GPU time with each dispatch isolated, calibrated for the empty
command-buffer cost, joined with the L4's nsys sums in `profile/joined.md` and `joined_late.md`):

* **Two-way: the frame cost grows during the run on Metal only.** Mean ms/frame per 20-frame window, captured,
  1st repetition: 36 at the start, 50-66 from 0.2 s, then about 53. On the L4 the 2 s median (31.0) is about its
  early cost (29.7). Synchronized parts, chunk-graph eager (`profile_example_parts.py`): early 72.0 ms (MPM step
  49.0, MuJoCo 13.5, body forces 4.2, collide 3.1); late (after 140 frames) 93.8 ms, with the MPM step at 67.2.
  * The growth is one kernel, `_bsr_mv_listed_rows`: 0.6 ms/frame early and 19.3 ms/frame late, over 100 launches
    per frame. It is the rigidity correction inside the rheology loop: two products per iteration,
    `IJtm @ delta_impulse` and `J @ delta_body_qd`.
  * Newton `80e79684` (section 8f) switched both products to a one-thread-per-row loop on Metal whenever J has at
    least 8,192 rows. That loop is Warp's off-CUDA `bsr_mv` body.
  * IJtm has one row per body degree of freedom, and each row holds one block per collider node the body touches. As
    the boxes sink into the sand those rows grow to thousands of blocks, and each is summed serially by one thread.
  * On CUDA, `bsr_mv` reduces such rows with its tiled kernel: `bsr_mv_tiled_kernel` (51 launches) and `bsr_mv_kernel`
    (51 launches), 1.45 ms/frame in all (early window).
  * This attribution is inferred from the launch count and the matrix shapes. The profile does not separate the two
    products.
  * Rest of the early gap: Gauss-Seidel 7.9 vs 3.2 ms (latency-bound, section 7), Metal's fixed-order body-force sum
    `compute_body_forces_lanes` 3.0 ms (the L4's atomic `compute_body_forces` is 0.04 ms; section 7b measured
    the fixed-order sum at 0.45 ms, why it is 3.0 ms here was not investigated), friction / impulse kernels 1.5-3.4x. Sorts, scans and BSR assembly are faster on
    Metal (6.0 vs 15.5 ms).
  * Kernel sums: early 31.9 ms (Metal, calibrated) vs 28.9 ms (L4); late 55.3 ms (Metal).
* **ANYmal captured** (kernel sum 21.6 ms vs the L4's 10.5): Gauss-Seidel 8.5 vs 3.3 ms (the same latency-bound color
  passes as every MPM scene, 400 launches). The MuJoCo Warp side is many small kernels at 3-11x the L4's time
  (`_crb_accumulate`, `_cfrc_backward`, `_primitive_narrowphase`, Cholesky updates). Sorts and scans take 2.3 vs
  1.6 ms. Parts: MPM step 25.6 ms, MuJoCo step 18.4 ms of 45.4 (eager, synchronized).

**What would close it** (as written on 2026-10-13; item 1 and the body-force sum are done, section 8k):
1. **Two-way growth.** On Metal, run the long-row product (IJtm) through a row-tiled reduction (CUDA's
   `bsr_mv_tiled_kernel` choice, as Warp `3c4e251e` did for warp.fem's dispatched assembly), or swap the IJtm product
   for a J^T product over J's short rows. Expected: the late window back to about the early cost (36-37 ms), giving
   two-way captured about 1.2x the L4. It changes summation order. A CUDA-faithful tiled order would bring Metal
   closer to the L4, but the floor ensemble must be re-run either way.
2. **The remaining ~1.2x.** The Gauss-Seidel color pass is near its latency floor at equal results (sections 7, 8).
   The MuJoCo Warp small-kernel cost belongs to the rigid stack's own performance work, not to MPM. The body-force sum
   (3.0 ms here against 0.45 ms in section 7b) needs a look first.

## 8k. Coupled scenes: closing the gap (2026-10-13; Warp `31cf79d7`, Newton `addd10cb`)

Research: `research_notes/coupled_speed/NOTE.md`. Logs, JSON and the batch scripts: `runs/mpm/coupled_speed/`
(batch_02 to batch_08, bf1/, hv1/, any1/, b08/). Every timing ran on the 16-inch as an exclusive `timing` job on an
idle GPU, with three interleaved repetitions unless noted. ms/frame figures use the L4's protocol (2 s, median).

**1. The two-way growth was the rigidity product IJtm** (section 8j, item 1). `probes/rigidity_mv_probe.py` times
the two in-loop products separately (eager, synchronized):
- IJtm has 12 rows (two per body), and the longest row grows to about 1,170 blocks as the boxes sink.
- Untiled, it takes 29-40 ms per 50 calls. J (655,360 rows, at most 2 blocks each) takes about 9-12 ms, either way.

Warp's `bsr_mv` sums such rows with its tiled kernel on CUDA only (`nnz * block_size > 2 * 64 * rows`; the L4's nsys
profile shows one tiled and one untiled product per iteration). Metal summed every row serially in one thread.
- Warp `31cf79d7`: `warp.config.metal_bsr_mv_tiles` (default on, `WP_METAL_BSR_MV_TILES=0` off) makes CUDA's choice
  on Metal. Metal's `tile_sum` is CUDA's shuffle tree, so the summation order is the L4's.
- Newton `addd10cb` (first of three commits): `_ListedRowsMv` sends a matrix that `bsr_mv` would tile through `bsr_mv`
  itself.
- First launches: `probes/bsr_mv_tiles_check.py`, under a watchdog.
- IJtm: 29-40 -> 12.1-12.9 ms per 50 calls.
- Late two-way GPU kernel sum per frame: 55.3 -> 36.4 ms.

**2. The ordered body-force sum** (two-way example, 3.0 ms per frame). Every body scanned all 655,360 grid nodes, in
every substep.
- Newton `addd10cb` (third commit): compact the nodes in contact with a body first (flag, integer scan, scatter;
  capture-safe), then the 256 fixed lanes run over that list.
- 1.05 -> 0.38 ms per call (`probes/body_force_compact_check.py`, exact compaction, 5e-7 from float64).
- `NEWTON_METAL_BODY_FORCE_COMPACT=0` restores the full scan.

**3. The proxy-wrench harvest** (`SolverImplicitMPM.coupling_harvest_proxy_wrenches`, used by Franka-Pour's
`SolverCoupledProxy` and section 11's MPMCoupling). With one world, every proxy body scanned every active collider
node.
- Newton `addd10cb` (second commit): the same compaction, per world range.
- `probes/harvest_compact_check.py`: one world, 16 or 64 bodies, 1.6 -> 0.33 ms per call.
- `NEWTON_METAL_HARVEST_COMPACT=0` restores the full scan.

**Results** (ms/frame, Metal / L4):

| scene | before (8j) | IJtm tiled | + body-force compaction | L4 |
|---|---|---|---|---|
| two-way, captured | 54.4-54.8 (1.76-1.77x) | 40.0-40.4 (1.29-1.30x) | **37.6 (1.21x)** | 31.0 |
| two-way, eager | 78.8-79.6 (1.08-1.09x) | 61.5-62.1 (0.84-0.85x) | **61.3 (0.84x)** | 73.0 |
| ANYmal, captured (control) | 29.9-31.2 (1.23-1.29x) | 29.4 (1.21x) | unchanged | 24.2 |

MPMCoupling (`probes/multibody_sand.py`, 1 world, about 73k particles), harvest compaction off -> on:
- The harvest alone: 15.6-15.8 -> 1.5-1.7 ms per call.
- Coupled step: 47.7 -> 32.3 ms (4 bodies), 49.3 -> 34.2 ms (16 bodies), 80.7 -> 65.2 ms (64 bodies).

Franka-Pour at 2,048 envs is unchanged: 1.80 s per policy step with the tiled choice on or off, and with the harvest
compaction on or off.

**Results: bitwise and parity.**
- Pure-MPM floor scenes (granular, multi_material, plate drag, column) and ANYmal are bitwise unchanged; they use no
  `bsr_mv` and no proxy harvest.
- Two-way and Franka-Pour change rounding, because the summation order changes:
  - for IJtm, it becomes CUDA's order;
  - for the two compactions, it becomes another fixed order. The L4 sums these with float atomics, so there is no CUDA
    order to match.
- All of them repeat bitwise run to run: two-way recordings, Franka-Pour 64 envs × 30 steps, MPMCoupling 30 steps.
- Two-way stays inside the L4's floor ensemble (worst-frame ratio 0.83, as before).
- Each switch off restores the previous behaviour bitwise:
  - Franka-Pour with the tiled choice off equals the trees without the change (batch_06);
  - with the harvest compaction off, it equals the trees without that change (batch_hv1).
- On the pinned commits: `tests/test_mpm.py` 15 passed, 1 xfailed; `tests/test_mpm_main.py` 15 passed. The vendored
  `implicit_mpm` equals `addd10cb` (`scripts/vendor_mpm152.py --check`).

**ANYmal: where the 1.2x is** (`probes/anymal_probe.py`, per frame, captured robot graph + eager sand step):
- Robot 8.3 ms, sand 22.2 ms.
- The colour launch with 1 / 2 / 4 / 8 threadgroups per GPU core gives bitwise equal states; 2, the current choice, is
  already the fastest (1 per core: sand 26.2 ms).
- Newton's reordered Gauss-Seidel (`solver="gs-soa"`) is bitwise equal to `"gs"` but slower (sand 26.0 ms).
- On the L4, the nsys kernel time for the same frame is about 10.5 ms (MPM about 9.1, robot about 1.4) against a
  24.2 ms frame, so most of the L4's frame is host launch cost of its eager sand step.
- What remains on Metal:
  - the MuJoCo Warp robot graph: 8.3 ms against about 1.4 ms of L4 kernel time; that is the rigid stack's own work;
  - the latency-bound Gauss-Seidel colour passes (sections 7, 8).
- The "captured" comparison is like for like: the example captures the sand step only for the `"fixed"` grid and only
  on CUDA, and the default grid is `"sparse"`. So both GPUs capture only the robot.

**Left on two-way** (late profile, `bf1/profile/joined_late_bf.md`):
- Gauss-Seidel: 7.0 vs 3.2 ms.
- Warp's tiled `bsr_mv`: 1.33 vs 0.12 ms over 51 launches. Only 12 rows x 3 x 64 lanes run, so the GPU sits mostly
  idle. A larger tile would cut it but would leave CUDA's summation order.
- Friction / impulse kernels: 1.5-3.4x.

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
scripts/setup_newton152.sh          # .venv-newton152: Newton fork metalsim-1.5.2 @addd10cb, shared Warp fork @31cf79d7
scripts/gpu_run.sh test_mpm check 15 -- .venv-newton152/bin/python -m pytest tests/test_mpm.py   # 15 passed, 1 xfailed
```

| what | where it lives | how a fresh setup gets it |
|---|---|---|
| Warp changes | pulipakaa24/warp `metalsim`, pinned `31cf79d7` | `scripts/setup_warp.sh` (fallback: innate-inc/warp `ce15f6bb` + `patches/warp/metalsim-31cf79d7.diff`) |
| Newton changes (MPM, Franka-Pour rigid) | pulipakaa24/newton `metalsim-1.5.2`, pinned `addd10cb` | `scripts/setup_newton152.sh` (fallback: v1.5.2 + `patches/newton/metalsim-1.5.2-addd10cb.diff`) |
| Newton 1.7 Metal fixes for the rigid stack | pulipakaa24/newton `metalsim-hashtable-metal` `a4be3604` (`.venv`'s `newton` extra) and `metalsim` `9f317390` (`.venv-newtonfork`) | `pip install -e ".[newton]"`, `scripts/setup_newton.sh` |
| compat layer, scenes, probes, tests, Isaac-side recorders | this repository | the clone |
| logs, JSON, reports of every run | `runs/mpm/`, `runs/parity/isaaclab3_mpm*/`, `runs/rigid_perf/`, `runs/fma_rigid_parity/` | the clone |
| Metal particle recordings (`.npz`, ~6.6 GB) | not committed | `record_metal_suite.sh`, `record_variant_suite.sh OUT - - .` (one GPU job per run; deterministic, so a re-recording is bitwise the original on the same trees) |
| L4 recordings (`runs/parity/isaaclab3_mpm*`, ~31 GB incl. tarballs) | not committed | on the L4 machine: copy `record_mpm.py`, `record_mpm_il3.py`, `scenes.py`, `newton_compat.py` to `~/parity3_scripts/mpm/`, run `isaaclab3/stage_mpm.sh`, `stage_mpm_floors.sh`, `stage_mpm_floors2.sh`, `stage_mpm_timing.sh` (Isaac Lab 3.0-EA install: `isaaclab3/`) |
| kernel caches, L4 scale logs > 1 MB | not committed | rebuilt on first run / re-run `probes/scale_pour.py` on the L4 |
| Franka panda model | MuJoCo Menagerie (not vendored) | `git clone https://github.com/google-deepmind/mujoco_menagerie upstream/mujoco_menagerie` or `PANDA_XML` |

## 11. MPM in the main simulator (`.venv`, BatchSim)

Design and research: `research_notes/MPM in main simulator/NOTE.md` (status: `STATUS.md` there). The main environment
has Newton 1.7, whose own implicit MPM is a later algorithm than Isaac Lab 3.0-EA's, so MetalSim carries the 1.5.2
fork's solver package as a generated, Apache-2.0 copy in `metalsim/physics/mpm/implicit_mpm/` (licence and the list of
modifications: its `NOTICE.txt`, `THIRD_PARTY_NOTICES.md`). `metalsim.physics.mpm.solver.solver_class()` returns
Newton's own `SolverImplicitMPM` on the 1.5.2 line (`.venv-newton152`) and the copy elsewhere
(`METALSIM_MPM_SOLVER=newton|vendored` forces one); `newton_compat.install` patches both. The copy run against Newton
1.7 is bitwise Newton 1.5.2's own solver (column a = 1, every frame, positions, velocities and iteration counts;
`runs/mpm_main/vendored_bitwise/`, `tests/test_mpm_main.py::test_vendored_column_bitwise_newton152`).

### Using it

```python
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.physics.mpm.coupled import MPMCoupling, MPMConfig, MPMMaterial, GridEmitter, PointsEmitter, SAND

sim = BatchSim(model, num_envs=64, options=BatchSimOptions(substeps=3))     # the coupled step = one BatchSim step
mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.02))   # Isaac's MPMSolverCfg fields + coupling settings
mpm.add_emitter(GridEmitter(lower=(-.1, -.1, 0), upper=(.1, .1, .05), material=SAND))   # Isaac's MPMGridCfg
mpm.set_colliders(bodies=["foot_l", "foot_r"])       # optional: which MuJoCo bodies are coupled (default: all moving)
mpm.finalize()
v = mpm.step()                                       # instead of sim.step(); returns sim.event's value
renderer = Tier2Renderer(model, 64, ...); mpm.attach_renderer(renderer)   # grains from the solver's buffer
mpm.wrenches(); mpm.particles(); mpm.reset(mask); mpm.check_status()
```

`metalsim/physics/mpm/coupled.py` documents the exchange: Isaac Lab's `SolverCoupledProxy` semantics (proxy bodies
with MuJoCo's effective mass, staggered or lagged, the solver's own fixed-order wrench harvest into `xfrc_applied`,
the caller's `xfrc_applied` restored after the step), `proxy_mass="effective" | "body" | "kinematic"`,
`harvest="last_substep"` (Isaac) or `"mean"`. Colliders are the MuJoCo geoms (`colliders.py`: box, sphere, capsule,
cylinder, plane, mesh, ellipsoid as a mesh); static geoms are static colliders. Nothing in `BatchSim` changes, and
building an MPM scene changes no process-wide Warp setting (`test_main_sim_settings_untouched`).

### Gauss-Seidel with separate worlds (Newton fork fix)

Two identical multi-world runs differed from the first step whenever the rheology solve was Gauss-Seidel (`"auto"`,
Newton's default) on a sparse grid with `separate_worlds`, in Newton's own 1.5.2 solver as in the copy; Jacobi
(Isaac Lab's Franka-Pour) and single-world runs repeated bitwise (`runs/mpm_main/repeat/`). Cause: the color blocks of
a discontinuous strain basis (P0) are built as `cell * nodes_per_cell`, while environment-first strain partitions
number a cell's nodes in environment-major order. A NanoVDB environment grid does not store cells environment by
environment (negative coordinates sort after positive ones), so the blocks were a permutation of other cells' nodes and
strain nodes solved concurrently in one color shared velocity nodes: a data race, on CUDA as on Metal. Fixed on the
fork (`99843d09`, blocks from each cell's partition node index; single-environment grids keep the old kernel) and
re-pinned. `probes/gs_coloring_check.py` (2 worlds, sparse grid, on Metal): the old fork's coloring has 930 velocity-node
write conflicts within colors (all 8 colors), the fixed one none, every strain node once per sweep; the fixed solver
repeats bitwise in all 12 multi-world configurations of `probes/multiworld_repeat.py` (`runs/mpm_main/batch_01/`).

### Measured (batch 01, 2026-10-13, 16-inch)

* **Isaac's coupling path** (box 2 kg / 0.1 m dropped on a 0.3 x 0.3 x 0.08 m bed, 2 worlds, Jacobi, 60 steps of
  1/120 s; `probes/box_on_sand.py`, `.venv` vs Newton's `SolverCoupledProxy` + `SolverMuJoCo` in `.venv-newton152`):
  every particle bitwise equal until the box reaches the sand (step 17); afterwards the two rigid stacks (MuJoCo Warp
  3.14 vs 3.11) differ by rounding: box position at most 1.7e-7 m apart, sand 1.0e-6 m RMS (max 2.5e-5 m) after 0.5 s.
* **Repeatability**: the coupled scene (box on a 2,592-particle bed per world) repeats bitwise at 1, 64, 256 and 1,024
  worlds with Gauss-Seidel. With Jacobi at 1,024 worlds the box reaches the floor and the run diverged at step 34:
  MuJoCo Warp's constraint solve is not run-to-run repeatable on Metal by default (MetalSim's rigid stack never was;
  `probes/rigid_neutral_mpm.py`: the panda differs between two plain runs; a box resting on the floor alone diverges
  at step 18 at 1,024 worlds). An opt-in deterministic `BatchSim` (Warp's RUN_TO_RUN mode for MuJoCo Warp's kernels)
  is being developed separately on branch `batchsim-deterministic` (open: a divergence in MuJoCo Warp's box-box CCD
  kernel under the two-phase deterministic counters, 2 of 256 worlds from step 139).
* **Rigid neutrality**: after an MPM scene was built and stepped, the MuJoCo Warp kernel modules a rigid scene loads
  have the same hashes (8 modules), and the box scene's trajectory is bitwise that of a plain run.
* **Speed** (ms per captured coupled step = 3 MuJoCo substeps + 1 MPM step of 1/120 s, box + 2,592 particles per
  world, median of 40 after 10 warm-up; `probes/scale_coupled.py --time`, timing class):

  | worlds | Gauss-Seidel (`"auto"`) | Jacobi | world-steps per s (GS / Jacobi) |
  |---|---|---|---|
  | 1 | 13.7 | 8.3 | 73 / 120 |
  | 64 | 24.3 (eager: 95.1) | 20.3 | 2,631 / 3,156 |
  | 256 | 60.6 | 54.7 | 4,226 / 4,678 |
  | 1,024 | 223.6 | 210.5 | 4,579 / 4,865 |

* **Bearing** (research `research_notes/MPM in main simulator/bearing_and_sinking.md`; runs `runs/mpm_main/bearing/`
  (batch 02), `runs/mpm_main/bearing3/` (batch 03); box 0.1 m, 2 kg on a 0.3 x 0.3 x 0.08 m bed, voxel 0.02).
  Expected: bearing capacity 5-10x the box's 2 kPa (mu 0.7 ~ phi 30-35 deg); a box at rest settles within about half a
  voxel; a 1.7 m/s drop stops after 1-2.5 cm; the bed fails below mu ~ 0.3-0.45.
  * **`"staggered"` exchange (Isaac's Franka-Pour setting): broken for resting bodies.** In every configuration of
    batch 02 (voxel 0.02 / 0.01, 50 / 250 iterations, friction 0.4 / 0.7 / 0.9, bed 8 / 16 cm) the resting box sank at
    a steady 0.082 m/s = g dt while the harvested sand force equalled its weight, i.e. through the bed. Cause: the MPM
    solve stops the proxy, but only the wrench reaches MuJoCo, in the next step, so v_{n+1} = -g dt whatever the sand.
    Isaac's coupling path does the same (MetalSim's Isaac-path comparison ran staggered): Isaac parity is not physical
    validity here. Isaac's Franka-Pour cups (held by the gripper) carry the same per-step drift relative to the sand.
  * **`"lagged"` exchange with the velocity rewind** (Newton's `SolverCoupledProxy.Proxy` default and its two-way
    example's scheme; `MPMCoupling`'s default since batch 03) removes the drift. Rest test, settlement after 1.5 s /
    creep over the last 0.3 s: 50 iterations, tolerance 1e-4: 25.3 mm / 16.1 mm/s; **250 iterations, tolerance 1e-6:
    10.2 mm (half a voxel) / 0.4 mm/s**; voxel 0.01: 19.7 mm / 5.8 mm/s; collider basis Q1 27.0 mm / 2.1 mm/s, pic27
    28.3 mm / 8.0 mm/s; friction 0.3: 74.2 mm (bearing failure, as predicted), 1.2: 14.2 mm / 11.6 mm/s. Drop from
    0.15 m: peak 38.8 mm, creep 8.7 mm/s (dt 1/240: 31.7 mm, 4.1 mm/s). The residual creep follows the rheology solve's
    convergence (an under-converged solve delivers a fraction alpha of the stopping impulse and drifts at
    (1 - alpha) / alpha g dt): bodies resting on sand need a converged solve (Isaac's default is 250 iterations,
    tolerance 1e-4; batch 04 measures it). Staggered in batch 03: 80 mm (the floor) in drop and rest.
* **Angle of repose** (research `research_notes/MPM in main simulator/angle_of_repose.md`; `probes/repose_lift.py`, runs
  `runs/mpm_main/repose3/`). Lifted-cylinder test: tube R = 5 cm (mocap, frictionless towards the sand), fill 5 cm, rough
  floor (friction 1.0), column-maximum surface fit in 8 sectors (30.0 deg on a synthetic 30 deg cone). The solver's
  yield surface is sqrt(J2) <= mu p, so the reference is atan(mu). Measured (fit mean, sector sd 0.0-0.4 deg):

  | mu (atan mu) | R/dx 5 | R/dx 10 | R/dx 15 |
  |---|---|---|---|
  | 0.40 (21.8 deg) | | 14.7 | |
  | 0.68 (34.2 deg) | 25.4 | 26.3 | 27.2 |
  | 0.90 (42.0 deg) | | 32.6 | |

  Lift speed 1.25 / 2.5 / 5 cm/s at R/dx 10: 26.6 / 26.3 / 25.4 deg (plateau within 0.3 deg below 2.5 cm/s); 4x the
  iterations: 26.6 deg. The angle rises monotonically with resolution (+0.9 deg per refinement) and with friction
  (slope 0.89 against atan(mu)), but sits 7-9 deg below atan(mu) at the finest resolution run: the published MPM heaps
  also stop at or below atan(mu) (Dunatunga & Kamrin 2015), and on a slope this cone can yield below the simple-shear
  stopping angle whenever the lateral stresses differ from the vertical one; the remaining gap to physical sand
  (repose ~ the friction angle) is a property of the solver Isaac Lab ships, not of the coupling. Not yet converged in
  resolution (R/dx 30 is the next step).

### Re-vendoring when the fork pin moves

The copy must always be the commit `scripts/setup_newton152.sh` pins. After a Newton fork change:

1. Push the fork branch (`metalsim-1.5.2`) and move `PIN` in `scripts/setup_newton152.sh` (and the fallback patch
   `patches/newton/metalsim-1.5.2-<sha>.diff`, as before).
2. Set the same commit as `PIN` in `scripts/vendor_mpm152.py`, then regenerate:
   `python scripts/vendor_mpm152.py upstream/newton-metalsim-1.5.2` (a checkout at the new pin; without an argument it
   downloads the pinned commit's archive from GitHub). Never edit the generated files by hand.
3. Update the commit in `THIRD_PARTY_NOTICES.md` (Newton section) and run `pytest tests/test_mpm_main.py`:
   `test_vendored_pin_is_the_setup_pin` fails until both pins agree, `test_vendored_copy_regenerates_from_the_pin`
   regenerates from the pinned commit and fails on any difference, `test_vendored_files_keep_the_licence_header`
   checks every file still starts with Newton's SPDX header and the modification notice.
4. If the fork's package now imports something new from outside `implicit_mpm`, extend `rewrite()` in the generator
   (and the modification list in its `NOTICE_TXT`).
