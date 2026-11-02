# Changelog

What landed between 2026-09-22 and 2026-10-13, grouped by area. The September sections list one line per commit in
commit order; the October sections, newest first, summarise each merged pull request with its main commits (`git log`
for the full messages). Dates are the commit dates (month-day).

## Physics: heightfield contacts against the exact geometry (2026-10-31, `docs/research/hfield_contacts_2026-10-31.md`)

- 10-31 MuJoCo Warp pin [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624) -> [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) (the `metalsim` branch of [my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) fast-forwarded by six commits; `patches/mujoco_warp/0018-0023`, whole-branch diff `metalsim-93e5ce1.diff`): a heightfield prism is a plain convex polytope as in [MuJoCo](https://github.com/google-deepmind/mujoco) C ([upstream MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)'s side-wall override returned penetrations for prisms clear of the geom), the prism support returns distinct vertex indices (discrete EPA stopped early for boxes and meshes; MuJoCo C has the same defect), only penetrating prisms are kept, contacts 1-3 skip duplicates instead of stopping within 1 mm, margin and gap as MuJoCo C main, and meshes take GJK/EPA instead of the plane path (`HFIELD_PLANE_CONTACTS=True` restores it). Contact sets on [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s rough tile equal the exact ones in 99.9-100 % of placements (before 53-89 %); Unitree's G1 on Isaac's bumps falls like on CPU MuJoCo; about -15 % of the G1 rough env step under random actions, 1.2-1.3x per PPO iteration in the Isaac Lab 2.3.2 rough configs and 1.68x in rough_il3 (indicative; 2.05x on [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147), which also dropped heightfield contacts with EPA_HORIZON when a geom gap was set: fixed by the EPA horizon and margin-shrink commits, next line, [research note](docs/research/hfield_contacts_2026-10-31.md) §12). Rough-terrain results on MuJoCo Warp before this pin: re-run on 10-31 where no other machine is needed (last line of this section), the rest pending ([PARITY](docs/PARITY.md) §1.4, §1.5, §1.8, [GAPS](docs/GAPS.md)).
- 10-31 MuJoCo Warp pin [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) -> [`787a5b3`](https://github.com/pulipakaa24/mujoco_warp/commit/787a5b3a37acc2551ffc8a2c1af1e9781d76723f) (my fork's `metalsim` fast-forwarded by eight commits; `patches/mujoco_warp/0024-0031`, whole-branch diff `metalsim-787a5b3.diff` replaces `metalsim-93e5ce1.diff`): EPA's horizon as MuJoCo C builds it: a face-by-face scan whose horizon is not one loop is undone and the region grown from the closest face through shared edges, the horizon buffer holds every edge of EPA's largest polytope (`3 (5 + ccd_iterations) - 6`, was 24), and a horizon that cannot be built (buffer full, or the region still not a disk) gives no contact, as MuJoCo C on horizon failure; a full face buffer keeps the closest face, as MuJoCo C, where the contact was dropped without a flag. Heightfield prism pairs with a margin or gap run GJK on the uninflated shapes first and EPA only when they overlap, as MuJoCo C's `mjc_ccd` (`collision_gjk.HFIELD_MARGIN_SHRINK`; off with `HFIELD_PRISM_POLYTOPE = False`; both read at import). On [`93e5ce1`](https://github.com/pulipakaa24/mujoco_warp/commit/93e5ce18e76854c430ce669314af00ddceb92147) Isaac Lab 3.0's like-for-like rough config (`rough_il3`, 1 cm geom gap) flagged `EPA_HORIZON` and dropped those contacts; now none. Without a margin or gap only non-loop scans and full face buffers change (bitwise unchanged on the heightfield contact checks and 20,000 G1-foot placements on box stairs). New tests in my fork: `test_horizon_is_loop`, `test_epa_nonloop_horizon`, `test_epa_horizon_size`, `test_hfield_mesh_gap_distance`; `test_box_box_horizon` updated (research note §12).
- 10-31 `scripts/fork_sync.sh` fast-forwards the shared fork checkouts to the pinned commits (was: to the branch heads), finds the fork remote by URL, handles `upstream/warp-metalsim` (and the older `upstream/warp-innate`), syncs the main checkout's `upstream/` when run from a worktree, and has a dry run (`-n`); `tests/test_fork_pins.py` fails while a fork checkout, or the checkout the installed package comes from, is off its pin.
- 10-31 Rough-terrain re-runs on the repaired heightfield kernel (research note §13; PARITY §1.5, §1.8; the runs themselves are local): the 2.3.2 rough preset's seeds 0 / 1 / 2 +16.0 / +20.6 / +17.4 at 1500, levels 6.11-6.13 (plane path +16.4 / +19.1 / +22.6, 6.03-6.10; Isaac +14.1 / 5.90), inside the seed spread, with the curriculum later (level 5.48 vs 5.83 at 1000); `rough_il3` with the initial reset +12.0 / 5.93 (plane path +8.4 / 5.96; Isaac Lab on [Newton](https://github.com/newton-physics/newton) +14.5 / 5.80), 67 late blow-ups (187); transfer falls barely change (re-trained seed 0: 0 / 24, and 0 / 96 on the wall cells). Pending: Isaac-side transfers (the NVIDIA L4 machine), the §1.4 throughput rows (the 16-inch MacBook Pro), the elliptic-default and final `rough_il3` runs. PARITY §1.8's "before the fixes" sentence no longer pairs the mixed-preset and corrected-preset runs.

## Licence: Apache-2.0, NOTICE and CITATION.cff (2026-10-28, `LICENSE`)

- 10-28 MetalSim's licence changes from MIT to the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (`LICENSE`, `pyproject.toml`'s `license` expression `Apache-2.0`); copies taken earlier stay under MIT. New `NOTICE` (attribution that redistributions must keep, Apache-2.0 section 4(d)); `CITATION.cff` for GitHub's "Cite this repository". SPDX headers of the derived files: `MIT AND Apache-2.0` -> `Apache-2.0`, `MIT AND BSD-3-Clause` -> `Apache-2.0 AND BSD-3-Clause`. README, `THIRD_PARTY_NOTICES.md` and the vendored Newton package's `NOTICE.txt` (with its template in `scripts/vendor_mpm152.py`) updated to match.

## MPM: SIMD Gauss-Seidel sweep removed (2026-10-14, `metalsim/physics/mpm/MPM.md`)

- 10-14 `metalsim/physics/mpm/simd_gs.py`, `newton_compat.install(simd=)`, `record_mpm.py --simd`, `loop_breakdown.py --simd`, `ab_timing.py`'s `+simd` / `+simdgraph` variants and the xfail `test_simd_sweep_bitwise` removed. The sweep was opt-in and never the default; its +10 % with the chunk graph (MPM.md, 2026-10-09) came with bitwise equality to Newton's serial loop, which FMA contraction on Metal (in [my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim)) ended. (The pull request committed only the file deletion; the rest of the removal landed in its follow-up.)

## Demo tag demo-2026-10 (2026-10-14, `docs/DEMO.md`)

- 10-14 `docs/DEMO.md`: every demo with its command (each run by the fresh-install check or the batch checks it names), the measured numbers from the shipped docs and the known limits; tag [`demo-2026-10`](https://github.com/pulipakaa24/MetalSim/tree/demo-2026-10) on the main commit that carries it.

## Scenes: composition from a config file (2026-10-14, `docs/SCENES.md`)

- 10-14 `metalsim.scene` (`cfg`, `compose`, `proxy`, `film`) and `python -m metalsim.run scene.yaml`: robot (MJCF / URDF / USD), movable objects with per-env reset randomization, cameras mounted on links (tier 0 / 1 / 2; rgb, depth, segmentation, normals), ray-traced lidar, ground / HDRI / lights, and a Gaussian-splat background behind a collision proxy (plane, mesh, [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) floor plan or occupancy map); `--film` hands over to `metalsim.render.film`, `--viewer` opens MuJoCo's viewer. Field names follow Isaac Lab 3.0's InteractiveSceneCfg / ArticulationCfg / RigidObjectCfg / CameraCfg / RayCasterCfg. Examples `configs/scenes/`; tests `tests/test_compose.py`.

## Rendering: filming tool and the Franka-Pour render (2026-10-14, `docs/renders/`)

- 10-14 `metalsim.render.film`: one command films a registered task (`python -m metalsim.render.film <task>`): record the policy rollout in the task's venv (Newton or MuJoCo), build a portable render twin (MJCF + per-frame geom poses + particles), render offline with the tier-2 path tracer, encode H.264 + stills, and write `docs/renders/<TASK>.md` from the stage logs (timings, hardware, pins, assets with sources and licences). Camera and recording settings follow Isaac Lab's `ViewerCfg` / `VideoRecorderCfg`. Per task: a `FilmTask` entry and a recorder adapter. First films: Franka-Pour (trained policy, 2 shots) and the G1 flat walk (smoke test). `franka_pour_scene.build_model(visuals=True)` loads the USD visual meshes for the render twin (physics unchanged). Tests `tests/test_film.py` (CPU).

## Physics: deterministic BatchSim (2026-10-13, `research_notes/MPM in main simulator/deterministic_ccd_divergence.md`)

- 10-13 `BatchSimOptions(deterministic=True)`: MuJoCo Warp's kernels in [Warp](https://github.com/NVIDIA/warp)'s RUN_TO_RUN mode (set before MuJoCo Warp's first launch, one mode per process, record bound for `smooth`, post-capture check of every replayed module including the `module="unique"` ones, deterministic overflow and phase-mismatch flags in `check_overflow`). Two stacked boxes at 256 worlds repeat bitwise in one process and across processes.
- 10-13 MuJoCo Warp pin [`f1b35e2`](https://github.com/pulipakaa24/mujoco_warp/commit/f1b35e2b3ddc80b197b0c665bf6d71523d175daf) -> [`4d05cb7`](https://github.com/pulipakaa24/mujoco_warp/commit/4d05cb7d4453d82c94d19ab9fb392df2c62b1acd) ([my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim)): deterministic-mode CCD as a counter-free solve kernel plus a contact-write kernel (the single kernel's EPA scratch row came from a slot counter, so its counting pass ran EPA on stale scratch and a contact could land in another world's slot). Warp pin [`31cf79d7`](https://github.com/pulipakaa24/warp/commit/31cf79d781e4b10d8ff85c8cbd98fc0e40f75953) -> [`52c09cdd`](https://github.com/pulipakaa24/warp/commit/52c09cdd82f79e8564b5ae17cdcee7b52d10d075) ([my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim)): a slot-counter call that differs between the counting and the execution pass raises ("Deterministic counter phase mismatch").
- 10-14 MuJoCo Warp pin [`4d05cb7`](https://github.com/pulipakaa24/mujoco_warp/commit/4d05cb7d4453d82c94d19ab9fb392df2c62b1acd) -> [`8eb51fe`](https://github.com/pulipakaa24/mujoco_warp/commit/8eb51fe637437108185f5a4750b7f5f64ef26624): `_solve_done` keeps the solver's iteration count in a local instead of reading it back after incrementing it in place (an increment Warp's counting pass suppresses); a world reaching the iteration cap then made the early exit's counter call in one pass only, and the counter tripwire stopped deterministic G1 runs. In Warp's deterministic mode the Newton update takes the unfused launches (the fused kernel reads back its own change counter within the launch; its plan at 1,024 worlds asked for more than 2^31 scratch elements) (`research_notes/MPM in main simulator/deterministic_g1_capture.md`).

## Rendering: 3D Gaussian splat scenes (2026-10-13, `metalsim/render/SPLAT.md`)

- 10-13 `metalsim/render/splat/` + `shaders/splat.metal`: a batched compute-tile 3DGS rasterizer (cull, EWA projection with SH, 24-bit depth keys, 5-pass radix sort, subtile-masked blend) writing tier-0 layout buffers, GPU path without host sync, tier-0 mesh composite; loaders for [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) / [SuperSplat](https://github.com/playcanvas/supersplat) PLY, SPZ, `.splat` and [NuRec](https://docs.nvidia.com/nurec/) `.nurec`; float64 reference (EWA and [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) kernels); `make_transform` / `floor_plane_transform`; `scripts/fetch_splat_scenes.sh` (InteriorGS reference scene, gated, and a CC0 object, never committed) and `scripts/splat_bench.py`; the GPU path now follows the composited mesh renderer's camera and restores it after host-path calls.

## Newton 1.7: SolverMuJoCo on MuJoCo Warp 3.14 (2026-10-13)
- 10-13 Newton pins [`9f317390`](https://github.com/pulipakaa24/newton/commit/9f317390da07432dfb5813bd0db6a47eeb7cb330) -> [`1301e8da`](https://github.com/pulipakaa24/newton/commit/1301e8da2002541f6f189b6e459e1ed5e6db5731) (`metalsim` branch of [my fork of Newton](https://github.com/pulipakaa24/newton/tree/metalsim), `.venv-newtonfork`) and [`a4be3604`](https://github.com/pulipakaa24/newton/commit/a4be3604bc218bb366d12164fa3e1025f333bdae) -> [`9d6616c6`](https://github.com/pulipakaa24/newton/commit/9d6616c64ca018dd9119f67fca3316d715a26704) ([`metalsim-hashtable-metal`](https://github.com/pulipakaa24/newton/tree/metalsim-hashtable-metal), `.venv`'s `newton` extra): `SolverMuJoCo` passes the contact adhesion that MuJoCo Warp 3.14's `contact_force_fn` takes (upstream Newton's [`90f809a`](https://github.com/newton-physics/newton/commit/90f809add3ab32fca9dd7aac82ed6cd18236246a)). Before, `SolverMuJoCo`'s kernels did not compile against [my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim), and the Newton-engine and Newton-bridge render tests skipped in `.venv`.

## MPM: coupled step and resets in one policy-step graph (2026-10-13, `metalsim/physics/mpm/MPM.md` 11)

- 10-13 `MPMCoupling.launch_step()`, a capturable per-world reset (`reset_mask`, `launch_reset(from_sim_mask=True)`, `reset(mask)` with numpy or torch masks and a reset graph) and `BatchSim.launch_reset()`; `scale_coupled.py --split` per-section timings; `policy_step_ab.py`. The coupling layer is about 0.4 ms per coupled step; one graph per policy step saves 0.18 ms at 1, 1.20 ms at 64, 2.11 ms at 1,024 worlds, bitwise. One-world masked reset fixed (it raised).

## MPM: coupled scenes closer to the L4 (2026-10-13, `metalsim/physics/mpm/MPM.md` 8k)

- 10-13 Warp pin [`95f6f121`](https://github.com/pulipakaa24/warp/commit/95f6f121b7e67025958335c7c15e338ac04a798a) -> [`31cf79d7`](https://github.com/pulipakaa24/warp/commit/31cf79d781e4b10d8ff85c8cbd98fc0e40f75953) ([my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim)): `bsr_mv` takes CUDA's tiled-kernel choice on Metal (`warp.config.metal_bsr_mv_tiles`). Newton pin [`99843d09`](https://github.com/pulipakaa24/newton/commit/99843d09840f89ac16c82104ad66ae792e45eb91) -> [`addd10cb`](https://github.com/pulipakaa24/newton/commit/addd10cbeb25e946b3a4c2bb1d4097acc7a98ca2) (`metalsim-1.5.2` branch of [my fork of Newton](https://github.com/pulipakaa24/newton/tree/metalsim-1.5.2)): the long-row rigidity product goes through `bsr_mv`; the proxy-wrench harvest and the two-way example's body-force sum compact their nodes first. Two-way captured 54.8 -> 37.6 ms/frame (1.77x -> 1.21x the L4), eager 0.84x; MPMCoupling coupled step 49.3 -> 34.2 ms (16 bodies); Franka-Pour unchanged (1.80 s per 2,048-env step). Pure-MPM floors and ANYmal bitwise unchanged; two-way and Franka-Pour repeat bitwise, two-way inside the L4 floor. Vendored implicit_mpm regenerated. Logs `runs/mpm/coupled_speed/`, research `research_notes/coupled_speed/NOTE.md`.

## Documentation: audit against the measured record (2026-10-13)

- 10-13 Every doc checked against main as of 10-13: STATUS rewritten to the current state (superseded text kept and marked), GAPS / PARITY / GUIDE / README / RENDER_COVERAGE / NEWTON_RENDER / MPM.md / DECISIONS / PHASES / RESEARCH / HANDOFF / notices corrected (pins, queued or pending items that landed, test counts re-measured into `runs/docs_audit/`); this section and the October sections below added

## MPM: close-out bookkeeping and coupled-scene re-timing (2026-10-13, `metalsim/physics/mpm/MPM.md` 8j)

- 10-13 STATUS / GAPS MPM rows brought to the measured record (all six floor scenes inside, current pins, 1.7 barrier fix verified in both pinned commits); MPM.md superseded "Open:" lines marked resolved with pointers; the 10-12 02:32 GPU hang re-attributed (most likely the 25 M-instance particle build, not Newton's cylinder test; evidence in `runs/rigid17/out/`); RENDER_COVERAGE particle rows (merged, T0 / T2); coupled scenes re-timed on the 16-inch against the L4 (`runs/mpm/coupled_retime/`)

## Physics: equality block at njmax (2026-10-13, `research_notes/capacity_overflow_guard/g1_bound_and_lost_flags.md`)

- 10-13 MuJoCo Warp pin [`cf0f8ad`](https://github.com/pulipakaa24/mujoco_warp/commit/cf0f8adc8f99f0c559cf8580cb667952e473bd07) -> [`f1b35e2`](https://github.com/pulipakaa24/mujoco_warp/commit/f1b35e2b3ddc80b197b0c665bf6d71523d175daf) (the `metalsim` branch of [my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) fast-forwarded to [`f52488b4`](https://github.com/pulipakaa24/mujoco_warp/commit/f52488b461fb3d609e9706bdd752a7dbf3c04837): a connect / weld block ending exactly at njmax is written; one that does not fit leaves nefc > njmax and MuJoCo Warp flags it). MetalSim's workaround removed (latch counted `nefc >= njmax` for models with such blocks, `Bounds.njmax` kept a spare row); the reset latch stays. `patches/mujoco_warp/metalsim-f1b35e2.diff`; `tests/test_capacity.py::test_equality_block_ending_at_njmax_is_flagged` covers connect and weld

## Rendering: tilted DomeLight (2026-10-13, `research_notes/dome_light_azimuth/tilted_dome.md`)

- 10-13 Tier 2 applies a DomeLight's full orientation: the importer records `rot` = R_prim . P for a dome whose pole leaves +z (authored tilt, DomeLight_1 `poleAxis` / stage upAxis), `set_environment(rotation=)` uploads R . R_z(yaw) to `PTConsts.env_rot` (`FLAG_ENVROT`, kernel variant `ENV_ROT`), and the lookup, importance sampler and its pdf all use it; upright domes keep the yaw-only path (renders byte-identical to before, `scripts/diagnostics/dome_untilted_bytes.py`). Tests: `test_env_map_tilted_dome_lookup`, `test_env_map_tilted_importance_sampling`, `test_env_map_usd_tilted_dome_import`

## Learning: checkpoint loading (2026-10-13, `docs/research/ppowarp_checkpoint_load_2026-10-13.md`)

- 10-13 PPOWarp: a state dict loaded into `algo.net` is published to the Warp rollout policy (load_state_dict post-hook, same event ordering as `update()`); before, a replayed checkpoint ran the initial weights (`tests/test_ppo_warp_rollout.py::test_loaded_weights_reach_the_warp_policy`). Overflow-guard research note's trained-policy rows corrected and re-run; `scripts/diagnostics/g1_policy_replay.py`

## Physics: capacity overflow guard (2026-10-13, `research_notes/capacity_overflow_guard/g1_bound_and_lost_flags.md`)

- 10-13 BatchSim latches MuJoCo Warp's capacity flags after every substep (`reset_data` had erased the flags of the worlds it reset, so 13 of 31 overflowed worlds went unseen in a 100-step probe); NVMAX and TACTILE count as capacity flags; the four long-standing rigid-suite failures root-caused (three stale tests: the G1 capacity bound written under the pyramidal preset, an `njmax` that never overflows under elliptic cones, the dome azimuth test before the measured Kit offset)
- 10-13 Real-scale training checks of the latched guard (G1 flat / rough boxes / rough heightfield 4096, Tron1 8192, SO-101 lift 64, a trained rough policy over all terrain levels): no capacity flag; latch cost within run-to-run spread; suite logs on the merged tree (rigid 396 passed / 5 skipped, render 208 passed / 1 skipped, `runs/fixfail/`)

## Franka-Pour: Isaac Lab 3.0-EA's MPM manipulation task (2026-10-12 .. 13, `docs/PARITY.md` §1b)

- 10-12 IsaacContrib-Franka-Pour on Metal: scene from Isaac's own assets, Isaac's coupled solver settings, MDP, [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 5.4.1 PPO transcription; term tests against Isaac's own code (`tests/test_franka_pour_terms.py`, `tests/franka_pour_isaac_ref.py`); Isaac-side stage scripts for the L4
- 10-12 Complete resume state, STOP-file pause, cooperative training driver that yields to waiting GPU jobs
- 10-12 Throughput on the M4 Max at 2,048 envs (idle device: step 986, rollout 999, PPO loop 979 env-steps/s); 1,024 envs step 1,054 / loop 992 (L4: 317)
- 10-12 Matched learning curve against Isaac on the L4 (1,024 envs, 60 iterations); 300-iteration run at 2,048 envs (14-inch MacBook Pro)
- 10-12 Opt-in MPM memory budgets (`mpm_grid_budget="measured"`, `mpm_cells_per_world`); overflow check once per PPO iteration
- 10-13 Deterministic evaluation of the iteration-299 policy (2,048 episodes: 0.477; the memory opt-ins give 0.470); why evaluation reads 0.48 and training 0.62 (reset distribution and episode counting)
- 10-13 Seed study: 5 seeds ours at the matched configuration; Isaac's single run inside the prediction interval through iteration 49, about 0.06 ahead at 50-59 (verdict partial); `stage_pour_seeds.sh` for Isaac seeds 1-4 on the L4
- 10-13 Isaac seeds 1 and 2 on two L4 machines (`fingerprint.sh` gate against the seed-0 setup): 5 seeds ours vs 3 Isaac's, no window differs through iteration 49, 50-59 success 0.376 vs 0.411 (Welch p = 0.09); verdict still partial, the gap is at most ~0.04

## Rendering: integration, coverage audit, render gaps (2026-10-12 .. 13; the tilted DomeLight has its own section above)

- 10-12 Render coverage audit: every simulated feature against what each renderer draws (`docs/RENDER_COVERAGE.md`)
- 10-12 Appearance against MuJoCo's renderer: mesh normals (hard edges without duplicated vertices), SDF geoms, texture mapping, skybox, transparency, every light, spot / point shadows (RENDER_COVERAGE §3b)
- 10-12 Deformables, cables, tendons, skins and sites in every renderer and sensor; dynamic geometry in the ray tracers (`dyn_layout`: sbvh for sensors and tier 2, per_env for tier 1; merged tiles dropped) (§5)
- 10-12 World extent (infinite planes, far plane, heightfield sides), per-env fields (size, colour, lights), camera models (fisheye), compact env grid (§3c)
- 10-12 Newton → renderer bridge: Newton models, cloth / soft surfaces and free particles in every renderer (`docs/NEWTON_RENDER.md`)
- 10-12 MPM particles as render grains in tiers 2 and 0, two-level grain acceleration structures within Metal's limits (`metalsim/render/PARTICLES.md`); grain chunks of 2^24 - 1, 25 M-grain validation (14-inch); demo ground tint; 25 M-grain benchmark on the 16-inch with logs (`runs/sand25m_16in/`)
- 10-13 Render gaps 2: rolling shutter, per-env texture pool, MuJoCo haze / fog, tendon catenary and limit colour, heightfield-side shadows; ray kernels specialized per scene (§3d)
- 10-13 Transparent geoms in the ray-traced depth camera and lidar; true USD area lights in tier 2 and tier 1 (§3e; `research_notes/Render sensor fidelity/`)

## MPM: parity close-out, step at scale, pins (2026-10-12, `metalsim/physics/mpm/MPM.md` 8d-8h)

- 10-12 Parity close-out: multi_material and ANYmal inside the L4's floor ensemble; Warp pin [`43ffc6dd`](https://github.com/pulipakaa24/warp/commit/43ffc6dd544ccaa8ae18772ba67b0d292bf01371) ([my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim); CUDA's warp.fem tiling, `metal_cuda_libm`) (8e)
- 10-12 MPM step at 2,048 worlds: profile and four bitwise speedups, [my fork of Newton](https://github.com/pulipakaa24/newton/tree/metalsim-1.5.2)'s `metalsim-1.5.2` pin [`3fd1f052`](https://github.com/pulipakaa24/newton/commit/3fd1f052a5ad031578960d9642aaaedc03834e4d) (8f)
- 10-12 Warp pins [`85b92466`](https://github.com/pulipakaa24/warp/commit/85b924669629838c2d2637f3e1e51ef650b3a8d6) (capture memory reuse, page fault fixed), [`f0df4cf8`](https://github.com/pulipakaa24/warp/commit/f0df4cf8b291c0fcc5590d5286f0e7341a692ab1), [`95f6f121`](https://github.com/pulipakaa24/warp/commit/95f6f121b7e67025958335c7c15e338ac04a798a) (tight warp.fem temporary pools; Franka-Pour 37.6 -> 34.5 GB at 2,048 worlds) (8g, 8h); deterministic coupled-solver contacts on Metal
- 10-12 Setup pins a commit of [my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) ([`cf0f8ad`](https://github.com/pulipakaa24/mujoco_warp/commit/cf0f8adc8f99f0c559cf8580cb667952e473bd07), replaced by [`f1b35e2`](https://github.com/pulipakaa24/mujoco_warp/commit/f1b35e2b3ddc80b197b0c665bf6d71523d175daf) on 10-13) and keeps MuJoCo below 3.15
- 10-12 My Newton fork's branches pinned with the 64-bit atomics protocols and rigid-scale ports; 2,048-world and deterministic timings of the 1.7-line ports (8d)

## Tooling: GPU queue (2026-10-12)

- 10-12 Check jobs waiting over 10 minutes go ahead of timing jobs; no starvation; jobs refused on a runaway GPU (work with no owning process), queue paused and `runs/GPU_RUNAWAY_ALERT` written

## MPM, scalable determinism, rigid scale (2026-10-09 .. 11, `metalsim/physics/mpm/MPM.md`)

- 10-11 MPM on Metal landed in the tree: `metalsim/physics/mpm` (compat layer, scenes, reports, probes), `tests/test_mpm.py`, Isaac-side recorders and stages, research (`docs/research/mpm_2026-10-09.md`, `reports/`, `research_notes/`), run logs and JSON (`runs/mpm`, `runs/parity/isaaclab3_mpm*`, `runs/rigid_perf`, `runs/fma_rigid_parity`)
- 10-11 Setup pins fork commits: [my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim) at `metalsim` [`152808b7`](https://github.com/pulipakaa24/warp/commit/152808b7e33ce08fcc17c898c5ce11d567d60795) (`scripts/setup_warp.sh`), [my fork of Newton](https://github.com/pulipakaa24/newton/tree/metalsim) at `metalsim` [`fe7f5bd4`](https://github.com/pulipakaa24/newton/commit/fe7f5bd48010fcab428eac6373e85ba3317f7a42) (`scripts/setup_newton.sh`) and at [`metalsim-1.5.2`](https://github.com/pulipakaa24/newton/tree/metalsim-1.5.2) [`35791a07`](https://github.com/pulipakaa24/newton/commit/35791a077dbc74d8e5b4cb9d320faa72af2f3492) (`scripts/setup_newton152.sh`, new: the MPM environment), the `newton` extra at `metalsim-hashtable-metal` [`24fb5367`](https://github.com/pulipakaa24/newton/commit/24fb53679efbb45de4bc08a1631759789ee9db2e); whole-branch diffs in `patches/warp`, `patches/newton` as the offline fallback
- 10-11 My fork of Warp (`metalsim`): CUDA-faithful `fast_math` (`warp.config.metal_fast_math`), default `"off"`; MPM chooses through `newton_compat.install(metal_fast_math=)`
- 10-12 My Newton fork's branches: mesh-SDF barrier fix and the 1.5.2 line's Metal collision launches on the 1.7 lines; Metal protocols for the last 64-bit CAS / exchange sites (contact reduction, XPBD restitution manifolds, Kamino metric argmax), bounded claim loops; pins `metalsim` [`9f317390`](https://github.com/pulipakaa24/newton/commit/9f317390da07432dfb5813bd0db6a47eeb7cb330), `metalsim-hashtable-metal` [`a4be3604`](https://github.com/pulipakaa24/newton/commit/a4be3604bc218bb366d12164fa3e1025f333bdae) (the `newton` extra), `metalsim-1.5.2` [`c0427fc7`](https://github.com/pulipakaa24/newton/commit/c0427fc78ba79d1f8482c7c9f18847568f135ef0) (MPM.md 8d)
- 10-12 Newton engine: rigid-contact buffer sized per contact pair (64 G1 rough worlds dropped about half the contacts at Newton's automatic 1,000 slots)
- 10-11 GPU queue: `check` class (correctness jobs share the GPU, up to 3) and job heartbeats

## Physics: elliptic-cone premium (2026-09-26, `docs/research/elliptic_warmstart_2026-09-26.md`)

- 09-26 [My fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim), branch `metalsim-rough-ellip` [`d9ec218`](https://github.com/pulipakaa24/mujoco_warp/commit/d9ec218f935f2bb791b0f2d4a50047d5f7f45b0c) (on `metalsim` [`f824af1`](https://github.com/pulipakaa24/mujoco_warp/commit/f824af1c5f482791cca90c1a6ca6fe5adb19ce63)): the convex narrowphase's EPA / multi-contact scratch is allocated once per Data on Metal; graph captures kept one 2.8 GB copy per substep at 4096 worlds, so the G1 rough task at 2.5 ms (8 substeps) overflowed the 55.7 GB GPU working set under the elliptic default (`docs/research/rough_ellip_oom_2026-09-26.md`)
- 09-26 My MuJoCo Warp fork, branch `metalsim-ellip-ws` ([`5d0d165`](https://github.com/pulipakaa24/mujoco_warp/commit/5d0d1656637247834ebb10e30aff2e8fa91cc066) … [`433c305`](https://github.com/pulipakaa24/mujoco_warp/commit/433c30524971267054f5fe02522278fb34d08c3e)): the fused per-iteration Newton launch covers elliptic cones (11 -> 5 launches per iteration, 128 lanes per world; bitwise on the CPU device, within the floor on Metal): G1 task ellip10 cap 20 loop 34.3 K -> 37.7 K env-steps/s; [my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim) [`6bceb39d`](https://github.com/pulipakaa24/warp/commit/6bceb39d5b114e42b0060b62a776fa3bd71af730) + MuJoCo Warp [`21be81b`](https://github.com/pulipakaa24/mujoco_warp/commit/21be81b1da2ff09e02c8fab403ce4bf0b29f2db6): exact per-world early exit of the Newton loop in Metal graph replay through ICB indirect execution ranges (elliptic physics only 55.4 -> 47.7 ms, pyramidal 43.8 -> 39.7); [`12cc1e2`](https://github.com/pulipakaa24/mujoco_warp/commit/12cc1e28529bb3912d3365ac4ff7bf77bdd6ba3f): line-search derivative noise floor `MJW_LS_NOISE_FLOOR` ([Genesis #3382](https://github.com/Genesis-Embodied-AI/genesis-world/pull/3382)'s rule; loop 37.5 K -> 38.1 K); extrapolated warm start `MJW_WARMSTART_EXTRAP` archived (tolerance-level state change, more worlds at the cap). Elliptic loop premium 1.68x -> 1.53x (1.50x with the floor). Scripts: `g1_newton_iters.py`, `ellip_fused_cpu_check.py`, `ls_noise_floor_check.py`, `cap_probe_variant.py`; `g1_solve_iteration_cost.py` rollout state.

## Install and packaging

- 09-25 GPU queue dashboard: `scripts/gpu_dashboard.py` (`--serve`, `--watch`, `--html`, `/json`), `scripts/gpu_dashboard.sh` launcher; queue records command lines, grants and releases with exit codes (`runs/gpu_history.jsonl`)
- 09-25 MIT licence (LICENSE, pyproject metadata), THIRD_PARTY_NOTICES.md with the IP audit and `licenses/`, source headers on the three ported files, README rewritten as the front page (what was built in each fork + branch, heads and commits); `METALSIM_CHANGES.md` pushed to every fork branch
- 09-23 Rename to MetalSim (package metalsim); setup script and patches for [my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim); heightfield defect localized
- 09-23 Point the Warp setup at my fork of Warp (branch `metalsim`)
- 09-23 Fetch NVIDIA Isaac assets instead of committing them; tron1 rename leftovers
- 09-24 Setup for [my fork of Newton](https://github.com/pulipakaa24/newton/tree/metalsim): scripts/setup_newton.sh installs pulipakaa24/newton@metalsim into .venv-newtonfork

## Physics (MuJoCo Warp on Metal) and interop

- 09-26 Elliptic premium 1.68× → 1.53×: fused per-iteration launch on the elliptic path ([my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) [`433c305`](https://github.com/pulipakaa24/mujoco_warp/commit/433c30524971267054f5fe02522278fb34d08c3e)), exact per-world early exit in Metal graph replay through indirect-command-buffer ranges ([my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim) [`6bceb39d`](https://github.com/pulipakaa24/warp/commit/6bceb39d5b114e42b0060b62a776fa3bd71af730), MuJoCo Warp [`21be81b`](https://github.com/pulipakaa24/mujoco_warp/commit/21be81b1da2ff09e02c8fab403ce4bf0b29f2db6)); independent cost review (`docs/research/elliptic_cost_review_2026-09-26.md`); fork heads mujoco_warp [`063ff98`](https://github.com/pulipakaa24/mujoco_warp/commit/063ff98f0f88cc15a1de205a9df65b11062494a5), warp [`c200d46b`](https://github.com/pulipakaa24/warp/commit/c200d46b31503c51406651ad7528ef7171a1f29f)
- 09-26 Throughput without fidelity cost: model-derived constraint capacities + overflow guard (`metalsim/physics/capacity.py`), register triangular solve (my Warp fork [`27e63fd1`](https://github.com/pulipakaa24/warp/commit/27e63fd1476596b08f11dfa89a822490adabf6e5)), fused Newton iteration launches (my MuJoCo Warp fork [`dd42c23`](https://github.com/pulipakaa24/mujoco_warp/commit/dd42c23b14287ebdc05fdb6a2c762e9aa3422bc5)), policy kernel mapping: full PPO loop 49.5–50.1 K → 56.4 K env-steps/s (+13.3 %), env step +12.4 %; elliptic cones on MuJoCo C's incremental Newton structure (my MuJoCo Warp fork [`b895bff`](https://github.com/pulipakaa24/mujoco_warp/commit/b895bff5a050c7419c3b06d038a07b0baf915fbb), premium 1.42× → 1.21× physics; the faithful cap of 20 makes the loop premium 1.60×); fork heads mujoco_warp [`a1daf88`](https://github.com/pulipakaa24/mujoco_warp/commit/a1daf88e13b50ae85243fca53f6d51e87766eefc), warp [`e29950ee`](https://github.com/pulipakaa24/warp/commit/e29950ee603bdea6e09e00a966595d62c1f105c9)
- 09-26 Elliptic-cone premium cut further (my MuJoCo Warp fork's branch `metalsim-elliptic2` [`b0150ac`](https://github.com/pulipakaa24/mujoco_warp/commit/b0150acd500679259a6154c828209d7fed17f523), pending merge): MuJoCo C's incremental Newton structure (quadratic part kept across iterations, cone term rebuilt per iteration) on Metal, mode 2 = per-entry deltas + cone term into `htot` and the plain register Cholesky; G1 task ellip10 55.4 K → 64.5 K env-steps/s (1.42× → 1.21× pyramidal), Go2 +7 %, humanoid +4 %, SO-101 +4 %; per-iteration constraint-state characterisation on walking and falling states (`scripts/diagnostics/competitors/g1_state_*.py`); rejected variants archived behind flags (`docs/DECISIONS.md`, `docs/research/elliptic_cones_2026-09-25.md` §8)
- 09-25 Elliptic follow-ups: `tau10_impact_hardlimits_ellip{1,10,100}` presets (+ field test), confirming training run `g1_flat_flatcfg_ellip10` queued, [PhysX](https://github.com/NVIDIA-Omniverse/PhysX)-checkpoint falls characterised over 3 seeds (+7–10 %, within one seed sd; `scripts/diagnostics/competitors/g1_falls*.py`), `tests/test_lift_ctrl_ordering.py` (the SO-101 env was already event-ordered), the two pre-existing test failures in my MuJoCo Warp fork root-caused (heightfield primitives fix on its branch `metalsim-hfield-test`; tendon-frictionloss test depends on a MuJoCo nightly)
- 09-25 Elliptic friction cones on Metal: reproduced the hand-off in `docs/HANDOFF_elliptic_cone_perf.md` (2.4–7× slower than pyramidal), fixed in a worktree of my MuJoCo Warp fork ([`c301880`](https://github.com/pulipakaa24/mujoco_warp/commit/c30188029f6729b9f7cfe69706d6b8069b7f53d2), branch `metalsim-elliptic`, pending merge: per-world cone-list Hessian term, full-width sparse assembly groups, every previous form behind `MJW_JTCJ_MODE` / `MJW_JTDAJ_ELLIPTIC_LANES`), audited every other non-CUDA fallback, re-ran the G1 fidelity protocol and an SO-101 creep protocol with elliptic cones; upstream PR draft `scripts/diagnostics/mjwarp_upstream/DRAFT_elliptic_launch.md`; `docs/research/elliptic_cones_2026-09-25.md`; scripts `scripts/diagnostics/competitors/elliptic_*`, `so101_creep.py`
- 09-26 Upstream: MuJoCo Warp issues [#1704](https://github.com/google-deepmind/mujoco_warp/issues/1704) (off-CUDA elliptic Hessian launch sized by naconmax) and [#1705](https://github.com/google-deepmind/mujoco_warp/issues/1705), [#1706](https://github.com/google-deepmind/mujoco_warp/issues/1706), [#1707](https://github.com/google-deepmind/mujoco_warp/issues/1707), [#1708](https://github.com/google-deepmind/mujoco_warp/issues/1708), [#1709](https://github.com/google-deepmind/mujoco_warp/issues/1709) (flex: element ids of later flexes, box-triangle contacts, cable capsules, active layers, per-pair cap vs [#1668](https://github.com/google-deepmind/mujoco_warp/pull/1668)), MuJoCo issue [#3628](https://github.com/google-deepmind/mujoco/issues/3628) (C Newton solver stops unconverged after an exhausted line search, elliptic impratio 10, ls_iterations 20), all re-measured on upstream main; line-search budget finding not filed (fixed upstream by [mujoco_warp #1700](https://github.com/google-deepmind/mujoco_warp/pull/1700): 10.0 % -> 0.4 %); elliptic PR branch `elliptic-jtcj-offcuda` [`6d8e29a`](https://github.com/pulipakaa24/mujoco_warp/commit/6d8e29a210ed0d1812118c3cd22b5e3450b6d2d3) ready, blocked on the owner's Google CLA; Newton PRs [#4316](https://github.com/newton-physics/newton/pull/4316), [#4317](https://github.com/newton-physics/newton/pull/4317), [#4318](https://github.com/newton-physics/newton/pull/4318): [CodeRabbit](https://www.coderabbit.ai) fixes prepared locally, [EasyCLA](https://github.com/linuxfoundation/easycla) still unsigned (`scripts/diagnostics/mjwarp_upstream/FILED.md`, `scripts/diagnostics/newton_upstream/REVIEW_2026-09-26.md`)
- 09-25 Shared fork checkouts fast-forwarded to the documented heads (mujoco_warp [`1791414`](https://github.com/pulipakaa24/mujoco_warp/commit/17914144d132dec4c0aa3bccfe743816faec72c9) -> [`07a51a6`](https://github.com/pulipakaa24/mujoco_warp/commit/07a51a63bce527d2d26b363a32c52765df4412fc), warp [`9050cb5`](https://github.com/pulipakaa24/warp/commit/9050cb5452de26ca6c80871bee15c389c06f9b10) -> [`4127c48`](https://github.com/pulipakaa24/warp/commit/4127c4818334080de6f3d23aef522e582ff6a2a7)) after the throughput A/B showed them identical for the G1 (flex changes gated on nflex > 0); the local branches had never been advanced
- 09-22 [`b773ae7`](https://github.com/pulipakaa24/MetalSim/commit/b773ae76d1d51745a8b6f8cd53a92537e539453e) orchard: project skeleton and SO-101 lift scene
- 09-22 [`7d64c48`](https://github.com/pulipakaa24/MetalSim/commit/7d64c48e45dea45429486122a7e12e63cab91406) interop: zero-copy MPS tensors over Warp Metal arrays and cross-queue event ordering
- 09-22 [`688a13c`](https://github.com/pulipakaa24/MetalSim/commit/688a13ca61154ace620af427c3eb2e0e854c0c2b) physics: BatchSim on MuJoCo Warp with graph replay and MPS-aliased state; render: native Metal tier-0 renderer
- 09-22 [`3f6c5e7`](https://github.com/pulipakaa24/MetalSim/commit/3f6c5e736bd8491b94000823e2a3626f062b9f53) physics: size constraint buffers (njmax=512) and expose overflow flags; tools: race finder; scene: MJCF->USD importer
- 09-22 [`1a2741a`](https://github.com/pulipakaa24/MetalSim/commit/1a2741a86d3dc6f5aec6543a9bbbcfec3cd1c224) physics: per-world model fields (physics DR) with recompute_constants; lift env randomizes box mass/friction per world; URDF->USD test; IMU/contact sensor tensors test
- 09-22 [`1359a52`](https://github.com/pulipakaa24/MetalSim/commit/1359a5284dda87d516c858b69e711bee56ab6cfe) G1 velocity task on Isaac's g1_minimal.usd with test-cited parity matrix; tier 2 path tracer; Metal backend memory fix
- 09-23 MuJoCo Warp heightfield plane-contact patch; heightfield contact test now passes (was xfail)
- 09-24 Uncontended benchmark pass 2026-09-24; rough terrain at 4096 envs (nconmax 32); docs updated from the clean pass
- 09-24 My MuJoCo Warp fork: plane_convex now reproduces MuJoCo C's contact set (`metalsim` [`284dcd1`](https://github.com/pulipakaa24/mujoco_warp/commit/284dcd12ff37d56ea0201ade4e7f941d772a0f28))
- 09-24 Contact/limit tuning module and tuned parity replays against Isaac's PhysX recordings
- 09-24 plane_convex patch: add the MJW_PLANE_CONVEX=legacy switch (my fork's `metalsim` [`a8e6485`](https://github.com/pulipakaa24/mujoco_warp/commit/a8e64857ea81c5277f55ec7a995adc1e99191e09)); regression test skips under it
- 09-25 Throughput "regression" settled (`docs/research/throughput_regression_2026-09-25.md`): the flex merge in my MuJoCo Warp fork costs the G1 nothing (interleaved A/B ±0.4 %); the 67.6 K → 54.2 K env step is the task's PhysX-parity contact preset (solver only); README / PARITY §1.4 carry both settings; `g1_tp_variants.py` / `g1_step_profile.py` take `contact_cfg` / `solver_cfg`; `g1_tp_generality.py` prints which fast paths a scene takes (Go2, SO-101, Panda at 4096: elliptic cones fall off the fused Hessian)

## Rendering

- 09-22 [`0656f1f`](https://github.com/pulipakaa24/MetalSim/commit/0656f1fa97e53ab3aaff005f920625455ef12417) render: model light list, headlight/ambient model, tiled shadow maps, radiance calibration; fidelity benchmark (PSNR/[FLIP](https://github.com/NVlabs/flip))
- 09-22 [`62857e0`](https://github.com/pulipakaa24/MetalSim/commit/62857e0678bdf755ec344fa25de8a65982abc323) render: tier 1 hybrid ray tracing (RT soft shadows, AO, mirror reflections from the fragment stage), skybox colour, Isaac-style lidar spec; bench suite
- 09-24 Parity scene: hide collision geometry in renders (Isaac draws visuals only; the foot collider plates were visible); post-queue video re-render
- 09-26 Tier 2: HDR equirectangular environment map (textured USD DomeLight) importance-sampled with MIS (`set_environment`, `env_sampling_table`, `metalsim/render/hdr.py` RGBE reader), oriented as Kit/RTX renders a dome on a z-up stage; USD import records DomeLights (`usd_dome` custom text, `set_environment_from_model`); `Randomizer.environment` per-episode map by key; `set_fovy`, `set_materials`, opt-in `firefly_clamp` (biased); parity presets bit-for-bit unchanged without a map; [Poly Haven](https://polyhaven.com) CC0 maps fetched (not committed); gallery `g1_hdri_tier2.png`; handoff `docs/HANDOFF_tier2_hdri_envmap.md` ported

## Sensors

- 09-22 [`fbc6c58`](https://github.com/pulipakaa24/MetalSim/commit/fbc6c58e9495abc548b8f27aec2955e8aa69a45d) sensors: Metal hardware ray tracing (per-mesh primitive AS, per-frame instance AS refit from physics), lidar and ray-cast depth kernels, tests vs mj_ray and raster depth
- 09-24 Contact sensor (Isaac Lab ContactSensor equivalent) on MuJoCo Warp as in-graph Warp kernels
- 09-24 Lidar extras (beam divergence, multi-return, reflectance table) and radar-lite on Metal RT
- 09-24 Sensor benches: record GPU queue status at start; 7 interleaved A/B repeats for the contact sensor; measured logs

## Scene layer (USD / MJCF), terrain, replicator

- 09-22 [`3c921be`](https://github.com/pulipakaa24/MetalSim/commit/3c921bebe37afe853e3cbda94e51516de328a543) scene: USD -> MuJoCo loader (lossless and generic UsdPhysics paths) with round-trip tests; learn: contiguous NCHW rollout buffer (2.3x faster updates), train_lift script
- 09-22 [`46768d0`](https://github.com/pulipakaa24/MetalSim/commit/46768d0be8fda6f1a2a413489d7560c9127f61af) replicator: randomizers, annotators (2D/3D boxes, semantic seg, camera params), [COCO](https://cocodataset.org)/[KITTI](https://www.cvlibs.net/datasets/kitti/)/basic writers with test; learn: Warp-rollout PPO with rsl_rl KL schedule (learning issue under investigation)
- 09-24 USD loader: instance proxies and MDL OmniPBR materials (G1 visual meshes render); kinematics-only reset (+20% step rate); clipped value loss; terrain curriculum; cartpole tier-2 video; G1 still
- 09-24 Rough terrain: port Isaac Lab v2.3.2's TerrainGenerator exactly (ROUGH_TERRAINS_CFG)

## G1 velocity task and PPO learning

- 09-26 G1 default contacts: elliptic friction cones, impratio 10, Newton cap 20 (`recommended` = `tau10_impact_hardlimits_ellip10`; `recommended_pyramidal` archives the previous default); contact presets carry their Newton cap; the isaaclab3 solver presets own the cone (pyramidal, impratio 1); headline re-runs queued
- 09-24 Blow-up guard by magnitude, non-finite rows dropped from the PPO update, per-iteration callback with periodic checkpoints; hero cartpole video
- 09-24 G1 1,500-iteration run recorded (learns to stand for 12 s by it 750, then degrades); G1 tier-2 policy video; camera-cartpole PPO brought to Isaac's [skrl](https://github.com/Toni-SM/skrl) config
- 09-24 G1 blow-up root cause: explicit actuator stiffness at 5 ms with Isaac's kp 200 (0/8 blow-ups at 2.5 ms); physics_dt option; camera PPO gets Isaac's image centering and skrl value scaler
- 09-24 G1 reward: termination penalty scaled by dt as Isaac's RewardManager does (was -200 unscaled, which made falling optimal)
- 09-24 Task-agnostic anomaly monitor (learning signals + physics invariants) hooked into training; G1 preflight; monitor findings recorded
- 09-24 Corrected G1 run recorded (stays up 11-15 s by it 1000, return still negative); monitor trend window 100 iterations
- 09-24 G1 learning: per-term reward comparison against Isaac's rsl_rl run (PARITY 1.5); reward-term diagnostic
- 09-24 G1 feet_slide: Isaac's quantity (foot body world linear velocity) on both engines
- 09-24 Learner differential: rsl_rl 3.1.2 on MetalSim G1 flat; PPOWarp rollout fixes
- 09-24 rsl_rl 3.1.2 on MetalSim G1 flat, 1500 iterations (Isaac's G1FlatPPORunnerCfg, seed 0, 4096 envs): log, per-term JSONL, curves script
- 09-24 G1 task: flat terrain uses Isaac's G1FlatEnvCfg reward set; all 13 reward terms exposed and asserted against Isaac's formulas
- 09-24 Fixed PPO demonstrated on G1 flat: +32.8 return / 995 length at iteration 1000, tracking rsl_rl on the same physics (PARITY 1.5, GAPS, STATUS)
- 09-24 G1 task: Isaac ContactSensor for the flat set's contact terms; Isaac's rough-terrain seed and env-column assignment
- 09-24 G1 flat: ContactSensor cost at 4096 envs, 5 interleaved A/B repeats: +0.62 ms per control step (+0.4 %), 29,479 vs 29,611 env-steps/s
- 09-24 rsl_rl 3.1.2 on the ported G1FlatEnvCfg task (ContactSensor contact terms), 400 iterations, 4096 envs, seed 0
- 09-25 Like-for-like training result: MuJoCo Warp + fixed PPO on Isaac's flat config reaches +28.4 at iteration 1000 vs Isaac's +27.3 (PARITY 1.5, STATUS, GAPS)
- 09-25 G1 training CLI: --seed
- 09-25 Per-term reward breakdown at it 300/1000 of the MuJoCo Warp flat-cfg fixed-PPO checkpoints (like-for-like with the Newton rows)

## Other learning tasks (lift, cartpole, camera cartpole, lidar navigation)

- 09-22 [`836c518`](https://github.com/pulipakaa24/MetalSim/commit/836c5182614e095f98e6e13d048fbd2d37b689f1) learn: GPU SO-101 lift task and MPS PPO; render: levels of detail, GPU-writable DR params; docs and gallery
- 09-22 [`7d2ee8b`](https://github.com/pulipakaa24/MetalSim/commit/7d2ee8bdaddcb8503685d568f7a1de98119cef1b) learn: Warp rollout policy with graph-replayed rollouts (zero torch launches per step); Isaac Cartpole-RGB task; physics launch_step for outer captures
- 09-22 [`72bad94`](https://github.com/pulipakaa24/MetalSim/commit/72bad94e7824f8476c28f3ff191831b08ee9400b) learn: cartpole learns with Isaac Lab's rsl_rl config on the Warp rollout path (collision-free cartpole MJCF)
- 09-24 Camera-based RL row: Isaac's camera cartpole agent config learns from tier-0 pixels
- 09-24 Lidar-based RL task (planar 64-beam Metal lidar, goal navigation among random obstacles) with scan-vs-mj_ray test
- 09-24 Camera-RL update: research on MPS conv backward and alternatives, update profiler and measurements (torch eager/fp16/bf16/compile, [MLX](https://github.com/ml-explore/mlx))
- 09-24 Camera-RL update on MPS: fast_update path (compiled loss and grad clip, fused Adam, per-rollout image means) and Metal kernels for conv1 weight grad and conv2 input grad; update 7.2 s -> 3.9 s
- 09-25 Camera-cartpole training with the fast update path: 11.4K env-steps/s incl. training (was 7.5K), learning curve matches the reference run
- 09-25 Camera-RL update: conv1 forward Metal kernel, minibatch gather fused into the compiled preprocessing, conv2/conv3 weight-grad kernels (measured, off by default); update 3.68 -> 3.20 s
- 09-25 Camera-RL round two: gather fused in graph, conv1 forward kernel, conv2/conv3 wgrad measured (off), MLX whole-update experiment (no gain in fp32); training 12.7K env-steps/s incl. training, curve unchanged
- 09-25 Camera-RL: 12.7K env-steps/s incl. training after the gather fusion and conv1 forward kernel; MLX not faster in fp32. GPU queue: release checks the job pid

## Parity against Isaac Sim / Isaac Lab (benchmarks, fidelity protocol, side-by-side, transfer)

- 09-22 [`50efb27`](https://github.com/pulipakaa24/MetalSim/commit/50efb277273964e6f56de100db69e1d8725f3c3b) bench: uncontended suite and Isaac comparisons recorded
- 09-24 Fidelity protocol: Isaac-side recorder and policy player, MetalSim replay, metrics (PSNR/SSIM/[LPIPS](https://github.com/richzhang/PerceptualSimilarity)/FLIP, depth, joint RMSE, divergence time, contact stats)
- 09-24 Isaac Lab measured on an L4: Cartpole-Direct 400.7K, G1 rough 39.1K env-steps/s (reference rows)
- 09-24 Side-by-side video composer (Isaac RTX vs MetalSim tier 2, same policy checkpoint and state)
- 09-24 Policy export for Isaac playback; Isaac-side stage 5 (play checkpoints from both trainings in Isaac)
- 09-24 Isaac-side recorder/player: no debug markers, plain grey ground, 20 m env spacing (fair frames)
- 09-24 Fidelity protocol results vs [Isaac Sim](https://github.com/isaac-sim/IsaacSim) 5.1 (PARITY 1.7): physics rows, whole-frame and robot-only render metrics, gallery composites; lidar RL run; Isaac-side grey ground + stage 6 re-record
- 09-24 Docs: Isaac's full rsl_rl run and cross-simulator playback results; L4 flat G1 benchmark; stage-video runner
- 09-24 Fidelity: grey-ground re-recording replaces the whole-frame rendering rows (tier 2 vs RTX 18.0 dB / 0.876 SSIM); gallery composites refreshed
- 09-24 Fidelity replay: full-extent ground plane; re-render runner
- 09-24 Gallery: side-by-side training-stage videos (Isaac Sim | MetalSim) for Isaac's and MetalSim's checkpoints; cross-simulator transfer table
- 09-24 MuJoCo Warp transfer of Isaac's checkpoints before / after the plane_convex contact-set fix ([my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim) [`f2716b4`](https://github.com/pulipakaa24/mujoco_warp/commit/f2716b4e7873ed40b8405c0840238610349c7e7e) vs [`284dcd1`](https://github.com/pulipakaa24/mujoco_warp/commit/284dcd12ff37d56ea0201ade4e7f941d772a0f28))
- 09-24 Transfer of Isaac checkpoints (500/1000/1499, 8 s) on MuJoCo Warp: old vs fixed plane_convex collider and each contact tuning; tuning throughput logs; --contact_tuning in g1_reward_terms and mjwarp:<tuning> in newton_transfer
- 09-25 Gallery: round-two side-by-side videos (fixed-PPO policies in Isaac Sim and MetalSim); transfer table both directions

## Newton XPBD engine (experimental; archived 2026-09-25)

- 09-24 Diagnostics: MuJoCo contact stiffness sweep on the G1 drop test; Newton XPBD on Metal with Isaac's G1 USD (feasibility)
- 09-24 Newton XPBD status: drives not engaging yet; throughput figure qualified
- 09-24 Newton XPBD drives: root causes and a faithful PD actuator for the G1
- 09-24 Diagnostics: Isaac actuator model under Newton XPBD (effort limits, armature) on the G1 hold and a 1 m drop
- 09-24 Diagnostics: G1 drop-test penetration under Newton XPBD on Metal (MuJoCo C protocol)
- 09-24 Diagnostics: G1 physics throughput, Newton XPBD with working drives vs MuJoCo Warp (graph replay)
- 09-24 Diagnostics: Newton mesh/convex narrow phase (GJK/MPR) on Metal vs CPU
- 09-24 Newton backend: optional native convex-mesh colliders (GJK/MPR on Metal); docstring with the measured recipe
- 09-24 Newton backend: module docstring with the measured drive recipe
- 09-24 Newton XPBD results in the parity and gap docs; Warp fixed-size-array patch; fidelity depth metric masks sky and separates the robot silhouette; Isaac-side player accepts MetalSim exports
- 09-24 G1 task: engine switch (mjwarp default | newton) with a BatchSim-compatible NewtonSim
- 09-24 Tests: G1 task invariants on both engines; Newton engine invariants
- 09-24 Newton engine: gravity-implicit actuator on collider-free subtrees, MuJoCo cvel semantics, 2.5 ms qacc; preflight/bench engine flags
- 09-24 G1 flat PPO on Newton XPBD: 1500 iterations, same config and seed as the MuJoCo Warp reference
- 09-24 Newton upstream issue drafts: XPBD joint relaxation defaults, iteration-dependent drive stiffness
- 09-24 Diagnostics: Newton XPBD joint-constraint drift on the G1 vs solver setting
- 09-24 Newton engine: torso contact history over the last 15 ms of substeps (Isaac's illegal_contact window)
- 09-24 Newton engine: rough terrain (heightfield collider, height-scan body pose)
- 09-24 g1_engine_bench: --rough (either engine)
- 09-24 Newton engine: solver/actuator knobs for the 3-sigma stability sweep
- 09-24 Diagnostics: transfer of Isaac's checkpoints to the Newton engine (side_by_side protocol); Newton column in g1_learning_curves
- 09-24 Newton engine: keep revolute limits inside +-(pi - 0.15) (XPBD angle wrap at pi); drift-remedy evaluation; Featherstone option
- 09-24 Newton engine: optional recentred revolute zeros (moves XPBD's +-pi angle wrap away from the limits)
- 09-24 Newton upstream issue draft: XPBD revolute angle wraps at +-pi (limit ranges or overshoot past pi explode)
- 09-24 Newton drift remedies evaluated on CPU (projection, Featherstone, recentring, 0.625 ms); throughput script
- 09-24 newton_remedy_throughput: drop Featherstone from the 4096-env timing (1 env-step/s at 16 envs, measured)
- 09-24 G1 engine benchmark, final Newton recipe: cost split at 4096 envs (power source at start and end in the log)
- 09-24 Mark the 21:19 final-engine benchmark as contended (not a result)
- 09-24 G1 engine benchmark, final Newton recipe (clean: MuJoCo Warp control at start and end within 5 % of 27,749)
- 09-24 Isaac fidelity protocol on the Newton engine (A_hold, B_random, C_drop vs Isaac's parity_out2/rt recording)
- 09-24 [My fork of Newton](https://github.com/pulipakaa24/newton/tree/metalsim), item 1 (hinge angle wrap, [`f844a4e6`](https://github.com/pulipakaa24/newton/commit/f844a4e67f738592c1eb3e5722d3327071fa1965)): G1 check harness and measurements
- 09-24 Newton engine: limit_margin pass-through (--newton_limit_margin none, sweep ':nolim'); install stamp in sweep logs
- 09-24 G1 flat PPO (fixed learner) on Newton XPBD, 1000 iterations, 4 it 1.25 ms, seed 0
- 09-24 g1_reward_terms: --engine newton (--newton_it, --newton_dt, --newton_limit_margin)
- 09-24 Transfer of Isaac's PhysX checkpoints to the Newton engine (side_by_side protocol, 4 envs, 400 steps)
- 09-24 My Newton fork, items 2, 3, 5, 6 ([`fda6658a`](https://github.com/pulipakaa24/newton/commit/fda6658a5654bc739e301ac9d8d13d4906623b4c), [`9f626ce2`](https://github.com/pulipakaa24/newton/commit/9f626ce23ad4ffe902f64c23fc7016f9eb727b15)): diagnostics and measurements
- 09-24 NewtonSim: switches for my Newton fork's solver-level drive (drive='solver', joint_coloring, relaxation_angular)
- 09-24 Newton transfer over-travel diagnostics: stance gap, fork drive, subtree joint inertia (opt-in)
- 09-24 Push-off diagnostic (Isaac ckpt 1000): forward ground-reaction impulses per touch-down, MuJoCo Warp vs Newton
- 09-24 newton_transfer: rlin=/rang= relaxation tokens
- 09-24 G1 flat fixed-PPO on Newton XPBD 4 it 0.625 ms, 1000 iterations, seed 0 (pinned install)
- 09-24 Per-term reward breakdown at it 300/1000: MuJoCo Warp vs Newton fixed-PPO policies, each on its own engine
- 09-24 Newton 3-sigma stability sweep: blow-ups of 1024 vs physics throughput
- 09-24 My Newton fork: pd drive damping fix ([`9b0901d3`](https://github.com/pulipakaa24/newton/commit/9b0901d36f01f8761cf52a5113f890e31c8d6d1b)) and throughput refactor ([`242eeda7`](https://github.com/pulipakaa24/newton/commit/242eeda7ba4b87cf94420ca9b0b4cd2c91b48811)): diagnostics and measurements
- 09-24 NewtonSim drive='solver': probe joint_drive_mode, read my fork's public solver.joint_drive_force (total drive torque)
- 09-24 3-sigma winner fails transfer: implicit leg stiffness makes Isaac's policies fall (functional check)
- 09-25 Transfer of Isaac's checkpoints on my Newton fork (fixed solver drive vs ActuatorPD, 1.25 / 0.625 ms, relaxation 0.5/0.4, no clamp)
- 09-25 My Newton fork final ([`90e23324`](https://github.com/pulipakaa24/newton/commit/90e23324521fa86710e483e01d02e80d1dac0060)): GPU throughput and 3-sigma
- 09-25 G1 flat fixed-PPO on my Newton fork with the fixed solver PD drive, 4 it 0.625 ms, 1000 it, seed 0 (throughput provisional: fork build ~30 % slower on GPU)
- 09-25 Per-term reward breakdown at it 300/1000, my Newton fork's solver-drive (0.625 ms) fixed-PPO policy
- 09-25 My Newton fork: 3-sigma check and throughput with the fixed solver PD drive (vs ActuatorPD), GPU, 1024 / 4096 envs
- 09-25 Swing-leg diagnostic (Isaac ckpt 1000): swing duration, clearance, touch-down velocity, step length, hip/knee trajectories, MuJoCo Warp vs Newton 0.625 ms
- 09-25 Newton track archived: remaining logs (job-chain status, install watch, partial held benchmark, PhysX compare outputs)
- 09-25 Newton upstream: issues [#4313](https://github.com/newton-physics/newton/issues/4313), [#4314](https://github.com/newton-physics/newton/issues/4314), [#4315](https://github.com/newton-physics/newton/issues/4315) and PRs [#4316](https://github.com/newton-physics/newton/pull/4316), [#4317](https://github.com/newton-physics/newton/pull/4317), [#4318](https://github.com/newton-physics/newton/pull/4318) filed (bodies, reproducers, regression logs)
- 09-25 Newton upstream: three issues and three pull requests filed (`scripts/diagnostics/newton_upstream/FILED.md`); gap table

## Tooling (GPU queue)

- 09-24 GPU arbitration: priority queue lock (timing > render > train) with atomic acquisition, stale-holder cleanup and a run wrapper
- 09-24 gpu_run.sh: record the job's own pid in the lock and kill the job if the wrapper dies (a cut-off wrapper caused a double grant)
- 09-25 GPU queue: low-priority class

## Documentation (status, parity matrix, gap ledger, gallery)

- 09-22 [`0460afb`](https://github.com/pulipakaa24/MetalSim/commit/0460afbd630423f7ba783025c78d58c8de54f70c) docs: status against the plan, acceptance tests, judgement calls
- 09-22 [`fa2df43`](https://github.com/pulipakaa24/MetalSim/commit/fa2df43b539df449eaba85cefdf95ad8fa2fbf7c) gallery: tier 1, lidar, learning curves, Isaac comparison
- 09-24 Contact-model section: MuJoCo stiffness limits and Newton XPBD on Metal (throughput table)
- 09-24 Gap ledger: closed, open, and Newton-track issues
- 09-24 Gap ledger: Newton-subsumed vs engine-independent items; camera-RL MPS update profile
- 09-24 Docs: status and gap table refreshed with the fidelity, Newton, lidar and learning-gap measurements; Isaac L4 camera rollout rate
- 09-24 Docs: Newton engine switch results (throughput at the training setting, 1500-iteration curve on both engines, preflight), feet-slide term defect, gap table
- 09-24 Docs: feet-slide term fix and Newton constraint-drift finding
- 09-24 Docs: learner differential verdict (rsl_rl on MetalSim learns like Isaac; three PPO rollout bugs fixed), flat-vs-rough reward config mismatch
- 09-24 Research: PyTorch-on-Metal interop survey (2026-09-22) saved under docs/research
- 09-24 Gap table: Newton constraint drift quantified, rough terrain on Newton closed, upstream issue drafts
- 09-24 Gap table: XPBD angle-wrap defect and workaround, drift remedies evaluated
- 09-24 Gap table: Newton learning parity pending the fixed learner, 3σ cause, Newton fidelity vs PhysX unmeasured, Featherstone verdict
- 09-24 Docs: clean Newton benchmark with control row; PhysX fidelity protocol on Newton (PARITY 1.4, 1.7); rough-terrain Newton cost as a new gap
- 09-24 Docs: exact Isaac terrain port (grid-exact, wall ramps remain) in the parity matrix and gap table
- 09-24 Docs: contact sensor, lidar extras and radar-lite rows; MuJoCo Warp plane_convex contact-set defect as a new gap
- 09-24 Gap table: camera-RL update profile, chosen route, MLX as a compounding next step
- 09-24 Gap table: [my fork of Newton](https://github.com/pulipakaa24/newton/tree/metalsim) fixes the angle wrap at source; relaxation fix in progress
- 09-24 Docs: fixed PPO on Newton learns tracking far slower (+7.8 vs +32.6 at it 1000); MuJoCo Warp stays the default
- 09-24 Gap ledger: engine position after the measurements (MuJoCo Warp is the parity engine; XPBD is a throughput option)
- 09-24 Gap table: my Newton fork fixes relaxation, drives, contact forces; rest-height explained; coloring as the drift remedy pending throughput
- 09-24 Docs: Isaac checkpoint transfer on Newton over-travels 20-48 % (drift changes the gait); MuJoCo Warp within 5 %
- 09-24 Docs: Isaac flat config ported; rsl_rl on the ported task within 3.2 return of Isaac at it 399; remaining term gaps are contact quantities
- 09-24 Gap table: Newton over-travel is a stride-length effect independent of the drift (unexplained); fork drive light-joint fault
- 09-24 Gap table: fork drive light-joint fix measured; halving the substep beats joint colouring; fork throughput regression under investigation
- 09-25 Gap table: my Newton fork's final state (six solver defects fixed at source, costs and remaining items); transfer with the in-solver drive
- 09-25 Docs: camera-RL update 1.86x faster with Metal gradient kernels; 11.4K env-steps/s incl. training
- 09-25 Docs: Newton learning at all settings vs MuJoCo Warp and Isaac (flat weights), 3σ closed with the fork drive, push-off diagnostic, collider A/B on transfer
- 09-25 Docs: Newton XPBD archived as an experimental option; MuJoCo Warp is the parity engine
- 09-25 Docs: like-for-like per-term rows and the swing-leg localisation of Newton's long stride; Newton validation closed
