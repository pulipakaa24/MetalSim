# MuJoCo Warp G1 training-loop throughput on the M4 Max: research notes (2026-09-25)

Scope: the settings and techniques that bear on the G1 flat training loop (MuJoCo Warp on Warp's Metal
backend, 4096 envs, 2.5 ms physics, decimation 8), what the reference stacks set for the G1 and why, and
which of them can be applied here **without changing results**. Measurements that follow from these
notes are in `runs/mjw_tp/` and summarised at the end.

## 1. MuJoCo Warp's own performance guidance

Source: MuJoCo documentation, "MuJoCo Warp (MJWarp)", performance sections
(https://mujoco.readthedocs.io/en/latest/mjwarp/).

- **Graph capture.** Functions like `mjw.step` are collections of launches; capture them as a graph when
  called repeatedly. (Done: one graph per env step; on Metal replayed through an indirect command
  buffer, Warp fork b1ec949.)
- **Capacity sizing first.** "Memory and computation scales with the values of these parameters
  [`nconmax`/`naconmax`, `njmax`]. For best performance, the values ... should be set as small as
  possible while ensuring the simulation does not exceed these limits." Tuning tool:
  `mjwarp-testspeed --measure_alloc`.
- **Solver budget second.** Reducing `iterations` / `ls_iterations` "may improve performance and should
  be secondary"; too low prevents convergence. On CUDA the doc notes that "once all worlds have
  converged the solver can early exit", so the settings have "comparatively less impact on
  performance". **On Metal this is not true**: the early exit is a conditional graph node
  (`wp.capture_while`), and MetalSim sets `m.opt.graph_conditional = False` (the condition would be read
  back on the host per iteration), so all 10 Newton iterations are launched every substep; converged
  worlds exit per thread (`ctx.done`), so results are identical to the early-exit schedule but the
  launches and their barriers are paid in full. Lowering the budget is therefore worth more here than
  on CUDA, and it changes results, so it is out of scope for this pass.
- **Sparse Jacobians** "reduce memory and skip zero arithmetic". MuJoCo Warp's `is_sparse` rule
  (`mujoco_warp/_src/io.py`): `jacobian=auto` selects sparse for `nv > 32`. The G1 here has nv = 43
  and the task forces `mjJAC_DENSE` (so MuJoCo Warp runs its dense path; MuJoCo C is dense for
  nv < 60 under `auto`). Dense and sparse layouts are the same equations; the difference is summation
  order (float noise).
- **Contact sensors**: keep `contact_sensor_maxmatch` small (not used: our ContactSensor is MetalSim's).

Upstream issues and pull requests read:
- google-deepmind/mujoco_warp#1671: after a selective `reset_data`, derived arrays (`xpos`,
  `subtree_com`, `qfrc_bias`, `qfrc_smooth`, `qacc_smooth`, `qfrc_constraint`) keep pre-reset values
  until the next complete `forward`. Consistent with MetalSim's choice (kinematics only after reset:
  the next `mjw.step` recomputes everything from qpos/qvel/qacc_warmstart before use).
- google-deepmind/mujoco#3572: selective `reset_data` corrupts the packed contact prefix when several
  worlds have contacts; "a later forward rebuilds contacts". Harmless for a reset followed by a step
  (collision runs first), relevant only to readers of `d.contact` between reset and step.
- google-deepmind/mujoco_warp#1638 (open): one warp per world for sparse contact-constraint
  construction, iterating active rows instead of capacity-sized grids: +2.2 % on a single Panda at
  `njmax` 64 on an RTX 5090, no numerical change. The theme matters more on Metal, where every
  capacity-sized launch (`(nworld, njmax)` grids) costs a full dispatch with a barrier.
- Changelog 3.11 (Newton-decrement early termination, PR 1520) and 3.13 (elliptic-cone refactorisation):
  both already in the fork (v3.14.0).

## 2. What the reference stacks set for the G1 at scale

| stack (source) | dt / decimation | solver | iterations / ls | integrator | `njmax` | contacts | Jacobian |
|---|---|---|---|---|---|---|---|
| MuJoCo Warp benchmark `unitree_g1_flat` (`benchmarks/unitree_g1/`, fork at v3.14.0) | 5 ms | Newton, pyramidal | 10 / 20 | implicitfast, eulerdamp off | 192 | `nconmax` 48 | auto → sparse (nv 35) |
| mjlab G1 velocity (`mjlab`, c2e1e06, `tasks/velocity`) | 5 ms / 4 | Newton, pyramidal | 10 / 20 | implicitfast | 300 (flat), 1500 default | `nconmax` None (flat), 70 (rough) | auto |
| MuJoCo Playground G1 joystick (`mujoco_playground` 4057c14, `locomotion/g1/joystick.py`) | 2 ms / 10 | Newton | 3 / 5 | Euler, eulerdamp off | 29·2 + 8·4 = 90 (sized: two limit rows per joint + 8 contacts × 4 pyramidal rows) | `naconmax` 8 per world | auto |
| Isaac Lab 3.0 Newton-MuJoCo-Warp G1 flat (`IsaacLab` 49bd35f, `isaaclab_tasks/core/velocity/config/g1/flat_env_cfg.py`, `velocity_env_cfg.py`, `isaaclab_newton/physics/mjwarp_manager_cfg.py`) | 5 ms / 4, 2 substeps (2.5 ms) | Newton, pyramidal, impratio 1 | 100 / 50 (defaults), tolerance 1e-6 | implicitfast | **95** (rough 300) | `nconmax` **10** | default |
| this stack (`metalsim/learn/g1_velocity.py`) | 2.5 ms / 8 | Newton, pyramidal | 10 / 20 | implicitfast, eulerdamp off | 256 | `nconmax` 32 | forced dense |

Why they differ: Playground and Isaac Lab size `njmax`/contacts to the model's worst case (Playground
spells the bound out), because the capacity is a cost on GPU as well as memory; mjlab keeps generous
capacities (it targets many robots) and relies on CUDA's early exit. Only Playground lowers the solver
budget (3/5, Euler), which is a physics change. Isaac Lab 3.0 keeps MuJoCo's 100/50 defaults with a
1e-6 tolerance, i.e. converged solves, relying on the conditional-graph early exit.

**The G1 here has four colliders** (ground plane, both foot meshes, the torso mesh; `ngeom` 4). Worst
case per world: plane × 3 meshes at ≤ 4 contacts each (the fork's `plane_convex` returns MuJoCo C's
≤ 4-point set) + 3 mesh–mesh pairs at 1 contact each (no multi-CCD) = 15 contacts, × 4 pyramidal rows
= 60, + 37 limited hinges at ≤ 1 active row each (margin 0), no equality / friction-loss / tendon rows:
**nefc ≤ 97**. Isaac Lab's 95 is this bound (minus two); our 256 is 2.6× it and our 32 contact slots 2.1×
the 15. A capacity above the bound cannot overflow, and without overflow the rows, their order rules
and every kernel's arithmetic are unchanged (the dense JTDAJ tile width is min(16, njmax), unchanged
for any njmax ≥ 16), so re-sizing is result-preserving by construction.

`mjw.forward` after reset: mjlab calls one full `forward` per env step for all envs (documented in
`ManagerBasedRlEnv.step`: it refreshes the one-substep-stale derived quantities for every env before
observations). MetalSim already replaced it with `mjw.kinematics` (reset worlds need body poses only
for the rough height scan); on flat terrain no observation reads body poses, so even the kinematics is
consumed by nothing in the training graph.

## 3. Warp on Metal specifics

- **Dispatch model.** The fork replays a captured graph from one indirect command buffer (ICB) with
  `concurrentDispatchThreads` commands, each followed by `setBarrier` (sequential semantics, like one
  CUDA stream). Every dispatch therefore drains the GPU before the next starts: many small launches
  (one thread per world = 4096 threads, about 100 threads per GPU core on a 40-core M4 Max) are
  latency-bound, and capacity-sized launches that mostly exit early still cost a full drain. Kernel
  fusion and fewer launches pay more than on CUDA, where graph launches overlap tails.
  Dispatch overhead on Metal is measured at tens of microseconds per dispatch in browser WebGPU
  stacks (arXiv 2604.02344, "Characterizing WebGPU Dispatch Overhead..."), which is an upper bound for
  native ICB replay (no CPU encode per dispatch) but shows why fusion is the general lever.
- **Threadgroup sizing.** Plain launches use Warp's `block_dim` (default 256) clamped to the pipeline's
  `maxTotalThreadsPerThreadgroup`; tiled launches (`wp.launch_tiled`) use MuJoCo Warp's per-kernel
  `BlockDim` (linesearch 32, JTDAJ dense 128, Cholesky 32/64, ...), tuned for NVIDIA warps/SMs. Apple
  SIMD groups are 32 wide and the fork checks `threadExecutionWidth == 32`. Changing a tiled kernel's
  width changes its reduction partition (float noise only).
- **Knobs in the fork** (7b5c828): `WP_METAL_INFLIGHT` (committed command buffers before back-pressure,
  default 64), `WP_METAL_ICB_BATCH` (dispatches per command buffer during ICB replay, default 0 = the
  whole graph in one), `WP_METAL_BATCH` (eager dispatches per command buffer, 128), `WP_METAL_ICB=0`
  (disable ICB replay), `WP_METAL_PROFILE=1` (one command buffer per dispatch with GPU timestamps; for
  attribution only). With one graph replay per env step and one host sync per rollout, the in-flight
  bound is not expected to bind.

## 4. Candidates ranked before measuring (result-preserving only)

1. Contact / constraint capacity to the provable bound (`njmax` 256 → ≥ 97, `nconmax` 32 → ≥ 15).
2. MuJoCo Warp's sparse Jacobian path (its own default for this nv); float-noise change, verify against
   MuJoCo C on the physics protocol.
3. Post-reset kinematics only where a reader exists (flat: none in the training graph).
4. Reset path: `reset_data` launches six all-world kernels (one capacity-sized over `naconmax`);
   fuse what the next step actually reads.
5. Warp fork knobs (in-flight bound, ICB chunking) and tiled `block_dim` for Apple's 32-wide SIMD groups.
6. MPS update: 5 epochs × 4 minibatches with a `.item()` (host sync) per minibatch for the adaptive KL
   schedule; 4–5 % of the loop.

Measured outcome: see the "Results" section appended after the runs.

## 5. Results (measured 2026-09-25 on the M4 Max, GPU through the queue; logs in `runs/mjw_tp/`)

### Profile of the starting point (`runs/mjw_tp/profile.log`)

- Outside physics the env step costs **6.5 ms of 128 ms** in graph replay (reward 0.07, `reset_data` 0.13,
  contact-sensor reset 0.02, `g1_reset` 0.02, kinematics 0.13, commands + obs 0.05 ms as separate graphs; the
  rest is dispatch drains). The "~18 ms / 23 % reset-obs overhead" was the difference between physics-only
  (robots standing from reset) and the full step (random actions, more contacts): physics measured in a
  different state, not task work. Removing the post-reset kinematics gains 0 %.
- Per-kernel (WP_METAL_PROFILE=1): dense 43×43 Cholesky kernels were about two thirds of physics:
  `_tile_cholesky_factorize_solve_block` (M and M − dt·D, 3.25 ms per call, 2 per substep), the Newton
  Hessian's `_update_gradient_cholesky` (2.6 ms initial + 0.4 ms per iteration), `_update_gradient_h_incremental`
  0.19 ms per iteration. Cause: the Warp fork's register Cholesky (one SIMD group, no barriers) was limited to
  n ≤ 40; the G1's 43 dofs took the cooperative path with a threadgroup barrier per column.
- Dispatch cost in a replayed ICB graph (`metal_dispatch_cost.py`): 1.9 µs for a 4,096-thread no-op, 26 µs at
  1 M threads, 92 µs at 3.9 M threads: dispatch count is cheap, capacity-sized empty grids are not.

### Steps (4096 envs, flat, 2.5 ms; env-steps/s: physics only | full env step | rollout + inference | full PPO loop)

| step | change | physics | step | rollout | loop | loop gain | checks |
|---|---|---|---|---|---|---|---|
| 0 | as committed before (same session: 36.2K / 31.5K / 29.1K / 27.9K; ms: 113 / 130 / 141 per step, update 145–155 per iteration) | 36,202 | 31,494 | 29,134 | 27,937 | – | – |
| 1 | Warp fork b9557cb: `metal_register_cholesky_max` 48 (register path for n = 43) | 50,663 | 45,272 | 42,955 | 40,297 | +44 % | floor, eager, MuJoCo C, probe |
| 2 | `m_dense_max` 0: M and M − dt·D by MuJoCo Warp's tree-sparse L'DL | 61,470 | 53,486 | 50,471 | 46,764 | +16 % | floor, eager, MuJoCo C, probe, rough |
| 3 | MuJoCo Warp fork ad22120: one-world-per-thread sparse L'DL factor/solve on Metal (the solve ran in 1-thread threadgroups); skip the no-op gravcomp ((nworld, nbody−1, nv) = 7.7 M threads) and tendon-damping launches | 71,463 | 61,324 | 56,981 | 52,464 | +12 % | floor, eager, MuJoCo C; solve bitwise, factor bitwise on CPU |
| 4 | MuJoCo Warp fork ccfaffb: incremental Hessian update fused into the register Cholesky launch | 80,994 | 67,620 | 61,676 | 56,575 | +7.7 % | bitwise on CPU; floor, eager, MuJoCo C, probe; fork tests 316 passed |
| final (same session as a re-measured original) | | 80,469–80,885 | 67,6xx | 61,059–61,510 | **55,769–56,250** vs 28,200 | **2.0×** | |

Per control step after: physics 50.6–50.9 ms, full step 60.6 ms, rollout + inference 66.6–67.1 ms, PPO update
149–153 ms per iteration (unchanged). Rough terrain (boxes): full loop 23.7K → 41.9K (1.77×), update 310 ms.

Correctness (every step): the variant against the previous configuration stays within the run-to-run floor of
either configuration (MuJoCo Warp's atomics order contacts and rows nondeterministically; after one control
step qpos differs by ≤ 3.6e-6 rad, median 0, the same as two instances of one configuration); graph replay
equals eager launches to the same floor; the MuJoCo C protocol (PD hold + fixed random targets) gives 4.08e-6
after one control step and the same envelope later for every configuration; 20-iteration training probes
give the same episode-length trend (it 5: 115.0, it 6: 91.5–91.9, it 20: 41.5–43.1); overflow flags unchanged;
`tests/test_g1_parity.py tests/test_g1_task_terms.py tests/test_physics.py tests/test_g1_fast_factorization.py`
28 passed.

### Rejected (kept behind flags or as patches, with numbers)

| option | loop effect | how to re-enable |
|---|---|---|
| `njmax` 128 / `nconmax` 16 (provable bound 97 rows / 15 contacts) | +1.8 % | `BatchSimOptions(njmax=128, nconmax=16)` |
| sparse constraint Jacobian | +2.2 % (before step 4; the fused path is dense-only) | `BatchSimOptions(jacobian="sparse")` |
| no post-reset kinematics (flat) | 0 % | – (measurement only) |
| `WP_METAL_INFLIGHT` 16 / 256, `WP_METAL_ICB_BATCH` 256 | 0 / 0 / −11 % | environment variables |
| `BlockDim.sparse_ldl_serial` 16 | +1.1 % | `BatchSimOptions(block_dim={"sparse_ldl_serial": 16})` |
| blocked (16-wide) Hessian Cholesky | −18 % | `MJW_METAL_DENSE_CHOL_MAX=32` |
| compact per-lane storage in the register Cholesky | −11 % (n = 43: 1.41 vs 1.11 ms) | `runs/mjw_tp/rejected_warp_compact_register_cholesky.patch` |
| Schur-complement Hessian solve (core 19 + arms 24, tile ops) | 1.35 vs 1.11 ms per 4096 solves | `scripts/diagnostics/metal_schur_cost.py` |
| removing the per-minibatch KL `.item()` in the update | ≤ 1 % (update 159 vs 142–148 ms) | – |

Stop: after step 4 every remaining candidate measured under 3 % or negative. Remaining time: the 43×43 Cholesky
(fused kernel 0.25 ms per iteration, initial 1.2 ms per substep; n = 32 costs 0.27 ms, n = 43 1.11 ms, so the
second column per lane is the cost), the one-thread-per-world sparse L'DL of M (0.5 ms per call), and the Warp
MLP inference in the rollout (~6 ms per step, `metalsim/learn/warp_policy.py`: one thread per (env, output)).
