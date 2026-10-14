# MPMCoupling coupling-layer cost: inventory, reference designs, plan (2026-10-13)

(Research by a sub-agent, source reading only; to be committed under `research_notes/MPM in main simulator/` on the
coupling-optimisation branch.)

**Bottom line.** The coupling layer (everything in `MPMCoupling.step()` except the MuJoCo Warp substeps, the MPM solve
and the solver's harvest) is about 17 dispatches per coupled step: ~25-40 us of GPU time captured, i.e. ~0.2-0.3 % of
the 13.7 ms step at 1 world and ~0.02 % of 223.6 ms at 1,024 worlds; eager ~0.3-0.5 ms of 95.1 ms at 64 worlds. The
eager/captured 3.9x gap is MuJoCo Warp's eager solver loop (graph_conditional off on Metal: all iterations run without
the ICB early exit) and the MPM step's eager launches, not the coupling. No micro-fusion inside the layer can move the
measured step by more than ~1 %.

**Inventory** (lagged, 1 coupled body): xfrc save copy (1), wrench add (1), begin-pose sync (1), BatchSim substeps (out
of scope), xfrc restore (1), end-of-step kinematics (12: MuJoCo Warp kinematics 4, com_pos 4 + tree depth, com_vel 2;
~20 for a G1-class tree), end-velocity sync (1), rewind (1, solver), MPM step (out of scope, dominant), harvest (out of
scope), one capture_launch + event signal per step. All coupling kernels are O(nworld x nbody) at the dispatch floor.

**References.** Newton's SolverCoupledProxy: rebuilds the entry's private body_f from the user's force + the proxy
wrench (never touches the user's buffer), gets the end pose from its own one-launch eval_fk (2 dispatches regardless
of tree depth), one gather kernel per sync, the coupled step captured as one graph: ~6-8 small launches. Genesis: one
fused grid kernel reads the rigid link state and writes coupling forces directly (explicit, atomics, no compliant
collider: not MetalSim's semantics).

**Ranked, worth doing** (bitwise unless noted):
1. Public `MPMCoupling.launch_step()` (no signal) so the coupled step can be captured into a policy-step graph with
   observation / policy / bookkeeping (BatchSim already has launch_step for this): one command buffer and one host
   submission per policy step instead of per coupled step.
2. Capturable per-world reset: persistent device mask written by torch (as BatchSim.reset_mask), reset graph; today
   `reset()` uploads a fresh mask from numpy and launches eagerly, which matters for RL loops resetting every step.
3. End-of-step kinematics as the MuJoCo Warp subset the proxies need (drop geom / site globals and cinert: -3 launches,
   bitwise), or a fused free-body proxy kernel (needs hash verification; general trees need an FK kernel, not bitwise
   with MuJoCo Warp's cvel). Caveat: geom poses would then lag the particles by one substep for a renderer.
4. Fuse save + add + begin-sync and restore + end-sync (-3 launches, bitwise).
Not recommended: reordering the MPM step between MuJoCo Warp's step1 / step2 (not bitwise, touches BatchSim, lags
particles); a separate wrench buffer (MuJoCo Warp patch, different summation order).

**Measurement plan** (16-inch, exclusive timing jobs): fix `scale_coupled.py --rigid-only --time` (calls mpm on None);
time rigid-only / GS / Jacobi at 1-1,024 worlds to split rigid / MPM / coupling shares; per-section eager GPU time with
`WP_METAL_PROFILE=1` as `probes/section_kprof.py` does; per-section captured graphs; interleaved A/B with ~200 steps
(the expected deltas are 0.02-0.1 ms against ~0.8 ms p10-p90 spreads); bitwise checks of rigid / wrench / particles /
proxy body_q, body_qd and the kinematics fields, Gauss-Seidel scenes only, with an A-vs-A control first.

## Results (2026-10-13, 16-inch; batch_07, batch_10)

Done: items 1 (`MPMCoupling.launch_step()`) and 2 (capturable per-world reset: persistent device mask, `reset_mask`
torch view, `launch_reset(from_sim_mask=True)`, `BatchSim.launch_reset()`, a reset graph after the first `reset`).
Items 3-4 not done: measured, the whole layer is about 0.4 ms per coupled step (2.9 % at 1 world, 0.2 % at 1,024),
so trimming or fusing its ~17 launches can save at most a fraction of that. Bitwise: the coupled trajectory with
resets is identical on main and on this branch (`hash_trajectory.py`, 8 worlds, 60 steps), and the one-graph policy
step equals the eager API at every world count measured.

Section split, ms (`scale_coupled.py --split`):

| worlds | coupled step | MuJoCo Warp substeps | coupling layer (share) | MPM + harvest | rigid-only scene |
|---|---|---|---|---|---|
| 1 | 14.01 | 2.43 | 0.41 (2.9 %) | 11.85 | 2.28 |
| 64 | 25.67 | 2.42 | 0.40 (1.5 %) | 23.27 | 2.47 |
| 256 | 66.15 | 2.64 | 0.42 (0.6 %) | 63.66 | 2.55 |
| 1,024 | 242.59 | 2.71 | 0.44 (0.2 %) | 239.49 | 2.76 |

Policy-step A/B, ms per policy step (`policy_step_ab.py`, 2 % of the worlds reset per step):

| worlds | A (eager API) | B (one graph) | saving | steps with resets | bitwise A = B |
|---|---|---|---|---|---|
| 1 | 12.92 | 12.74 | 0.18 (1.014x) | 4 / 200 | yes |
| 64 | 25.70 | 24.49 | 1.20 (1.049x) | 137 / 200 | yes |
| 1,024 | 220.03 | 217.92 | 2.11 (1.010x) | 60 / 60 | yes |

Found on the way: with one world the solver's masked reset refused grid warm starts (no separate FEM environments
below two worlds), so `mpm.reset(mask)` raised at one world on main; fixed in `MPMCoupling._solver_reset`
(`one_world_reset.py`: selected = the solver's full reset bitwise, unselected = no reset). batch_07's 1,024-world A/B
ran out of memory because every run of the probe stayed in one process; each run is its own process now.
