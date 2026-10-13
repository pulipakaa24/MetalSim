# G1 capacity bound and the overflow guard (2026-10-13)

Three `tests/test_capacity.py` tests had failed on every tree since at least 2026-10-11: the G1 bound (109 rows where
the test wanted 37 + 24·4 = 133), the task default (njmax 112, test 144), and "a too-small njmax raises" (no raise).
Versions: Warp 95f6f121 (1.18.0.dev20260917), MuJoCo Warp cf0f8ad (`upstream/mujoco_warp`), MuJoCo 3.14.0, `.venv`.

## Rows per contact

MuJoCo's efc layout (mujoco_warp `constraint.py`, same as MuJoCo C): condim 1 → 1 row; otherwise elliptic → `condim`
rows (normal + condim − 1 frictions), pyramidal → `2·(condim − 1)` (two edges per friction direction).
`capacity._rows_per_contact` does exactly that for both cones. The G1 colliders are condim 3: 3 rows elliptic, 4
pyramidal. Bound: 6 pairs × ≤ 4 contacts = 24 contacts; elliptic 37 + 72 = 109 → njmax 112, pyramidal 37 + 96 =
133 → njmax 144. The code was right; the tests were written for the pyramidal cone.

## Timeline

- 2026-09-26 04:23 `9a3ef746`: capacity bound, guard and tests, measured under the then-default pyramidal preset
  (`recommended` = `tau10_impact_hardlimits`): njmax 144, +2.5 % full env step.
- 2026-09-26 11:00 `7a385560`: G1 default `contact_cfg="recommended"` becomes elliptic (owner's decision). Bound →
  109 / 112. Tests not updated. (`302bab5c`, 01:34, was the elliptic confirming run, not the switch.)

## Test 3: why it did not raise

`scripts/diagnostics/capacity_overflow_probe.py` (64 worlds, the test's random actions, every control step):
with njmax 32 the largest nefc seen is 27 and no world ever overflows (`runs/fixfail/probe1.log`). The test's
comment "the build pose needs 32 rows (8 foot contacts)" is 8 × 4 pyramidal rows; elliptic needs 24. The scenario
no longer overflowed, so not raising was correct. `put_data` refuses njmax below the initial state's nefc (24), so
the rewritten test uses njmax 24, which genuinely overflows (nefc up to 27) within ~10 control steps.

## The real guard bug: resets erase the evidence

`mjw.reset_data(m, d, reset=mask)` sets `overflow_out[worldid] = 0` for every reset world (io.py, the reset kernel),
and the G1 / Tron1 / PPO tasks call it every control step for terminated worlds. A world whose rows were dropped
often falls and is reset, which wipes its NEFC bit before `check_overflow()` (called at PPO log points) can read it.

Measured (`runs/fixfail/cap_tests_and_probe.log`, njmax 24, 100 control steps, 64 worlds, same seed):

| tree | worlds that overflowed (seen after each step's physics) | worlds `check_overflow` reports at the end |
|---|---|---|
| origin/main 94b75be8 | 31 | 18 (13 lost to resets) |
| this branch | 31 | 31 |

In a training run with resets between log points, an overflow confined to worlds that then terminated would be
silent. Fix: `BatchSim` latches the capacity bits into its own per-world array after every substep (a substep hook,
so it is inside the step graph and also runs under `solver_presets`' substep loop); `overflow_flags()` ORs it in,
`check_overflow()` raises on it. Solver-cap bits (ITERATIONS, LS_ITERATIONS) stay live-only (`monitor.py` reports
them per log window). `CAPACITY_FLAGS` gains NVMAX ("behavior undefined") and TACTILE (dropped sensor pairs).

## Second silent drop: connect / weld blocks that end exactly at njmax

mujoco_warp `constraint.py` writes a connect equality's 3 rows (weld: 6) with
`efcid = atomic_add(nefc, worldid, 3); if efcid >= njmax - 3: return`. A block that would occupy rows
njmax − 3 … njmax − 1 fits but is dropped, and then nefc == njmax, so `_next_time`'s `nefc > njmax` check sets no
NEFC bit. The solver then runs on 3 stale rows (put_data's copy of MuJoCo C's initial rows). Demonstrated: a ball
hung by a connect equality with njmax 3 sags to z = 0.921 in 0.4 s (njmax 16: 0.9996) with no flag
(`test_equality_block_ending_at_njmax_is_flagged`). Proper fix on the fork: branch `metalsim-eq-block-njmax` of
pulipakaa24/mujoco_warp, commit `f52488b4` on top of the pinned cf0f8ad (`efcid + 3 > njmax`, `efcid + 6 > njmax`),
with `constraint_test.test_equality_block_ending_at_njmax` (connect / weld x dense / sparse, njmax = block size): 4
failed on cf0f8ad (efc rows left NaN), 4 passed with the fix; constraint_test.py 96 passed / 2 skipped, the
equality forward / solver tests 4 passed / 1 skipped (`runs/fixfail/mjw_eqfix_tests.log`). Upstream
google-deepmind/mujoco_warp main (76707759, 2026-10-12) has the same check. The pinned `metalsim` branch is not
moved yet. Workaround here until that is pinned: for models with connect / weld equalities the latch counts `nefc >= njmax` as NEFC
(conservative: an exactly full, intact buffer also raises), and `Bounds.njmax` keeps one spare row above nefc for
such models so the auto bound never trips it.

## Cost

One dim-N launch per substep (8 per control step on the G1). Measured at 4096 G1 worlds, default task (elliptic,
njmax 112), exclusive timing job, 5 interleaved A/B repeats (`scripts/diagnostics/overflow_latch_cost.py`,
`runs/fixfail/latch_cost.log`): with latch 40,653 env-steps/s (100.75 ms per control step), without 40,628
(100.82 ms): within run-to-run noise (−0.06 %).

Once the fork fix is pinned, the `nefc >= njmax` rule and the spare row (`Bounds.eq_blocks`) can be dropped: with the
fix, any block that does not fit leaves nefc > njmax, which MuJoCo Warp flags itself. The reset latch stays (it is
independent of this bug).

## Real-scale training with the latched guard (2026-10-13)

Do the sticky flags trip in real training on the shipped defaults? `scripts/diagnostics/overflow_training_check.py`
runs each task through its own training path; the PPOWarp loop calls `check_overflow()` at every log point (G1: every
iteration), and the run reports `overflow_flags()` (latched, so every world since construction) at the end.

| task (defaults) | scale | run | capacity flags | solver caps (worlds, last read) |
|---|---|---|---|---|
| G1 flat (njmax 112 / nconmax 24) | 4096, 50 PPO it | train_g1 | none at all 50 log points | ITERATIONS ≤ 335, LS_ITERATIONS ≤ 4056 |
| G1 rough, boxes_local (256 / 128) | 4096, 50 it | train_g1, curriculum | none | ITERATIONS ≤ 334 |
| G1 rough, heightfield (256 / 32) | 4096, 50 it | train_g1, curriculum | none | ITERATIONS ≤ 301 |
| Tron1 WF (64 / 16) | 8192, 50 it | PPOWarp (its own loop has no guard) | none | ITERATIONS 1 |
| SO-101 lift (512 / 48) | 64, 30 it | torch PPO of train_lift (no guard in its loop) | none | ITERATIONS 20 |
| G1 rough, boxes_local / heightfield, trained rough checkpoint, envs spread over levels 0–9 | 4096, 150 rollouts | rollout only, curriculum on | none | – |
| same, levels frozen (≈ 410 envs on each of the 10 levels for 3,600 control steps), checkpoint's own training settings | 4096, 150 rollouts | 14-inch, macOS 27 | none | – |

Logs: `runs/fixfail/train/real_*.log` (+ `.anomalies.jsonl`, every log point's flags), `runs/fixfail/train_real.log`,
`runs/fixfail/train/g1_rough_policy*.log`, `runs/fixfail/train14/` (14-inch). Those two policy rows did NOT run the
checkpoint: PPOWarp's parameters live in Warp buffers written through PyTorch's MPS queue, and nothing ordered the
`load_state_dict` copy before the Warp rollout, so the rollouts ran the random initial weights (episodes ≈ 40 control
steps, like an untrained policy; the Warp weights still differed from the checkpoint after the run). They cover
~370 K falls on every terrain level, not walking. Fixed 2026-10-13 (PPOWarp publishes loaded weights,
`tests/test_ppo_warp_rollout.py::test_loaded_weights_reach_the_warp_policy`). Re-run with the fix (16-inch, 4096 envs
over levels 0–9, boxes_local, curriculum on, 60 rollouts, the 2026-09-26 elliptic-default checkpoint
`g1_rough_ellip_default_s0.pt`; `runs/g1regress/harness_rough_policy_ellip_fixed.log`): 4,044 of 4,096 first episodes
run to the 1,000-step time-out, mean level 4.50 → 4.66, no capacity flag. The 2026-09-25 checkpoints also walk on
today's task once loaded (`runs/g1regress/`). No default capacity needed changing.

Throughput (16-inch only). Same process, latch hook attached / detached, interleaved, 5 repeats
(`scripts/diagnostics/overflow_latch_cost_tasks.py`, `runs/fixfail/latch_cost_tasks.log`; the latch is the only
per-step change of the branch):

| task | with latch | without | cost | spread without |
|---|---|---|---|---|
| G1 flat 4096 (PPO rollout) | 75,746 | 75,681 | −0.09 % | 0.3 % |
| G1 rough 4096 | 64,280 | 64,331 | +0.08 % | 0.2 % |
| Tron1 8192 | 245,713 | 246,399 | +0.28 % | 0.9 % |
| SO-101 lift 64 (env step) | 5,122 | 5,157 | +0.69 % | 7.7 % |

Branch vs main (f3b5c62e) as whole training runs, one run each, 50 iterations (`runs/fixfail/train_real.log`): G1 flat
63,073 vs 63,595, rough 50,692 vs 50,766, rough heightfield 52,623 vs 52,719, Tron1 135,654 vs 138,114, lift 2,039 vs
2,138 env-steps/s (cumulative incl. startup; single runs, so within their noise; the A/B above is the measurement).
