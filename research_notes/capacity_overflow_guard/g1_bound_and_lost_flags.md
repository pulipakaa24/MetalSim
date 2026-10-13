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
(`test_equality_block_ending_at_njmax_is_flagged`). Upstream fix would be `efcid > njmax - 3`; not patched here (the
fork is shared). Ours: for models with connect / weld equalities the latch counts `nefc >= njmax` as NEFC
(conservative: an exactly full, intact buffer also raises), and `Bounds.njmax` keeps one spare row above nefc for
such models so the auto bound never trips it.

## Cost

One dim-N launch per substep (8 per control step on the G1). Measured at 4096 G1 worlds, default task (elliptic,
njmax 112), exclusive timing job, 5 interleaved A/B repeats (`scripts/diagnostics/overflow_latch_cost.py`,
`runs/fixfail/latch_cost.log`): with latch 40,653 env-steps/s (100.75 ms per control step), without 40,628
(100.82 ms): within run-to-run noise (−0.06 %).
