# BatchSimOptions(deterministic=True): how Warp's deterministic mode reaches MuJoCo Warp's kernels (2026-10-13)

Why: batch_02's diagnosis showed MuJoCo Warp itself is not run-to-run repeatable at 1,024 worlds (a box resting on the
floor, no sand: two runs differ from step 18; the coupled Jacobi scene from step 34 in every part at once,
`runs/mpm_main/batch_02/STATUS.txt`). The coupled scene can only repeat if the rigid side does. Source reading of the
Warp fork (`95f6f121`), MuJoCo Warp fork and Newton 1.5.2's `SolverMuJoCo`; the coupled-speed agent's note
(`research_notes/coupled_speed/NOTE.md` on its branch) reached the same conclusions independently.
`W` = `upstream/warp-innate/warp/_src`, `MJW` = `upstream/mujoco_warp/mujoco_warp/_src`,
`N` = `upstream/newton-metalsim-1.5.2/newton/_src`.

## How the mode reaches a kernel (verified)

* `Module.__init__` copies `wp.config.deterministic` / `deterministic_max_records` into `module.options` when the module
  object is created, i.e. at import for ordinary modules (W/context.py:4264-4265); `wp.config.deterministic` is not
  read again on the launch path.
* Both options are resolved into the module hash (`resolve_options` 4288-4349, `ModuleHasher` ~3095; a non-default mode
  also salts `metal_deterministic:2`, 3102-3107) and the same resolved dict drives codegen (W/deterministic.py:428-441),
  so hash and build agree on the mode.
* `set_module_options` after load: updates options and `mark_modified()` clears hashers / resolved options but not the
  loaded executables (13870-13916, 5348-5355). The next `Module.load` re-hashes and loads or compiles the other variant
  (5132-5143). Launch metadata (`det_meta`) is snapshotted at load: from the compile on a cold cache, rebuilt from the
  adjoints on a cache hit (4655-4682). Upstream documents "set before the module compiles"; changing the mode after the
  first launch is undocumented.
* `module="unique"` kernels copy `wp.config` at decoration and hash immediately (2093-2160), so they take whatever config
  is active when MuJoCo Warp's factories create them. MuJoCo Warp's `@cache_kernel` factories (~89) cache by argument
  sizes and function name only (MJW/warp_util.py:148-165): a kernel generated in one mode is handed out in the other
  unless `_KERNEL_CACHE` is cleared. `reset_data` creates unique kernels on every call (MJW/io.py ~2553).
* A captured graph holds its `ModuleExec`s and replays them (W/context.py 6670, 6734, 12701, 12756): a mode change needs a
  re-capture.

## How Newton's SolverMuJoCo applies it (verified, N/solvers/mujoco/solver_mujoco.py)

Options on every loaded `mujoco_warp._src.*` module *before* `put_model` / `put_data` (3543-3552, 600-611); record bound
for `smooth` computed after `put_data` (305-326) and re-applied (7483-7488); `_prepare_generated_kernels` clears
`_KERNEL_CACHE` whenever the class-level (mode, records) pair changes (587-598); every step / reset / model change runs
inside `_scoped_mujoco_warp_execution` (3820-3825): re-apply options changed by another solver (N/solvers/solver.py:
204-219, set-only-if-changed 13-18), clear generated kernels if needed, and hold `wp.config.deterministic` and
`deterministic_max_records` (the computed bound, not 0) for lazily generated kernels (3803-3817).

## Where MetalSim's first version (batch_02) is exposed

1. Options set after `put_model` / `put_data` (both launch kernels): those modules load once in default mode; on a cold
   cache they compile twice. The coupled-speed agent's open bug (`test_coupled_contact_filter_keeps_source_order`) is in
   this family ("mode changed after first launch"), although its logs show the failure is not cache temperature.
2. `_KERNEL_CACHE` cleared once: a default-mode BatchSim alive in the same process refills it on any uncaptured launch or
   re-capture, and the deterministic sim then picks those kernels up.
3. Shared module options can be switched back by any later default-mode solver; never re-applied.
4. `det_scope` sets `deterministic_max_records = 0` (Newton: the computed bound). On Metal RUN_TO_RUN float sums use exact
   integer bins (no records); only consumed-return counters in dynamic loops use the bound
   (W/metal_deterministic.py:1231), so the gap is small but real.
5. Deterministic overflow is only checked on uncaptured launches and reported once per process
   (W/metal_deterministic.py:1295-1332); `BatchSim.check_overflow` does not read it.
6. Not about determinism but found on the way: `BatchSim` sets `wp.config.metal_register_cholesky_max` process-wide,
   and that value is read both at hash time and at build time; a module hashed under one value and built after a change
   poisons a cold cache. Pre-existing; logged for the owner of BatchSim.

Metal limits of the fork's deterministic mode (verified): 64-bit CAS / exchange and int64 min/max refused at launch;
deterministic float accumulation for f16 / f32 / bf16 only; negative-stride destinations refused; at most 8
consumed-return counter targets per kernel; `deterministic_debug` is a no-op on Metal; RUN_TO_RUN = exact integer bins
(bitwise independent of thread order), GPU_TO_GPU = record-and-sort path sized by max_records.

## Design for batch_03 (`metalsim/physics/batch.py`)

* One mode per process, enforced: the first `BatchSim` fixes the process's MuJoCo Warp mode (RUN_TO_RUN with
  `deterministic=True`); constructing a `BatchSim` with the other mode afterwards raises (mixing would need Newton's full
  revision machinery, and no MetalSim workflow mixes them). A deterministic `BatchSim` also raises if any MuJoCo Warp
  module has already been loaded (launched) in default mode.
* Set the mode (records 0) on every loaded MuJoCo Warp module and clear `_KERNEL_CACHE` *before* `put_model`, and hold
  `det_scope` around `put_model` / `put_data`; after `put_data`, set `smooth`'s record bound (it is not launched by
  `put_model` / `put_data`; checked by the runtime verification below) and use it inside `det_scope`.
* Runtime verification after capture: every MuJoCo Warp / unique module executable referenced by the captured graphs has
  `resolved_options["deterministic"] == RUN_TO_RUN`; `check_overflow()` also reads the deterministic overflow flag.
* Tests (`tests/test_batch_deterministic.py`, each in a subprocess): repeatability with a cold kernel cache (fresh
  `WARP_CACHE_PATH`, twice) and warm, all three hashes equal; the refusal when a default-mode MuJoCo Warp module was
  already launched; a negative control (default mode diverges on the same scene at 1,024 worlds, as batch_02 measured).
