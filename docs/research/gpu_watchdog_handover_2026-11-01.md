# The GPU watchdog and PPOWarp's rollout-to-update hand-over (2026-11-01)

**Symptom.** G1 PPO's first update after a rollout trained on zeros without any error: the logged KL at iteration 1
was 34.4 (flat) and 34.1 (rough) against 0.035 and 0.22 for the same seed, with identical rollout statistics, and the
rough run then raised a NaN in torch's `Normal` at iteration 2. It was seen on 2026-11-01 while other GPU processes ran
and, found afterwards, in three `runs/il3/` trainings of 2026-09-25 that ran alone in the GPU queue.

**Cause** (M4 Max, macOS 27.0.1, PyTorch 2.14.0). Probes and outputs are in
[`runs/gpu_watchdog_2026-11-01/`](../../runs/gpu_watchdog_2026-11-01).

1. **The hand-over waited on the GPU only.** `PPOWarp.rollout` queues the rollout's 24 graph replays in about 0.1 s
   and ended with `sim.after(v)`, an event wait encoded into PyTorch's command buffer
   (`metalsim/native/torch_metal_bridge.mm`, `wait_event`). PyTorch's first update command buffer (GAE, the clones
   of observations, actions and log-probabilities) was committed at the update's first `.item()` and then sat on the
   GPU for the rest of the rollout: 2.5 s for G1 flat at 4,096 envs alone (`j1_flat.out`, `tail_s`).
2. **macOS's GPU watchdog aborts a command buffer that stays on the GPU for more than 4 to 5 s, and time spent
   waiting on an event counts.** A PyTorch command buffer waiting 1, 2 or 4 s ran; at 5, 6, 7, 8 or 16 s it was
   aborted with `kIOGPUCommandBufferCallbackErrorTimeout` and the copy behind the wait never ran (`e0_wait_timeout.py`,
   `j1_e0_torch.out`, `j2_e0.out`; no other GPU load). Measured on this Mac only; a Warp command buffer is aborted the
   same way, but Warp raises for it at the next synchronize (`j1_e0_warp.out`).
3. **PyTorch 2.14 never reads a command buffer's error** (`MPSStream::commitAndWait` checks only the kernel error
   buffer, [MPSStream.mm at v2.14.0](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/mps/MPSStream.mm)).
   The aborted buffer's outputs, fresh allocations, stay zero: the stored log-probabilities are 0, the recomputed
   ones 37 x log N(0; ~0, 1) = -34.0, so the logged KL is 34.4, and zero advantages give `pg 0.000`. After repeated
   errors the driver ignores the process's submissions (`kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`), which
   gives the NaN at the next iteration. Only the MPS framework's stderr line ("Caused GPU Timeout Error") showed it.
   PyTorch 2.15.0 raises for the buffers `MPSStream` commits
   ([pytorch#198883](https://github.com/pytorch/pytorch/pull/198883)); the buffers MPSGraph commits itself are
   covered after 2.15.0 ([pytorch#200398](https://github.com/pytorch/pytorch/pull/200398)).

**The trigger is a rollout longer than the watchdog, not concurrency or memory.** Other GPU processes are one way: next
to one 8192² matmul loop the G1 flat rollout took 50.7 s instead of 2.5 s and the first update logged exactly the
incident's `pg 0.000 vf 0.024 kl 34.4073` (`j2_c1.out`). A slow task is another: `runs/il3/early_rough232task` ran
alone at 14.7 K env-steps/s, about 6.7 s per rollout, and logged `kl 34.1435`. A Mac with a slower GPU would be a third
(an inference, not measured): the 2.5 s G1 flat rollout passes 5 s on a GPU at under half the M4 Max's speed. 30 GB of
touched, idle Metal memory next to the probe left it healthy (`j2_m30.out`).

**Fix.**
- `PPOWarp.rollout` waits on the host (`sim.synchronize()`) before handing over. The update synchronizes the host
  right away anyway, so no overlap is lost, and Warp's own GPU errors now surface at the hand-over.
- The torch bridge records the command buffers it encodes a wait or signal into, and `torch_bridge.check_errors()`
  raises `MPSCommandBufferError` for one that failed. `wait_event` / `signal_event` call it, as do the update after
  its first synchronization and PPOWarp's log point.
- Next to the matmul load, flat and rough now train normally: rough iteration 1 `kl 0.2169` against 0.2170 alone
  (`j3_fix_c1.out`, `j7_rough_c1.out`). The old hand-over with the new check raises `MPSCommandBufferError` instead
  of training on zeros (`j3_guard_c1.out`).
- Tests: `tests/test_interop.py::test_watchdog_killed_torch_buffer_raises` (a child process holds a torch wait for
  8 s), `tests/test_ppo_warp_rollout.py::test_rollout_hands_over_after_the_host_waited`. The host wait exposed a race
  in that file's `_rollout` helper (its reads were never synchronized and raced the next rollout); fixed.

Rule: never encode a GPU-side wait for a producer that may need more than a few seconds; wait on the host first.

**Results it touched.** A search of every run log on the development Mac (this repository's `runs/` and the other
working trees there) for "Caused GPU Timeout Error" found today's incident and three trainings of 2026-09-25:
`runs/il3/early_rough232task` and `early_rough232task_it12` (iteration 1), and `early_recommended` (iteration 11,
`kl 54.2187`); the other hits were copies of these three in other clones.
[`isaaclab3_like_for_like_2026-09-25.md`](isaaclab3_like_for_like_2026-09-25.md) §6.3 drew on them: it called the
rough_isaac task with Isaac's settings "unstable too" from its KL 34–53, and could not take that task's policy side.
Re-run with the fix on the tree of 2026-11-01 (`rerun_il3_early.sh`): rough_isaac with Isaac's settings trains 20
iterations with 0 blow-ups and KL 0.010–0.016 after iteration 1 (`rerun_rough232task.log`), and the "recommended" row's
settings give 0 blow-ups (`rerun_recommended.log`). §6.3 now says so; its solref conclusion rests on the other rows
and stands.

**Open.**
- The per-step loops that hand over to PyTorch on the GPU (`cartpole_rgb.py`, `so101_lift.py`, `lidar_nav.py`,
  `rslrl_adapter.py`, `render/film/adapters.py`) commit one PyTorch command buffer per step, each holding a wait. If
  the host runs far ahead (up to `WP_METAL_INFLIGHT` = 64 Warp command buffers), one wait could span many steps and
  pass the watchdog under heavy load. Not tested; the new check makes it raise instead of corrupting.
- Moving to PyTorch 2.15 would add PyTorch's own check.
- The watchdog threshold was measured on one Mac with no other load.
