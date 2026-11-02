# G1 rough training out of Metal memory under the elliptic default (2026-09-26)

**Symptom.** `python -m metalsim.learn.g1_velocity 4096 rough train ... 0.0025 --reward_cfg rough_isaac` failed every time
at start-up under `contact_cfg="recommended"` (elliptic, impratio 10, cap 20) with `Metal command buffer failed:
Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`, surfacing at the first synchronizing copy after
the step graphs were captured (`BoxWindow.__init__`, terrain.py:451). Pyramidal ran; the bare constructor ran.

**What the probes missed.** The CLI passes `physics_dt=0.0025` (8 substeps per 20 ms control step); the constructor
default is 5 ms (4 substeps). `G1VelocityTask(4096, terrain="rough", contact_cfg="recommended", physics_dt=0.0025)`
fails alone in a fresh process; the same with `recommended_pyramidal` builds (`probe1.log`; this note's probe scripts and
logs were not kept).

**Cause.** The Metal backend of [my fork of Warp](https://github.com/pulipakaa24/warp/tree/metalsim) keeps every buffer allocated inside a graph capture alive for the graph's
lifetime (`Graph::retained`, metal.mm; the CUDA-graph-allocation contract), and every Warp buffer sits in the queue's
single `MTLResidencySet`, so a command buffer fails once the resident total passes `recommendedMaxWorkingSetSize`
(55.7 GB on the M4 Max 64 GB). [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)'s `convex_narrowphase` allocates its EPA / multi-contact scratch afresh on
every call, sized by `naccdmax` = `naconmax` (128 contacts per world on rough terrain): 0.70 GB per call at 1024
worlds (epa_pr 285 MB, epa_vert 126 MB, epa_face / epa_norm2 95 MB each, ...), 2.8 GB at 4096. One captured
`mjw.step` allocates 776 MB (elliptic) / 751 MB (pyramidal) at 1024 worlds (`probe3.log`), so the 8-substep step graph
retains about 25 GB at 4096 worlds, and the re-capture when the contact sensor registers its substep hook holds the old
graph's buffers until the capture ends (the destroyed graph's frees are deferred while the device is capturing):
measured `MTLDevice.currentAllocatedSize` 0.16 -> 7.14 GB for the first capture and 7.21 -> 14.18 GB during the
re-capture at 1024 worlds (`probe2.log`), i.e. about 57 GB at 4096 elliptic against 55.7 GB. Elliptic's extra solver
context (hc, htot, hfactor, cone lists: +25 MB per substep per 1024 worlds) was the last 3 %; pyramidal sat just below
the limit. At 5 ms (4 substeps) it is about 32 GB, which is why the probes built.

**Fix** ([my fork of MuJoCo Warp](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim), [`d9ec218`](https://github.com/pulipakaa24/mujoco_warp/commit/d9ec218f935f2bb791b0f2d4a50047d5f7f45b0c) on [`metalsim-rough-ellip`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim-rough-ellip), now on its `metalsim` branch at
[`f824af1`](https://github.com/pulipakaa24/mujoco_warp/commit/f824af1c5f482791cca90c1a6ca6fe5adb19ce63)): on Metal the
narrowphase scratch is allocated once per `Data` and reused by every call and every graph
(`MJW_METAL_CCD_SCRATCH_CACHE=0` restores the per-call allocation). The arrays are pure per-call scratch (written
before read, indexed by the call's own ccd slot) and a Metal device runs every launch on one in-order queue, so no
result changes. Construction at 4096 elliptic, 2.5 ms: 5.5 GB resident after build, 7.6 GB peak during re-capture
(`probe4.log`). No change on CUDA or the CPU device.

**Proof.** Zero-action rollouts from the task's reset, 1024 worlds, 100 control steps, two runs per setting: mean base
drop -0.68654 / -0.68653 (cache) vs -0.68650 / -0.68652 (per-call), mean |qvel| 0.01225 / 0.01276 vs 0.01224 /
0.01267, mean contacts 5582.7 / 5581.7 vs 5583.3 / 5581.2: inside the run-to-run spread (contact append order makes
MuJoCo Warp non-bitwise across runs either way; `probe6.log`). My fork's collision_driver / collision_gjk tests pass
(99 passed, `tests.log`). Training at 4096 on the frozen worktree (`proof.log`): rough elliptic 20 iterations,
31.0 K env-steps/s; flat elliptic 20 iterations, 36.7 K env-steps/s (the three 1000-iteration flat runs: 36.3–36.8 K
at it 20).

**Headline run.** `g1_rough_ellip_default_s0` (rough, 1500 iterations, seed 0, `runs/ellip_default/run.sh`) re-queued
after `scripts/fork_sync.sh` brought the shared checkout to [`f824af1`](https://github.com/pulipakaa24/mujoco_warp/commit/f824af1c5f482791cca90c1a6ca6fe5adb19ce63): the test gate passed and it trains on the first
attempt, 30.8 K env-steps/s at iteration 20.

Probe scripts and logs (not kept): `probe_ctor.py`, `probe_mem.py`, `probe_alloc.py`, `probe_equiv.py`, and the logs
named above.
