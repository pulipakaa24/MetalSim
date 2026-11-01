# Hand-off: g1-switch-2 (2026-10-31)

## Status
**Blocked: a concurrent agent is doing this same increment in the same checkout.** I stopped before any GPU job or
any edit to shared take files, to avoid overwriting its takes and racing its edits.

Evidence, 04:52 PDT:
- `python3 scripts/gpu_lock.py status`: the holder is `g1-switch-render-flat` (pid 17344, `.venv/bin/python -m
  metalsim.render.film g1_flat --stage render`, started 04:51:29). `g1-switch-rec-rough` (`--stage record`) is waiting.
- `ps`: both wrappers are children of Claude process 40333, the same process as my own shell. They were issued as
  separate tool calls 32 s apart, so a live sibling agent is driving them; neither job is a leftover of a dead agent.
- `runs/film/g1_flat/rec/rec.json` was written at 04:51:18. Frames are landing in `runs/film/g1_flat/hero/frames/`.

## Changed
- `metalsim/render/film/provenance.py`, two edits:
  - The power-mode source comment now says what the cited sources actually support, checked with WebFetch:
    - In https://developer.apple.com/forums/thread/835131, "powermode 2 (the High Power equivalent)" is the
      **original poster's** reading. No reply documents the 0/1/2 values.
    - Apple DTS states that pmset returns "the kernel's current configuration from IOKit", and that system_profiler's
      status is the current state ("idle machines show as 'Low = Yes'"). High Power engages only near the thermal
      limit.
  - `power_text` says "power-mode setting High Power (pmset `powermode 2`)" instead of "power mode High Power", because
    it reports a configured setting, not the mode the machine ran in.
  - The stamp's dict keys are unchanged, so the sibling's record and render stamps are unaffected; only the doc
    stage's wording changes. No test references the old wording.
- `runs/readme_media/handoff/g1-switch-2.md`: this file.

## Verified
- Live state, 04:5x:
  - `pmset -g batt`: "Now drawing from 'Battery Power'", 67 %, discharging.
  - `pmset -g`: `powermode 2`.
- Fetched checkpoints, `shasum -a 256`, match `tasks.ISAAC_PRETRAINED`:
  - flat: 2,027,170 bytes, `fdf0f242...`;
  - rough: 7,842,530 bytes, `d85a9ade...`.
- The sibling's flat take, read only (`runs/film/g1_flat/rec/rec.json`):
  - Policy and take: `checkpoint` is the Isaac flat checkpoint, iteration 1499; 301 frames; `envs` [3, 1, 6, 2].
  - `gait.natural_gait_envs` is [1, 3]. Filmed env 3: cadence 3.784 / 3.784 Hz, touch-downs 3.59 / 3.79 per s, apex
    7.1 / 7.0 cm, slip 0.019 / 0.015 m/s, 0.853 m/s (command 0.982), stride 0.225 m, not terminated.
  - `hardware.power`: Battery Power, 66 %, discharging, `powermode 2`, 04:51:18.
  - Not looked at: no frames or strips. The render was still running.
- Step 1 check (`tasks.py` notes), **gaps found and not fixed**, because the sibling owns the same file:
  - The **The policy**, **Task settings** and **How the env was chosen** paragraphs of both G1_FLAT and G1_ROUGH still
    describe our g1regress checkpoints: their sha256 values, training runs, the old ranking with no natural-gait
    step, and the old envs 6 and 8.
  - Nothing yet cites Isaac Lab 2.3 rsl_rl iteration 1499 / 2999 with URL and sha256 in the prose (only the
    `AssetRef`s and `summary` do), `ISAAC_TASK_SETTINGS`, the natural-gait preference, or
    `docs/research/speculative_contacts_2026-09-27.md`.
  - The figures for that citation: §2 cadence table, and the Verdict at lines 17-20. Isaac's own L4 flat checkpoints
    (iterations 500 / 1000 / 1499) in PhysX: 3.846 / 3.597 / 3.231 Hz, strides 0.108 / 0.110 / 0.123 m. MuJoCo Warp
    adds 3.4-5.9 % cadence under every contact setting.
  - `G1_FLAT_POWER` and `G1_ROUGH_POWER` are still "PENDING".
  - There is no battery or Low Power Mode claim in the notes or summaries. `_g1_take_notes` quotes the stamped
    power only.

## Open
- Steps 1 to 5 of the brief: none done by me. Whether the sibling finishes them is unknown to me.
- The `parse_power` unit test is not added. `tests/test_film.py` is the right home, since it tests the film tool's CPU
  stages.

## Next
For the orchestrator:
1. Find the sibling agent that issued `g1-switch-render-flat` / `g1-switch-rec-rough` and let exactly one agent own
   this increment.
2. If the sibling continues, give it the `tasks.py` gaps and the provenance wording change above.
3. If it is stopped, a fresh agent can resume from its `runs/film/g1_flat/rec` take or re-record. Check
   `gpu_lock.py status` first, and verify that no `metalsim.render.film` process is running.
