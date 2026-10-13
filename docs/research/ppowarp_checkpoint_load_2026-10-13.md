# Checkpoints loaded into PPOWarp ran the untrained weights (2026-10-13)

## Finding

While the latched overflow guard was being checked (PR #7), `scripts/diagnostics/overflow_training_check.py
g1_rough_policy*` replayed G1 rough checkpoints on today's task. Every episode ended in a fall after about 40 control
steps, on every terrain level. The open question was whether the task had changed since the checkpoints were trained
(cone, observations, gains, ...) or whether physics or task code had regressed.

It was neither. The replay never ran the checkpoint:

- `WarpMLPPolicy` (since `7d2ee8bd`, 2026-09-22) re-homes every parameter of the torch net onto a Warp buffer
  (`p.data = tb.mps_tensor(warp_array)`), so torch and the Warp rollout share the memory.
- Torch writes to that memory go through PyTorch's MPS queue. `PPOWarp.update()` orders the optimizer's writes before
  the next Warp rollout with a shared event (`tb.signal_event` + `sim.wait`). A `net.load_state_dict(...)` had no such
  ordering. The copy was not visible to Warp before the rollout, and it was still not visible after a 20-rollout replay
  (`runs/g1regress/eval_ellip_s0_nosync.log`: Warp first-layer weight != checkpoint at the start and at the end).
- The rollout therefore ran the random initial weights, which nothing else writes into the bound buffers: episode
  length 40.7, 100 % falls, as in iterations 5–10 of a training run (41.8–42.4 in the reference run).

Fix (`metalsim/learn/ppo_warp.py`): a `load_state_dict` post-hook on `PPOWarp.net` publishes the weights the same way
`update()` does, so every caller of `algo.net.load_state_dict` is covered. Test:
`tests/test_ppo_warp_rollout.py::test_loaded_weights_reach_the_warp_policy` loads random weights without any torch
synchronization and compares the rollout's Warp actor output with the reference net. It fails on main (all 64 rows off,
up to 94 absolute; `runs/g1regress/test_before_fix.log`) and passes with the fix. The `test_ppo_warp_rollout.py`,
`test_warp_policy.py` and `test_g1_task_terms.py` gate passes: 38 passed (`runs/g1regress/test_gate.log`).

## The checkpoints walk on today's main (16-inch, check jobs)

| checkpoint | settings | envs, control steps | result | log |
|---|---|---|---|---|
| `runs/ellip_default/g1_rough_ellip_default_s0.pt` (2026-09-26, elliptic default) | saved in the checkpoint (= today's defaults) | 1024, 1,200 | 1,019 / 1,024 episodes to the 1,000-step time-out, 8 falls in 1,200 steps; mean level 2.61 → 3.21 | `replay_ellip_s0_fixed.log` |
| same, load without the fix | same | 1024, 480 | ep_len 40.7, 100 % falls, mean level → 0.00 | `eval_ellip_s0_nosync.log` |
| `g1_rough_ppowarp_fixed_it1500.pt` (2026-09-25, pyramidal era, no saved settings) | its training settings: `reward_cfg=rough_isaac scan_ordering=ij terrain_collision=hfield scan_surface=grid contact_cfg=None base_velocity=origin feet_slide_velocity=origin` | 1024, 1,200 | 1,023 / 1,024 to time-out, 1 fall | `eval_legacy_it1500_contactNone.log` |
| elliptic checkpoint through the overflow harness: levels 0–9, boxes / heightfield, curriculum on / frozen | today's defaults | 4096, 3,600 | 4,025–4,043 / 4,096 first episodes to time-out in all four runs, no capacity flag | `fixed_g1_rough_policy*.log` |

All logs are under `runs/g1regress/`. Bisecting over commits was not needed: the 2026-09-25 checkpoint walks on today's
code when it is given its own settings. (The harness's `--legacy` flag does not reproduce that checkpoint's settings:
it uses the `xy` scan order, the exact scan and box collision, while that run used `ij`, grid and the heightfield.)

## Training on today's main reproduces the documented curve

G1 rough, today's main `ad97888e`, trained on the 14-inch (macOS 27, correctness run; throughput not comparable) with
the documented run's exact settings (`runs/ellip_default/run.sh`: 4096 envs, physics dt 0.0025, `--reward_cfg
rough_isaac`, seed 0; the run header is identical to the reference's), 1000 iterations. Log
`runs/g1regress/ckpt14/rough_main_s0.log`. Reference: `runs/ellip_default/g1_rough_ellip_default_s0.log` (2026-09-26,
PARITY §1.5). Columns are return / episode length / mean terrain level.

| iteration | reference (2026-09-26) | today's main |
|---|---|---|
| 100 | −4.71 / 71 / 0.00 | −4.69 / 70 / 0.00 |
| 150 | −6.47 / 203 / 0.00 | −5.84 / 176 / 0.00 |
| 200 | −9.33 / 846 / 0.08 | −9.81 / 803 / 0.13 |
| 300 | −2.77 / 971 / 1.08 | −3.85 / 965 / 1.09 |
| 400 | +2.82 / 974 / 2.34 | +2.29 / 978 / 2.38 |
| 500 | +4.19 / 948 / 3.55 | +4.36 / 976 / 3.56 |
| 700 | +7.01 / 966 / 5.03 | +6.63 / 983 / 5.29 |
| 1000 | +8.23 / 973 / 5.65 | +7.55 / 985 / 5.86 |

The resulting policy walks when replayed (16-inch, `runs/g1regress/replay_main_rough_it1000.log`): 1,016 of 1,024
episodes run to the time-out, 9 falls over 1,200 control steps. Spread over levels 0–9 at 4096 envs
(`main_rough_it1000_all_levels.log`): 3,983 of 4,096 first episodes run to the time-out.

G1 flat, same protocol (`run.sh` flat settings: dt 0.0025, `--reward_cfg flat`, seed 0, identical run header),
500 iterations on the 14-inch (`runs/g1regress/ckpt14/flat_main_s0.log`). Reference: the three elliptic-default seeds
`runs/ellip_default/g1_flat_ellip_default_s{0,1,2}.log` (PARITY §1.5). Columns are return / episode length.

| iteration | reference s0 / s1 / s2 | today's main (s0) |
|---|---|---|
| 100 | −5.09 / 84, −5.11 / 85, −5.12 / 86 | −5.12 / 84 |
| 200 | −5.92 / 957, −5.46 / 934, −5.63 / 903 | −4.43 / 942 |
| 300 | +6.41 / 987, +7.17 / 992, +8.65 / 1000 | +9.68 / 997 |
| 400 | – | +14.83 / 984 |
| 500 | +17.43 / 1000, +17.37 / 1000, +17.02 / 1000 | +18.64 / 996 |

Today's flat run is at or slightly above the documented seeds from iteration 200 on, so it is not worse. Replayed
(`runs/g1regress/replay_main_flat_it500.log`): 1,012 of 1,024 episodes run to the time-out, 17 falls over 1,200 control
steps.

## Blast radius

Affected: code that loads a state dict into a net after a `WarpMLPPolicy` has bound it, i.e. `PPOWarp.net` after
`PPOWarp(...)` is constructed. Unaffected: training itself (the initial weights are copied into Warp at bind time
through a synchronizing `.cpu().numpy()`, and every update is event-ordered), and every evaluator that builds its own
torch module.

| path | loads into a Warp-bound net? | documented results produced through it | verdict |
|---|---|---|---|
| `scripts/diagnostics/overflow_training_check.py g1_rough_policy*` (`f4afcff1`, 2026-10-13 02:54) | yes | the two trained-policy rows of `research_notes/capacity_overflow_guard/g1_bound_and_lost_flags.md` (`runs/fixfail/train/g1_rough_policy*`, `runs/fixfail/train14/`) | **invalid**: they ran the untrained weights. Corrected in the note, re-run with the fix (four runs, last two rows). The conclusion is unchanged: no capacity flag, and no default changed |
| `metalsim/learn/tron1_wf.py train(resume=...)` / `--resume` (since `293d38a5`, 2026-09-24) | yes | none: `runs/tron1_wf/{easy1,v1,v2}/train.log` each have one start header, so none was resumed, and `docs/TRON1.md` documents no resume | would have run the first rollout of a resumed run on the initial weights (the first update then queues behind the copy). Fixed by the hook |
| `scripts/diagnostics/ppo_update_profile.py` (`9a3ef746`, 2026-09-26) | yes, to restore the net between timed repeats | PPO update timings | not affected: the timed work is the torch update, which reads the parameters on the same MPS queue as the copy, and `torch.mps.synchronize()` follows each load inside the timed loop. No Warp rollout reads the restored weights |
| `scripts/diagnostics/g1_policy_replay.py` (new here) | yes | this note | written after the fix; it prints whether the Warp weights match the checkpoint |
| G1 evaluators: `g1_reward_terms.py`, `g1_rough_transfer.py` (`runs/g1_rough_transfer*.jsonl`, PARITY §1.5 transfer numbers), `il3_transfer.py`, `newton_transfer.py`, `newton_stance_gap.py`, `g1_swing.py`, `g1_pushoff.py`, `contact_research/air_time_rollout.py`, `competitors/{g1_falls,g1_state_dump,cap_probe_*}.py`, `speculative/overtravel_probe.py`, `runs/il3/{blowup_probe,policy_stats}.py`, `metalsim/parity/side_by_side.py` (stage videos), `docs/gallery/scripts/14_g1_policy_video.py` | no: each builds its own `ActorCriticMLP` / `nn.Sequential` and evaluates it in torch, with no `WarpMLPPolicy` | PARITY §1.5 transfer, reward-term and swing / push-off results; stage videos; gallery video | not affected |
| Isaac-side plays of our checkpoints (`runs/parity/isaac/rough/play_rough/L*/metalsim_ours_it*`, `metalsim/parity/isaac_side/play_policy.py`) | no: torch `nn.Sequential` on CUDA inside Isaac | PARITY §1.5 transfer into Isaac | not affected |
| Franka-Pour (`metalsim/learn/franka_pour_ppo.py`, `franka_pour_eval.py`, PR #6) | no: `PourPPO` keeps its actor and critic as ordinary torch modules on the env device and acts in torch. `franka_pour*.py` contain no `WarpMLPPolicy`, `PPOWarp`, `warp_policy` or `.data =` (grep, exit 1). Eval: `PPO.Actor(...).load_state_dict(ckpt["actor_state_dict"])` then torch inference; resume: `PourPPO.load_state_dict` into torch modules + optimizer | Franka-Pour eval and resume numbers (PARITY §1b) | not affected |

Searched: every `load_state_dict` / `torch.load` in `metalsim/`, `scripts/`, `tests/`, `runs/` and `docs/gallery/`,
and every user of `WarpMLPPolicy` / `PPOWarp(` (`git grep`).
