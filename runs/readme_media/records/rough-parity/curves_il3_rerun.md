Isaac Lab 3.0 (newton_mjwarp; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim runs/rough_rerun_2026-10-31/logs/il3fix_rough_s0.log (PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)
| iteration | Isaac newton_mjwarp: length / return / lin track / yaw track / level | MetalSim: length / return / lin track / yaw track / level |
|---|---|---|
| 50 | 51 / -4.8 / 0.012 / 0.008 / 0.00 | 49 (49) / -4.9 (-4.9) / 0.011 / 0.008 / 0.00 |
| 100 | 69 / -4.6 / 0.021 / 0.015 / 0.00 | 62 (63) / -4.6 (-4.6) / 0.019 / 0.013 / 0.00 |
| 150 | 292 / -6.5 / 0.109 / 0.065 / 0.06 | 273 (284) / -7.1 (-7.2) / 0.094 / 0.060 / 0.01 |
| 200 | 906 / -5.9 / 0.452 / 0.219 / 0.38 | 910 (921) / -9.0 (-8.8) / 0.413 / 0.212 / 0.25 |
| 250 | 953 / -0.6 / 0.592 / 0.300 / 1.05 | 953 (956) / -3.9 (-4.1) / 0.548 / 0.265 / 0.79 |
| 300 | 964 / +3.0 / 0.681 / 0.373 / 1.71 | 978 (979) / -0.3 (-0.1) / 0.656 / 0.323 / 1.46 |
| 400 | 983 / +7.0 / 0.736 / 0.499 / 2.99 | 966 (968) / +2.2 (+2.1) / 0.681 / 0.392 / 2.68 |
| 500 | 994 / +8.9 / 0.752 / 0.584 / 4.15 | 968 (961) / +3.6 (+3.4) / 0.694 / 0.453 / 3.73 |
| 750 | 979 / +8.2 / 0.749 / 0.617 / 5.46 | 958 (968) / +4.5 (+4.4) / 0.704 / 0.541 / 5.33 |
| 1000 | 973 / +8.9 / 0.762 / 0.657 / 5.48 | 999 (976) / +7.1 (+6.5) / 0.736 / 0.604 / 5.83 |
| 1250 | 1000 / +11.6 / 0.795 / 0.722 / 5.74 | 992 (983) / +9.6 (+9.1) / 0.766 / 0.668 / 6.04 |
| 1499 | 989 / +14.5 / 0.811 / 0.808 / 5.80 | 985 (981) / +12.0 (+11.5) / 0.780 / 0.730 / 5.93 |

Per-term at iteration 1000 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | MetalSim |
|---|---|---|
| track_lin_vel_xy_exp | +0.7616 | +0.7362 |
| track_ang_vel_z_exp | +0.6573 | +0.6044 |
| feet_air_time | +0.0041 | +0.0042 |
| feet_slide | -0.0291 | -0.0340 |
| joint_deviation (hip+arms+fingers+torso) | -0.2482 | -0.2532 |
| flat_orientation_l2 | -0.0104 | -0.0163 |
| action_rate_l2 | -0.5641 | -0.5757 |
| termination_penalty | -0.0131 | -0.0107 |
| lin_vel_z_l2 | +0.0000 | +0.0000 |
| ang_vel_xy_l2 | -0.0407 | -0.0455 |
| dof_torques_l2 | -0.0008 | -0.0007 |
| dof_acc_l2 | -0.0282 | -0.0636 |
| dof_pos_limits | -0.0204 | -0.0192 |
| falls (base_contact fraction of episode ends) | 0.0569 | 0.0537 |

Per-term at iteration 1499 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | MetalSim |
|---|---|---|
| track_lin_vel_xy_exp | +0.8110 | +0.7802 |
| track_ang_vel_z_exp | +0.8081 | +0.7297 |
| feet_air_time | +0.0047 | +0.0046 |
| feet_slide | -0.0270 | -0.0311 |
| joint_deviation (hip+arms+fingers+torso) | -0.2489 | -0.2444 |
| flat_orientation_l2 | -0.0101 | -0.0168 |
| action_rate_l2 | -0.5331 | -0.5305 |
| termination_penalty | -0.0024 | -0.0067 |
| lin_vel_z_l2 | +0.0000 | +0.0000 |
| ang_vel_xy_l2 | -0.0334 | -0.0381 |
| dof_torques_l2 | -0.0007 | -0.0006 |
| dof_acc_l2 | -0.0264 | -0.0531 |
| dof_pos_limits | -0.0210 | -0.0185 |
| falls (base_contact fraction of episode ends) | 0.0291 | 0.0333 |

Throughput (training loop, env-steps/s):
  Isaac newton_mjwarp: median 42,039 (L4), total iteration time 58.3 min over 1500 iterations
  MetalSim: median 27,330 (M4 Max, incl. the monitor), 1500 iterations
