Isaac Lab 3.0 (newton_mjwarp; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim runs/il3/train_rough_il3_isaaclab3_every_substep_cap20_s0.log (PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)
| iteration | Isaac newton_mjwarp: length / return / lin track / yaw track / level | MetalSim: length / return / lin track / yaw track / level |
|---|---|---|
| 50 | 51 / -4.8 / 0.012 / 0.008 / 0.00 | 48 (49) / -5.1 (-5.1) / 0.011 / 0.008 / 0.00 |
| 100 | 69 / -4.6 / 0.021 / 0.015 / 0.00 | 57 (58) / -4.6 (-4.7) / 0.017 / 0.012 / 0.00 |
| 150 | 292 / -6.5 / 0.109 / 0.065 / 0.06 | 153 (158) / -5.7 (-5.8) / 0.048 / 0.035 / 0.00 |
| 200 | 906 / -5.9 / 0.452 / 0.219 / 0.38 | 833 (829) / -9.8 (-9.6) / 0.338 / 0.195 / 0.05 |
| 250 | 953 / -0.6 / 0.592 / 0.300 / 1.05 | 925 (895) / -7.2 (-7.4) / 0.446 / 0.244 / 0.53 |
| 300 | 964 / +3.0 / 0.681 / 0.373 / 1.71 | 949 (932) / -3.7 (-3.7) / 0.587 / 0.295 / 1.21 |
| 400 | 983 / +7.0 / 0.736 / 0.499 / 2.99 | 949 (958) / +0.0 (+0.1) / 0.673 / 0.370 / 2.62 |
| 500 | 994 / +8.9 / 0.752 / 0.584 / 4.15 | 918 (939) / -0.1 (-0.3) / 0.660 / 0.395 / 3.84 |
| 750 | 979 / +8.2 / 0.749 / 0.617 / 5.46 | 977 (951) / +1.9 (+1.0) / 0.671 / 0.477 / 5.01 |
| 1000 | 973 / +8.9 / 0.762 / 0.657 / 5.48 | 962 (964) / +2.3 (+1.7) / 0.694 / 0.518 / 5.41 |
| 1250 | 1000 / +11.6 / 0.795 / 0.722 / 5.74 | 970 (956) / +3.9 (+3.3) / 0.709 / 0.556 / 5.58 |
| 1499 | 989 / +14.5 / 0.811 / 0.808 / 5.80 | 997 (974) / +6.1 (+5.4) / 0.747 / 0.604 / 5.70 |

Per-term at iteration 1000 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | MetalSim |
|---|---|---|
| track_lin_vel_xy_exp | +0.7616 | +0.6944 |
| track_ang_vel_z_exp | +0.6573 | +0.5183 |
| feet_air_time | +0.0041 | +0.0035 |
| feet_slide | -0.0291 | -0.0410 |
| joint_deviation (hip+arms+fingers+torso) | -0.2482 | -0.2714 |
| flat_orientation_l2 | -0.0104 | -0.0153 |
| action_rate_l2 | -0.5641 | -0.6350 |
| termination_penalty | -0.0131 | -0.0172 |
| lin_vel_z_l2 | +0.0000 | +0.0000 |
| ang_vel_xy_l2 | -0.0407 | -0.0527 |
| dof_torques_l2 | -0.0008 | -0.0008 |
| dof_acc_l2 | -0.0282 | -0.0771 |
| dof_pos_limits | -0.0204 | -0.0205 |
| falls (base_contact fraction of episode ends) | 0.0569 | 0.0860 |

Per-term at iteration 1499 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | MetalSim |
|---|---|---|
| track_lin_vel_xy_exp | +0.8110 | +0.7475 |
| track_ang_vel_z_exp | +0.8081 | +0.6045 |
| feet_air_time | +0.0047 | +0.0041 |
| feet_slide | -0.0270 | -0.0386 |
| joint_deviation (hip+arms+fingers+torso) | -0.2489 | -0.2738 |
| flat_orientation_l2 | -0.0101 | -0.0148 |
| action_rate_l2 | -0.5331 | -0.6113 |
| termination_penalty | -0.0024 | -0.0100 |
| lin_vel_z_l2 | +0.0000 | +0.0000 |
| ang_vel_xy_l2 | -0.0334 | -0.0461 |
| dof_torques_l2 | -0.0007 | -0.0007 |
| dof_acc_l2 | -0.0264 | -0.0688 |
| dof_pos_limits | -0.0210 | -0.0221 |
| falls (base_contact fraction of episode ends) | 0.0291 | 0.0501 |

Throughput (training loop, env-steps/s):
  Isaac newton_mjwarp: median 42,039 (L4), total iteration time 58.3 min over 1500 iterations
  MetalSim: median 34,775 (M4 Max, incl. the monitor), 1500 iterations
