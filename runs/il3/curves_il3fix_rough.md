Isaac Lab 3.0 (newton_mjwarp; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim runs/il3/il3fix_rough_s0.log (PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)
| iteration | Isaac newton_mjwarp: length / return / lin track / yaw track / level | MetalSim: length / return / lin track / yaw track / level |
|---|---|---|
| 50 | 51 / -4.8 / 0.012 / 0.008 / 0.00 | 49 (49) / -5.1 (-5.0) / 0.011 / 0.007 / 0.00 |
| 100 | 69 / -4.6 / 0.021 / 0.015 / 0.00 | 59 (60) / -4.6 (-4.6) / 0.018 / 0.012 / 0.00 |
| 150 | 292 / -6.5 / 0.109 / 0.065 / 0.06 | 235 (245) / -7.1 (-7.1) / 0.069 / 0.051 / 0.00 |
| 200 | 906 / -5.9 / 0.452 / 0.219 / 0.38 | 906 (924) / -9.3 (-9.2) / 0.404 / 0.214 / 0.18 |
| 250 | 953 / -0.6 / 0.592 / 0.300 / 1.05 | 920 (937) / -3.9 (-4.1) / 0.548 / 0.262 / 0.78 |
| 300 | 964 / +3.0 / 0.681 / 0.373 / 1.71 | 977 (974) / +0.3 (+0.1) / 0.662 / 0.323 / 1.47 |
| 400 | 983 / +7.0 / 0.736 / 0.499 / 2.99 | 939 (969) / +2.4 (+3.0) / 0.694 / 0.407 / 2.87 |
| 500 | 994 / +8.9 / 0.752 / 0.584 / 4.15 | 960 (969) / +3.5 (+3.6) / 0.695 / 0.468 / 4.01 |
| 750 | 979 / +8.2 / 0.749 / 0.617 / 5.46 | 969 (969) / +3.4 (+3.4) / 0.688 / 0.540 / 5.47 |
| 1000 | 973 / +8.9 / 0.762 / 0.657 / 5.48 | 937 (961) / +4.4 (+4.0) / 0.698 / 0.570 / 5.73 |
| 1250 | 1000 / +11.6 / 0.795 / 0.722 / 5.74 | 970 (971) / +5.9 (+6.4) / 0.730 / 0.627 / 5.85 |
| 1499 | 989 / +14.5 / 0.811 / 0.808 / 5.80 | 976 (976) / +8.4 (+8.2) / 0.745 / 0.681 / 5.96 |

Per-term at iteration 1000 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | MetalSim |
|---|---|---|
| track_lin_vel_xy_exp | +0.7616 | +0.6977 |
| track_ang_vel_z_exp | +0.6573 | +0.5700 |
| feet_air_time | +0.0041 | +0.0043 |
| feet_slide | -0.0291 | -0.0357 |
| joint_deviation (hip+arms+fingers+torso) | -0.2482 | -0.2652 |
| flat_orientation_l2 | -0.0104 | -0.0166 |
| action_rate_l2 | -0.5641 | -0.6017 |
| termination_penalty | -0.0131 | -0.0177 |
| lin_vel_z_l2 | +0.0000 | +0.0000 |
| ang_vel_xy_l2 | -0.0407 | -0.0465 |
| dof_torques_l2 | -0.0008 | -0.0008 |
| dof_acc_l2 | -0.0282 | -0.0690 |
| dof_pos_limits | -0.0204 | -0.0212 |
| falls (base_contact fraction of episode ends) | 0.0569 | 0.0883 |

Per-term at iteration 1499 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | MetalSim |
|---|---|---|
| track_lin_vel_xy_exp | +0.8110 | +0.7448 |
| track_ang_vel_z_exp | +0.8081 | +0.6806 |
| feet_air_time | +0.0047 | +0.0053 |
| feet_slide | -0.0270 | -0.0324 |
| joint_deviation (hip+arms+fingers+torso) | -0.2489 | -0.2589 |
| flat_orientation_l2 | -0.0101 | -0.0202 |
| action_rate_l2 | -0.5331 | -0.5744 |
| termination_penalty | -0.0024 | -0.0138 |
| lin_vel_z_l2 | +0.0000 | +0.0000 |
| ang_vel_xy_l2 | -0.0334 | -0.0403 |
| dof_torques_l2 | -0.0007 | -0.0007 |
| dof_acc_l2 | -0.0264 | -0.0600 |
| dof_pos_limits | -0.0210 | -0.0220 |
| falls (base_contact fraction of episode ends) | 0.0291 | 0.0688 |

Throughput (training loop, env-steps/s):
  Isaac newton_mjwarp: median 42,039 (L4), total iteration time 58.3 min over 1500 iterations
  MetalSim: median 38,715 (M4 Max, incl. the monitor), 1500 iterations
