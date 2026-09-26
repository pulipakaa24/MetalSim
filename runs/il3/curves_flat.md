Isaac Lab 3.0 (newton_mjwarp, isaacsim_physx; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim runs/il3/train_flat_il3_isaaclab3_every_substep_cap20_s0.log (PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)
| iteration | Isaac newton_mjwarp: length / return / lin track / yaw track | Isaac isaacsim_physx: length / return / lin track / yaw track | MetalSim: length / return / lin track / yaw track |
|---|---|---|---|
| 50 | 53 / -5.0 / 0.012 / 0.004 | 51 / -5.0 / 0.013 / 0.004 | 50 (50) / -5.3 (-5.3) / 0.010 / 0.004 |
| 100 | 73 / -5.0 / 0.022 / 0.007 | 177 / -6.5 / 0.044 / 0.015 | 64 (66) / -5.0 (-5.0) / 0.019 / 0.007 |
| 150 | 536 / -9.4 / 0.218 / 0.061 | 978 / -6.6 / 0.486 / 0.105 | 319 (333) / -8.4 (-8.5) / 0.099 / 0.036 |
| 200 | 988 / -4.6 / 0.526 / 0.128 | 977 / +4.3 / 0.739 / 0.179 | 875 (883) / -8.1 (-8.0) / 0.380 / 0.117 |
| 250 | 982 / +3.4 / 0.729 / 0.175 | 972 / +9.4 / 0.809 / 0.245 | 951 (958) / -1.5 (-1.7) / 0.593 / 0.158 |
| 300 | 998 / +8.4 / 0.830 / 0.234 | 1000 / +14.4 / 0.872 / 0.374 | 996 (980) / +3.5 (+3.4) / 0.746 / 0.189 |
| 400 | 968 / +14.0 / 0.841 / 0.372 | 982 / +20.4 / 0.904 / 0.569 | 991 (980) / +9.1 (+8.7) / 0.823 / 0.261 |
| 500 | 1000 / +19.6 / 0.917 / 0.530 | 1000 / +23.8 / 0.922 / 0.672 | 988 (992) / +13.6 (+13.7) / 0.876 / 0.364 |
| 750 | 1000 / +24.6 / 0.935 / 0.679 | 982 / +26.2 / 0.929 / 0.745 | 977 (985) / +19.0 (+19.2) / 0.904 / 0.516 |
| 1000 | 989 / +25.9 / 0.932 / 0.723 | 998 / +28.0 / 0.937 / 0.778 | 990 (992) / +22.2 (+22.2) / 0.922 / 0.610 |
| 1250 | 997 / +27.1 / 0.940 / 0.748 | 996 / +28.3 / 0.943 / 0.789 | 996 (987) / +23.5 (+23.3) / 0.920 / 0.651 |
| 1499 | 1000 / +27.5 / 0.940 / 0.753 | 1000 / +28.9 / 0.944 / 0.795 | 995 (990) / +24.9 (+24.7) / 0.927 / 0.689 |

Per-term at iteration 1000 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9323 | +0.9367 | +0.9221 |
| track_ang_vel_z_exp | +0.7231 | +0.7784 | +0.6103 |
| feet_air_time | +0.0417 | +0.0462 | +0.0542 |
| feet_slide | -0.0121 | -0.0115 | -0.0146 |
| joint_deviation (hip+arms+fingers+torso) | -0.1446 | -0.1394 | -0.1690 |
| flat_orientation_l2 | -0.0076 | -0.0064 | -0.0049 |
| action_rate_l2 | -0.1872 | -0.1664 | -0.2319 |
| termination_penalty | -0.0022 | +0.0000 | -0.0038 |
| lin_vel_z_l2 | -0.0044 | -0.0044 | -0.0060 |
| ang_vel_xy_l2 | -0.0123 | -0.0109 | -0.0138 |
| dof_torques_l2 | -0.0080 | -0.0095 | -0.0081 |
| dof_acc_l2 | -0.0108 | -0.0103 | -0.0193 |
| dof_pos_limits | -0.0021 | -0.0029 | -0.0031 |
| falls (base_contact fraction of episode ends) | 0.0064 | 0.0012 | 0.0189 |

Per-term at iteration 1499 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9399 | +0.9436 | +0.9270 |
| track_ang_vel_z_exp | +0.7526 | +0.7946 | +0.6894 |
| feet_air_time | +0.0488 | +0.0491 | +0.0670 |
| feet_slide | -0.0107 | -0.0105 | -0.0120 |
| joint_deviation (hip+arms+fingers+torso) | -0.1399 | -0.1321 | -0.1643 |
| flat_orientation_l2 | -0.0060 | -0.0049 | -0.0048 |
| action_rate_l2 | -0.1802 | -0.1565 | -0.2166 |
| termination_penalty | -0.0017 | +0.0000 | -0.0040 |
| lin_vel_z_l2 | -0.0041 | -0.0052 | -0.0058 |
| ang_vel_xy_l2 | -0.0122 | -0.0111 | -0.0136 |
| dof_torques_l2 | -0.0079 | -0.0091 | -0.0083 |
| dof_acc_l2 | -0.0108 | -0.0101 | -0.0171 |
| dof_pos_limits | -0.0018 | -0.0022 | -0.0030 |
| falls (base_contact fraction of episode ends) | 0.0060 | 0.0027 | 0.0198 |

Throughput (training loop, env-steps/s):
  Isaac newton_mjwarp: median 59,405 (L4), total iteration time 41.3 min over 1500 iterations
  Isaac isaacsim_physx: median 47,257 (L4), total iteration time 52.2 min over 1500 iterations
  MetalSim: median 42,589 (M4 Max, incl. the monitor), 1500 iterations
