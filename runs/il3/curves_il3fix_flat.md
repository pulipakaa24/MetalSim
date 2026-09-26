Isaac Lab 3.0 (newton_mjwarp, isaacsim_physx; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim runs/il3/il3fix_flat_s0.log (PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)
| iteration | Isaac newton_mjwarp: length / return / lin track / yaw track | Isaac isaacsim_physx: length / return / lin track / yaw track | MetalSim: length / return / lin track / yaw track |
|---|---|---|---|
| 50 | 53 / -5.0 / 0.012 / 0.004 | 51 / -5.0 / 0.013 / 0.004 | 50 (50) / -5.3 (-5.3) / 0.010 / 0.004 |
| 100 | 73 / -5.0 / 0.022 / 0.007 | 177 / -6.5 / 0.044 / 0.015 | 63 (64) / -5.0 (-4.9) / 0.018 / 0.007 |
| 150 | 536 / -9.4 / 0.218 / 0.061 | 978 / -6.6 / 0.486 / 0.105 | 305 (286) / -8.0 (-7.8) / 0.078 / 0.031 |
| 200 | 988 / -4.6 / 0.526 / 0.128 | 977 / +4.3 / 0.739 / 0.179 | 899 (927) / -8.9 (-9.2) / 0.339 / 0.120 |
| 250 | 982 / +3.4 / 0.729 / 0.175 | 972 / +9.4 / 0.809 / 0.245 | 943 (949) / -4.4 (-4.3) / 0.483 / 0.149 |
| 300 | 998 / +8.4 / 0.830 / 0.234 | 1000 / +14.4 / 0.872 / 0.374 | 988 (981) / +0.6 (+0.5) / 0.666 / 0.175 |
| 400 | 968 / +14.0 / 0.841 / 0.372 | 982 / +20.4 / 0.904 / 0.569 | 994 (991) / +7.9 (+7.9) / 0.824 / 0.245 |
| 500 | 1000 / +19.6 / 0.917 / 0.530 | 1000 / +23.8 / 0.922 / 0.672 | 1000 (994) / +13.8 (+13.7) / 0.878 / 0.359 |
| 750 | 1000 / +24.6 / 0.935 / 0.679 | 982 / +26.2 / 0.929 / 0.745 | 994 (994) / +21.1 (+21.0) / 0.921 / 0.547 |
| 1000 | 989 / +25.9 / 0.932 / 0.723 | 998 / +28.0 / 0.937 / 0.778 | 993 (997) / +24.6 (+24.8) / 0.937 / 0.661 |
| 1250 | 997 / +27.1 / 0.940 / 0.748 | 996 / +28.3 / 0.943 / 0.789 | 1000 (996) / +27.1 (+26.9) / 0.943 / 0.724 |
| 1499 | 1000 / +27.5 / 0.940 / 0.753 | 1000 / +28.9 / 0.944 / 0.795 | 990 (997) / +27.5 (+27.8) / 0.945 / 0.757 |

Per-term at iteration 1000 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9323 | +0.9367 | +0.9375 |
| track_ang_vel_z_exp | +0.7231 | +0.7784 | +0.6605 |
| feet_air_time | +0.0417 | +0.0462 | +0.0474 |
| feet_slide | -0.0121 | -0.0115 | -0.0149 |
| joint_deviation (hip+arms+fingers+torso) | -0.1446 | -0.1394 | -0.1568 |
| flat_orientation_l2 | -0.0076 | -0.0064 | -0.0040 |
| action_rate_l2 | -0.1872 | -0.1664 | -0.1841 |
| termination_penalty | -0.0022 | +0.0000 | -0.0014 |
| lin_vel_z_l2 | -0.0044 | -0.0044 | -0.0051 |
| ang_vel_xy_l2 | -0.0123 | -0.0109 | -0.0122 |
| dof_torques_l2 | -0.0080 | -0.0095 | -0.0088 |
| dof_acc_l2 | -0.0108 | -0.0103 | -0.0162 |
| dof_pos_limits | -0.0021 | -0.0029 | -0.0023 |
| falls (base_contact fraction of episode ends) | 0.0064 | 0.0012 | 0.0072 |

Per-term at iteration 1499 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9399 | +0.9436 | +0.9453 |
| track_ang_vel_z_exp | +0.7526 | +0.7946 | +0.7570 |
| feet_air_time | +0.0488 | +0.0491 | +0.0575 |
| feet_slide | -0.0107 | -0.0105 | -0.0118 |
| joint_deviation (hip+arms+fingers+torso) | -0.1399 | -0.1321 | -0.1448 |
| flat_orientation_l2 | -0.0060 | -0.0049 | -0.0036 |
| action_rate_l2 | -0.1802 | -0.1565 | -0.1631 |
| termination_penalty | -0.0017 | +0.0000 | -0.0018 |
| lin_vel_z_l2 | -0.0041 | -0.0052 | -0.0051 |
| ang_vel_xy_l2 | -0.0122 | -0.0111 | -0.0105 |
| dof_torques_l2 | -0.0079 | -0.0091 | -0.0094 |
| dof_acc_l2 | -0.0108 | -0.0101 | -0.0149 |
| dof_pos_limits | -0.0018 | -0.0022 | -0.0028 |
| falls (base_contact fraction of episode ends) | 0.0060 | 0.0027 | 0.0088 |

Throughput (training loop, env-steps/s):
  Isaac newton_mjwarp: median 59,405 (L4), total iteration time 41.3 min over 1500 iterations
  Isaac isaacsim_physx: median 47,257 (L4), total iteration time 52.2 min over 1500 iterations
  MetalSim: median 48,940 (M4 Max, incl. the monitor), 1500 iterations
