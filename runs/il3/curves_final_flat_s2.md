Isaac Lab 3.0 (newton_mjwarp, isaacsim_physx; rsl_rl 5.4.1, last-100-episode means, 0-based it) vs MetalSim runs/il3/final_flat_il3_s2.log (PPOWarp, episodes finished in the iteration, 1-based it; ±5-iteration mean in brackets)
| iteration | Isaac newton_mjwarp: length / return / lin track / yaw track | Isaac isaacsim_physx: length / return / lin track / yaw track | MetalSim: length / return / lin track / yaw track |
|---|---|---|---|
| 50 | 53 / -5.0 / 0.012 / 0.004 | 51 / -5.0 / 0.013 / 0.004 | 52 (52) / -5.2 (-5.1) / 0.012 / 0.004 |
| 100 | 73 / -5.0 / 0.022 / 0.007 | 177 / -6.5 / 0.044 / 0.015 | 73 (74) / -5.0 (-5.0) / 0.023 / 0.007 |
| 150 | 536 / -9.4 / 0.218 / 0.061 | 978 / -6.6 / 0.486 / 0.105 | 665 (720) / -11.4 (-11.3) / 0.232 / 0.073 |
| 200 | 988 / -4.6 / 0.526 / 0.128 | 977 / +4.3 / 0.739 / 0.179 | 947 (967) / -5.5 (-5.3) / 0.528 / 0.123 |
| 250 | 982 / +3.4 / 0.729 / 0.175 | 972 / +9.4 / 0.809 / 0.245 | 997 (990) / +1.7 (+1.8) / 0.717 / 0.162 |
| 300 | 998 / +8.4 / 0.830 / 0.234 | 1000 / +14.4 / 0.872 / 0.374 | 983 (988) / +5.9 (+6.1) / 0.791 / 0.199 |
| 400 | 968 / +14.0 / 0.841 / 0.372 | 982 / +20.4 / 0.904 / 0.569 | 980 (985) / +11.6 (+11.9) / 0.851 / 0.308 |
| 500 | 1000 / +19.6 / 0.917 / 0.530 | 1000 / +23.8 / 0.922 / 0.672 | 1000 (994) / +16.6 (+16.4) / 0.894 / 0.408 |
| 750 | 1000 / +24.6 / 0.935 / 0.679 | 982 / +26.2 / 0.929 / 0.745 | 987 (993) / +21.0 (+21.2) / 0.919 / 0.553 |
| 1000 | 989 / +25.9 / 0.932 / 0.723 | 998 / +28.0 / 0.937 / 0.778 | 989 (997) / +23.3 (+23.4) / 0.931 / 0.633 |
| 1250 | 997 / +27.1 / 0.940 / 0.748 | 996 / +28.3 / 0.943 / 0.789 | 1000 (996) / +24.8 (+24.8) / 0.934 / 0.680 |
| 1499 | 1000 / +27.5 / 0.940 / 0.753 | 1000 / +28.9 / 0.944 / 0.795 | 998 (998) / +25.6 (+25.7) / 0.938 / 0.709 |

Per-term at iteration 1000 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9323 | +0.9367 | +0.9306 |
| track_ang_vel_z_exp | +0.7231 | +0.7784 | +0.6333 |
| feet_air_time | +0.0417 | +0.0462 | +0.0532 |
| feet_slide | -0.0121 | -0.0115 | -0.0137 |
| joint_deviation (hip+arms+fingers+torso) | -0.1446 | -0.1394 | -0.1562 |
| flat_orientation_l2 | -0.0076 | -0.0064 | -0.0047 |
| action_rate_l2 | -0.1872 | -0.1664 | -0.2208 |
| termination_penalty | -0.0022 | +0.0000 | -0.0015 |
| lin_vel_z_l2 | -0.0044 | -0.0044 | -0.0047 |
| ang_vel_xy_l2 | -0.0123 | -0.0109 | -0.0141 |
| dof_torques_l2 | -0.0080 | -0.0095 | -0.0079 |
| dof_acc_l2 | -0.0108 | -0.0103 | -0.0181 |
| dof_pos_limits | -0.0021 | -0.0029 | -0.0029 |
| falls (base_contact fraction of episode ends) | 0.0064 | 0.0012 | 0.0074 |

Per-term at iteration 1499 (Isaac: that iteration's log; MetalSim: ±5-iteration mean of the per-iteration episode means)
| term | Isaac newton_mjwarp | Isaac isaacsim_physx | MetalSim |
|---|---|---|---|
| track_lin_vel_xy_exp | +0.9399 | +0.9436 | +0.9380 |
| track_ang_vel_z_exp | +0.7526 | +0.7946 | +0.7085 |
| feet_air_time | +0.0488 | +0.0491 | +0.0659 |
| feet_slide | -0.0107 | -0.0105 | -0.0112 |
| joint_deviation (hip+arms+fingers+torso) | -0.1399 | -0.1321 | -0.1569 |
| flat_orientation_l2 | -0.0060 | -0.0049 | -0.0037 |
| action_rate_l2 | -0.1802 | -0.1565 | -0.2123 |
| termination_penalty | -0.0017 | +0.0000 | -0.0014 |
| lin_vel_z_l2 | -0.0041 | -0.0052 | -0.0047 |
| ang_vel_xy_l2 | -0.0122 | -0.0111 | -0.0133 |
| dof_torques_l2 | -0.0079 | -0.0091 | -0.0081 |
| dof_acc_l2 | -0.0108 | -0.0101 | -0.0155 |
| dof_pos_limits | -0.0018 | -0.0022 | -0.0028 |
| falls (base_contact fraction of episode ends) | 0.0060 | 0.0027 | 0.0071 |

Throughput (training loop, env-steps/s):
  Isaac newton_mjwarp: median 59,405 (L4), total iteration time 41.3 min over 1500 iterations
  Isaac isaacsim_physx: median 47,257 (L4), total iteration time 52.2 min over 1500 iterations
  MetalSim: median 57,508 (M4 Max, incl. the monitor), 1500 iterations
