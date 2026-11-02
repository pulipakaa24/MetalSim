
## 2.3.2 seed 0
- old: `runs/g1_rough_ppowarp_fixed.log` (1500 iterations; restarts cut: 0)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 0, reward_cfg rough_isaac
- new: `runs/rough_rerun_2026-10-31/logs/g1_rough_ppowarp_fixed_s0.log` (1500 iterations; restarts cut: 0)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 0, reward_cfg rough_isaac, height-scan ray order ij, terrain collision hfield, height scan grid, solver_cfg None, contact_cfg default, base velocity origin, feet-slide velocity origin, initial reset True, effort limit on the actuator

| it | old len / ret / level | new len / ret / level | old ret ±5 | new ret ±5 |
|---|---|---|---|---|
| 100 | 80 / -4.82 / 0.000 | 85 / -4.88 / 0.000 | -4.83 | -4.90 |
| 150 | 510 / -9.98 / 0.014 | 522 / -9.91 / 0.019 | -9.82 | -9.96 |
| 200 | 949 / -7.17 / 0.114 | 966 / -7.42 / 0.122 | -6.29 | -7.48 |
| 300 | 984 / +4.03 / 1.608 | 957 / +0.79 / 1.183 | +3.81 | +0.93 |
| 500 | 976 / +8.44 / 4.407 | 984 / +5.44 / 3.530 | +8.55 | +5.36 |
| 750 | 982 / +8.59 / 5.790 | 984 / +6.69 / 5.146 | +8.60 | +6.72 |
| 1000 | 991 / +11.33 / 6.014 | 997 / +9.71 / 5.506 | +11.15 | +9.64 |
| 1250 | 994 / +14.16 / 6.092 | 998 / +13.52 / 5.818 | +13.79 | +13.47 |
| 1499 | 992 / +16.63 / 6.098 | 989 / +16.35 / 6.125 | +16.58 | +15.93 |
| 1500 | 992 / +16.42 / 6.097 | 1000 / +15.96 / 6.125 | +16.62 | +15.91 |
- old: blow-ups 0 in 0 iterations (it <= 30: 0, it >= 100: 0, first/last None); dropped-row lines 0; penetration anomalies {'all': (1497, 2.9, 21.9), 'it1-50': (50, 2.9, 16.2), 'it1400-1500': (101, 4.8, 18.9)}; contact-force anomalies {'all': (0, None, None), 'it1-50': (0, None, None), 'it1400-1500': (0, None, None)}; joint-speed anomalies {'all': (0, None, None)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 4.22 (14-inch for new: not a performance figure)
- new: blow-ups 0 in 0 iterations (it <= 30: 0, it >= 100: 0, first/last None); dropped-row lines 0; penetration anomalies {'all': (1499, 3.1, 8.4), 'it1-50': (50, 3.6, 6.2), 'it1400-1500': (101, 3.9, 7.5)}; contact-force anomalies {'all': (0, None, None), 'it1-50': (0, None, None), 'it1400-1500': (0, None, None)}; joint-speed anomalies {'all': (0, None, None)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 2.75 (14-inch for new: not a performance figure)

## 2.3.2 seed 1
- old: `runs/g1_rough_ppowarp_fixed_seed1.log` (1500 iterations; restarts cut: 0)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 1, reward_cfg rough_isaac, height-scan ray order xy, terrain collision boxes_local, height scan exact
- new: `runs/rough_rerun_2026-10-31/logs/g1_rough_ppowarp_fixed_s1.log` (1500 iterations; restarts cut: 0)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 1, reward_cfg rough_isaac, height-scan ray order xy, terrain collision boxes_local, height scan exact, solver_cfg None, contact_cfg default, base velocity origin, feet-slide velocity origin, initial reset True, effort limit on the actuator

| it | old len / ret / level | new len / ret / level | old ret ±5 | new ret ±5 |
|---|---|---|---|---|
| 100 | 74 / -4.70 / 0.000 | 77 / -4.71 / 0.000 | -4.71 | -4.72 |
| 150 | 377 / -8.56 / 0.008 | 427 / -8.84 / 0.009 | -8.86 | -9.13 |
| 200 | 830 / -7.36 / 0.133 | 845 / -8.46 / 0.068 | -7.10 | -8.32 |
| 300 | 951 / -0.37 / 1.293 | 933 / -1.75 / 1.046 | -0.20 | -1.41 |
| 500 | 968 / +4.21 / 3.413 | 971 / +3.68 / 3.255 | +4.38 | +3.87 |
| 750 | 973 / +7.97 / 5.109 | 980 / +6.41 / 5.023 | +7.57 | +6.39 |
| 1000 | 999 / +11.01 / 5.601 | 991 / +11.39 / 5.561 | +10.99 | +11.35 |
| 1250 | 992 / +15.11 / 5.876 | 993 / +16.44 / 6.007 | +15.23 | +16.62 |
| 1499 | 1000 / +18.87 / 6.024 | 998 / +20.85 / 6.106 | +18.89 | +20.54 |
| 1500 | 996 / +19.08 / 6.026 | 998 / +20.61 / 6.108 | +18.93 | +20.58 |
- old: blow-ups 0 in 0 iterations (it <= 30: 0, it >= 100: 0, first/last None); dropped-row lines 1; penetration anomalies {'all': (1487, 3.1, 16.5), 'it1-50': (49, 3.8, 9.0), 'it1400-1500': (100, 4.3, 9.7)}; contact-force anomalies {'all': (602, 15870.0, 56259.0), 'it1-50': (6, 17159.0, 52093.0), 'it1400-1500': (12, 20796.0, 50477.0)}; joint-speed anomalies {'all': (0, None, None)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 2.59 (14-inch for new: not a performance figure)
- new: blow-ups 0 in 0 iterations (it <= 30: 0, it >= 100: 0, first/last None); dropped-row lines 0; penetration anomalies {'all': (1496, 3.0, 27.2), 'it1-50': (48, 3.4, 6.9), 'it1400-1500': (101, 4.5, 11.1)}; contact-force anomalies {'all': (282, 15822.0, 44202.0), 'it1-50': (0, None, None), 'it1400-1500': (15, 17601.0, 32162.0)}; joint-speed anomalies {'all': (0, None, None)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 2.74 (14-inch for new: not a performance figure)

## 2.3.2 seed 2
- old: `runs/g1_rough_ppowarp_fixed_seed2.log` (1500 iterations; restarts cut: 1)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 2, reward_cfg rough_isaac, height-scan ray order xy, terrain collision boxes_local, height scan exact
- new: `runs/rough_rerun_2026-10-31/logs/g1_rough_ppowarp_fixed_s2.log` (1500 iterations; restarts cut: 0)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 2, reward_cfg rough_isaac, height-scan ray order xy, terrain collision boxes_local, height scan exact, solver_cfg None, contact_cfg default, base velocity origin, feet-slide velocity origin, initial reset True, effort limit on the actuator

| it | old len / ret / level | new len / ret / level | old ret ±5 | new ret ±5 |
|---|---|---|---|---|
| 100 | 76 / -4.72 / 0.000 | 74 / -4.67 / 0.000 | -4.70 | -4.66 |
| 150 | 349 / -8.40 / 0.004 | 321 / -7.96 / 0.006 | -8.19 | -7.90 |
| 200 | 832 / -8.35 / 0.050 | 814 / -9.29 / 0.046 | -8.14 | -8.89 |
| 300 | 971 / -0.10 / 0.669 | 952 / -2.62 / 0.582 | -0.09 | -2.85 |
| 500 | 958 / +4.98 / 3.051 | 973 / +2.89 / 2.457 | +5.02 | +3.04 |
| 750 | 974 / +10.31 / 5.477 | 977 / +5.61 / 4.684 | +10.76 | +6.42 |
| 1000 | 991 / +14.25 / 5.881 | 998 / +10.34 / 5.382 | +14.08 | +10.17 |
| 1250 | 992 / +18.21 / 6.128 | 1000 / +13.84 / 5.696 | +18.35 | +13.98 |
| 1499 | 998 / +22.17 / 6.076 | 991 / +17.38 / 6.117 | +22.45 | +17.30 |
| 1500 | 998 / +22.62 / 6.079 | 1000 / +17.43 / 6.111 | +22.50 | +17.28 |
- old: blow-ups 0 in 0 iterations (it <= 30: 0, it >= 100: 0, first/last None); dropped-row lines 1; penetration anomalies {'all': (1500, 3.4, 13.8), 'it1-50': (50, 3.8, 9.0), 'it1400-1500': (101, 4.4, 11.4)}; contact-force anomalies {'all': (440, 15877.0, 51504.0), 'it1-50': (8, 16027.0, 51504.0), 'it1400-1500': (2, 22107.0, 22166.0)}; joint-speed anomalies {'all': (0, None, None)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 2.59 (14-inch for new: not a performance figure)
- new: blow-ups 0 in 0 iterations (it <= 30: 0, it >= 100: 0, first/last None); dropped-row lines 0; penetration anomalies {'all': (1488, 3.1, 13.8), 'it1-50': (47, 3.9, 9.3), 'it1400-1500': (101, 4.4, 12.1)}; contact-force anomalies {'all': (306, 15830.0, 46274.0), 'it1-50': (0, None, None), 'it1400-1500': (14, 19692.0, 22313.0)}; joint-speed anomalies {'all': (0, None, None)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 2.69 (14-inch for new: not a performance figure)

## rough_il3 seed 0
- old: `runs/il3/il3fix_rough_s0.log` (1500 iterations; restarts cut: 0)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 0, reward_cfg rough_il3, height-scan ray order xy, terrain collision hfield, height scan exact, solver_cfg isaaclab3_every_substep_cap20, contact_cfg recommended, base velocity com, feet-slide velocity com, Isaac Lab 3.0 events True (backend newton_mjwarp), add_base_mass 
- new: `runs/rough_rerun_2026-10-31/logs/il3fix_rough_s0.log` (1500 iterations; restarts cut: 0)
  header: G1 rough PPO: N=4096 obs_dim 310 act_dim 37 rollout 24 x 1500 iterations, engine mjwarp, physics dt 0.0025 (decimation 8), seed 0, reward_cfg rough_il3, height-scan ray order xy, terrain collision hfield, height scan exact, solver_cfg isaaclab3_every_substep_cap20, contact_cfg recommended_pyramidal, base velocity com, feet-slide velocity com, initial reset True, effort limit on the actuator, Isaac

| it | old len / ret / level | new len / ret / level | old ret ±5 | new ret ±5 |
|---|---|---|---|---|
| 100 | 59 / -4.64 / 0.000 | 62 / -4.62 / 0.000 | -4.65 | -4.62 |
| 150 | 235 / -7.06 / 0.003 | 273 / -7.14 / 0.006 | -7.13 | -7.25 |
| 200 | 906 / -9.31 / 0.185 | 910 / -9.04 / 0.246 | -9.25 | -8.84 |
| 300 | 977 / +0.27 / 1.467 | 978 / -0.28 / 1.460 | +0.10 | -0.07 |
| 500 | 960 / +3.47 / 4.012 | 968 / +3.65 / 3.728 | +3.65 | +3.44 |
| 750 | 969 / +3.36 / 5.466 | 958 / +4.53 / 5.331 | +3.39 | +4.38 |
| 1000 | 937 / +4.42 / 5.734 | 999 / +7.11 / 5.829 | +3.95 | +6.52 |
| 1250 | 970 / +5.89 / 5.846 | 992 / +9.60 / 6.036 | +6.38 | +9.11 |
| 1499 | 969 / +8.00 / 5.954 | 973 / +11.17 / 5.933 | +8.21 | +11.46 |
| 1500 | 976 / +8.38 / 5.957 | 985 / +12.02 / 5.933 | +8.16 | +11.49 |
- old: blow-ups 1010 in 163 iterations (it <= 30: 823, it >= 100: 187, first/last (10, 1500)); dropped-row lines 0; penetration anomalies {'all': (67, 2.0, 17.3), 'it1-50': (3, 2.1, 17.3), 'it1400-1500': (6, 2.1, 5.0)}; contact-force anomalies {'all': (9, 16055.0, 277931.0), 'it1-50': (1, 277931.0, 277931.0), 'it1400-1500': (0, None, None)}; joint-speed anomalies {'all': (201, 122.0, 931.0)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 2.54 (14-inch for new: not a performance figure)
- new: blow-ups 67 in 61 iterations (it <= 30: 0, it >= 100: 67, first/last (1012, 1500)); dropped-row lines 0; penetration anomalies {'all': (48, 2.0, 3.3), 'it1-50': (2, 2.1, 2.6), 'it1400-1500': (4, 2.2, 2.7)}; contact-force anomalies {'all': (31, 15826.0, 41069.0), 'it1-50': (0, None, None), 'it1400-1500': (4, 17052.0, 21326.0)}; joint-speed anomalies {'all': (53, 200.0, 725.0)}; EPA_HORIZON lines 0; other overflow lines []; median s/it 100-1500 3.66 (14-inch for new: not a performance figure)

per-term ±5 mean at 1000:
  action_rate_l2: old -0.6017 new -0.5757
  ang_vel_xy_l2: old -0.0465 new -0.0455
  base_contact: old +0.0883 new +0.0537
  dof_acc_l2: old -0.0690 new -0.0636
  dof_pos_limits: old -0.0212 new -0.0192
  dof_torques_l2: old -0.0008 new -0.0007
  episodes: old +86.4000 new +104.5000
  feet_air_time: old +0.0043 new +0.0042
  feet_slide: old -0.0357 new -0.0340
  flat_orientation_l2: old -0.0166 new -0.0163
  joint_deviation: old -0.2652 new -0.2532
  lin_vel_z_l2: old +0.0000 new +0.0000
  pushes: old +116.1000 new +99.2000
  termination_penalty: old -0.0177 new -0.0107
  track_ang_vel_z_exp: old +0.5700 new +0.6044
  track_lin_vel_xy_exp: old +0.6977 new +0.7362

per-term ±5 mean at 1499:
  action_rate_l2: old -0.5759 new -0.5305
  ang_vel_xy_l2: old -0.0404 new -0.0381
  base_contact: old +0.0606 new +0.0380
  dof_acc_l2: old -0.0602 new -0.0528
  dof_pos_limits: old -0.0221 new -0.0186
  dof_torques_l2: old -0.0007 new -0.0006
  episodes: old +89.1667 new +100.5000
  feet_air_time: old +0.0053 new +0.0046
  feet_slide: old -0.0324 new -0.0311
  flat_orientation_l2: old -0.0204 new -0.0166
  joint_deviation: old -0.2595 new -0.2444
  lin_vel_z_l2: old +0.0000 new +0.0000
  pushes: old +111.0000 new +97.1667
  termination_penalty: old -0.0121 new -0.0076
  track_ang_vel_z_exp: old +0.6825 new +0.7295
  track_lin_vel_xy_exp: old +0.7463 new +0.7796

## 2.3.2 three-seed statistics (point values; sample sd)
it 300 old: return +4.03, -0.37, -0.10 mean +1.19 sd 2.47 (range 4.40); ±5 return mean +1.18 sd 2.29; level 1.608, 1.293, 0.669 mean 1.190 sd 0.478; length 984, 951, 971 mean 969
it 300 new: return +0.79, -1.75, -2.62 mean -1.19 sd 1.77 (range 3.41); ±5 return mean -1.11 sd 1.91; level 1.183, 1.046, 0.582 mean 0.937 sd 0.315; length 957, 933, 952 mean 947
it 500 old: return +8.44, +4.21, +4.98 mean +5.88 sd 2.25 (range 4.23); ±5 return mean +5.98 sd 2.24; level 4.407, 3.413, 3.051 mean 3.624 sd 0.702; length 976, 968, 958 mean 967
it 500 new: return +5.44, +3.68, +2.89 mean +4.00 sd 1.31 (range 2.55); ±5 return mean +4.09 sd 1.17; level 3.530, 3.255, 2.457 mean 3.081 sd 0.558; length 984, 971, 973 mean 976
it 1000 old: return +11.33, +11.01, +14.25 mean +12.20 sd 1.79 (range 3.24); ±5 return mean +12.07 sd 1.74; level 6.014, 5.601, 5.881 mean 5.832 sd 0.211; length 991, 999, 991 mean 994
it 1000 new: return +9.71, +11.39, +10.34 mean +10.48 sd 0.85 (range 1.68); ±5 return mean +10.39 sd 0.88; level 5.506, 5.561, 5.382 mean 5.483 sd 0.092; length 997, 991, 998 mean 996
it 1500 old: return +16.42, +19.08, +22.62 mean +19.37 sd 3.11 (range 6.20); ±5 return mean +19.35 sd 2.96; level 6.097, 6.026, 6.079 mean 6.067 sd 0.037; length 992, 996, 998 mean 995
it 1500 new: return +15.96, +20.61, +17.43 mean +18.00 sd 2.38 (range 4.65); ±5 return mean +17.92 sd 2.40; level 6.125, 6.108, 6.111 mean 6.115 sd 0.010; length 1000, 998, 1000 mean 999
