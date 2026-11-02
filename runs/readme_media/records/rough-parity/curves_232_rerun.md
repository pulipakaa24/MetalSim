Isaac (PhysX + rsl_rl): runs/parity/isaac/rough/isaac_train_g1_rough_terms.txt, 1500 iterations; MetalSim (MuJoCo Warp + PPOWarp): runs/rough_rerun_2026-10-31/logs/g1_rough_ppowarp_fixed_s0.log, 1500 iterations
Isaac: mean of the last 100 finished episodes (0-based it); MetalSim: episodes finished in that iteration (1-based it), point value and mean over it-4..it+5; terrain level: mean over all envs at that iteration on both sides
| iteration | Isaac length | Isaac return | Isaac terrain level | MetalSim length (point / ±5) | MetalSim return (point / ±5) | MetalSim terrain level |
|---|---|---|---|---|---|---|
| 50 | 51.1 | -4.79 | 0.000 | 53.2 / 53.5 | -4.98 / -4.99 | 0.000 |
| 100 | 143.8 | -5.77 | 0.036 | 84.8 / 87.0 | -4.88 / -4.90 | 0.000 |
| 150 | 761.6 | -9.30 | 0.548 | 521.7 / 523.2 | -9.91 / -9.96 | 0.019 |
| 200 | 934.7 | -4.62 | 1.077 | 966.5 / 950.6 | -7.42 / -7.48 | 0.122 |
| 250 | 879.7 | -1.07 | 1.593 | 987.6 / 979.8 | -2.61 / -2.52 | 0.610 |
| 300 | 918.6 | +2.14 | 2.185 | 957.0 / 976.3 | +0.79 / +0.93 | 1.183 |
| 400 | 955.6 | +6.01 | 3.530 | 979.0 / 963.1 | +4.37 / +4.01 | 2.343 |
| 500 | 917.3 | +5.83 | 4.831 | 984.0 / 974.6 | +5.44 / +5.36 | 3.530 |
| 750 | 930.2 | +5.99 | 6.131 | 983.5 / 984.0 | +6.69 / +6.72 | 5.146 |
| 1000 | 882.3 | +5.36 | 5.962 | 997.4 / 991.0 | +9.71 / +9.64 | 5.506 |
| 1500 | 966.1 | +14.14 | 5.896 | 1000.0 / 992.2 | +15.96 / +15.91 | 6.125 |

Isaac per-iteration detail: it | linear tracking | yaw tracking | falls (base_contact fraction) | action std | iteration time
50 | 0.015 | 0.008 | 1.000 | 0.98 | 2.62 s
100 | 0.036 | 0.026 | 1.000 | 1.01 | 2.67 s
150 | 0.250 | 0.153 | 0.891 | 1.07 | 2.69 s
200 | 0.492 | 0.237 | 0.331 | 1.09 | 2.65 s
250 | 0.546 | 0.290 | 0.178 | 1.09 | 2.70 s
300 | 0.636 | 0.378 | 0.178 | 1.10 | 2.65 s
400 | 0.709 | 0.500 | 0.145 | 1.12 | 2.66 s
500 | 0.670 | 0.522 | 0.149 | 1.15 | 2.71 s
750 | 0.689 | 0.567 | 0.191 | 1.23 | 2.68 s
1000 | 0.664 | 0.569 | 0.175 | 1.27 | 2.74 s
Isaac (PhysX + rsl_rl): runs/parity/isaac/rough/isaac_train_g1_rough_terms.txt, 1500 iterations; MetalSim (MuJoCo Warp + PPOWarp): runs/rough_rerun_2026-10-31/logs/g1_rough_ppowarp_fixed_s1.log, 1500 iterations
Isaac: mean of the last 100 finished episodes (0-based it); MetalSim: episodes finished in that iteration (1-based it), point value and mean over it-4..it+5; terrain level: mean over all envs at that iteration on both sides
| iteration | Isaac length | Isaac return | Isaac terrain level | MetalSim length (point / ±5) | MetalSim return (point / ±5) | MetalSim terrain level |
|---|---|---|---|---|---|---|
| 50 | 51.1 | -4.79 | 0.000 | 53.5 / 54.1 | -4.94 / -4.94 | 0.000 |
| 100 | 143.8 | -5.77 | 0.036 | 76.7 / 77.2 | -4.71 / -4.72 | 0.000 |
| 150 | 761.6 | -9.30 | 0.548 | 427.0 / 441.0 | -8.84 / -9.13 | 0.009 |
| 200 | 934.7 | -4.62 | 1.077 | 845.4 / 879.7 | -8.46 / -8.32 | 0.068 |
| 250 | 879.7 | -1.07 | 1.593 | 948.5 / 957.8 | -4.76 / -4.39 | 0.520 |
| 300 | 918.6 | +2.14 | 2.185 | 933.1 / 947.2 | -1.75 / -1.41 | 1.046 |
| 400 | 955.6 | +6.01 | 3.530 | 924.1 / 946.5 | +0.33 / +1.13 | 2.148 |
| 500 | 917.3 | +5.83 | 4.831 | 970.6 / 977.9 | +3.68 / +3.87 | 3.255 |
| 750 | 930.2 | +5.99 | 6.131 | 979.5 / 983.8 | +6.41 / +6.39 | 5.023 |
| 1000 | 882.3 | +5.36 | 5.962 | 991.3 / 990.3 | +11.39 / +11.35 | 5.561 |
| 1500 | 966.1 | +14.14 | 5.896 | 997.8 / 998.4 | +20.61 / +20.58 | 6.108 |

Isaac per-iteration detail: it | linear tracking | yaw tracking | falls (base_contact fraction) | action std | iteration time
50 | 0.015 | 0.008 | 1.000 | 0.98 | 2.62 s
100 | 0.036 | 0.026 | 1.000 | 1.01 | 2.67 s
150 | 0.250 | 0.153 | 0.891 | 1.07 | 2.69 s
200 | 0.492 | 0.237 | 0.331 | 1.09 | 2.65 s
250 | 0.546 | 0.290 | 0.178 | 1.09 | 2.70 s
300 | 0.636 | 0.378 | 0.178 | 1.10 | 2.65 s
400 | 0.709 | 0.500 | 0.145 | 1.12 | 2.66 s
500 | 0.670 | 0.522 | 0.149 | 1.15 | 2.71 s
750 | 0.689 | 0.567 | 0.191 | 1.23 | 2.68 s
1000 | 0.664 | 0.569 | 0.175 | 1.27 | 2.74 s
Isaac (PhysX + rsl_rl): runs/parity/isaac/rough/isaac_train_g1_rough_terms.txt, 1500 iterations; MetalSim (MuJoCo Warp + PPOWarp): runs/rough_rerun_2026-10-31/logs/g1_rough_ppowarp_fixed_s2.log, 1500 iterations
Isaac: mean of the last 100 finished episodes (0-based it); MetalSim: episodes finished in that iteration (1-based it), point value and mean over it-4..it+5; terrain level: mean over all envs at that iteration on both sides
| iteration | Isaac length | Isaac return | Isaac terrain level | MetalSim length (point / ±5) | MetalSim return (point / ±5) | MetalSim terrain level |
|---|---|---|---|---|---|---|
| 50 | 51.1 | -4.79 | 0.000 | 53.5 / 53.5 | -4.95 / -4.93 | 0.000 |
| 100 | 143.8 | -5.77 | 0.036 | 73.6 / 73.7 | -4.67 / -4.66 | 0.000 |
| 150 | 761.6 | -9.30 | 0.548 | 321.3 / 321.5 | -7.96 / -7.90 | 0.006 |
| 200 | 934.7 | -4.62 | 1.077 | 814.1 / 832.3 | -9.29 / -8.89 | 0.046 |
| 250 | 879.7 | -1.07 | 1.593 | 926.2 / 916.4 | -6.70 / -6.12 | 0.265 |
| 300 | 918.6 | +2.14 | 2.185 | 952.1 / 954.9 | -2.62 / -2.85 | 0.582 |
| 400 | 955.6 | +6.01 | 3.530 | 972.9 / 969.4 | +0.73 / +0.67 | 1.396 |
| 500 | 917.3 | +5.83 | 4.831 | 972.8 / 966.6 | +2.89 / +3.04 | 2.457 |
| 750 | 930.2 | +5.99 | 6.131 | 977.1 / 984.1 | +5.61 / +6.42 | 4.684 |
| 1000 | 882.3 | +5.36 | 5.962 | 997.8 / 989.6 | +10.34 / +10.17 | 5.382 |
| 1500 | 966.1 | +14.14 | 5.896 | 1000.0 / 991.1 | +17.43 / +17.28 | 6.111 |

Isaac per-iteration detail: it | linear tracking | yaw tracking | falls (base_contact fraction) | action std | iteration time
50 | 0.015 | 0.008 | 1.000 | 0.98 | 2.62 s
100 | 0.036 | 0.026 | 1.000 | 1.01 | 2.67 s
150 | 0.250 | 0.153 | 0.891 | 1.07 | 2.69 s
200 | 0.492 | 0.237 | 0.331 | 1.09 | 2.65 s
250 | 0.546 | 0.290 | 0.178 | 1.09 | 2.70 s
300 | 0.636 | 0.378 | 0.178 | 1.10 | 2.65 s
400 | 0.709 | 0.500 | 0.145 | 1.12 | 2.66 s
500 | 0.670 | 0.522 | 0.149 | 1.15 | 2.71 s
750 | 0.689 | 0.567 | 0.191 | 1.23 | 2.68 s
1000 | 0.664 | 0.569 | 0.175 | 1.27 | 2.74 s
