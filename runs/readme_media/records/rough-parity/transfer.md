
### Protocol A, 2026-09-25 record (plane-path kernel; `runs/g1_rough_transfer.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| isaac_it500 (levels [3, 6]) | 2 / 8 | 2 / 4 | 0 / 2, 1.91 m | 2 / 2, 0.20 m | 0 / 2, 1.78 m | 0 / 2, 1.08 m | 0 / 8 |
| isaac_it1000 (levels [3, 6]) | 3 / 8 | 3 / 4 | 0 / 2, 4.43 m | 2 / 2, 0.22 m | 1 / 2, 1.48 m | 0 / 2, 3.65 m | 0 / 8 |
| isaac_it1499 (levels [3, 6]) | 2 / 8 | 2 / 4 | 0 / 2, 4.49 m | 1 / 2, 1.82 m | 1 / 2, 3.19 m | 0 / 2, 4.23 m | 0 / 8 |
| metalsim_ours_it500 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 3.86 m | 0 / 2, 1.76 m | 0 / 2, 2.80 m | 0 / 2, 2.94 m | 6 / 8 |
| metalsim_ours_it1000 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 2.42 m | 0 / 2, 2.40 m | 0 / 2, 2.72 m | 0 / 2, 2.91 m | 3 / 8 |
| metalsim_ours_it1500 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 4.09 m | 0 / 2, 2.72 m | 0 / 2, 3.64 m | 0 / 2, 3.74 m | 3 / 8 |
| total | 7 / 48 | 7 / 24 | | | | | |

### Protocol A, pinned fork 8eb51fe (plane path), today's code (`runs/rough_rerun_2026-10-31/transfer/old_pin_protocol24.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| metalsim_ours_it1500 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 4.12 m | 0 / 2, 2.72 m | 0 / 2, 3.73 m | 0 / 2, 3.72 m | 3 / 8 |
| nvidia_pretrained_rough_it2999 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 4.40 m | 0 / 2, 2.59 m | 0 / 2, 2.96 m | 0 / 2, 4.17 m | - |
| total | 0 / 16 | 0 / 8 | | | | | |

### Protocol A, fixed fork 93e5ce1 (`runs/rough_rerun_2026-10-31/transfer/old_fix_protocol24.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| metalsim_ours_it1500 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 2.84 m | 0 / 2, 2.70 m | 0 / 2, 3.10 m | 0 / 2, 2.94 m | 3 / 8 |
| nvidia_pretrained_rough_it2999 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 3.30 m | 0 / 2, 3.74 m | 0 / 2, 3.47 m | 0 / 2, 3.84 m | - |
| total | 0 / 16 | 0 / 8 | | | | | |

### Protocol A, fixed fork 93e5ce1, re-trained seed 0 (`runs/rough_rerun_2026-10-31/transfer/new_s0_protocol24.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| metalsim_rerun_s0_it500 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 0.30 m | 0 / 2, 1.63 m | 0 / 2, -0.93 m | 0 / 2, -0.32 m | - |
| metalsim_rerun_s0_it1000 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 3.78 m | 0 / 2, 1.22 m | 0 / 2, 3.23 m | 0 / 2, 3.17 m | - |
| metalsim_rerun_s0_it1500 (levels [3, 6]) | 0 / 8 | 0 / 4 | 0 / 2, 3.86 m | 0 / 2, 2.61 m | 0 / 2, 3.27 m | 0 / 2, 3.12 m | - |
| total | 0 / 24 | 0 / 12 | | | | | |

### Protocol B, 2026-09-25 record (boxes_local + exact; `runs/terrain_walls/transfer_reps.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| isaac_it500 (levels [3, 6]) | 8 / 64 | 8 / 32 | 0 / 16, 1.55 m | 8 / 16, 1.87 m | 0 / 16, 2.17 m | 0 / 16, 1.37 m | - |
| isaac_it1000 (levels [3, 6]) | 6 / 64 | 6 / 32 | 0 / 16, 2.36 m | 4 / 16, 3.11 m | 2 / 16, 3.48 m | 0 / 16, 3.79 m | - |
| isaac_it1499 (levels [3, 6]) | 1 / 64 | 0 / 32 | 1 / 16, 3.37 m | 0 / 16, 3.53 m | 0 / 16, 4.01 m | 0 / 16, 4.21 m | - |
| metalsim_ours_it500 (levels [3, 6]) | 3 / 64 | 3 / 32 | 0 / 16, 3.44 m | 1 / 16, 1.56 m | 2 / 16, 3.45 m | 0 / 16, 2.90 m | - |
| metalsim_ours_it1000 (levels [3, 6]) | 0 / 64 | 0 / 32 | 0 / 16, 0.96 m | 0 / 16, 2.91 m | 0 / 16, 2.58 m | 0 / 16, 3.03 m | - |
| metalsim_ours_it1500 (levels [3, 6]) | 1 / 64 | 1 / 32 | 0 / 16, 2.28 m | 1 / 16, 3.79 m | 0 / 16, 3.71 m | 0 / 16, 3.72 m | - |
| total | 19 / 384 | 18 / 192 | | | | | |

### Protocol B, pinned fork 8eb51fe (`runs/rough_rerun_2026-10-31/transfer/old_pin_reps8.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| metalsim_ours_it1500 (levels [3, 6]) | 0 / 64 | 0 / 32 | 0 / 16, 2.34 m | 0 / 16, 4.11 m | 0 / 16, 3.76 m | 0 / 16, 3.71 m | - |
| nvidia_pretrained_rough_it2999 (levels [3, 6]) | 0 / 64 | 0 / 32 | 0 / 16, 3.20 m | 0 / 16, 3.83 m | 0 / 16, 2.99 m | 0 / 16, 3.94 m | - |
| total | 0 / 128 | 0 / 64 | | | | | |

### Protocol B, fixed fork 93e5ce1 (`runs/rough_rerun_2026-10-31/transfer/old_fix_reps8.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| metalsim_ours_it1500 (levels [3, 6]) | 1 / 64 | 1 / 32 | 0 / 16, 2.18 m | 1 / 16, 4.01 m | 0 / 16, 3.91 m | 0 / 16, 2.97 m | - |
| nvidia_pretrained_rough_it2999 (levels [3, 6]) | 1 / 64 | 1 / 32 | 0 / 16, 3.18 m | 0 / 16, 3.72 m | 1 / 16, 2.94 m | 0 / 16, 3.17 m | - |
| total | 2 / 128 | 2 / 64 | | | | | |

### Protocol B, fixed fork 93e5ce1, re-trained seed 0 (`runs/rough_rerun_2026-10-31/transfer/new_s0_reps8.jsonl`)
| checkpoint | falls in MetalSim | wall-cell falls (inv. stairs + boxes) | pyr. stairs: falls, mean x | inv. stairs: falls, mean x | boxes: falls, mean x | random rough: falls, mean x | falls in Isaac (play records) |
|---|---|---|---|---|---|---|---|
| metalsim_rerun_s0_it500 (levels [3, 6]) | 0 / 64 | 0 / 32 | 0 / 16, 0.37 m | 0 / 16, 1.49 m | 0 / 16, -0.30 m | 0 / 16, -0.36 m | - |
| metalsim_rerun_s0_it1000 (levels [3, 6]) | 0 / 64 | 0 / 32 | 0 / 16, 3.70 m | 0 / 16, 2.28 m | 0 / 16, 3.86 m | 0 / 16, 3.21 m | - |
| metalsim_rerun_s0_it1500 (levels [3, 6]) | 0 / 64 | 0 / 32 | 0 / 16, 3.72 m | 0 / 16, 3.93 m | 0 / 16, 3.78 m | 0 / 16, 3.04 m | - |
| total | 0 / 192 | 0 / 96 | | | | | |

### Same checkpoint, fixed minus pinned, protocol A and B (per env)
protocol24 metalsim_ours_it1500 pyr. stairs: mean dx -1.28 m (min -1.61, max -0.96, n 2)
protocol24 metalsim_ours_it1500 inv. stairs: mean dx -0.02 m (min -1.17, max +1.13, n 2)
protocol24 metalsim_ours_it1500 boxes: mean dx -0.64 m (min -0.73, max -0.54, n 2)
protocol24 metalsim_ours_it1500 random rough: mean dx -0.77 m (min -0.93, max -0.61, n 2)
protocol24 nvidia_pretrained_rough_it2999 pyr. stairs: mean dx -1.10 m (min -1.42, max -0.78, n 2)
protocol24 nvidia_pretrained_rough_it2999 inv. stairs: mean dx +1.15 m (min +0.67, max +1.64, n 2)
protocol24 nvidia_pretrained_rough_it2999 boxes: mean dx +0.51 m (min -1.01, max +2.04, n 2)
protocol24 nvidia_pretrained_rough_it2999 random rough: mean dx -0.33 m (min -0.89, max +0.23, n 2)
reps8 metalsim_ours_it1500 pyr. stairs: mean dx -0.16 m (min -0.64, max +0.01, n 16)
reps8 metalsim_ours_it1500 inv. stairs: mean dx -0.09 m (min -0.79, max +0.41, n 16)
reps8 metalsim_ours_it1500 boxes: mean dx +0.15 m (min -0.56, max +0.77, n 16)
reps8 metalsim_ours_it1500 random rough: mean dx -0.74 m (min -1.00, max -0.45, n 16)
reps8 nvidia_pretrained_rough_it2999 pyr. stairs: mean dx -0.02 m (min -0.21, max +0.38, n 16)
reps8 nvidia_pretrained_rough_it2999 inv. stairs: mean dx -0.11 m (min -0.60, max +0.13, n 16)
reps8 nvidia_pretrained_rough_it2999 boxes: mean dx -0.05 m (min -0.66, max +1.18, n 16)
reps8 nvidia_pretrained_rough_it2999 random rough: mean dx -0.77 m (min -2.43, max +0.38, n 16)
