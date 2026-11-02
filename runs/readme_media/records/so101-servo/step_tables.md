| servo | joint | overshoot | crossings | settle to 2 % | 90 % at | sag before step | peak speed / acc, 0.06 step | peak speed / acc, 0.5 rad step |
|---|---|---|---|---|---|---|---|---|
| 12v (old) | shoulder_lift | 52 % | 6 | 190 ms | 40 ms | 1 mrad | 2.31 rad/s, 96 rad/s² | 3.78 rad/s, 121 rad/s² |
| 12v (old) | elbow_flex | 52 % | 5 | 170 ms | 40 ms | 0 mrad | 2.47 rad/s, 113 rad/s² | 3.81 rad/s, 144 rad/s² |
| 12v (old) | wrist_flex | 33 % | 5 | 120 ms | 35 ms | 0 mrad | 2.49 rad/s, 130 rad/s² | 4.39 rad/s, 180 rad/s² |
| 7v4 (old) | shoulder_lift | 89 % | 6 | 315 ms | 50 ms | 1 mrad | 2.09 rad/s, 68 rad/s² | 3.37 rad/s, 77 rad/s² |
| 7v4 (old) | elbow_flex | 72 % | 6 | 240 ms | 45 ms | 0 mrad | 2.26 rad/s, 78 rad/s² | 3.48 rad/s, 99 rad/s² |
| 7v4 (old) | wrist_flex | 52 % | 5 | 190 ms | 45 ms | 0 mrad | 2.32 rad/s, 87 rad/s² | 4.31 rad/s, 117 rad/s² |
| 12v_real | shoulder_lift | 0 % | 0 | 275 ms | 235 ms | 79 mrad | 0.35 rad/s, 6 rad/s² | 3.04 rad/s, 28 rad/s² |
| 12v_real | elbow_flex | 0 % | 0 | 250 ms | 215 ms | 61 mrad | 0.38 rad/s, 7 rad/s² | 3.05 rad/s, 29 rad/s² |
| 12v_real | wrist_flex | 0 % | 0 | 240 ms | 200 ms | 13 mrad | 0.45 rad/s, 9 rad/s² | 3.09 rad/s, 31 rad/s² |
| 7v4_real | shoulder_lift | 0 % | 0 | 310 ms | 275 ms | 121 mrad | 0.31 rad/s, 5 rad/s² | 2.81 rad/s, 22 rad/s² |
| 7v4_real | elbow_flex | 0 % | 0 | 280 ms | 245 ms | 96 mrad | 0.31 rad/s, 5 rad/s² | 2.95 rad/s, 26 rad/s² |
| 7v4_real | wrist_flex | 0 % | 0 | 255 ms | 220 ms | 21 mrad | 0.41 rad/s, 7 rad/s² | 3.24 rad/s, 30 rad/s² |

| 7v4_real ablation | lift: overshoot, settle | elbow | wrist flex | wrist, 0.5 rad step: peak speed, acc |
|---|---|---|---|---|
| 7v4_real | 0.4 %, 310 ms | 0.3 %, 280 ms | 0.2 %, 255 ms | 3.24, 30 |
| A0 gains+line only | 0.4 %, 270 ms | 0.3 %, 235 ms | 0.3 %, 215 ms | 3.54, 60 |
| A1 +(a) servo delay 1.05 ms | 0.4 %, 270 ms | 0.3 %, 240 ms | 0.3 %, 220 ms | 3.54, 60 |
| A2 +(b) speed+accel profile | 0.4 %, 310 ms | 0.3 %, 280 ms | 0.2 %, 255 ms | 3.24, 30 |
| A3 +(c) host latency 3 ms (= 7v4_real) | 0.4 %, 310 ms | 0.3 %, 280 ms | 0.2 %, 255 ms | 3.24, 30 |
| full -(d) no back-EMF line | 33.1 %, 580 ms | 44.0 %, 510 ms | 48.9 %, 460 ms | 5.04, 62 |
| full -(b) accel only off | 0.4 %, 275 ms | 0.3 %, 245 ms | 0.3 %, 225 ms | 3.48, 40 |
| full, D=32 @ 10 ms loop | 0.0 %, 320 ms | 0.0 %, 285 ms | 0.0 %, 265 ms | 3.03, 28 |
| full, BAM kt/R stall 3.41 N.m | 0.7 %, 250 ms | 0.5 %, 230 ms | 0.0 %, 215 ms | 3.31, 33 |
| full, host latency 10 ms | 0.3 %, 320 ms | 0.3 %, 290 ms | 0.2 %, 265 ms | 3.24, 30 |

max_ncon during steps: {'12v (old)': 4, '7v4 (old)': 4, '12v_real': 4, '7v4_real': 4, 'A0 gains+line only': 4, 'A1 +(a) servo delay 1.05 ms': 4, 'A2 +(b) speed+accel profile': 4, 'A3 +(c) host latency 3 ms (= 7v4_real)': 4, 'full -(d) no back-EMF line': 4, 'full -(b) accel only off': 4, 'full, D=32 @ 10 ms loop': 4, 'full, BAM kt/R stall 3.41 N.m': 4, 'full, host latency 10 ms': 4}
