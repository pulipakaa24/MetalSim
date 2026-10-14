# sand_fidelity batch 01: results

Plan: research_notes/sand_fidelity/PLAN_2026-10-14.md section 6. No solver changes: existing code paths and new probes (metalsim/physics/mpm/probes/{sand_repose,tilt_box,bearing_fidelity,sphere_drop}.py). Machine: the 16-inch M4 Max (correctness runs; wall times below are queue-wrapped, not timing-class).

## Predictions stated before the runs

* **1 repose profile + stress (R1, R3).** Gate: R/dx 10 baseline reproduces the earlier run (fit 26.33 +- 0.3 deg). R1 (convex crest, measurement) predicts: lower-flank band angle (5-50 % of h_max) >= fit + 2 deg and within 2 deg of the cone-volume angle; the taller fill raises the fit by >= 1.5 deg and shrinks the estimator spread by >= 30 %. Against R1: band angles within 1 deg of each other and the tall fill within +-0.7 deg of the baseline. R3 (Drucker-Prager stress geometry) predicts: flank surface shells (0-2 cells) at yield (median sqrt(J2)/(mu p) >= 0.95) while the local angle is >= 3 deg below atan(mu), with a Lode parameter off pure shear (|cos 3theta| > 0.3). Flank surface below yield (median < 0.9) means the heap stopped for a non-rheological reason (R2).
* **2 tilting box (B1, R3).** Correct plane-shear rheology: theta_stop (post-avalanche surface angle) within +-1 deg of atan(mu) at 10 cells deep, +-2 deg at 5 cells (plan B1, applied to theta_stop). theta_fail is reported against the plan's +-1 deg but is not decisive (rigid-plastic onset is not unique; see tilt_box.py). Discrimination with arm 1: theta_stop at atan(mu) while the heap is >= 3 deg low -> deficit is heap-specific (R1/R3); theta_stop as low as the heap (within 2 deg of the heap's fit, >= 3 deg below atan(mu)) -> general (R2 discretization or R4 solver).
* **3 basis variants (R2).** R2 (weak free-surface skin from P0 strain / Q1 velocity / PIC quadrature / ppc) predicts at least one variant raises the fit by >= 1 deg over the R/dx 10 baseline. All variants within +-0.5 deg: R2 not supported by these settings. P1d/Q1d with a sector sd > 1 deg: the partially-filled-cell strain defect fixed upstream in Newton draft PR #4660, not physics.
* **4 drop dt sweep (C1).** One-step exchange lag (H_lag): peak penetration linear in dt with slope within +-30 % of the impact speed (1.72 m/s) and intercept d0 in 10-25 mm; d(1/480) - d(1/960) < 3 mm. Against: slope < 0.5 v (the lag is not the cause) or d0 > 25 mm (bed too weak, R3/C4). Voxel 0.01 at 1/480 within 3 mm of voxel 0.02: penetration not set by grid contact resolution. dt 1/120 reproduces 29.8 mm (+-1 mm).
* **5 resting box (C2).** Gate: voxel 0.02 S2 without projection reproduces 10.2 mm (+-1 mm). H_grid (sub-voxel contact resolution, Newton #3473): settlement / voxel constant (within +-30 %) over voxel 0.02 / 0.01 / 0.005, and the box overlaps the particles (deepest particle inside the box >= half the settlement); project_outside then removes the overlap. H_bed (real bed deformation): settlement in mm constant (+-30 %) over the voxels, small overlap, ring heave > 0. pic27 (Newton #1251's advice) reduces settlement below S2 at every voxel only under H_grid. Plan B5 pass: settlement <= 0.25 voxel and <= 3 mm, creep < 0.1 mm/s.
* **6 sphere drop smoke (B4).** Smoke pass: finite, the ball stops (|v| < 1 cm/s) within 0.5 s, never reaches the floor, final penetration within 0.5-2x Uehara's law with mu = 0.7 (h = 0.2 m, D = 0.05 m: ~16 mm). Reported, not judged: d - v dt (lag removed, from arm 4) against Uehara with mu = tan(heap angle of arm 1) (~22 mm) and mu = 0.7 (~16 mm).

## Verdicts

* Arm 1: gate PASS; R1 mixed / undecided; R3 (stress) consistent.
* Arm 2: theta_stop {}; theta_fail {}; heap vs plane shear: undecided.
* Arm 3: R2 supported by a3_b2, a3_q1d, a3_p1d.
* Arm 4: H_lag not supported.
* Arm 5: gate PASS; C2 not run.
* Arm 6: smoke not run.

## 1. Repose profile and stress (R1, R3)

| run | R/dx | fill cm | particles | fit (sd) | cone vol. | h/flank | band angles 5-25/25-50/50-75/75-95 % | max local | flank 0-1 cell ratio / Lode | at rest |
|---|---|---|---|---|---|---|---|---|---|---|
| a1_dx10 | 10 | 5 | 80040 | 26.3 ± 0.1 | 31.5 | 26.4 | 29.8 / 27.9 / 24.7 / 19.3 | 30.5 | 1.00 / -0.42 | False |
| a1_dx20 | 20 | 5 | 658080 | 27.8 ± 0.0 | 33.3 | 27.8 | 28.2 / 28.1 / 27.5 / 23.1 | 28.7 | 1.00 / -0.33 | False |
| a1_tall | 10 | 10 | 160080 | 26.4 ± 0.1 | 33.8 | 26.5 | 28.6 / 27.3 / 25.3 / 19.7 | 29.8 | 1.00 / -0.39 | False |

* Gate (reproduces repose3 26.33 deg within 0.3): **PASS** (26.33 deg).
* a1_dx10: lower-flank band angle 29.8 vs fit 26.3 / cone volume 31.5 -> convex (R1 prediction met).
* a1_dx20: lower-flank band angle 28.2 vs fit 27.8 / cone volume 33.3 -> R1 prediction not met.
* Tall fill: fit +0.03 deg, estimator spread -43 % smaller -> against R1.
* a1_dx10: flank surface (0-2 cells) yield ratio 1.00, Lode cos3theta -0.43, p_base/(rho g h) 0.86 -> R3 consistent (at yield below atan mu, off pure shear).
* a1_dx20: flank surface (0-2 cells) yield ratio 1.00, Lode cos3theta -0.35, p_base/(rho g h) 0.79 -> R3 consistent (at yield below atan mu, off pure shear).

## 2. Tilting box (B1)

| run | mu | atan mu | depth cells | theta_fail | theta_stop | at rest | stop / fail vs atan mu |
|---|---|---|---|---|---|---|---|
| a2_mu040 | not run |||||||
| a2_mu068 | not run |||||||
| a2_mu090 | not run |||||||
| a2_mu068_v01 | not run |||||||


## 3. Discretization variants (R2)

| run | velocity / strain / integration / ppc | fit (sd) | vs baseline | cone vol. | at rest | note |
|---|---|---|---|---|---|---|
| a3_b2 | B2 / P0 / pic / 27 | 28.3 ± 0.2 | +1.94 | 35.9 | False |  |
| a3_q1d | Q1 / Q1d / pic / 27 | 29.0 ± 0.3 | +2.65 | 36.7 | False |  |
| a3_p1d | Q1 / P1d / pic / 27 | 31.8 ± 0.3 | +5.43 | 33.4 | False |  |
| a3_gimp | Q1 / P0 / gimp / 27 | 26.3 ± 0.1 | +0.00 | 30.8 | True |  |
| a3_ppc64 | Q1 / P0 / pic / 64 | 26.2 ± 0.1 | -0.13 | 30.1 | True |  |

* Baseline fit 26.33 deg (a1_dx10). R2: supported by a3_b2, a3_q1d, a3_p1d.

## 4. Drop dt sweep (C1)

| run | dt | voxel | impact m/s | peak mm | final mm | creep mm/s | floor |
|---|---|---|---|---|---|---|---|
| a4_dt120 | 1/120 | 0.02 | 1.64 | 29.8 | 28.6 | 1.29 | False |
| a4_dt240 | 1/240 | 0.02 | 0.87 | 27.9 | 25.2 | 0.51 | False |
| a4_dt480 | 1/480 | 0.02 | 0.61 | 24.7 | 24.7 | 15.91 | False |
| a4_dt960 | 1/960 | 0.02 | 0.68 | 17.8 | 17.8 | 0.43 | False |
| a4_dt480_v01 | 1/480 | 0.01 | 0.98 | 23.9 | 22.9 | 0.10 | False |

* Fit peak = d0 + k dt over voxel 0.02: d0 19.6 mm, k 1.39 m/s vs impact speed 0.95 m/s; d(1/480)-d(1/960) 6.9 mm -> H_lag not supported.
* Voxel 0.01 vs 0.02 at 1/480: -0.8 mm -> not set by grid contact resolution.
* dt 1/120 reproduction of bearing4 (29.8 mm): 29.8 mm (ok).

## 5. Resting box (C2)

| run | voxel | collider basis | project | settlement mm | / voxel | creep mm/s | inside box: n / deepest mm | ring heave mm | B5 |
|---|---|---|---|---|---|---|---|---|---|
| a5_v02_S2 | 0.02 | S2 | False | 10.2 | 0.51 | 0.40 | 40 / 0.6 | -16.9 | FAIL |
| a5_v02_S2_proj | not run ||||||||
| a5_v02_pic27 | not run ||||||||
| a5_v02_pic27_proj | not run ||||||||
| a5_v01_S2 | not run ||||||||
| a5_v01_S2_proj | not run ||||||||
| a5_v01_pic27 | not run ||||||||
| a5_v01_pic27_proj | not run ||||||||
| a5_v005_S2 | not run ||||||||
| a5_v005_S2_proj | not run ||||||||
| a5_v005_pic27 | not run ||||||||
| a5_v005_pic27_proj | not run ||||||||

* Gate (reproduces bearing3 rest 10.2 mm within 1 mm): **PASS** (10.2 mm).

## 6. Sphere drop smoke (B4)

a6_sphere: not run


## Jobs (wall time, s, gpu_run + watchdog wrapped; estimate from runs/gpu_history.jsonl)

| job | rc | measured | estimate |
|---|---|---|---|
| a1_dx10 | 0 | 98 | 80 |
| a5_v02_S2 | 0 | 48 | 28 |
| a1_dx20 | 0 | 315 | 410 |
| a1_tall | 0 | 169 | 159 |
| a4_dt120 | 0 | 109 | 28 |
| a4_dt240 | 0 | 193 | 43 |
| a4_dt480 | 0 | 129 | 73 |
| a4_dt960 | 0 | 159 | 133 |
| a4_dt480_v01 | 0 | 284 | 173 |
| a3_b2 | 0 | 163 | 78 |
| a3_q1d | 0 | 264 | 78 |
| a3_p1d | 0 | 164 | 78 |
| a3_gimp | 0 | 99 | 78 |
| a3_ppc64 | 0 | 138 | 141 |
| a2_mu040 | 3 | 29 | 195 |
