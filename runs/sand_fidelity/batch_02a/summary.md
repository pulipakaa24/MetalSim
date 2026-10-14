# sand_fidelity batch 02a: results

Plan: research_notes/sand_fidelity/PLAN_2026-10-14.md sections 8-9. 16-inch, correctness runs.

## Predictions stated before the runs

* **Repeat check.** The 1.5 s box drop at dt 1/480 (l_slip) repeats batch 01's a4_dt480 within 0.1 mm at every step (bitwise reported separately): the probe changes add only reads.
* **S sphere (8.6).** S-bed: walls reduce the final depth by >= 15 mm, and the unwalled ball-free control subsides >= 10 mm at the centre within 0.2 s. S-res: depth / Uehara falls >= 30 % from 10 to 20 cells, in both 20-cell variants. S-coupling: kinematic F(16 mm) >= 13 N while the walled free drop still overshoots Uehara by >= 1.5x; F(16 mm) <= 7 N means the bed itself is weak. Steel (7,800 kg/m^3): Uehara predicts 1.77x the 2,500 kg/m^3 depth; a sim ratio outside 1.77 +- 0.35 points at the coupling.
* **L lag (8.7).** Impact-phase peak (within 0.3 s after free-fall contact t_c = sqrt(2h/g)) = d0 + k dt at each drop height, with k within +-30 % of v = sqrt(2gh) (1.72 and 0.99 m/s), so k(0.05)/k(0.15) = 0.58 +- 0.2; d0 at voxel 0.02 in 10-25 mm; d0 at voxel 0.01 = d0(0.02) + 5 +- 3 mm. Against: k independent of v.
* **C mu 0.4 ladder (8.3).** H-edge: C1 (plane floor, frictionless side walls, fixed tilt 20.3 deg) goes non-finite within the 8 s hold; C2 (box floor) or C3 (rough side walls) stays finite; C4 (no side walls) stays finite. H-solver: C1-C4 all non-finite. C5 = C1 with logged iterations.
* **T tilt and iterations (8.1, 8.2).** Runaway (surface speed > 10 mm/s and rising for 0.5 s) within atan(mu) +- 1 deg and the deposit angle within 1 deg below it (mu 0.68, 0.9). >= 95 % of logged solves stop at the iteration cap. At a fixed tilt of atan(mu) - 4 deg (mu 0.68) the creep speed at 1000 iterations is <= half that at 250. Resting box at voxel 0.01, 1000 iterations: creep <= half of 250's 0.41 mm/s, settlement 0.45 +- 0.1 voxel.

## S. Sphere (8.6)

| run | case | particles | final mm | Uehara mm (mu 0.7) | | centre subsidence mm |
|---|---|---|---|---|---|---|
| s_walled | D 0.05 m (10 cells), 2500 kg/m^3, walls | 307200 | 21.3 | 16.3 | ratio 1.31; stopped True | 13.1 |
| s_ctrl_walled | control | 307200 | — | — | centre subsidence 0.1 mm @0.2 s, 0.1 mm end | |
| s_ctrl_unwalled | control | 307200 | — | — | centre subsidence 26.1 mm @0.2 s, 36.6 mm end | |
| s_kin | not run ||||||
| s_steel | D 0.05 m (10 cells), 7800 kg/m^3, walls | 307200 | 38.2 | 29.3 | ratio 1.30; stopped True | 27.2 |
| s20a | not run ||||||
| s20b | not run ||||||
| a6_sphere (batch 01, unwalled) | D 0.05 m (10 cells), 2500 kg/m^3 | 307200 | 60.4 | 16.3 | ratio 3.70; stopped True | — |

* S-bed: walls change the depth by -39.1 mm (unwalled 60.4 -> walled 21.3); unwalled control subsides 26.1 mm at 0.2 s, walled 0.1 mm -> supported.
* Steel / glass-density depth ratio 1.79 (Uehara 1.77): consistent.

## L. Lag (8.7)

| run | h m | dt | voxel | impact-phase peak mm | max tilt deg |
|---|---|---|---|---|---|
| l_h15_dt120 | 0.15 | 1/120 | 0.02 | 29.8 | 0.0 |
| l_h15_dt240 | 0.15 | 1/240 | 0.02 | 27.9 | 1.0 |
| l_h15_dt480 | 0.15 | 1/480 | 0.02 | 19.0 | 0.1 |
| l_h15_dt960 | 0.15 | 1/960 | 0.02 | 17.4 | 0.0 |
| l_h05_dt120 | 0.05 | 1/120 | 0.02 | 15.2 | 0.0 |
| l_h05_dt240 | 0.05 | 1/240 | 0.02 | 11.2 | 0.0 |
| l_h05_dt480 | 0.05 | 1/480 | 0.02 | 9.1 | 0.0 |
| l_h05_dt960 | 0.05 | 1/960 | 0.02 | 8.5 | 0.0 |
| l_v01_dt240 | 0.15 | 1/240 | 0.01 | 25.6 | 0.0 |
| l_v01_dt960 | 0.15 | 1/960 | 0.01 | 23.3 | 0.0 |

* h 0.15 m: d0 16.6 mm, k 1.76 m/s vs v 1.72 m/s (within +-30 %).
* h 0.05 m: d0 7.3 mm, k 0.94 m/s vs v 0.99 m/s (within +-30 %).
* k(0.05)/k(0.15) = 0.53 (prediction 0.58 +- 0.2) -> H_lag supported.
* Voxel 0.01: d0 22.5 mm, +5.9 mm vs voxel 0.02 (prediction +5 +- 3).
* Repeat of batch 01 a4_dt480: max |diff| 0.000 mm (bitwise) -> PASS; late sink 5.7 mm, max box tilt 3.0 deg (a rotation).

## C. mu 0.4 ladder (8.3)

| run | variant | outcome | where (last finite sample / fast particles) | solves at cap |
|---|---|---|---|---|
| c1 | plane floor, frictionless sides | NON-FINITE after 4.85 s of hold | n 41526, near floor edge 0.06, below floor 0.05 | — |
| c2 | box floor | finite | n 121, near floor edge 0.50, below floor 0.99 | — |
| c3 | rough side walls | finite | n 0, near floor edge —, below floor — | — |
| c4 | no side walls (width 0.1) | finite | n 1, near floor edge 0.00, below floor 1.00 | — |
| c5 | C1, iterations logged | NON-FINITE after 4.85 s of hold | n 41526, near floor edge 0.06, below floor 0.05 | 1.00 |

* Cause: H-edge (floor / side-wall junction).

## T. Tilt and iterations (8.1, 8.2)

| run | mu | atan mu | creep / runaway onset deg | deposit deg | at rest | solves at cap |
|---|---|---|---|---|---|---|
| t1 | NON-FINITE ||||||
| t2 | NON-FINITE ||||||

* Solves at the 250 cap: 1.00, 1.00, 1.00 (prediction >= 0.95) -> cap-bound.
* Creep at atan(mu) - 4 deg: 250 iterations 0.77 mm/s, 1000 iterations 0.06 mm/s, ratio 12.34 (prediction >= 2: under-convergence) -> supported.
* Resting box, voxel 0.01, 1000 iterations: settlement 2.9 mm (0.29 voxel), creep 0.43 vs 0.41 mm/s at 250 -> not halved.
