# sand_fidelity batch 02c: results

## Predictions stated before the runs

* **Heap (preset).** Reference atan(mu) = 34.2 deg. Under-convergence (H-iter): R/dx 10 rises >= 1.5 deg over the 250-iteration P1d value (29.9 -> >= 31.4) and R/dx 20 ends within 2 deg of atan(mu) (>= 32.2). Against: both within +-0.5 deg of their 250-iteration values (30.7 at R/dx 20).
* **Tilt (preset, box floor).** Runaway within atan(mu) +- 1 deg and the deposit angle within 1 deg below the runaway.
* **Sphere glass (preset).** Final depth within +-15 % (Uehara's scatter) of Uehara's law with mu = tan(the preset heap angle at the finest resolution run), and of mu = 0.7; reported both ways.
* **Sphere mass scaling.** Steel / glass depth ratio 1.77 +- 0.2 (Uehara: d ~ rho_ball^1/2) at 250 iterations with P1d and with the preset.
* **Sphere resolution (P1d, 20 cells).** Within +-15 % of the 10-cell P1d depth (16.3 mm): the 10-cell hit is resolution-converged. Deepening like parity P0 (1.31 -> 1.46x Uehara) means the 10-cell match was a resolution coincidence.
* **Box drop (preset, dt 1/960, voxel 0.01).** Lag-free penetration (impact-phase peak minus v dt = 1.8 mm) in the 10-25 mm band and below parity's 22.5 mm at the same voxel.
* **Rest.** Parity at voxel 0.005 with 8 particles per cell: 0.45-0.5 voxel (grid contact resolution); the voxel-0.01 bridge at 8 ppc reproduces 27-ppc's 4.5 mm within 0.7 mm; the preset at voxel 0.005 passes B5 (<= 0.25 voxel, <= 3 mm, creep < 0.1 mm/s).
* **Edge leak.** C1 with the collider-only variant (pr4660c) stays finite with no particle below the floor: the collider distance-sign fix is what stops the leak.
* **Deposit metric.** Parity t1b rerun (fixed deposit fit): deposit within 1 deg below its runaway.

## Results

* rep10_pre: fit 29.91 deg (+0.02 vs 250 iterations; atan(mu) - fit = 4.30); toe / mid 37.3 / 32.7; cone volume 26.4; at rest False.
* rep20_pre: not run.
* H-iter: against; preset heap within 2 deg of atan(mu): None.
* tilt_pre: creep —, runaway — (hold 40.2), deposit 36.7 vs atan(mu) 34.2 -> FAIL.
* t1b_fix: creep 33.6, runaway 36.7 (hold 37.7), deposit -52.2 vs atan(mu) 34.2 -> FAIL.
* Sphere glass (preset): 15.1 mm; Uehara with mu = tan(heap) = 0.575: 19.9 mm (0.76x), with mu 0.7: 16.3 mm (0.93x) -> outside +-15 % of the self-consistent reference.
* Steel / glass (P1d, 250 iterations): 1.91 (Uehara 1.77 +- 0.2) -> PASS.
* Steel / glass (preset): 1.95 (Uehara 1.77 +- 0.2) -> PASS.
* Sphere P1d at 20 cells: 17.0 mm, 1.04x the 10-cell P1d depth -> converged.
* Box drop (preset): impact-phase peak 14.6 mm, lag-free 12.8 mm (parity d0 at voxel 0.01: 22.5) -> PASS; late sink -0.0 mm, max tilt 1.4 deg.
* rest_v005_ppc2: settlement 10.5 mm (2.11 voxel), creep 0.93 mm/s, overlap 1586, 468512 particles -> FAIL.
* rest_v01_ppc2: settlement 6.4 mm (0.64 voxel), creep 0.79 mm/s, overlap 23, 59536 particles -> FAIL.
* rest_v005_pre: settlement -0.7 mm (-0.14 voxel), creep -1.01 mm/s, overlap 1, 468512 particles -> FAIL.
* C1 with the collider-only variant: finite; -> the collider sign fix is the cause.
