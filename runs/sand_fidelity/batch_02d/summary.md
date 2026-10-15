# sand_fidelity batch 02d: results

## Predictions stated before the runs

* **ppc, sphere.** P1d + #4660 sphere at 27 ppc (10 cells) within +-10 % of its 8-ppc depth (16.3 mm, 250 iterations).
* **ppc, rest.** Parity resting box at voxel 0.01: settlement converges between 27 ppc (4.5 mm) and 64 ppc within 0.1 voxel (1 mm); 8 ppc (6.4 mm) is then the unconverged one.
* **Preset rest.** The preset (P1d + #4660 + 1000 iterations) resting box at voxel 0.01 and 27 ppc tilts < 0.5 deg (the 02c preset box at 8 ppc tilted 3 deg); lowest-corner settlement and creep reported against B5.
* **Heap / layer vs mu.** No prediction of the outcome: the gap (layer deposit - heap angle) at mu 0.68 / 0.90 and the heap at 0.80 / 0.90 are reported with the calibration facts. Reading agreed beforehand: a gap >= 4 deg at every mu means Drucker-Prager's geometry dependence is the limit; < 2 deg at some mu means that mu can define the preset.

## Results (facts for the decision; no choice made here)

* Sphere P1d + #4660 at 27 ppc (1036800 particles): 15.9 mm, 0.98x the 8-ppc depth -> ppc-converged.
* Parity rest, voxel 0.01: 8 / 27 / 64 ppc = 6.4 / 4.5 / 4.2 mm (lowest corner 4.3 mm, tilt 0.08 deg) -> 27-64 converged within 0.1 voxel.
* Preset rest, voxel 0.01, 27 ppc: lowest-corner settlement 1.1 mm (0.11 voxel), centre 1.0 mm, max tilt 0.17 deg, lowest-corner creep 0.56 mm/s -> tilt < 0.5 deg; B5 FAIL.

| mu | atan(mu) | heap R/dx 10 (R/dx 20 est.) | layer deposit | layer - heap | heap - atan | layer - atan |
|---|---|---|---|---|---|---|
| 0.68 | 34.2 | 29.9 (30.7) | 36.7 | 5.9 | -3.5 | 2.5 |
| 0.80 | 38.7 | 33.5 (34.3) | — | — | -4.3 | — |
| 0.90 | 42.0 | 37.6 (38.4) | 48.0 | 9.5 | -3.5 | 6.0 |

* Gap (layer - heap at R/dx 20 est.): 5.9 deg at mu 0.68, 9.5 deg at mu 0.90 -> >= 4 deg at every mu measured: the agreed reading is a Drucker-Prager geometry limit.
* Option A facts, heap target 32 deg: mu = 0.722 (atan 35.8 deg); a plane-shear layer at that mu deposits at ~38.9 deg (6.9 deg above the heap); sphere depth scales ~1/mu (Uehara) from 15.1 mm at mu 0.7.
* Option A facts, heap target 34 deg: mu = 0.789 (atan 38.3 deg); a plane-shear layer at that mu deposits at ~42.3 deg (8.3 deg above the heap); sphere depth scales ~1/mu (Uehara) from 15.1 mm at mu 0.7.
* Option B facts: a Lode-dependent yield surface (Mohr-Coulomb-like / Matsuoka-Nakai) replaces the second-order-cone projection in the DB16 local solver (plan section 2, R3): a new projection kernel in rheology_solver_kernels, flag-gated like the #4660 variant, with the heap / layer / sphere / drop arms as its acceptance tests.
