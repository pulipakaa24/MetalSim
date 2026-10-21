# sand_fidelity batch 02b: results

Plan: research_notes/sand_fidelity/PLAN_2026-10-14.md section 11.

## Predictions stated before the runs

* **P1 tests.** The variant's own tests (Newton PR #4660's, CPU and Metal) pass; the parity suites (tests/test_mpm_main.py, tests/test_mpm.py) pass unchanged: the default path is bitwise as before.
* **R2 with #4660 (repose).** With the PR's P1d/Q1d fix the toe band (5-25 % of h_max) is <= atan(mu) + 1 deg (the locking is gone). R2 confirmed if the mid band (25-50 %) stays >= 33 deg and, at R/dx 20, the 25-75 % fit is >= 31 deg. If the mid band falls back to ~28 deg, batch 01's P1d/Q1d gain was the defect, not the strain basis.
* **S-weak (sphere, P1d + #4660).** Depth falls >= 30 % from the parity walled run (21.3 mm), i.e. <= 14.9 mm, and closer to Uehara's 16.3 mm.
* **S-res (sphere, 20 cells, P2).** Depth / Uehara falls >= 30 % from 10 cells (1.31) in both 20-cell variants.
* **S-coupling (kinematic, rerun).** Quasi-static F(16 mm) against the Uehara-implied mean stopping force of 26 N: >= 13 N means the bed is about as strong as the drop needs (the 1.31x residual is not bed weakness); <= 7 N means the bed is weak.
* **Edge leak and #4660.** C1 (plane floor + frictionless side walls, mu 0.4, fixed 20.3 deg) with the variant: still non-finite -> #4660 does not cover the concave two-collider junction (one collider per node); a candidate upstream issue with C1 as the repro. Finite -> covered.
* **Tilt runaway, box floor.** Runaway within atan(mu) +- 1 deg and the deposit within 1 deg below it (mu 0.68, 0.9); no non-finite state; particles below the floor < 0.2 % of the material.
* **Rest, voxel 0.005, S2 (P2).** Runs (no int32 scratch failure); settlement 0.45-0.5 voxel (2.2-2.5 mm) at 250 iterations, no overlap; project_outside changes it by < 0.5 mm.

## Results

* Tests: variant on Metal PASS; parity main PASS; parity 1.5.2 PASS.
* R/dx 10 P1d (#4660): fit 29.89 deg (sd 0.28), toe 36.6, mid 32.0, cone volume 29.1; toe lock still there; at rest False.
* R/dx 10 Q1d (#4660): fit 31.00 deg (sd 0.67), toe 41.8, mid 33.0, cone volume 30.1; toe lock still there; at rest False.
* R/dx 20 P1d (#4660): fit 30.73 deg (sd 0.29), toe 34.4, mid 31.0, cone volume 30.5; toe lock gone; at rest False.
* R2: partial.
* Sphere with P1d + #4660: 16.3 mm vs parity 21.3 mm, Uehara 16.3 -> S-weak not supported.
* 20 cells: s20a 1.46x Uehara, s20b 1.31x Uehara (10 cells 1.31x) -> S-res not supported.
* Kinematic sphere: F(16 mm) 11.2 N, F(30 mm) 25.2 N (Uehara-implied mean 26 N) -> in between.
* C1 with #4660: finite -> covered.
* t1b: creep 33.6, runaway 36.7, deposit -52.2 vs atan(mu) 34.2 -> FAIL; below floor 0.000 %; solves at cap 0.90.
* t2b: creep 42.2, runaway 44.8, deposit 43.8 vs atan(mu) 42.0 -> FAIL; below floor 0.045 %; solves at cap —.
* tilt_p1d_v: creep 38.0, runaway —, deposit 36.9 vs atan(mu) 34.2 -> FAIL; below floor 0.001 %; solves at cap —.
* rest_v005_S2: not run.
* rest_v005_S2_proj: not run.
