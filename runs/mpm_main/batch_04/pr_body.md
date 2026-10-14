## MPM material in the main simulator (BatchSim), first slice

MPM material (sand, snow, viscous media, ...) can now be added to a scene whose robot runs on MetalSim's own stack
(`BatchSim`: MuJoCo Warp on Metal, `.venv`), in N worlds, two-way coupled with the robot and rendered by the particle
renderer. Until now MPM only ran through the separate Newton 1.5.2 environment (`.venv-newton152`).

**Design** (`research_notes/MPM in main simulator/NOTE.md`): Isaac Lab 3.0-EA's MPM is Newton 1.5.2's implicit
solver; `.venv` has Newton 1.7, whose implicit MPM is a later algorithm. MetalSim therefore carries the 1.5.2 fork's
solver package as a generated Apache-2.0 copy (`metalsim/physics/mpm/implicit_mpm/`, `scripts/vendor_mpm152.py`:
only the six package-external imports rewritten, Newton's SPDX headers kept, `NOTICE.txt`, THIRD_PARTY_NOTICES
entry), which runs bitwise like Newton's own solver. The coupler (`metalsim/physics/mpm/coupled.py`) reproduces
Isaac Lab's `SolverCoupledProxy` exchange with BatchSim as the rigid entry: proxy bodies from the MuJoCo geoms with
MuJoCo's effective mass (or link mass, or kinematic), staggered or lagged, the solver's fixed-order wrench harvest
into `xfrc_applied`.

**Newton fork fix**: Gauss-Seidel with `separate_worlds` on a sparse grid colored the strain nodes in cell order
while the solver indexes them environment-first, so colors were invalid (a data race; two identical multi-world runs
differed from the first step, in Newton's own 1.5.2 too). Fixed on pulipakaa24/newton (`metalsim-1.5.2`), re-pinned
and re-vendored; Jacobi (Isaac's Franka-Pour) and single-world runs are unaffected.

**Tests** (`tests/test_mpm_main.py`): vendored copy regenerates exactly from the pinned fork commit, licence headers,
pin equals `setup_newton152.sh`'s; column collapse bitwise vs Newton 1.5.2's own solver; box dropped on sand vs
Isaac's coupling path (bitwise sand until contact, then box height / sand force within rounding of the two rigid
stacks); run-to-run bitwise repeatability (1, 2, 64 worlds; 1,024 in the batch logs); valid GS coloring with separate
worlds; free-fall momentum (exact with the solver's numerical drag off); a column collapse on the MuJoCo floor bitwise
equal to the same collapse in Newton alone, and its deposit's response to friction; renderer from the solver's
buffer; no process-wide Warp setting changed. Rigid scenes without MPM: same MuJoCo Warp kernels (module hashes) after
an MPM scene was built, and `BatchSim` unchanged (`probes/rigid_neutral_mpm.py`, `tests/test_physics.py`).

**Rigid determinism is not in this PR.** MuJoCo Warp's BatchSim is not run-to-run repeatable on Metal (contact solve
float atomics in arrival order); an opt-in deterministic mode is developed on branch `batchsim-deterministic` (open:
2 of 256 worlds diverge from step 139 in a two-box stack, suspected in MuJoCo Warp's box-box CCD kernel; GAPS row).
Coupled scenes here repeat bitwise as long as their bodies touch nothing but sand (tested at 1 / 64 worlds, measured
at 1,024).

**Speed** (16-inch, captured coupled step = 3 MuJoCo substeps + 1 MPM step, 2,592 particles + a box per world):
13.7 / 24.3 / 60.6 / 223.6 ms at 1 / 64 / 256 / 1,024 worlds with Gauss-Seidel, 8.3 / 20.3 / 54.7 / 210.5 ms with
Jacobi. A coupling-layer optimization pass follows in a separate PR.

**Known physics limitations, stated honestly** (`MPM.md` section 11; research in `research_notes/MPM in main
simulator/`: `bearing_and_sinking.md`, `angle_of_repose.md`):
* **Isaac's Franka-Pour exchange (`mode="staggered"`) sinks resting bodies** at g dt per step (0.082 m/s at 1/120 s)
  through any bed while the harvested sand force equals their weight: the coupling lag, not the sand (bearing capacity
  is 5-10x the load). Isaac's own coupling path does the same, so parity with Isaac is not physical validity here.
  `MPMCoupling` defaults to Newton's `"lagged"` exchange with the velocity rewind: a resting box settles 10 mm (half a
  voxel) and creeps 0.4 mm/s with a converged solve (250 iterations, tolerance 1e-6), but 25 mm and 16 mm/s with 50
  iterations; a 1.7 m/s drop penetrates 39 mm (31.7 at dt 1/240) against an expected 10-25 mm. The bed fails at
  friction 0.3 as bearing theory predicts.
* **The solver's angle of repose is 7-9 deg below atan(mu)** by a lifted-cylinder test (26.3 / 27.2 deg at R/dx
  10 / 15 for mu 0.68, atan = 34.2; 14.7 for 0.4, 32.6 for 0.9; lift-speed plateau within 0.3 deg; 4x iterations no
  change): it follows friction (slope 0.89) and rises slowly with resolution, but physical sand repose is close to its
  friction angle. This is Newton 1.5.2's solver as Isaac Lab ships it, so Isaac Lab users get the same.

Measurements and logs: `runs/mpm_main/` (timings on the 16-inch only), `metalsim/physics/mpm/MPM.md` section 11.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
