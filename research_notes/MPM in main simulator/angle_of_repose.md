# Angle of repose for Newton's implicit MPM: what to expect and how to measure it (2026-10-13)

Research by a sub-agent (source reading + literature; URLs in its report, summarised here). Context: batch_01's
column-collapse deposit gave 10-14 deg for friction 0.4-0.9, and my batch script marked it PASS without judging it.

## What `mpm:friction` means

The solver enforces sqrt(J2) <= mu p (cohesionless, cap at yield_pressure / 2, never reached): `implicit_mpm/
implicit_mpm_solver_kernels.py:249-260` (mu passed through), `rheology_solver_kernels.py:43-77, 100, 107-109, 626,
659-681`, the DB16 stress mapping (Warp `fem/space/dof_mapper.py:218-228`: r0 = tr(sigma)/sqrt(6), |r_T|^2 = J2).
This is the tau_bar <= mu p of Jop et al. 2006, Dunatunga & Kamrin 2015 and Daviet & Bertails-Descoubes 2016; with zero
dilatancy the plastic flow is deviatoric. A heap flank stops flowing in near-simple shear, where tau/sigma_n = mu: the
**expected repose angle is atan(mu)** = 21.8 / 34.2 / 42.0 deg for mu = 0.4 / 0.68 / 0.9 (Dunatunga & Kamrin: "the
material must settle at or below ... arctan(mu_s)"; their MPM deposits all at or below it). Not the plane-strain
(asin mu) or Klar et al. 2016 compression-meridian mappings (other stress paths; Klar's alpha is not Newton's mu).
Static Drucker-Prager states can hold slightly steeper slopes (Modaressi & Evesque), so "angle > atan mu" is not
automatically a bug, but flow-formed heaps should not.

## Why the collapse deposit was not a repose test, and what still looks wrong

A column of aspect ratio 2 collapses inertially and leaves a curved deposit flatter than repose (Lube 2004, Lajeunesse
2004/2005; Sarlin et al. 2021: quasi-static only for release speeds well below 0.4 sqrt(g H)). But the deposit slope
moved only 0.15 deg per deg of atan(mu) (10.4 -> 13.9 deg for 21.8 -> 42.0 deg); inertial runout alone does not explain
that weak a sensitivity. Suspects, all lowering the angle: grid smoothing (quadratic support, ~3 cells) over a flank
spanning only ~6 cells at voxel 0.02; partly filled surface cells carrying no pressure (the unilateral offset switches
cohesion off in non-compact cells, `rheology_solver_kernels.py:270-277`), so a one-cell skin can creep at any slope;
an unconverged solve acting like a lower mu; and the floor friction (0.7) below mu = 0.9 letting the toe slide.

## Protocol (lifted cylinder) and acceptance

* Lift slowly: V / sqrt(g H) <= 0.05 (0.05 m/s with H = 0.08 m is 0.056); prove a plateau with 0.0125 / 0.025 /
  0.05 m/s (< 1 deg between the two slowest). Experiments use 1.5-2 cm/s.
* Resolution: >= 20 cells across the heap radius. Ladder R / dx = 5, 10, 20 at fixed geometry; the angle must rise
  monotonically towards atan(mu).
* Fill aspect H/R ~ 1-1.6, enough volume that every mu produces an avalanche flank (at mu = 0.9 the predicted cone height
  equals the current fill height).
* Rough floor: friction >= 1.0 (above every tested mu), so the toe cannot slide.
* Rest: max particle speed < 1e-3 m/s and fitted angle drift < 0.3 deg over the last 0.5 s; log the angle against rest
  time (a steady fall is surface creep, to be reported, not averaged away).
* Fit: 95th-percentile surface per radial bin, between apex + 1.5 dx and toe - 1.5 dx (or 25-75 % of h_max), per
  azimuthal sector (8 sectors: mean +- sd); cross-check with the cone-volume angle atan(3V / (pi r_toe^3)) and
  atan(h_max / (r_toe - r_apex)); distrust the heap if the three differ by > 2 deg.
* Solver: log the rheology residual; repeat one case with 2-4x the iterations.
* Gate at the finest resolution: angle in [atan(mu) - 3 deg, atan(mu) + 1.5 deg], monotone with resolution, slope
  d(angle)/d(atan mu) in 0.8-1.1 over the three frictions, lift-speed plateau within 1 deg. At R/dx = 5 allow -5 deg
  but require the trend. Experimental repeatability is about +-1-2 deg.

batch_03 implements this in `probes/repose_lift.py` (rough floor, sectors, three estimators, rest criterion, angle vs
rest time) and runs the resolution ladder and the lift-speed plateau at mu = 0.68, then mu 0.4 / 0.9 at the finest
affordable resolution.
