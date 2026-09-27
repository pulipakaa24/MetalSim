# Peak contact penetration at impact: closing the gap to PhysX (G1 1 m drop), 2026-09-26

Question: find the most PhysX-faithful MuJoCo Warp contact setting that closes the landing-penetration gap to PhysX
(Isaac Sim 5.1 GPU PhysX 0.51 mm, `isaac_side_session_2026-09-26.md` §1) as far as MuJoCo's constraint model allows,
without losing what already matches (impulses, 20 ms impact forces, joint limits, slide, transfer, learning), and
recommend one. All numbers are **measured** on this machine (M4 Max, MuJoCo Warp fork f824af1 on Metal, every GPU job
through `scripts/gpu_run.sh`) unless marked otherwise. Scope: one asset (Isaac's G1 minimal USD, mesh-box feet), the three
open-loop protocols of PARITY §1.7 replayed from Isaac's recording (4 identical envs per run, each run repeated for the
key settings), Isaac's PhysX-trained checkpoint 1000 in closed loop (1024 envs × 1000 steps, seeds 1 and 2), Isaac's
checkpoints 500 / 1000 / 1499 for transfer (4 envs).

## Verdict

1. **Definition question: settled, the zero is the same.** PhysX's separation is the raw geometric shape-to-plane
   distance (rest offset 0; the solver subtracts restDistance internally, the report does not), and the G1 settles at
   −0.006 … −0.013 mm, i.e. at rest offset 0. The speculative contact does start the impulse **before** geometric
   contact: 6.5 kN are applied in the step whose pre-solve separation is +1.82 mm. That is the mechanism that bounds
   PhysX's penetration, not a shifted zero, so 0.51 mm is directly comparable to MuJoCo's `contact.dist`. No rest-offset
   accounting is needed. (§1)
2. **On the stock fork, nothing closes the gap without losing what already matches.** The landing foot peak under the
   default is **13.25 mm** (12.79 mm at PhysX's 5 ms cadence). No setting of MuJoCo's soft constraint goes below
   **8.4 mm** at the 2.5 ms step. Every setting that gets near that floor loses a match that holds today:
   - foot slide of Isaac's policy drops 20–25 % below Isaac's (at 5–10× the seed noise);
   - or closed-loop foot chatter rises by +42 %;
   - or the torso-impact forces rise (20 ms 1.04–1.12 → 1.2–1.8× PhysX's; the global τ 5 ms variants also exceed
     PhysX's bounded 5 ms force);
   - or margins lift the robot off the floor by the margin (1.75–7.9 mm at rest, where PhysX rests at 0).

   **Recommendation: keep `recommended` (`tau10_impact_hardlimits_ellip10`) unchanged.** No training run was queued,
   because the recommendation does not differ from the default. (§4, §6)
3. **What the floor is.** MuJoCo creates a contact row only once a corner is inside `margin`, and at τ = 2 dt the row
   removes the approach velocity in one substep. The G1 foot lands on one edge first and slaps down onto the other. So the second edge's corners
   travel a full substep past the plane before they have any row: v·h, with v = 2.3 m/s fall speed plus the rotation
   onto that edge, i.e. more than 5.75 mm. At τ 5 ms the peak is 8.97 mm
   (the first edge is detected at 1.16 mm; the other edge is first seen at 8.23 mm). PhysX detects every corner 13–20 mm early (contact offset) and
   stops each one at the surface. A `margin` detects early too, but MuJoCo centres the spring on dist = margin, so the
   robot then rests a margin high. (§2, §4)
4. **A speculative-contact prototype closes most of the gap, but it is not default-ready.** The prototype is a fork
   worktree, not a preset (`upstream/mujoco_warp-spec`, branch `metalsim-spec-proto`, bc83fb0). It gives contacts
   inside the ground `gap` constraint rows whose reference is PhysX's speculative bias a_ref = −(v + dist/h)/h.
   With τ 5 ms it reaches:

   | metric | prototype (τ 5 ms) | default | PhysX |
   |---|---|---|---|
   | foot peak | **1.56 mm** | 13.25 mm | 0.51 mm (5.1), 2.08 mm (6.1) |
   | torso peak | 0.33 mm | 10.7 mm | 0.10 mm |
   | depth 55 ms after the landing peak | 0.08 mm | 1.53 mm | 0.08 mm |
   | open-loop chatter | 12 | 120 | – |
   | closed-loop short contact phases | 0.4 % | 36 % | – |
   | physics cost | 1.00× | 1.00× | – |

   The 40 ms impact forces are closer to PhysX's on all three events. Two things stop it becoming the default:
   - Isaac's checkpoints walk 10–14 cm further than in Isaac: transfer error 0.117 m vs 0.060 m for the default.
   - The drop-torso 5 ms force is 4 % over PhysX's bound.

   It is also an unreviewed fork change. It is archived as the next step (§5).

## 1. The definition question

**Sources.**
- PhysX 5.6.1 TGS contact prep (`upstream/PhysX-5.6.1/physx/source/lowleveldynamics/src/DyTGSContactPrep.cpp:371–373`):
  `penetration = FSub(separation, restDistance); isSeparated = FIsGrtr(penetration, zero)`.
- The same file, :394: `biasCoeff = FNeg(FSel(isSeparated, invStepDt, invDtp8))`. A separated contact therefore gets the
  bias −separation/dt, and the solve (:1529–1553) makes it the unilateral constraint v_n ≥ −(separation − restDistance)/dt.
  It pushes only on the part of the approach that would cross the surface within the step. The GPU path does the same:
  `contactConstraintBlockPrep.cuh:924–941`, `solverBlockTGS.cuh:299–306`.
- restDistance is the sum of the shapes' rest offsets (`ScShapeInteraction.cpp:866`). The reported separation is the
  narrow-phase value: "The separation of the shapes at the contact point. A negative separation denotes a penetration"
  (`PxSimulationEventCallback.h:466`). The tensor view copies it unchanged (`CpuRigidContactView.cpp:443`).
- `PxShape.h:423`: "Two shapes will come to rest at a distance equal to the sum of their restOffset values. If the
  restOffset is 0, they should converge to touching exactly."
- Isaac Lab's G1_CFG sets no collision offsets (`isaaclab_assets/robots/unitree.py:272–288`). The recorded foot colliders
  author −inf, i.e. PhysX's automatic contact offset with rest offset 0 (session note §1).
- On CPU PhysX the full contact report and the tensor view agree to 1e-7 m (session note). So the GPU tensor numbers
  report the same quantity.

**The recording, read step by step** (`runs/parity3/isaac/penetration_il2/C_drop_penetration.npz`; step k reports the
contacts generated from the pose after step k−1):

| PhysX step | pose after the previous step: min foot separation | foot origin z at end of step | foot v_z over the step | Σ normal force applied in the step |
|---|---|---|---|---|
| 46 | – (outside the contact offset) | 48.36 mm | −2.26 m/s | 0 |
| 47 | +13.34 mm | 36.84 mm | −2.31 m/s | 0 (the gap is not closed within the step: 13.3 mm > 11.5 mm of travel) |
| 48 | **+1.82 mm** | 34.78 mm | −0.41 m/s | **6497 N** (speculative: the approach would cross the plane) |
| 49 | **−0.51 mm** (peak) | 34.21 mm | −0.11 m/s | 1376 N |
| 50–60 | −0.48 → −0.08 mm | 34.3 mm | ≈ 0 | 1472 → 659 N |
| 401–601 | −0.013 mm (settled) | | | 316 N |

So the impulse starts before geometric contact, as the question suspected. It starts one step before the foot crosses
the plane, and only as much as that step's travel requires. The foot still ends that step 0.2–0.5 mm inside
(articulation and the 8 TGS iterations). Both engines measure penetration from the same zero, geometric contact, and
PhysX's settled −0.013 mm confirms rest offset 0. The like-for-like MuJoCo quantity is `contact.dist` of the deepest
foot-ground contact, from the same kind of pose (the start of the substep, i.e. after the previous one), sampled at
PhysX's 5 ms cadence. The "5 ms tick" columns below do exactly that. The per-substep maxima are reported next to them.

## 2. MuJoCo's constraint model and what bounds penetration

- **Force law** (modeling.html#solver-parameters): a_ref = −b v − k r with b = 2/(d_w τ), k = d(r)/(d_w² τ² ζ²)
  (standard form). The realised acceleration is a = d·a_ref + (1 − d)·a_u. The resting violation is
  r = a_u (1 − d) τ² ζ². The impedance d(r) ramps from d0 to d_w over `width` and is clamped to [0.0001, 0.9999].
- **refsafe**: "the solver uses max(solref[0], 2*timestep)" (XMLreference, option/flag). MuJoCo Warp applies the same
  clamp (`constraint.py:103–104`). The direct form (negative solref) is not clamped, but only b ≤ 1/h avoids overshoot,
  so the direct form adds only stiffer springs (ζ < 1) at the same damping. At τ = 2h the contact is deadbeat: it removes
  the approach velocity in one substep.
- **Hard constraints**: "If we forced the solution to reach y = a_ref, which we could do by taking the limit R → 0, we
  would obtain a hard constraint model. This limit is not allowed in MuJoCo" (computation/index.html).
- **margin / gap**: "Active contact (distance ≤ margin) … The constraint impedance function is applied to the quantity
  distance − margin"; contacts in (margin, margin + gap] are detected but generate no force (computation/index.html
  #coMarginGap). MuJoCo Warp: `pos = dist − includemargin` with includemargin = margin (`constraint.py:4359–4360`), and
  rows only where pos < 0 (`constraint.py:2785–2792`). So the spring is centred on dist = margin. MuJoCo has no
  equivalent of PhysX's "detect early, rest at 0".
- **What bounds the landing peak** (measured, MuJoCo C and MuJoCo Warp CPU device on the same drop, per-corner contacts):
  - The foot lands on one edge. Only those two corners are inside the plane at the first detection (−1.16 mm).
  - Even at the deadbeat τ 5 ms, the other edge first appears 8.23 mm deep one substep later, because the foot rotates
    onto it at >3 m/s with no row to stop it.
  - The peak therefore has two parts: a detection lag of up to v·h per corner, and the soft spring's compliance under
    the body's follow-through load. The second part is 4 mm at τ 10 ms and ~0 at τ 5 ms.
  - Consequences: (i) τ < 10 ms cuts only the compliance part, down to 8.4–9.0 mm at τ 5 ms; (ii) impedance and width do
    nothing to the peak (the corner that has no row sets it); (iii) only early detection (margin, or a smaller dt)
    reaches the lag.
- **Published MuJoCo humanoid / quadruped settings** (sources checked by a research sub-agent, quotes in the run log):
  - MuJoCo Playground G1 / H1 / T1 / Berkeley Humanoid / Op3 / Go1 keep MuJoCo's default solref / solimp and pyramidal
    cones (G1: 2 ms, 3 iterations). Go1's only non-default is foot solimp width 0.023 (the foot radius). The paper
    (arXiv 2502.08844) states no contact rationale.
  - Menagerie `unitree_g1/scene_mjx.xml` uses foot-floor `solref="0.008 1"` at a 4 ms step, i.e. the refsafe floor 2 dt.
  - Menagerie Go2 uses elliptic cones with impratio 100, `margin="0.001"`.
  - mjlab keeps defaults for the G1. For the Go1 it sets `solref=(0.01, 1)` with the comment "Harden all collision
    geoms", plus elliptic cones and impratio 10: the same τ and cone as our default.
  - arXiv 2503.04613 (MuJoCo MPC, H1): "the moderate ground penetration from the softness of the contact model does not
    cause issues in the sim-to-real transfer".
  - arXiv 2504.13619 randomizes foot solref in (0.02, 0.4) and attributes a sharp force peak to "contact penetration".
  - None of them targets millimetre penetration. Our default is already as stiff as the stiffest published setting that
    keeps τ above the floor.
- **PhysX TGS** reaches near-rigid contact through the speculative bias above, applied at each TGS sub-iteration
  (8 position iterations per 5 ms step for the G1), with restitution 0 and depenetration capped at 1 m/s
  (`max_depenetration_velocity`). It cannot bounce by construction.

## 3. Protocol and tooling (what changed)

- `metalsim/parity/record_g1.py` now also records, for every substep:
  - the signed distance of the deepest ground contact per body class (left foot, right foot, torso), counting every
    detected contact, so positive values inside a margin or gap show up like PhysX's contact-offset separations;
  - the |net normal force| of those bodies;
  - the Newton iterations.

  It also uses the contact preset's Newton cap (20 for the elliptic presets). **Before this, record_g1 ran every preset
  at cap 10.** The elliptic rows of the earlier tables were therefore at cap 10, which is where the "1.55 cm" came from
  (env 0, max over all contacts, torso included). At cap 20 the default's drop maximum is 13.25 mm, from the feet.
- `scripts/diagnostics/bench_contact_tuning.py` uses the preset's cap too (before: always 10).
- `scripts/diagnostics/penetration/`:
  - `sweep.sh` runs the stages rec / transfer / air / cost / cap; `MJW=` selects a fork worktree, `DT=` the step,
    `TAG=_r2` makes a repeat run.
  - `summarize.py` builds the full table (`runs/penetration/sweep_table.md`, all 48 rows × 14 columns, and
    `summary.jsonl`).
  - `profile_table.py` builds the landing profile; `cap_table.py` the cap-probe table.
- Presets (all archived in `metalsim/physics/contact_tuning.py`, one change each on top of the elliptic default):
  `ellip10_{tau7p5, tau6, tau5, d095, d099, dmax9999, w2mm, w1mm, imp30, margin2mm/4mm/6mm, tau5_*, feet_*, tau2p5,
  *specgap*}`. A `Tuning.geom_contact` field gives per-geom (feet-only) parameters through MuJoCo's priority rule.
- Noise floor:
  - The open-loop landing is deterministic: the 4 envs and the repeat runs give identical landing numbers to 0.01 mm.
  - What varies after the fall is the torso impact. Across 8 env-runs the default's drop-torso 20 ms force spans
    1.04–1.12× PhysX's, and its 40 ms force 1.06–1.09×.
  - Closed loop, seeds 1 / 2: air ±0.0006, slide ±0.0002, short-phase fractions ±0.004, falls ±13.
  - Transfer: the 4-env spread is ±0.05 m, and the default measured 0.060 m here vs 0.033 m in the 2026-09-25 run.

## 4. The sweep (G1 1 m drop and the hold / random protocols against Isaac 5.1 PhysX)

Depths in mm: the deepest foot (or torso) contact, positive = penetration. The 20 ms and 40 ms forces are
momentum-derived: ours ÷ PhysX's (1996 / 2200 / 2190 N for landing / drop-torso / hold-torso), env range in brackets.
"5 ms force" = the largest 5 ms mean contact force per event against the bounds on PhysX's (2174–6522 / 2618–7854 /
2682–8045 N). "Chatter" = foot-flag transitions in A_hold + C_drop at 2.5 ms / on 5 ms averages. Impulse ratios are
0.993–0.995 / 0.969–0.978 / 0.987–0.990 for **every** row (momentum conservation), and the drop limit excursion is
0.0024–0.0040 rad for every row (PhysX 0.004). Full columns: `runs/penetration/sweep_table.md`.

| setting | foot peak substep / 5 ms tick | torso peak drop / hold | settle (last 0.5 s) | depth 55 ms after peak | 20 ms force land / drop-torso / hold | 40 ms force land / drop-torso / hold | 5 ms force > PhysX bound? | chatter | substeps at cap 20 |
|---|---|---|---|---|---|---|---|---|---|
| **PhysX 5.1 GPU** | – / **0.51** | 0.10 / – | 0.013 | 0.08 | 1 | 1 | – | – | – |
| **recommended** (default) | 13.25 / 12.79 | 10.7 (8.9–16.5) / 13.6 | 0.25 | 1.53 | 1.35 / 1.04–1.12 / 1.07 | 1.11 / 1.06–1.09 / 1.07 | no | 120 / 58 | 0.02 % |
| τ 7.5 ms | 11.02 / 11.02 | 7.5 / 12.7 | 0.14 | 0.69 | 1.27 / 1.23 / 1.20 | 1.06 / 1.10 / 1.07 | no | 116 / 50 | 0.02 % |
| τ 6 ms | 9.79 / 9.20 | 5.7 / 2.7 | 0.09 | 0.26 | 1.19 / 1.22 / 1.16 | 1.03 / 1.08 / 1.06 | **yes** (8237, 9355) | 106 / 42 | 0 |
| τ 5 ms | 8.97 / 8.23 | 5.1 / 2.6 | 0.064 | 0.10 | 1.08 / 1.22 / 1.15 | 1.00 / 1.10 / 1.06 | **yes** (10029, 9663) | 116 / 42 | 0 |
| τ 5 ms, ζ 0.5 / 0.7 | 8.63 / 8.37 | 5.9 / 4.3 | 0.05 / 0.03 | 0.67 / 0.33 | 0.99–1.13 / 1.53–1.83 / 1.43–1.45 | 0.96–0.99 / 0.88–1.12 / 1.08–1.15 | **yes** (up to 13184) | 200 / 56 (ζ 0.5) | 0 |
| τ 5 ms, ζ 1.5 / 2 | 9.74 / 10.01 | 3.8 / 8.2 | 0.14 / 0.25 | 0.53 / 1.89 | 1.11–1.12 / 1.20–1.23 / 1.15–1.16 | 1.00 / 1.10–1.11 / 1.04–1.05 | **yes** | 86–90 / 36–38 | 0 |
| τ 5 ms, d0 0.8 / 0.95 | 8.60 / 9.14 | 6.1 / 2.7 | 0.16 / 0.03 | 0.37 / 0.92 | 1.11–1.15 / 1.11–1.25 / 1.14–1.18 | 0.99–1.01 / 1.06–1.09 / 1.05–1.07 | **yes** | 78 / 156 | 0 |
| τ 5 ms + width 2 / 10 mm, dmax 0.9999, impratio 30 | 8.96–9.15 | 3.5–6.2 | 0.064 | 0.10–0.14 | 1.08–1.13 / 1.19–1.22 / 1.14–1.16 | 0.98–1.00 / 1.08–1.10 / 1.05–1.06 | **yes** | 106–116 | ≤ 0.05 % |
| d0 0.95 / 0.99 | 13.42 / 13.55 | 9.2 / 12.5 | 0.12 / 0.022 | 1.38 / 0.74 | 1.33–1.34 / 1.12–1.20 / 1.08–1.09 | 1.11–1.12 / 1.05–1.09 / 1.07–1.08 | no | 144 / **168** | 0.02 % |
| dmax 0.9999 | 13.25 / 12.79 | 15.4 | 0.25 | 1.53 | 1.35 / 1.08 / 1.07 | 1.11 / 1.08 / 1.07 | no | 120 / 60 | **0.31 %** |
| width 2 / 1 mm | 13.45 / 13.59 | 12.9–14.8 | 0.22–0.25 | 1.04–1.40 | 1.33–1.34 / 1.06–1.07 / 1.07 | 1.11 / 1.08 / 1.06–1.07 | no | 116–122 | 0.07–0.09 % |
| impratio 30 | 13.25 / 12.79 | 14.1 | 0.25 | 1.53 | 1.35 / 1.06 / 1.06 | 1.11 / 1.07 / 1.07 | no | 118 / 52 | 0.01 % |
| ground margin 2 / 4 / 6 mm | 6.20 / 5.54 / 7.14 | 3.2–13.2 | **rests 1.75 / 3.75 / 5.75 mm above the floor** | −0.4 … −4.5 | 1.03–1.15 / 1.13–1.67 / 1.09–1.15 | 0.98–0.99 / 1.08–1.13 / 1.07–1.08 | no | 106–118 | ≤ 0.05 % |
| τ 5 ms + margin 6 / 8 mm | 2.90 / −4.32 | −1.1 / 2.3 | **rests 5.9 / 7.9 mm above** | – | 0.94–1.03 / 1.22–1.94 / 1.26–1.33 | 0.95–0.96 / 1.09–1.13 / 1.05–1.06 | **yes** | 106–110 | ≤ 0.02 % |
| feet-only τ 7.5 ms | 11.02 / 11.02 | 11.4 / – | 0.14 | 0.69 | 1.27 / 1.21 / 1.08 | 1.06 / 1.11 / 1.07 | no | 116 / 50 | 0.01 % |
| feet-only τ 5 ms | 8.97 / 8.23 | 11.9 / 10.0 | 0.064 | 0.10 | 1.08 / 1.22 / 1.11 | 1.00 / 1.11 / 1.08 | no | 110 / 42 | 0 |
| feet-only τ 5 ms, ζ 2 | 10.01 / 9.96 | 14.5 / 9.8 | 0.25 | 1.89 (still 2.0 mm at 60 ms: overdamped creep) | 1.11 / 1.66 / 1.12 | 1.00 / 1.12 / 1.08 | no | 86 / 40 | 0.01 % |
| feet-only τ 5 ms, ζ 1.5 / d0 0.8 | 9.74 / 8.60 | 10.9–12.7 | 0.14–0.16 | 0.37–0.53 | 1.12–1.15 / 1.48–1.52 / 1.11–1.14 | 1.00–1.01 / 1.11–1.13 / 1.07–1.08 | no | 78–90 | ≤ 0.01 % |
| *what it takes*: τ 5 ms @ 1.25 ms step | 4.76 / 4.17 | 4.6 / 5.3 | 0.064 | 0.11 | 1.09 / 1.15 / 1.21 | 0.97 / 1.00 / 1.07 | **yes** | 176 / 50 | 0 |
| *what it takes*: τ 2.5 ms @ 1.25 ms (feet-only: same feet) | 2.63 / 2.50 | 5.0 / 3.8 | 0.016 | 0.03 | 0.85 / 1.23 / 1.25 | 0.91 / 1.00 / 1.04 | **yes** (feet-only: no) | 156 / 36 | 0 |
| **prototype** speculative gap 10 mm, τ 10 ms | 5.46 / 5.46 | 0.93 / 1.9 | 0.25 | 0.18 | 1.02 / 1.67 / 1.10 | 0.94 / 1.11 / 1.03 | yes (8261; hold 7892 ok) | **12 / 12** | 0.01 % |
| **prototype** speculative gap 10 mm, τ 7.5 ms | 3.59 / 3.46 | 0.76 / 4.7 | 0.14 | 0.14 | 0.93 / 1.39 / 1.17 | 0.92 / 0.92 / 1.04 | yes (hold 8594) | 12 / 8 | 0 |
| **prototype** speculative gap 10 mm, τ 5 ms (20 mm: identical) | **1.56 / 1.56** | **0.33 / 0.32** | 0.064 | **0.08** | 0.76 / 0.99 / 1.11 | 0.91 / 0.98 / 1.04 | +4 % (8168 vs 7854) | **12 / 12** | 0 |

Closed loop (Isaac's PhysX-trained checkpoint 1000, mean action, 1024 envs × 1000 steps). Isaac's own log: air 0.0446,
slide −0.0127. Transfer: Isaac's checkpoints 500 / 1000 / 1499 walk 3.19 / 3.02 / 3.11 m in Isaac. Physics-only cost:
4096 envs, each preset at its own cap 20, 3 interleaved repeats.

| setting | feet_air_time (s1 / s2) | feet_slide (s1 / s2) | falls / episodes | contact / air phases < 20 ms | median contact phase | dof_acc term | transfer x (err) | physics cost |
|---|---|---|---|---|---|---|---|---|
| recommended | 0.0373 / 0.0384 | **−0.0130 / −0.0134** | 169 / 144 | 36 % / 32 % | 120 ms | −0.0117 | 3.23 / 2.96 / 3.03 (0.060) | 46.6 ms (1.00×) |
| feet-only τ 5 ms | 0.0389 / 0.0392 | −0.0102 / −0.0104 | 132 / 130 | **51 %** / 41 % | **17 ms** | −0.0239 | 3.25 / 3.07 / 3.12 (0.040) | 0.99× |
| global τ 5 ms | 0.0390 | −0.0103 | 129 | 51 % / 41 % | 17 ms | −0.0242 | 3.27 / 3.07 / 3.12 (0.047) | 0.98× |
| feet-only τ 5 ms, ζ 2 | 0.0389 / 0.0394 | −0.0096 / −0.0098 | 149 / 138 | 35 % / 32 % | 120 ms | −0.0207 | 3.26 / 3.05 / 3.12 (0.037) | 0.96× |
| feet-only τ 5 ms, ζ 1.5 | 0.0393 | −0.0098 | 135 | 42 % / 36 % | 112 ms | −0.0216 | 3.28 / 3.07 / 3.13 (0.053) | – |
| feet-only τ 5 ms, d0 0.8 | 0.0394 | −0.0099 | 138 | 40 % / 34 % | 115 ms | −0.0235 | – | – |
| feet-only τ 7.5 ms (global: same) | 0.0383 | −0.0116 | 152 | 44 % / 36 % | 107 ms | −0.0151 | 3.24 / 3.01 / 3.07 (0.033) | 1.00× |
| **prototype**, τ 5 ms, gap 10 mm | 0.0390 / 0.0393 | −0.0137 / −0.0137 | 132 / 130 | **0.4 % / 0.7 %** | 145 ms | −0.0118 | 3.32 / 3.16 / 3.19 (**0.117**) | 1.00× |
| **prototype**, τ 10 ms, gap 10 mm | 0.0392 | −0.0153 | 139 | 3 % / 3 % | 142 ms | −0.0108 | 3.31 / 3.07 / 3.13 (0.063) | 1.02× |
| **prototype**, τ 7.5 ms, gap 10 mm | 0.0395 | −0.0143 | 129 | 1 % / 1 % | 142 ms | −0.0112 | 3.32 / 3.12 / 3.16 (0.093) | – |

Newton cap probe (4096 envs; `cap_probe_ellip.py` protocol; random actions and the elliptic-trained walking policy):
cap 20 is at the cap-100 noise floor for every candidate. Worlds at the cap per step, random / policy:
- default: 4.4 / 0
- feet τ 5 ms: 0.4 / 0
- feet τ 5 ms ζ 2: 1.9 / 0
- feet τ 7.5 ms: 1.5 / 0
- prototype τ 5 ms: 0 / 0; prototype τ 10 ms: 0.5 / 0

Mean iterations are 3.3–3.9 everywhere. The stiff-feet presets raise the floor itself on walking states: p99 |Δqvel|
0.0154 vs 0.0022 for feet τ 5 ms, i.e. the gait becomes more sensitive (`runs/penetration/cap/`).

Landing time profile, like for like: deepest foot contact per 5 ms tick in mm. Negative = the separation of a contact
already inside the contact offset / gap. PhysX's time axis includes its settle step.

| t [ms] | 230 | 235 | 240 | 245 | 250 | 255 | 260 | 265 | 270 | 280 | 295 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isaac 5.1 PhysX | −13.34 | −1.82 | **0.51** | 0.48 | 0.26 | 0.23 | 0.18 | 0.15 | 0.13 | 0.11 | 0.08 |
| recommended | – | – | 6.62 | **12.79** | 12.75 | 10.32 | 7.43 | 4.90 | 2.95 | 0.73 | 1.75 |
| τ 5 ms | – | – | **8.23** | 7.40 | 3.85 | 1.75 | 0.63 | 0.58 | 0.34 | 0.89 | 0.13 |
| feet τ 5 ms, ζ 2 | – | – | 8.13 | **9.96** | 8.67 | 7.34 | 6.21 | 5.26 | 4.46 | 3.21 | 2.03 |
| margin 4 mm | – | – | 4.25 | **5.54** | 3.79 | 1.34 | −0.80 | −2.22 | −3.13 | −3.88 | −2.04 |
| τ 2.5 ms @ 1.25 ms | – | – | **2.50** | 0.76 | 0.22 | 0.07 | 0.40 | 0.15 | 0.05 | 0.03 | 0.03 |
| prototype, τ 10 ms | – | −4.67 | 3.42 | **5.46** | 4.76 | 3.26 | 1.90 | 0.96 | 0.40 | 0.01 | 0.11 |
| prototype, τ 5 ms | – | −4.67 | **1.56** | 1.14 | 0.51 | 0.16 | 0.00 | −0.01 | 0.00 | 0.02 | 0.08 |

## 5. Speculative contacts in MuJoCo Warp (prototype, EXPERIMENTAL)

**Mechanism.** MuJoCo's primal solver already treats a contact row as unilateral through its reference acceleration: a
row pushes only while J·a would fall below a_ref. So a row at positive distance behaves speculatively by itself, if it
exists. The prototype makes it exist and gives it PhysX's reference (fork worktree `upstream/mujoco_warp-spec`, branch
`metalsim-spec-proto`, commits 2b424a4 + bc83fb0, patch `runs/penetration/spec_prototype.patch`, active only with
`MJW_SPECULATIVE_GAP=1`):
1. `plane_convex` detects out to margin + gap. The stock fork passes only margin, so gap-zone plane contacts never
   existed.
2. Every detected contact gets rows (`constraint.py`, the two row-allocation kernels).
3. A row with pos = dist − margin > 0 gets a_ref = −(v + pos/h)/h at impedance d0.

Item 3 is PhysX's v(t+h) ≥ −pos/h. Rows at pos ≤ 0 are unchanged, so resting contact is MuJoCo's. The presets
`ellip10_*specgap*` put the gap on the ground. On the stock fork gap contacts are inactive, and those presets equal
their base presets.

**Validation.**
- CPU device, box drop, elliptic, τ 5 ms: the box is caught at +2.26 mm (503 N), lands at 0.00 mm and rests at −0.01 mm.
  efc forces equal the momentum change to 0.1 %.
- First version, impedance dmax on the speculative rows: on the G1, when feet and torso were all inside the gap, the
  elliptic Newton line search stalled. `efc.force` reported 540 kN while the momentum changed by 5 kN (CPU device,
  `scripts/diagnostics/penetration/spec_g1_force_check.py`; box case `spec_box_check.py`; drop profile `spec_cpu_check.py`). That corrupts contact sensors. Those runs are archived in `runs/penetration/spec_dmax/`
  and are not used.
- At impedance d0 the forces match the momentum balance to < 1 % over the torso impact.

**Result (τ 5 ms, gap 10 mm).** Numbers are in the tables above:
- foot peak 1.56 mm (PhysX 0.51 / 2.08 mm by Isaac version), torso 0.33 mm (PhysX 0.10 mm);
- recovery matches PhysX's 0.08 mm at 55 ms; settle 0.064 mm (PhysX 0.013 mm);
- chatter −90 % open loop, and closed-loop short contact phases 36 % → 0.4 %;
- 40 ms forces 0.91 / 0.98 / 1.04 (default 1.11 / 1.06 / 1.07);
- impulses and limits unchanged, cap 20 at the floor, physics cost 1.00×.

**Why it is not the recommendation:**
- (a) Transfer regresses: Isaac's checkpoints walk 3.32 / 3.16 / 3.19 m against 3.19 / 3.02 / 3.11. The error is 0.117 m
  vs 0.060 m, outside the 4-env spread for checkpoints 1000 and 1499. Double stance rises 8.7 → 11.5 % and flight falls
  1.5 → 0.6 %. Slide moves −0.0130 → −0.0137 (Isaac −0.0127). The cause is not established. Hypothesis: speculative rows
  also brake swing feet descending within 10 mm, a softer touch-down than PhysX's final-substep-only bias, since
  MuJoCo's 2.5 ms step stands in for PhysX's 0.625 ms TGS sub-iteration.
- (b) The drop-torso 5 ms force is 8168 N vs PhysX's upper bound 7854 N (+4 %).
- (c) It is a MuJoCo Warp change: MetalSim fork policy, a C oracle (MuJoCo C has no such rows) and tests are all
  missing.
- (d) The dt-equivalence question in (a): PhysX closes the gap per 0.625 ms sub-iteration.

Next steps:
- Brake only when the gap closes within the step (already the rule) *and* use h_sub = h/4 in the bias, to mimic TGS.
- Measure transfer again.
- A confirming training run.
- A MuJoCo C reference implementation.

## 6. Recommendation

**Keep `recommended` = `tau10_impact_hardlimits_ellip10` (no change; no training run).** Evidence: §4. On the stock fork
every penetration reduction below the default's 13.25 mm costs one of the matches the default holds:
- **Slide.** Isaac's policy slides −0.0130 in ours vs −0.0127 in Isaac (2 %). Every τ ≤ 7.5 ms foot setting gives
  −0.0096 … −0.0116, 9–25 % less sliding than Isaac, at 5–10× the seed noise.
- **Chatter.** τ 5 ms at ζ = 1 raises closed-loop short contact phases 36 → 51 %, the median contact phase 120 → 17 ms,
  and the joint-acceleration term 2×.
- **Torso-impact forces.** All τ ≤ 6 ms global settings exceed PhysX's bounded 5 ms torso force by 5–68 %. Every
  stiffer-foot setting except τ 5 ms / d0 0.95 moves the drop-torso 20 ms force from 1.04–1.12 to 1.19–1.84. That
  one exception stays at 1.11 but exceeds the 5 ms bound instead. The 40 ms force moves from 1.06–1.09 to 1.08–1.13,
  which overlaps at the low end.
- **Rest height** (margins): the robot rests the margin above the floor.
- **Impedance, width, dmax 0.9999, impratio 30**: no peak reduction at all (the corner without a row sets the peak).
  d0 0.99 raises open-loop chatter 120 → 168. dmax 0.9999 raises the substeps at the cap 15×. Unlike joint limits, it
  does not diverge for contacts.

**Best trade-off on the stock fork, quantified** (for an owner who weights penetration higher): **feet-only τ 7.5 ms**
(`ellip10_feet_tau7p5`).
- Gains:
  - foot peak −17 % (13.25 → 11.02 mm), settle −43 % (0.25 → 0.14 mm), depth at 55 ms 1.53 → 0.69 mm;
  - landing 20 ms force 1.35 → 1.27;
  - within all 5 ms bounds; transfer 0.033 m; falls 169 → 152; cost 1.00×.
- Costs:
  - slide −0.0116 (9 % below Isaac's);
  - short contact phases 36 → 44 %;
  - drop-torso 20 ms force 1.21, 40 ms 1.11;
  - dof_acc +30 %.

The remaining gap to PhysX (11 mm vs 0.5 mm) stays open. Physically it needs early detection without a rest offset,
i.e. speculative contacts (§5), or a smaller step: 1.25 ms halves the lag (τ 5 ms: 4.8 mm), at ~2× physics cost, with
its own torso-force excess.

Artifacts: `runs/penetration/` holds `rec/<preset>[_dt…][_r2][_spec]/`, `air/`, `cap/`, `*.log`, `sweep_table.md`,
`profile_table.md`, `summary.jsonl`, `spec_prototype.patch`, and `spec_dmax/` (the rejected first prototype).
