# Hand-off: so101-tilt (2026-10-31, 21:40 to 22:30)

## Status

**Done** (research increment, CPU only; nothing committed; no GPU job).

**Answer.** The box does not swing in the jaws. It rotates with the hand: the expert's lift pitches the hand about
the arm's pitch axis, and the box goes with it.
- Hand pitch change from lift start to 6 s: median **−35.8°** (seed 10051: −45.7°), over a ~15 cm lift.
- The box's tilt in that plane changes by **−39.2°**.
- In-hand rotation: median 8° (seed 10051: 4.6°).

**Contact parameters are not the cause.** condim, torsional friction, μ, solref, impratio and cone each move the 6 s
median by at most about ±5°.

**Real SO-101 teleop lifts pitch the hand the same way.** hungho77's 15 cm lifts: median −44.6°. IlyaGusev's 8 cm
lifts: −17.2°.

**Verdict:** the tilt's size is **realistic** for this lift (kinematic, and seen in real teleop data). Two parts are
less realistic, each second order:
- the expert grasps with the hand **26-28° off vertical** (real operators: 4-13°), so the box sits about 28° to the
  jaw axis;
- MuJoCo's documented slow slippage adds about 5°.

**No contact-model change is proposed.** The one sourced remedy, noslip, does not exist on MuJoCo Warp and also
stiffens BAM's joint friction.

## Changed

All new, in `runs/readme_media/scratch/so101-tilt/` (git-ignored), plus this hand-off:

- `tilt_probe.py`: grasp_cpu.py's loop.
  - Adds ablation hooks (`opt`, `grip`, `grasp_dz`, `tcp`, `pitch_hold`, `box_mu`, `servo`, `over`).
  - Decomposes the tilt into the arm plane and out of it.
  - Measures in-hand rotation and dumps contact detail.
- `run_ablations.py` (driver), `table.py` → `table.md`, `torque_budget.py`, `keyframe.py`, `pitch_hold.py` (an
  experiment, not a proposal), `pitch_feasible.py`, `real_pitch.py`.
- `out/*.json` (one per run), `logs/*.log`.
- `runs/readme_media/handoff/so101-tilt.md` (this file).
- Session scratchpad only: `mjdocs_3.14.0/`, MuJoCo 3.14.0 doc and engine sources fetched from GitHub for the
  citations.
- **Not touched:** `so101.py`, `so101_expert.py` (md5 9884575e, unchanged), `assets/`, every so101Sim file.

## Verified

All sim runs: main venv, MuJoCo 3.14.0, film scene `assets/so101/scene_box_rl.xml`, `reset_draw`, servo `7v4_bam`,
seeds 10000-10099, 6 s.
- Baseline command: `.venv/bin/python runs/readme_media/scratch/so101-tilt/tilt_probe.py 100 '{}' out/base.json`
- **The baseline reproduces so101-film-bam's `grasp_cpu.py` exactly:** median tilt at lift start / 3 s / 6 s =
  11.0 / 36.6 / 39.9°; seed 10051 = 14.66 / 42.38 / 43.42°; 100/100 successes, 1 retry.

### Q1. Which geoms touch the box during the hold

Seed 10051, 7v4_bam, t = 3 s (`out/base.json` → `episodes[51].detail`):
- **11 contacts, all jaw geoms, no floor.**
  - Fixed jaw: `fixed_jaw_sph_tip2`, `fixed_jaw_box4`, `fixed_jaw_box5` ×2, `fixed_jaw_box6`.
  - Moving jaw: `moving_jaw_sph_tip2`, `moving_jaw_sph_tip3`, `moving_jaw_box2` ×2, and 2 on geom #41, the
    collision mesh `moving_jaw_so101_gripper_part0_v1`.
  - 13 contacts at 4 s and at 6 s.
  - So the box touches both the primitive pads and the moving jaw's collision mesh.
  - Over the 100 seeds (grasp_cpu.py's count), the touching geoms are the same set plus `fixed_jaw_box3`,
    `fixed_jaw_sph_tip1`, `moving_jaw_box3`, and rarely `gripper#31` (7 seeds), `moving_jaw_sph_tip1` (3) and the
    floor (1).
- **Where the contacts are:**
  - Box frame, mm from the COM: x from −13.2 to +11.0, z from +0.8 (the moving jaw's tip spheres) to +30.0 (the top
    edge: the jaw mesh and `fixed_jaw_box6`), on both y faces (±20).
  - **Centre of pressure: (2.6, −0.2, 22.3) mm.** The jaws hold the box's upper half, centred across the face and
    22 mm above the COM.
  - Normal force sum 40.0 N; penetration up to 0.76 mm.
- **Contact parameters**, read from `d.contact`, the same on every jaw contact:
  - dim 6, friction (1, 1, 0.005, 5e-4, 5e-4), solref (0.01, 1), solimp (0.9, 0.95, 0.001).
  - The box's own friction (1·fs, 0.03, 0.003) and condim 3 are ignored: the gripper classes have `priority="1"`
    (`assets/so101/so101.xml:23-28`), and the higher-priority geom sets condim, friction, solref and solimp (MuJoCo
    `doc/modeling.rst:424-470`, "Contact parameters").
  - `efc_R`: normal about 1.0, tangential about 0.1, torsional about 4000. That matches
    `engine_core_constraint.c:2244-2254`: R[1] = R[0]/impratio, R[j] = R[1]·μ0²/μj², here (1/0.005)² = 4×10⁴. The
    MuJoCo Warp port is the same (`upstream/mujoco_warp/mujoco_warp/_src/constraint.py:4806-4820`).
- **Model origin:** `so101.xml` is byte-identical to so101Sim's Menagerie clone, which has no local changes
  (`git -C ~/so101Sim/mujoco_menagerie status`). Menagerie README step 6: "Add default collision solver parameters for
  the gripper that work well for manipulation" (`robotstudio_so101/README.md:27`).

### Q2. Is the tilt mechanically expected?

**Decomposition.** Columns: in-plane = rotation about the shoulder_lift axis (signed); out-of-plane = rotation about
the radial axis.

| quantity, median over 100 seeds | lift start | 6 s |
|---|---|---|
| box tilt | 11.0 | 39.9 |
| box in-plane / out-of-plane | +1.0 / 10.5 | −39.4 / 3.5 |
| hand (gripper z) in-plane / out-of-plane | −28.3 / 0.0 | −65.6 / 0.0 |

- **The box is upright in the arm plane when lifted and ends where the hand's pitch takes it.**
- The hand's in-plane change equals the change of shoulder_lift + elbow_flex + wrist_flex on every seed (the three
  axes are parallel; identical percentiles: −46.8 / −35.8 / −28.3°).
- The jaws close mostly along the pitch axis: |gripper x · pitch axis| median 0.92 (p10 0.72). So the hand's pitch
  rotates the box in the pad plane, which gives the "hangs from a corner" look.

**Seed 10051 time series** (`series`):

| time | phase | box tilt | hand off vertical | box in hand |
|---|---|---|---|---|
| 1.44 s | grasp | 0° | 18.0° | |
| 1.56-1.74 s | jaws close | 0 → 14.7° | 18 → 22.6° | 18 → 28.6° |
| 1.92-2.34 s | lift | 13 → 42.7° | 23 → 69° | in-hand change ≤ 2.4° |
| 3-6 s | hold | | | creep about 0.3°/s |

**Torque budget about the closing axis** (seed 10051, box 0.12 kg; `torque_budget.py` → `logs/torque_budget.log`):

| time | gravity torque | at 90° | torsional limit Σμ_t·fn | torsion used | patch sliding limit Σμ·fn·r (r 4-20 mm) |
|---|---|---|---|---|---|
| 3 s | 0.019 N·m | 0.026 N·m | 0.20 N·m | 0.0016 N·m | 0.50 N·m |
| 6 s | 0.018 N·m | | 0.21 N·m | 0.0028 N·m | 0.54 N·m |

- **Gravity cannot turn the box:** about 10 % of the torsional limit, about 4 % of the sliding limit. The tilt is
  not gravity overcoming torsional friction.
- Some contacts do sit at the tangential cone limit (|ft|/(μ·fn) up to 1.0 at 2.5-4 s, median 0.2-0.5). The load is
  the jaws' **wedge**, not gravity:
  - on a 4 cm box (gripper q 0.342 rad) the moving pad's normal is 19.6° off the fixed pad's;
  - the fixed jaw's pad faces step 6.7° along the jaw (box4 to box7);
  - the box face sits 12.6° off the fixed-pad normal.
  - This is the SO-101 CAD geometry (pivoting jaw, tapered fixed jaw).
  - **Inference:** this wedge drives the slow in-hand rotation, mostly about the hinge axis (median 6.2°).
- **MuJoCo torsional friction:** for condim 4, friction[1] is a torsional coefficient with "units of length which can
  be interpreted as the diameter of the surface contact patch"; for condim 6, friction[2] is rolling, also a length,
  "the depth of the local deformation" (`doc/computation/index.rst:1072-1084`, Contact, condim; cones from `:1090`). The elliptic cone
  bounds the torsion at μ_t·fn when there is no tangential load (`index.rst`, Friction cones). So 5e-3 means a
  5 mm patch.
- **Real SO-101 jaws:**
  - Printed in PLA+ at 15 % infill, with no pads in the BOM. The optional compliant gripper has a TPU 95A finger and
    "optional 3M gripper tape for higher friction" (https://github.com/TheRobotStudio/SO-ARM100 README).
  - Menagerie's pads approximate the jaw faces: fixed jaw box4 8×8 mm, box5 10×12, box6 18×16, box7 20×16 mm;
    moving jaw box2 8×8, box3 10×12 mm.
  - **Torsional friction for PLA or TPU jaws: unknown.** No measured value was found (web search; the closest result,
    https://www.mmscience.eu/2021127, says PLA jaws have a low friction coefficient; only the search summary was read).
    It does not matter here: condim 3 (no torsion) gives 40.7° against 39.9°.

### Q3. Grasp location

- **Above the COM by design.** so101Sim `rl/native/scripted_expert.py:29` (d4c2990) sets `GRASP_DZ = 0.008`
  ("gripperframe target height above cube center"). The port keeps it at `so101_expert.py:49`, used at `:236-238`.
- **The TCP comes from the `pickup` keyframe** (`scripted_expert.py:37-47`, "where the cube center IS the grasp
  point"). `keyframe.py` shows that in that keyframe:
  - the cube lies on its side (long axis along the hinge axis) and touches only the floor, no jaw;
  - the gripper is open at 0.728 rad.
  - The TCP is therefore, in the gripper body frame:
    - 10 mm above the fixed jaw's tip sphere;
    - 33 mm from the fixed pad face, i.e. 13 mm more than a 4 cm box's half width;
    - 10.8 mm off the jaw centreline along the hinge axis.
- **Consequences, measured:**
  - Box COM in the gripper frame at grasp start: x 24.4 mm; at lift start: 13.4 mm.
  - The closing jaw pushes the box a median 6.6 mm (p90 8.1) across the table, and it tips 10.5° out of plane.
  - With box-table μ 0.5 the tip disappears (0.3°), but the 6 s tilt is unchanged (39.4°).
  - By design the jaw tips sit about 2 mm below the COM (TCP 10 mm above the tip, 8 mm above the COM). Measured:
    lowest contact z +0.8 mm, patch up to +30 mm, CoP +22 mm.
- **Real operators** (`real_pitch.py` → `out/real_pitch*.json`, `logs/real_pitch*.log`). Data: LeRobot v3.0 teleop
  data, robot_type so_follower, the parquets so101-expert downloaded (`hfdata/` in this session's scratchpad).
  Columns: hand tilt off vertical (deg) and Δpitch = change of lift + elbow + wrist_flex, grasp to highest point
  within 2 s; p10 / 50 / 90.

  | dataset | episodes | rise cm | hand tilt at grasp | hand tilt after lift | Δpitch |
  |---|---|---|---|---|---|
  | hungho77 pick red cube into pot | 50 / 50 | 13-18 | 1 / 4 / 10 | 33 / 41 / 50 | **−54 / −45 / −32** (92 % ≥ 30°) |
  | IlyaGusev pick gum | 88 / 104 | 6-12 | 1 / 7 / 17 | 5 / 13 / 21 | **−30 / −17 / −9** |
  | expert (sim, 7v4_bam) | 100 | 15 | 26 (grasp start), 28 (lift start) | 66 | −47 / −36 / −28 |

  - Same sign and similar size as the expert for a lift of similar height.
  - Δpitch does not depend on calibration offsets. The absolute tilts do: with the owner's offsets
    (`~/so101Sim/rl/real/joint_map.json`, Σ over the pitch joints 15.2°), hungho77's grasp tilt is 13° and its lift
    tilt 57°; IlyaGusev's 10° and 28°.
  - **Real grasps are nearer top-down than the expert's.** The objects' grasp height cannot be read from joint data.
- **LeRobot's own EE-space IK** weights orientation 0.01 against position 1.0 (`~/lerobot` bf31dd79,
  `src/lerobot/model/kinematics.py:98-129`, `robots/so_follower/robot_kinematic_processor.py:300`). Like the expert,
  it lets the pitch drift.
- **Can the hand pitch be held?** `pitch_feasible.py`: bounded least squares, roll held. At the lift target (TCP at
  0.20 m over the grasp xy) the joint limits allow a median 10.1° minimum pitch change (p90 26.8°). Only 10/100 seeds
  can hold the pitch within 2°. Seed 10051 needs ≥ 24.4° (wrist_flex at its 1.658 rad limit). So the expert's −36° is
  about 25° more than the kinematic minimum: a choice of position-only IK, not forced by the arm.

### Q4. MuJoCo specifics (sources at the installed 3.14.0 tag, https://github.com/google-deepmind/mujoco/tree/3.14.0/doc)

- **Soft friction creeps.** For friction dimensions of elliptic cones r ≡ 0, so the reference acceleration is pure
  damping (`computation/index.rst:1648-1653`). "Slow slippage: … a persistent tangential load can produce a nonzero
  steady slip velocity. This is expected behavior" (`modeling.rst:1740-1748`).
  - **Measured:** in-hand rotation median 8.0° (p90 13.6°). With noslip it is 0.1°.
- **Remedies the docs give** (`modeling.rst:1750-1762`, `:549-554`):
  - (a) elliptic cones and a larger impratio ("reduces slow slippage but does not guarantee exact sticking");
  - (b) noslip, "A small number (1, 2 or 3) is usually sufficient", which "can occasionally cause instabilities".
  - The scene already uses (a): `so101.xml:2` sets cone elliptic, impratio 10.
  - impratio: "not recommended to use high impratio values with pyramidal cones" (`XMLreference.rst:336-345`).
  - Elliptic "correspond more closely to physical reality" (`modeling.rst:549-551`; `index.rst:1734`).
- **noslip does not fit the film:**
  - MuJoCo Warp raises `NotImplementedError("noslip solver not implemented.")` when noslip_iterations > 0
    (`upstream/mujoco_warp/mujoco_warp/_src/io.py:311-312`, fork at 787a5b3a).
  - MuJoCo's noslip sweeps the dry-friction rows (dof frictionloss, where BAM writes its load-dependent budget) before
    the contact friction (`src/engine/engine_solver.c:770`, sweeps at `:805` "dry friction" and `:828` "contact friction"), so it would also
    change the servo model.

### Ablation table

`table.md`, from `python run_ablations.py` and `table.py`. One change per row (except the noslip10_pitch_hold
combination); 100 seeds; angles in degrees, medians, p90 in brackets.

| run | success | tilt lift start | tilt 3 s | tilt 6 s (p90) | hand pitch change lift→6 s | box in-plane change | box out-of-plane lift / 6 s | in-hand rotation (p90) | 10051 tilt lift / 6 s |
|---|---|---|---|---|---|---|---|---|---|
| base (7v4_bam) | 100 | 11.0 | 36.6 | 39.9 (49.5) | −35.8 | −39.2 | 10.5 / 3.5 | 8.0 (13.6) | 14.7 / 43.4 |
| servo 7v4_real | 99 | 5.3 | 28.8 | 28.6 (42.0) | −31.4 | −28.8 | 4.2 / 0.6 | 11.5 (20.5) | 6.6 / 26.6 |
| condim 3 (jaws) | 100 | 11.9 | 37.0 | 40.7 (49.4) | −35.9 | −37.8 | 10.5 / 3.4 | 9.2 (14.6) | 15.7 / 43.6 |
| condim 4 | 100 | 11.1 | 36.4 | 40.0 (49.5) | −35.8 | −39.2 | 10.5 / 3.3 | 8.1 (13.7) | 15.4 / 42.8 |
| torsional 1e-5 | 100 | 11.7 | 36.5 | 40.5 (49.3) | −35.9 | −37.8 | 10.3 / 3.4 | 8.6 (14.1) | 15.6 / 43.6 |
| torsional 0.03 | 99 | 9.3 | 36.6 | 38.6 (49.0) | −35.8 | −37.1 | 9.0 / 4.6 | 8.3 (10.4) | 9.8 / 42.6 |
| sliding μ 0.5 (jaws) | 100 | 10.6 | 40.7 | 42.6 (53.3) | −37.9 | −39.6 | 10.0 / 6.2 | 7.3 (18.1) | 11.6 / 43.8 |
| solref 0.02 (jaws) | 100 | 11.6 | 38.3 | 40.8 (50.3) | −35.7 | −40.2 | 11.1 / 3.7 | 7.3 (12.8) | 16.2 / 44.0 |
| impratio 1 | 100 | 10.4 | 44.3 | 45.4 (56.0) | −36.3 | −43.7 | 9.5 / 6.7 | 8.9 (23.3) | 11.4 / 45.4 |
| impratio 100 | **94** | 11.5 | 35.4 | 36.8 (68.3) | −37.4 | −35.5 | 8.4 / 0.5 | 2.2 (101.8) | **104.5 / 115.0** |
| pyramidal | 99 | 10.3 | 41.4 | 42.1 (54.6) | −36.4 | −39.8 | 9.7 / 7.4 | 10.3 (25.1) | 10.6 / 42.2 |
| noslip 2 | 100 | 10.9 | 35.0 | **35.0** (46.2) | −35.9 | −35.1 | 10.3 / 1.4 | **0.1** (0.6) | 14.7 / 41.2 |
| noslip 10 | 100 | 11.1 | 35.0 | 35.1 (45.9) | −36.0 | −35.0 | 10.5 / 1.6 | 0.1 (0.8) | 14.9 / 41.1 |
| grasp dz +16 mm | 100 | 12.3 | 36.9 | 39.4 (50.3) | −36.4 | −37.9 | 11.3 / 3.9 | 7.7 (10.3) | 16.5 / 41.9 |
| grasp dz 0 | 100 | 9.8 | 34.0 | 37.6 (48.2) | −35.8 | −38.6 | 9.2 / 3.1 | 11.1 (17.4) | 13.6 / 40.3 |
| grasp dz −8 mm | 95 | 10.1 | 33.7 | 32.5 (46.0) | −36.8 | −36.0 | 5.8 / −4.6 | 10.8 (97.3) | 12.9 / 36.0 |
| box-table μ 0.5 | 100 | **0.4** | 39.2 | 39.4 (50.6) | −38.5 | −39.1 | 0.3 / −0.3 | 11.2 (15.3) | 0.5 / 46.0 |
| pitch-hold lift (experiment) | 99 | 11.0 | 59.1 | 58.7 (127.5) | −9.1 | −57.0 | 10.5 / 18.2 | 8.2 (15.6) | 14.7 / 81.7 |

Reading the table:
- **The hand's pitch change is −35.7° to −38.5° on every contact ablation.** The box follows it.
- Contact parameters move the 6 s median between 35.0 and 45.4°.
- **impratio 100** flips the box during the jaw closure: seed 10051 jumps 5.5° → 43° in 40 ms at 1.64-1.68 s; 10
  seeds end flipped or failed. It is no option. Not investigated further.
- **The pitch-hold lift is confounded**, so it is not evidence either way. wrist_flex hits its 1.658 rad limit, then
  the 4-row damped least squares spins the wrist roll (median 86°), so the tilt grows.
- **7v4_real vs 7v4_bam** (28.6° vs 39.9°): a smaller hand pitch change (−31.4 vs −35.8°: 7v4_real's elbow and
  wrist sag more) and a smaller closure tip (4.2 vs 10.5°).

### Marked experiment-only

- The decomposition and every number in the table.
- The wedge as the driver of in-hand creep (an inference from cone use and the pad angles).
- The real-data detection rules (squeeze ≥ 3 points for ≥ 0.5 s, the highest point within 2 s, rise ≥ 3 cm). Not
  checked against the datasets' videos.

## Open

1. **No real measurement of a held box's tilt.** Settle it by replaying seed 10051's command stream on a real follower
   (LeRobot registers, 30 Hz absolute goals; the stream is the expert's `cmd` in `tilt_probe.py`). Use a 4×4×6 cm box
   of ~0.1 kg at the seed's pose. Photograph from the side, along the pitch axis, at the lift start and at 6 s, and
   measure:
   - (a) box tilt and hand tilt from vertical;
   - (b) the box's angle to the jaws, before and after a 4 s hold (in-hand creep; the sim gives 3-6°).
2. **Real grip force and PLA μ unknown.** The sim gives 40 N on 7v4_bam and 23 N with the 0.986 N·m limit. At 40 N
   the patch's sliding-moment limit (0.50 N·m × μ) stays above the 0.019 N·m gravity torque for μ > 0.04. The wedge's
   tangential load on a real jaw is not known. Inference only.
3. **The expert grasps 26-28° off vertical where real operators grasp at 4-13°.** A top-down grasp would put the box
   in line with the jaws; the lift's ~36° pitch would then tilt it about as much as now (inference, not run).
   Feasibility at spawn radii 0.15-0.26 m within the wrist_flex limit: not checked.
4. **The 7v4_bam elbow sags about 45 mrad, against 7v4_real's 123.** so101-expert found real elbows sag about as
   7v4_real predicts. So 7v4_bam may over-pitch by about 4° (inference).
5. impratio 100 box flip and the grasp dz −8 mm failures (5 seeds): not investigated.

## Next

1. **Film notes, no code change:** "The box rotates with the hand: the scripted lift pitches the gripper by about 36°
   (real teleop lifts of the same height pitch it by 32-54°, hungho77). Box-in-jaw creep from MuJoCo's soft contacts
   adds about 5°." Source: this hand-off's Q2-Q4.
2. **If the owner wants the box in line with the jaws** (look, not angle), a fresh agent on so101-expert's files
   could:
   - add an orientation row to `DeployExpert`'s descend IK, the hand's z within 5-13° of vertical (target from
     `out/real_pitch.json`);
   - check reachability with `pitch_feasible.py` adapted to the grasp pose first;
   - run `tilt_probe.py` against `base`.
   - That would change so101Sim's plan: the owner must agree.
3. **Measurement** (Open 1) on the real arm, if available.
