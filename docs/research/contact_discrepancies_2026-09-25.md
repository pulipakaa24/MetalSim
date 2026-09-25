# The two contact discrepancies that survived the tuning sweep: impact peaks and feet air time

2026-09-25. Scope: the two gaps to Isaac's PhysX recordings that no setting of the 11-setting sweep closed
(PARITY §1.7, `docs/research/mujoco_contact_vs_physx_2026-09-25.md`): (A) contact-force peaks 3–5× Isaac's at equal
mean force; (B) feet_air_time 0.016–0.022 vs Isaac 0.035 and feet_slide −0.025 vs −0.017. Every number is labelled
**measured** (run here, with the script and log named) or **estimated**. Diagnostics: `scripts/diagnostics/contact_research/`;
outputs: `runs/contact_research/`.

## Verdict

| discrepancy | verdict | evidence (short) |
|---|---|---|
| A. impact peak (3.0–3.7 kN vs 0.6–1.0 kN) | **reporting artefact** (plus a different sub-step time profile that no control-rate recording can resolve); **no physics difference found** | Isaac's own recorded states give its contact impulses by momentum balance: its sensor saw 42–45 % less impulse than its bodies received during the torso impacts. Impulses agree to 1–4 % for every setting. Force averaged over each 20 ms control step peaks at 2.0–2.2 kN in PhysX and 2.4–2.6 kN in MuJoCo (default); MuJoCo's 5 ms peak lies inside the bounds PhysX's own data allow. The "drop peak" in the tables is the torso hitting the floor at ~1.4 s, not the 1 m landing. The landing itself reads 917 vs 1030 N per foot. |
| B. feet_air_time / feet_slide | **mainly a policy difference**, with a **secondary physics difference in transfer** and a small **convention difference** in slide | With the default contact model held fixed, five policies span 0.022 to 0.048 (measured, B(1)). Our own iteration-1000 policy reaches 0.048, above Isaac's 0.0446. The table's 0.016–0.022 is one policy at iteration 400: the flat-config run, played with mean actions. Isaac's own checkpoint 1000 gets 0.029 with default contacts and 0.037 with impact-only stiffening, against its log value 0.0446. For that policy the contact model moves air time by 26 %, slide from −0.021 to −0.016 (Isaac −0.013) and falls from 310 to 142. Our slide uses the foot frame-origin velocity, while Isaac uses the COM velocity: 5–15 % more slide on our side. Stiff settings make the contact flag chatter (B(3)). |

## A. Impact peak

### A(1) What each side reports (the question that settles it)

**Isaac (primary sources).**
* Isaac Lab v2.3.2, `ManagerBasedRLEnv.step`: for each of the 4 decimation steps it runs `sim.step()` and then `scene.update(dt=physics_dt)`
  ([manager_based_rl_env.py L182–197](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)).
* `SensorBase.update` refreshes the buffers whenever `history_length > 0` ([sensor_base.py L184–192](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/sensors/sensor_base.py)).
  The velocity task's `contact_forces` has `history_length=3`, `update_period = sim.dt` and `force_threshold` 1.0 (velocity_env_cfg.py).
* `ContactSensor._update_buffers_impl` calls `get_net_contact_forces(dt=self._sim_physics_dt)` ([contact_sensor.py L373](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py)).
* omni.physics.tensors: "dt – The time step of the simulation to use to convert impulses to forces"
  ([API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)).
* PhysX `PxContactPairPoint::impulse`: "Divide by the simulation time step to get a force value"
  ([PhysX 5.6.1 API](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxContactPairPoint.html)).
* The GPU kernel behind the call sums the per-patch **normal** impulses and multiplies by 1/dt:
  `upstream/PhysX-omni-5.6.1/omni/extensions/runtime/source/omni.physx.tensors/plugins/gpu/CudaKernels.cu:1526–1541`
  (`impulse += impulseMag * patch.normal; return timeStepInv * impulse;`, tag 107.3-omni-and-physx-5.6.1).
* In the TGS solver the accumulated `appliedForce` is reset once per `simulate()` in constraint prep, and
  `writeBackContactBlockTGS` writes the sum over all position and velocity iterations
  (`physx/source/gpusolver/src/CUDA/solverBlockTGS.cuh`, `contactConstraintBlockPrep.cuh`).

So what `record_g1.py` stored after each `env.step()` (`contacts.data.net_forces_w`) is **the normal impulse of the
last 5 ms PhysX step of the 20 ms control step, divided by 5 ms**. The first 15 ms of every control step are never
reported.

**Ours.** `metalsim.sensors.contact.ContactSensor` runs after every 2.5 ms substep. Each value is MuJoCo's constraint
force of that substep, which for a velocity-level solve is the substep's impulse / 2.5 ms.
`metalsim.parity.record_g1` stores the value of the **last** substep (8 of 8). `compare.py` then takes, per body,
the max over the 150–250 control steps.

**Where the tables' peaks come from** (measured, `peak_windows.py`, `runs/contact_research/peak_windows.json`):
* The "drop peak" 3499–3746 N (default) and 3255 N (τ10 impact-only) is the **torso** striking the floor after the
  robot tips forward (control step 71, t = 1.42 s). It is not the 1 m landing.
* In the landing (step 11–12) the per-foot sample is 917 N (default) against Isaac's 1030 N.
* The hold peaks are the torso impact as well: default 3715 N, τ10 impact-only 848 N.
* Isaac's "712 N" hold peak is a **foot** (step 44), and its torso impact reads 587 N.

**Per-substep re-recording** (measured):
* `substep_forces.py` replays the three protocols at 2.5 ms and records every substep: MuJoCo C on the CPU, and
  MuJoCo Warp through the GPU queue (see the Warp confirmation below).
* Its last-substep samples reproduce the Warp recordings used in the tables: torso 3745.8 vs 3745.7 N (default),
  3233.6 vs 3254.9 (τ10 impact-only), feet 3073.6 vs 3073.6 (τ5).

The torso impact of A_hold under different reporting windows (N, per body; `peak_windows.py`):

| setting | 2.5 ms peak | max 5 ms window | max 20 ms mean | last substep (tables) | Isaac-style (last 5 ms of the step) | Isaac recorded |
|---|---|---|---|---|---|---|
| default | 4808 | 4080 | 2483 | 3715 | 4080 | 587 |
| τ10 impact-only + hard limits | 9722 | 4861 | 2573 | 849 | 1221 | 587 |
| τ5 imp 0.99 + hard limits | 19452 | 9785 | 2446 | 203 | 235 | 587 |

Reading:
* The number in the tables depends on where in the control step the impact falls: 203 to 3715 N for the same
  impulse. Stiffer contacts finish the impact inside the step, so the last-substep sample misses it, exactly as PhysX's
  last-5 ms sample does. Soft contacts are still pushing at the end of the step.
* Applying Isaac's reporting (the last 5 ms) to our physics does **not** bring our number down: 4080 N for default.
  So the two sensors differ not by their window but by **when inside the control step each engine's impulse arrives**.

**Isaac's impulses from its own states** (measured, `momentum_impulse.py`):
* Isaac recorded joint positions and velocities plus the root pose and velocity every control step. Isaac Lab 2.3.2's
  `root_lin_vel_b` is the root **COM** velocity, and the script converts it to the frame origin.
* Loading those states into MuJoCo on the same asset gives the robot's total linear momentum P(t). The contact impulse
  over each control step is then J = ΔP_z + M g · 20 ms, with M = 32.24 kg.
* The method checks out on both engines: in steady contact J/20 ms equals the sensor (Isaac 247 vs 247 N, ours 257 vs
  257 N), and on ours the impulse sums match the sensor sums (151.6 vs 153.3 N s).

| event | Isaac: impulse (momentum) / from its sensor samples × 20 ms | ours (default): impulse (momentum) / samples × 20 ms | max 20 ms-mean force Isaac / ours |
|---|---|---|---|
| A_hold torso impact (steps 61–76) | 154.2 / **89.6** N s | 151.6 / 153.3 | 2190 / 2578 N |
| C_drop landing (steps 9–20) | 143.0 / 123.1 | 142.7 / 144.5 | 1996 / 2430 |
| C_drop torso impact (steps 63–76) | 143.1 / **79.2** | 133.3 / 136.5 | 2200 / 2600 |

* During the torso impacts **Isaac's sensor missed 42–45 % of the impulse its bodies received**. At step 66 of the
  hold, the momentum balance says 2171 N averaged over 20 ms while the sensor reads 641 N (all bodies).
* The impulse over each event agrees with ours to 0.1–7 %. Across all 7 recorded settings, impulses over the event
  windows agree to 1–4 % (`summary_table.py`, `runs/contact_research/summary_table.md`,
  `runs/contact_research/momentum_impulse_presets.log`).
* The remaining 18–21 % in the 20 ms peaks follows the impact velocity. Our robot reaches the floor slightly faster:
  pelvis v_z −3.05 vs −2.67 m/s one step before the torso impact.

**Bounds on PhysX's 5 ms peak** (measured, `win5_bounds.py`):
* In each control step, the first three PhysX steps carried J − 5 ms · F_last. That puts Isaac's largest 5 ms force
  between an even spread over three steps and all of it in one step.

| event (total over bodies) | Isaac max 5 ms force (bounds) | default | τ10 impact-only | τ5 imp 0.99 | Isaac Lab 3.0 mapping (0.005, 1.375) |
|---|---|---|---|---|---|
| C_drop landing | 2174 – 6522 N | 2961 | 5063 | 5625 | 5255 |
| C_drop torso impact | 2618 – 7854 | 4404 | 7862 | 10517 | 9520 |
| A_hold torso impact | 2682 – 8045 | 4364 | 4861 | 9785 | 9412 |

* **Conclusion A(1):** the 3–5× peak ratio is a reporting artefact. It comes from sampling one short window per
  control step on two engines whose impacts have different sub-step time profiles.
* Measured on the quantities both recordings determine, the physics agrees: impulse (1–4 %), 20 ms-mean peak (+18–21 %,
  tracking the impact velocity) and mean force (≈10 %). Default MuJoCo's 5 ms peak lies inside PhysX's bounds.
* The stiffest settings (τ5, and Isaac Lab 3.0's own mapping) **exceed** PhysX's upper bound at the torso impact.
  At the 2·dt floor MuJoCo's contact is deadbeat, so it removes the approach velocity in one 2.5 ms step, where
  PhysX's step is 5 ms (next section).
* Consequence for the tables: the "contact peak" column (control-step sample, per body) should not be used to rank
  contact settings. The impulse and the momentum-derived 20 ms force are like-for-like.

**Warp confirmation.** The same per-substep recording on MuJoCo Warp through the GPU queue: see the addendum at the end.

### A(2) PhysX's impact model (primary sources)

* **Contacts are rigid unless compliant contact is enabled**: "Since PhysX 5.1 it is possible to use a compliant contact
  model … by assigning a negative restitution value" ([RigidBodyDynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/RigidBodyDynamics.html)).
  Isaac Lab's `compliant_contact_stiffness` defaults to 0 (rigid).
* **TGS**: "The number of substeps is equal to the number of position iterations" (G1: 8 position iterations → 0.625 ms
  substeps, 4 velocity iterations). The impulse accumulated over them is reported per 5 ms step (A(1)).
* **Offsets**:
  * `g1_minimal.usd` authors no `physxCollision:contactOffset` / `restOffset`, no material, and exactly 3 colliders
    (two foot boxes, one torso box); G1_CFG sets no `collision_props` (read with pxr).
  * Omniverse then computes the offset itself (`omni.physx` `usdLoad/Collision.cpp:275`, `UsdInterface.cpp`):
    planes get 0.02 m; the foot box gets max(2·dt²·g, 0.02 × smallest extent) ≈ 0.5 mm (**estimated** from that rule).
    Rest offset is 0.
  * A contact generated at positive separation is solved speculatively: TGS uses the inverse *substep* dt as bias for
    separated contacts (`DyTGSContactPrep.cpp:421`), so the gap closes within a substep and the approach stops there.
* **max_depenetration_velocity = 1.0** (G1_CFG) clamps only the penetration-recovery bias term (`maxPenBias`), not the
  velocity term, so it does not cap how fast the impact velocity is removed ([PxRigidBody::setMaxDepenetrationVelocity](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxRigidBody.html)).
* **Restitution 0** (terrain material, multiply) and `bounce_threshold_velocity` 0.5: no bounce.
* **Reading (estimated):**
  * A foot or torso that reaches the floor has its normal approach velocity removed within one TGS substep, as a hard
    constraint, and the reported impulse is concentrated in the 5 ms step in which that happened.
  * The articulated body then decelerates over the following steps through the joints. That is why Isaac's own
    momentum shows the torso-impact impulse spread over two control steps (2171 and 2190 N), not one.
  * MuJoCo's soft contact spreads the same impulse over τ ≈ 10–40 ms. That is the design difference ("impact spread over
    several substeps by design" is MuJoCo, not PhysX).
  * At τ = 2·dt MuJoCo is also deadbeat, but its step is 2.5 ms against PhysX's 5 ms reporting step. So its 5 ms peak
    can exceed PhysX's, as measured for τ5 and the Isaac Lab 3.0 mapping.

### A(3) MuJoCo-side options not yet tried (docs, issues, published configs)

Facts:
* **Force law** ([modeling.html#solver-parameters](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters)):
  * a_ref = −b v − k r, with b = 2/(d_w·τ) and k = d(r)/(d_w²·τ²·ζ²).
  * The per-contact force is ≈ m_eff·d·(b|v| + k r + a₀), since R = A(1−d)/d
    ([issue #2630](https://github.com/google-deepmind/mujoco/issues/2630)).
  * The velocity removed per step is (d/d_w)(2h/τ)|v|. **At the refsafe floor τ = 2h the contact is deadbeat**, and the
    first-step force is ≈ m_eff·v/h.
  * ζ only lowers k; b depends on τ alone. So "overdamped" ζ > 1 does not reduce the impact spike.
* **refsafe**: "the solver uses max(solref[0], 2*timestep)" ([XMLreference](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-flag-refsafe)).
  MuJoCo Warp has the same clamp (`mujoco_warp/_src/constraint.py:103–104`). The direct form (−k, −b) is not clamped.
* **o_solref / o_solimp / override**: not supported by MuJoCo Warp (`types.py:318`
  `# unsupported: OVERRIDE, FWDINV, ISLAND`; `io.py` rejects the flag). The same effect is available per geom.
* **Mixing**: the geom with higher `priority` supplies solref/solimp; at equal priority they are solmix-weighted; a
  non-positive solref on either side takes the element-wise minimum
  ([modeling.html#contact-parameters](https://mujoco.readthedocs.io/en/latest/modeling.html#contact-parameters)).
  Foot-only or torso-only solref therefore needs `priority` on those geoms.
* **condim** does not change the normal law.
* **implicitfast**: its derivatives exclude constraint forces ("D includes derivatives of all forces except the
  constraint forces", [computation](https://mujoco.readthedocs.io/en/latest/computation/index.html)). Contact damping is
  effectively explicit, which is why refsafe exists. The integrator does not stiffen contacts.
* **Changelog** 3.2–3.14: nativeccd default (3.3.0); margins summed (3.5.0); margin/gap redesign (3.9.0); an
  `integrator="discrete"` whose "constraints are stable at any timeconst" (3.13.0, **C only**, not in MuJoCo Warp).
  Nothing else changes impact behaviour.
* **Maintainers on peaks**: "looking at contact force magnitude in this situation is useless. This is a
  timestep-dependent quantity. You should be looking at the integral of the impulse (force*dt)" (Tassa,
  [#1620](https://github.com/google-deepmind/mujoco/issues/1620); likewise [#1743](https://github.com/google-deepmind/mujoco/issues/1743)).
* **Newton's ke/kd → MuJoCo mapping** (`upstream/newton/newton/_src/solvers/mujoco/kernels.py:189–203`; newton-1.5.2 `:192–206`):
  * Formula: `timeconst = 2/(kd·d_width)`, `dampratio = kd/2·sqrt(d_r/ke)`.
  * Applied per geom as `convert_solref(ke, kd, 1.0, 1.0)` (`solver_mujoco.py:6602`, the default
    `SOLREF_MODE_MJCF_DEFAULT`, no mass scaling); solimp stays MuJoCo's default.
  * Isaac Lab 3.0's `NewtonShapeCfg(margin=0.0, ke=160000, kd=1100)`
    (`scratch/deformable/il3/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:71`, 2 substeps of
    5 ms) maps to **solref (1.82 ms, 1.375)**. refsafe raises that to **(5 ms, 1.375)**, default solimp (0.9, 0.95, 0.001).
  * That is close to our τ5 preset with a softer impedance. The earlier reading of ke/kd as the direct form
    (−160000, −1100) was not Newton's mapping, which is why that robot never fell.
* **Published humanoid configs**:
  * MuJoCo Playground G1 (`g1_mjx_feetonly.xml`): 2 ms, Euler, no solref/solimp (default), foot pairs condim 3,
    friction 0.6.
  * mjlab G1 (`g1_constants.py:220–247`): seven r = 1 cm capsules per foot, priority 1, condim 3, friction 0.6, default
    solref/solimp, 5 ms step.
  * None tunes impact forces.

Measured on MuJoCo C, same protocols (`summary_table.py`; MuJoCo C reproduces the Warp recordings, above). The columns
are the 5 ms peak / 20 ms-mean peak / event impulse, and foot-flag flicker (1 N flag transitions in A_hold + C_drop, both
feet, at 2.5 ms / on 5 ms averages):

| setting | drop landing | drop torso | hold torso | flicker |
|---|---|---|---|---|
| Isaac PhysX (5 ms: bounds; momentum) | 2174–6522 / 1996 / 194.5 N s | 2618–7854 / 2200 / 245.8 | 2682–8045 / 2190 / 253.6 | – |
| default | 2961 / 2421 / 193.1 | 4404 / 2602 / 237.9 | 4364 / 2581 / 250.9 | 104 / 52 |
| hard limits | 2961 / 2421 / 193.1 | 4411 / 2605 / 238.6 | 4336 / 2567 / 251.4 | 124 / 44 |
| τ10 impact-only + hard limits (provisional pick) | 5063 / 2702 / 193.3 | 7862 / 3716 / 239.2 | 4861 / 2693 / 251.2 | 112 / 60 |
| τ5 imp 0.99 + hard limits | 5625 / 2078 / 193.3 | 10517 / 4251 / 239.7 | 9785 / 2564 / 250.8 | 212 / 42 |
| **Isaac Lab 3.0 mapping (0.005, 1.375), default solimp** + hard limits | 5255 / 2191 / 193.1 | 9520 / 2517 / 241.0 | 9412 / 2557 / 251.1 | 94 / 38 |
| τ5, default solimp + hard limits | 5656 / 2172 / 193.1 | 11105 / 2900 / 241.2 | 9752 / 2584 / 251.1 | 110 / 40 |
| τ10 impact-only, ζ = 2 + hard limits | 4539 / 2496 / 193.0 | 6972 / 2541 / 240.1 | 4666 / 2512 / 251.1 | 96 / 44 |
| τ10 impact-only, width 20 mm + hard limits | 4668 / 2679 / 193.2 | 7662 / 4230 / 239.8 | 4840 / 3246 / 251.5 | 110 / 56 |

* The open-loop protocols are chaotic after the fall, so the flicker counts carry noise of roughly ±20 %
  (**estimated** from the spread between the two feet).
* Impulses are identical across settings, as momentum requires. The settings differ only in how the impulse is
  distributed in time.

### A(4) Hardware sanity bound (literature; the numbers for the robot are estimates)

* **Humans, force plates at ~1 kHz:**
  * Stiff landings from 0.59 m: ≈ 3.3 body weights (BW); soft ≈ 2.3 BW (DeVita & Skelly 1992, MSSE 24(1):108,
    doi:10.1249/00005768-199201000-00018).
  * Barefoot drops: heel peak 4.1 / 2.8 BW from 0.6 m and 5.7 / 3.8 BW from 0.9 m, gymnasts / recreational athletes,
    sampled at 960 Hz (Seegmiller & McCaw 2003, [PMC314389](https://pmc.ncbi.nlm.nih.gov/articles/PMC314389/)).
  * 11 BW from 1.28 m (McNitt-Gray 1993, J Biomech 26:1037).
  * A meta-analysis of 26 studies finds lower peaks when plates sample below 1 kHz (Niu et al. 2014, doi:10.1155/2014/126860).
  * Human pulses last 20–50 ms because of soft tissue.
* **Robots:**
  * A 65 kg biped walking with stiff feet: 3.85 BW footfall (Honda, US5455497).
  * Humanoid falls without padding reach >100 g at the torso (Kajita et al., Humanoids 2016).
  * A 5 kg rigid impactor dropped from 1 m reads 86.8 kN at 25 kHz on a bare surface
    ([arXiv 2601.02857](https://arxiv.org/html/2601.02857)).
  * No published force-plate measurement of a G1-size humanoid dropped on stiff legs was found.
* **Bound for this robot (estimated;** M = 32.2 kg, weight 316 N, touchdown ≈ 2.1 m/s, momentum ≈ 67 N s):
  * At 1 kHz the landing peak is plausibly 2–7 kN (6–22 BW).
  * Averaged over 5 ms: 1.5–4.5 kN. Averaged over 20 ms: 1.5–3.5 kN, with a momentum ceiling ≈ 3.7 kN.
  * The torso impact after a tip-over is ≤ ~16 kN over 5 ms and 4–5 kN over 20 ms.
* **Which engine is closer to reality:**
  * Both engines' 20 ms-mean landing forces (PhysX 1996 N = 6.3 BW; MuJoCo default 2421 N = 7.7 BW) sit inside the
    plausible band.
  * Isaac's reported **1030 N (3.3 BW) is below what a stiff-legged 2 m/s landing requires**: humans reach 3.3 BW from
    0.59 m with soft tissue. It is below its own momentum-derived 1996 N, which is the reporting artefact again.
  * On the evidence, neither engine is shown to be closer to hardware. The reported PhysX peak understates PhysX's own
    physics.

## B. Feet air time and slide

### B(2) Term semantics against Isaac Lab v2.3.2 (primary source) and our kernel

* `feet_air_time_positive_biped` ([rewards.py L49–68](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py)):
  * `in_contact = current_contact_time > 0`.
  * in_mode_time = contact time if in contact, else air time.
  * Single stance means exactly one foot in contact.
  * reward = min over feet of (single_stance ? in_mode_time : 0), clamped at `threshold` (0.4 in the G1 configs;
    weight 0.75 flat, 0.25 rough).
  * The reward is zeroed when ‖command[:, :2]‖ ≤ 0.1.
  * The times come from `ContactSensor` with `is_contact = ‖net_forces_w‖ > force_threshold` (1.0 N), updated every
    physics step (5 ms) with elapsed_time = 5 ms.
* `feet_slide`: ‖`body_lin_vel_w`[foot].xy‖ summed over feet whose `net_forces_w_history` max norm exceeds 1 N.
  * The history has 3 entries of 5 ms, i.e. physics steps 2–4 of the control step.
  * In v2.3.2 `body_lin_vel_w` is the **COM** velocity ("Same as body_com_lin_vel_w", articulation_data.py L1071).
* Ours (`metalsim/learn/g1_velocity.py` reward kernel with `use_sensor=1`; `metalsim/sensors/contact.py`):
  * Same formula, clamp, command gate (‖cmd_xy‖), 1 N threshold and Isaac's timer update rule, applied every 2.5 ms.
  * Slide window: 6 × 2.5 ms = 15 ms = substeps 3–8, the same span as Isaac's steps 2–4.
  * **One difference found:** our slide uses the foot body's **frame-origin** velocity, Isaac uses its **COM** velocity.
    `air_time_rollout.py` measures both (below).

### B(3) Detection threshold, history and a foot-tap (measured, MuJoCo C, `foot_tap.py`, `runs/contact_research/foot_tap.md`)

Setup: the G1 foot box (0.2031 × 0.0655 × 0.0185 m, 3 kg), touch-down at 0.5 m/s, 250 N stance load, heel-off roll onto
the toe edge, lift.

* **Detection latency is 0 for every setting.** The 1 N flag switches on the first substep with geometric contact and
  off at geometric separation, so the threshold never delays or masks a MuJoCo contact.
* On PhysX's 5 ms averages the flag is at most one 5 ms step coarser.
* At lift-off the slide window stays set for 15 ms in both conventions (12.5 ms for Isaac-style averaging in two settings).
* **Chatter is the difference.**
  * Default, τ10 impact-only and τ10 ζ = 2: 4 flag transitions (touch-down, a single one-substep release during the
    roll, lift).
  * τ5 imp 0.99 and the Isaac Lab 3.0 mapping: 8 transitions. Examples: a deadbeat touch-down (2036 N, then one
    substep in the air, then back); repeated 2.5 ms separations on the rolling toe edge.
* The same holds in the G1 protocols: τ5 doubles the 2.5 ms flag transitions of A_hold + C_drop (212 vs 104).
  * Most of those are 1–3 substep gaps. Averaging over 5 ms like PhysX's report removes 50–80 % of them
    (212 → 42, 104 → 52).
  * Each gap resets Isaac's `current_contact_time` or `current_air_time`, which cuts the in-mode time that
    feet_air_time rewards.
  * PhysX's rigid, restitution-0 contacts cannot bounce by construction, and their flag is a 5 ms impulse average.
* **Reading:** stiff MuJoCo contact settings lower the air-time term through chatter. That is a contact-model effect,
  and it explains why the stiffer presets moved air time *away* from Isaac's (0.022 → 0.016).

### B(1) Isaac's checkpoints in our simulator (measured, GPU)

Isaac's rsl_rl checkpoints were converted to our joint order (`isaac_ckpt_to_metalsim.py`; the actions match the
`side_by_side.py` mapping to 1.4e-6). They were played with `air_time_rollout.py`: 1024 envs × 1000 control steps, the task's own
random commands and resets, mean action unless noted, seed 1 unless noted. The rollout uses the same per-term accounting as
`g1_reward_terms.py`. feet_slide is given both ways: with the foot frame-origin velocity (the task kernel) and with the foot COM
velocity (Isaac Lab 2.3.2).
* Runs: `run_air_time_batch.sh`, logs in `runs/contact_research/air_time/`, table `runs/contact_research/air_time_table.md`.
* Rows marked † are the contact-fidelity agent's rollout of the flat-config iteration-400 policy that the tuning table used
  (`runs/contact_research/flatcfg400/`, same script).
* The first attempt of these runs died of a Metal out-of-memory error while a training job was releasing the GPU
  (`air_time/failed_oom/`) and was re-run.

| policy (trained in) | contact | actions | feet_air_time | feet_slide origin / COM | Isaac log at that iteration (air / slide) | falls / episodes | median air phase | air / contact phases < 20 ms |
|---|---|---|---|---|---|---|---|---|
| Isaac it 1000 (PhysX) | default | mean | 0.0293 (seed 2: 0.0300) | −0.0211 / −0.0201 | 0.0446 / −0.0127 | 310 / 1244 | 105 ms | 17 % / 17 % |
| Isaac it 1000 | τ10 impact-only + hard limits | mean | **0.0368** | −0.0161 / **−0.0144** | 0.0446 / −0.0127 | 142 / 1122 | 110 ms | 32 % / 35 % |
| Isaac it 1000 | τ5 imp 0.99 + hard limits | mean | 0.0339 | −0.0131 / −0.0113 | 0.0446 / −0.0127 | 125 / 1103 | **12 ms** | 51 % / 57 % |
| Isaac it 1000 | default | stochastic | 0.0226 | −0.0222 / −0.0206 | 0.0446 / −0.0127 | 610 / 1463 | 95 ms | 26 % / 23 % |
| Isaac it 1000 | τ10 impact-only | stochastic | 0.0302 | −0.0200 / −0.0176 | 0.0446 / −0.0127 | 289 / 1214 | 92 ms | 36 % / 38 % |
| Isaac it 500 (PhysX) | default | mean | 0.0218 | −0.0173 / −0.0163 | 0.0364 / −0.0148 | 508 / 1408 | 97 ms | 16 % / 16 % |
| ours it 1000 (`g1_flat_rslrl`, MuJoCo Warp) | default | mean | **0.0479** | −0.0123 / −0.0118 | 0.0446 / −0.0127 | 37 / 1050 | 177 ms | 13 % / 13 % |
| ours it 1000 | τ10 impact-only | mean | 0.0465 | −0.0089 / −0.0082 | 0.0446 / −0.0127 | 47 / 1053 | 172 ms | 25 % / 27 % |
| ours it 400 (`g1_flat_rslrl`) | default | mean | 0.0388 | −0.0234 / −0.0215 | 0.0349 / −0.0175 | 27 / 1041 | 155 ms | 19 % / 20 % |
| ours it 400 | τ10 impact-only | mean | 0.0367 | −0.0176 / −0.0154 | 0.0349 / −0.0175 | 51 / 1062 | 142 ms | 36 % / 42 % |
| ours it 400, flat config† (the tuning table's policy) | default (seeds 1/2/3) | mean | 0.0217 / 0.0220 / 0.0218 | −0.0480 / −0.0444 | 0.0349 / −0.0175 | 31 / 1041 | 72 ms | 9 % / 8 % |
| same† | τ10 impact-only (seeds 1/2/3) | mean | 0.0202 / 0.0200 / 0.0192 | −0.0355 / −0.0321 | | 53 / 1054 | 65 ms | 38 % / 42 % |
| same† | τ5 imp 0.99 (seeds 1/2) | mean | 0.0157 / 0.0151 | −0.0258 / −0.0222 | | 125 / 1098 | 10 ms | 54 % / 65 % |

Reading:
1. **Air time is mainly a property of the policy.**
   * With the default contact model fixed, five policies give 0.022–0.048.
   * The seed-to-seed spread is ±0.0005 (three seeds of the flat-config policy, two of Isaac it 1000).
   * The tuning table's gap (0.016–0.022 vs 0.035) came from one early policy whose gait takes short steps: median air phase
     72 ms, against 105–177 ms for the others.
   * Our other iteration-1000 policy (`g1_flat_rslrl`) exceeds Isaac's logged air time (0.048 vs 0.0446) and matches its slide
     (−0.0118 vs −0.0127, COM convention).
2. **Isaac's policy does not reach its own log value in our simulator.** It gets 0.029–0.037 with mean actions, and 0.023–0.030
   with the stochastic actions its log was made with. Two things make up that gap:
   * **Transfer falls, a physics effect.** 142–610 falls in 1103–1463 episodes, where Isaac's training at iteration 1000 had a
     mean episode length of 991 of 1000 steps. The contact model changes this: impact-only stiffening halves the falls
     (310 → 142), raises air time 26 % (0.029 → 0.037) and brings slide from −0.020 to −0.014 (COM convention). That is
     Isaac's −0.0127 within 13 %, and closer than any other setting except τ5.
   * **The log's own conditions.** Isaac's log comes from training: stochastic actions under training randomization, as
     configured in Isaac's velocity env cfg. Those cannot be reproduced in a mean-action playback. Its value is therefore an
     upper reference, not a like-for-like target.
3. **Stiff contacts cut air phases into pieces.**
   * Under τ5 the median air phase falls to 10–12 ms and more than half of all phases last under 20 ms. That is the chatter of
     B(3), now measured in closed loop.
   * Air time still stays at 0.034 for Isaac's policy, because the reward takes the minimum over feet only during single
     stance, but the timers are corrupted.
   * Impact-only stiffening also raises short phases, from 13–19 % to 25–42 %.
4. **Slide convention.** Isaac's COM-velocity convention gives 5–15 % less slide than our frame-origin convention on every
   run. It is a small systematic bias in our kernel (`metalsim/learn/g1_velocity.py`, `g1_foot_vel_mjwarp` uses `xpos`
   rather than `xipos`); fixing it is for the task owner, since the file is not mine to edit.

### B: Isaac Lab 3.0's own MuJoCo Warp training (measured by NVIDIA's pipeline, L4, `runs/parity3/isaac/train/summary_flat_newton_mjwarp.json`, commit c6d17fd)

| iteration | PhysX air / slide (Isaac Lab 2.3.2) | Newton-MuJoCo Warp air / slide (Isaac Lab 3.0, solref ≈ (5 ms, 1.375)) |
|---|---|---|
| 300 | 0.0283 / −0.0224 | 0.0150 / −0.0334 |
| 500 | 0.0364 / −0.0148 | 0.0262 / −0.0198 |
| 1000 | 0.0446 / −0.0127 | 0.0423 / −0.0119 |
| 1499 | 0.0459 / −0.0126 | 0.0490 / −0.0106 |

* Both are training-log values: stochastic actions, training randomization.
* With NVIDIA's own MuJoCo Warp settings the same task reaches PhysX's air time and slide by iteration 1000 and
  exceeds them by 1499.
* Early in training it lags: at iteration ~400 it is ≈ 0.02, where our tuning table measured 0.016–0.022.
* The tuning table's comparison was therefore made at the training stage where MuJoCo Warp–trained policies lag.
  It also compared a deterministic playback with a stochastic training log.

## Ranked remedies (expected effects are estimates unless marked measured)

| # | remedy | discrepancy | expected effect | cost |
|---|---|---|---|---|
| 1 | Compare impacts by **impulse and momentum-derived 20 ms force** (both engines, from recorded states), not by the control-step sensor sample. Report per-body peaks only with the event named. | A | Closes the "3–5×" to the measured 1–4 % (impulse) and +18–21 % (20 ms peak). No physics change. | A column in `compare.py` (≈40 lines, the `momentum_impulse.py` logic); zero runtime cost |
| 2 | Compare air time / slide on **several policies at matched iteration and action mode**, and always include Isaac's checkpoint played in our sim (`air_time_rollout.py`). Do not rank contact presets on one policy's gait terms. | B | The policy spread (0.022–0.048 at fixed contacts, measured) is 10× the preset effect on the table's policy (0.016–0.022) | Existing scripts, ~75 s per rollout |
| 2b | feet_slide on the foot **COM velocity** (`xipos` instead of `xpos` in `g1_foot_vel_mjwarp`), as Isaac Lab 2.3.2's `body_lin_vel_w` | B | −5 to −15 % slide magnitude (measured on 13 runs) | One line in `metalsim/learn/g1_velocity.py` (task owner) |
| 3 | Keep **τ10 impact-only + hard limits** (the provisional pick) on the evidence of Isaac's own policy: fewest transfer falls besides τ5 (142 vs 310), air time 0.037 and slide −0.014 (COM) against Isaac's −0.013. Do not adopt τ5 / 0.99 impedance or the Isaac Lab 3.0 mapping. | A, B | τ5 and the mapping exceed PhysX's own 5 ms upper bound at torso impacts, and give a 10–12 ms median air phase from chatter (measured) | none |
| 4 | Train to ≥ 1000 iterations before comparing gait terms | B | Measured: ours it 1000 at 0.048 / −0.012; Isaac Lab 3.0's MuJoCo Warp run at 0.042 / −0.012 | Training time |
| 5 | Reduce touch-down chatter (and with it transfer falls, estimated) with **direct-form damping below deadbeat** (−k, −b with b·h ≈ 0.4–0.7 at the same k) on foot geoms with `priority` 1 | B (and A if stiff contacts are wanted) | Fewer 2.5 ms gaps and ≈ 30–60 % lower first-step spike at equal resting depth (estimate from the force law) | Preset plus a GPU check; no throughput change expected |
| 6 | Per-geom torso-only softening (priority + solref) | A (torso impact only) | Lowers the torso's 5 ms peak toward the lower half of PhysX's bounds. Not needed for fidelity. | Preset; none |
| 7 | A PhysX-style sensor window (mean of the last 5 ms) | A | **None measured**: 4080 vs 3715 N for default; the sub-step time profile differs, not the window | Not recommended |
| 8 | `integrator="discrete"` (MuJoCo 3.13) | A, B | Stiff contacts without the deadbeat spike (per the docs) | Not available in MuJoCo Warp; would need porting |
| – | archived, not recommended: o_solref/o_solimp override (unsupported in MuJoCo Warp); ζ > 1 at the τ floor (does not change b); margin (lifts the robot) | | | |

## Addendum: GPU checks

**MuJoCo Warp confirmation (measured, GPU queue, `substep_forces.py --engine warp`, `runs/contact_research/substep_warp_{default,tau10_impact_hardlimits}.npz`, log `substep_warp.log`).**
Warp's per-substep recordings match MuJoCo C to within 0.3 % on every quantity used above:
* default: identical, 5 ms / 20 ms peak and impulse (2961 / 2421 N / 193.1 N s landing; 4404 / 2602 N / 237.9 N s drop
  torso; 4364 / 2581 N / 250.9 N s hold torso);
* τ10 impact-only: drop torso 5 ms peak 7880 vs 7862 N, 20 ms peak 3727 vs 3716 N; landing and hold identical.

The A(1) conclusions therefore hold for MuJoCo Warp on Metal. B(1) above ran on MuJoCo Warp on the GPU.
