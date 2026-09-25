# MuJoCo (Warp) contacts vs PhysX for the G1: what the docs and published configs say, and candidate settings

2026-09-25. Scope: close MetalSim's remaining contact-fidelity gaps to Isaac Lab / PhysX on the G1 velocity task
(MuJoCo Warp is the parity engine). Gaps as measured before this note: impact peaks 3–5× Isaac's at equal mean
contact force (PARITY §1.7), feet_slide −0.059 vs Isaac −0.017 and feet_air_time 0.018 vs 0.035 on the rsl_rl
flat-config policy (learner agent, it 399), joint limits soft (MuJoCo default solref), and the plane-convex
contact-set decision (fork commit 284dcd1; MuJoCo C's contact set moved checkpoint transfer from within 5 % of Isaac
to 9–12 % short). Everything in the "expected effect" column is **estimated**; measurements follow in the
metric table referenced at the end.

Sources: [MX] https://mujoco.readthedocs.io/en/latest/XMLreference.html, [MM] https://mujoco.readthedocs.io/en/latest/modeling.html,
[MC] https://mujoco.readthedocs.io/en/latest/computation/index.html, [MJW] https://github.com/google-deepmind/mujoco_warp (README),
[PG] https://github.com/google-deepmind/mujoco_playground and arXiv 2502.08844, [MJL] https://github.com/mujocolab/mjlab,
[IL] https://github.com/isaac-sim/IsaacLab (isaaclab_assets/robots/unitree.py, isaaclab/sim/simulation_cfg.py,
velocity_env_cfg.py; develop: core/velocity/velocity_env_cfg.py and docs/source/concepts/solver-tuning/tune_mjwarp.rst),
[PX] https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html and .../RigidBodyDynamics.html.

## 1. MuJoCo's contact model (docs)

* **solref** (default 0.02, 1): positive = (time constant, damping ratio), negative = (−stiffness, −damping); the time
  constant "should be at least two times larger than the simulation time step", enforced unless `refsafe` is off [MM].
  Resting penetration r = a_u(1−d)·timeconst²·dampratio² [MM]. At our 2.5 ms step the floor is 5 ms.
* **solimp** (0.9, 0.95, 0.001, 0.5, 2): impedance d ∈ [0.0001, 0.9999] ramps from d0 to d_width over `width` of
  penetration; for pyramidal cones it applies to all directions [MM].
* **cone**: pyramidal is the default; "Elliptic cones are a better model of the physical reality, but pyramidal cones
  sometimes make the solver faster and more robust" [MX]. "When contact slip is a problem, the best way to suppress it is
  to use elliptic cones, large impratio, and the Newton algorithm with very small tolerance. If that is not sufficient,
  enable the Noslip solver" [MM]; the soft model "does not guarantee an exact zero-velocity stick state" [MM].
* **impratio** (1): frictional-to-normal impedance ratio, for elliptic cones; "not recommended to use high impratio values
  with pyramidal cones" [MX].
* **noslip_iterations**: post-processing PGS on friction; **not supported by MuJoCo Warp** [MJW] (put_model raises
  NotImplementedError in our fork too).
* **margin / gap**: a pair's values are the sums of the two geoms' (C engine_collision_driver.c:166/175, MuJoCo Warp
  collision_core.contact_margin_gap); contacts with dist < margin are active and the impedance acts on dist − margin;
  margin < dist ≤ margin+gap are inactive [MC]. Measured here: margin 1 cm makes the G1 stand 1 cm higher (pelvis
  0.7191 vs 0.7090 m) with or without gap. MuJoCo has no analogue of PhysX's speculative contact.
* **multiccd / nativeccd** (default on since 3.8): convex-convex multi-contact; MuJoCo Warp rejects non-zero margins on
  mesh-mesh pairs with multiccd.

## 2. Published MuJoCo humanoid locomotion configs

* **MuJoCo Playground G1** [PG]: 2 ms step, iterations 3, ls 5, Euler, default solref/solimp, pyramidal, impratio 1;
  box feet, only feet–floor contacts (condim 3, friction 0.6, randomized U(0.4, 1.0)). H1/T1/Berkeley similar
  (1–3 iterations; capsule/sphere feet). No stated contact-parameter rationale.
* **mjlab G1 velocity** [MJL]: 5 ms step, decimation 4, iterations 10, ls 20, implicitfast, pyramidal, impratio 1, default
  solref/solimp, 7 capsules per foot (r 0.01), friction 0.6 randomized 0.3–1.2. (Our budget follows this.)
* **Isaac Lab's own MuJoCo Warp config** (develop) [IL]: 2 substeps of 5 ms (2.5 ms), pyramidal, impratio 1,
  `NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)`; ke/kd reach MuJoCo through Newton's solref mapping, which we
  could not verify (our direct-form reading keeps the G1 standing after the drop: not the mapping).
* **Humanoid-Gym sim2sim** (arXiv 2404.05695): MuJoCo 1 ms, PGS 50 iterations, condim 4, solref (0.001, 2), friction 0.9.

Reading: the published MuJoCo humanoid configs all use default, soft contacts (τ 20 ms) and pyramidal cones and
randomize friction; none aims at PhysX parity of contact forces.

## 3. PhysX / Isaac Lab settings for this task

* G1_CFG [IL]: TGS; 8 position / 4 velocity iterations per articulation; `max_depenetration_velocity=1.0`; self
  collisions off. With TGS "the number of substeps is equal to the number of position iterations" [PX]: 5 ms → 0.625 ms
  sub-iterations, friction applied every iteration.
* Velocity env [IL]: dt 5 ms, decimation 4; ground multiply 1.0/1.0, robot 0.8 static / 0.6 dynamic, restitution 0 →
  effective ≈ 0.8/0.6. PhysxCfg: bounce threshold 0.5 m/s, friction offset threshold 0.04, friction correlation distance
  0.025, stabilization off, GPU contact buffers 2^23 contacts.
* Offsets [PX]: contactOffset default 0.02 m (contact distance 4 cm for a pair), restOffset 0 for rigid bodies (the
  values authored in g1_minimal.usd were not verified): speculative contacts limit approach velocity before touching,
  rest at zero distance; depenetration capped at 1 m/s. Physically: near-rigid, ~zero penetration, impacts resolved over
  several position iterations and capped by the depenetration velocity, friction per patch (2 anchors), isotropic-like
  combined friction (closer to an elliptic cone than MuJoCo's pyramid).

## 4. Published comparisons (numbers)

* Erez, Tassa, Todorov, ICRA 2015 (https://roboti.us/lab/papers/ErezICRA15.pdf): MuJoCo fastest and most accurate on
  robotics-relevant constrained systems; pre-PhysX 5/TGS.
* PolySim (arXiv 2510.01708), G1 motion tracking: IsaacSim-only policy in MuJoCo 3.6 % success (MPJPE 80.6 mm); training
  across three simulators 56.4 % (65.6 mm).
* Quadruped loco-manipulation sim-to-sim (arXiv 2512.18938): MuJoCo shows foot "sliding artifacts" (pose RMSE 0.113/0.123)
  vs Isaac Gym yaw oscillation (0.144/0.231); impratio 100 removes the error but is called "unrealistically high".
* No published engine comparison of contact-force peaks or penetration with numbers was found (Collins et al. 2021,
  Kaup et al. 2024 are qualitative).

## 5. Candidate settings and expected effect (estimated)

All at the task's 2.5 ms step, 10 Newton / 20 LS iterations, fixed (C-exact) plane-convex collider unless stated.
Joint limits "hard" = limit solref (5 ms, 1), solimp (0.99, 0.999): chosen in MuJoCo C so the drop keeps excursions
< 0.01 rad (0.0034 rad measured; 0.999 impedance oscillates, 0.9999 diverges).

| candidate | impact peak (sampled at control steps) | penetration | feet_slide | feet_air_time | limits | transfer | cost |
|---|---|---|---|---|---|---|---|
| default | reference (3–5× Isaac) | ~3 cm | reference | reference | ~0.03 rad | reference | 1.0× |
| hardlimits | ≈ default | ≈ default | ≈ default | ≈ default | < 0.01 rad | small change | ≈ 1.0× |
| tau5_imp99 (+hardlimits) | lower at control-step sampling, higher per substep | < 1 cm | lower (stiffer normal → less creep) | slightly lower (touch-downs shorter) | < 0.01 rad | closer (stiffer support) | +20 % (unconverged worlds) |
| margin (= gap) 1 cm | lower | 0 (rests 1 cm high) | ≈ tau5 | ≈ tau5 | – | root-height offset | ≈ tau5 |
| elliptic (+hardlimits) | ≈ default | ≈ default | lower (docs: elliptic suppresses slip) | ≈ default | < 0.01 rad | unknown | +10–30 % |
| elliptic impratio 10 (+hardlimits) | ≈ default | ≈ default | lowest (stiffer friction) | ≈ default | < 0.01 rad | unknown | +10–30 % |
| tau5_imp99 + elliptic (+hardlimits) | as tau5 | < 1 cm | lowest | as tau5 | < 0.01 rad | unknown | +30 % |
| old (heuristic) collider, any of the above | – | – | higher (fewer contacts per foot) | – | – | measured closer on the default row | ≈ 1.0× |

Not offered: tau5 + elliptic impratio 10 (diverges in the MuJoCo C drop at 2.08 s); Isaac Lab's ke/kd read as a
direct-form solref (the G1 does not fall; not Newton's mapping); noslip (not in MuJoCo Warp).
