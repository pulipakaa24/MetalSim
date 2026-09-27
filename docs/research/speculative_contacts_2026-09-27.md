# Speculative contacts in MuJoCo Warp: the over-travel explained, the rule corrected, the prototype stays archived (2026-09-27)

Question (owner): explore the speculative-contact prototype of `penetration_2026-09-26.md` §5 to a decision. It cuts the
G1 landing penetration from 13.25 mm to 1.56 mm (PhysX 0.51 mm), but Isaac's PhysX-trained checkpoints walked 10–14 cm
further than in Isaac under it (transfer error 0.117 m vs 0.060 m for the default), and its drop-torso 5 ms force was 4 %
over PhysX's bound. Fidelity first; the references are PhysX and the real robot.

All numbers are **measured** on this machine (M4 Max; MuJoCo Warp fork f824af1 plus the prototype commits on branch
`metalsim-spec-proto`; every GPU job went through `scripts/gpu_run.sh` and imported a frozen per-job worktree:
`upstream/mujoco_warp-f824af1`, `-spec-d99186b`, `-spec-27f1fcd`, `-spec-e636feb`). Scope: one asset (Isaac's G1 minimal USD, mesh-box
feet), the PARITY §1.7 protocols, and Isaac's checkpoints 500 / 1000 / 1499.

## Verdict

1. **The over-travel is not caused by the speculative rule.** It is an existing cadence mismatch that the default contacts
   hide.
   - Isaac's own play recordings (`runs/parity/isaac/parity_out/play/isaac_it*/traj.npz`) give PhysX's gait: cadence
     3.846 / 3.597 / 3.231 Hz and stride 0.108 / 0.110 / 0.123 m.
   - In MuJoCo Warp the cadence is 3.4–5.9 % higher under **every** contact setting measured: default, τ 5 ms, and the
     speculative variants (standard errors 0.004–0.025 Hz).
   - Under the default, the stride is also 4.7–8.9 % too short. The two errors cancel in the one number the transfer
     test reports, x travelled.
   - Speculative rows bring the stride to PhysX's (final rule: −0.4 / +0.6 / +1.1 %). The excess cadence is then no
     longer cancelled, so the robot over-travels by exactly the cadence excess.
   - By the leg-joint gait distance to PhysX the speculative settings are the **closest** of all: 0.0088–0.0118 rad,
     against 0.0151–0.0170 rad for the default (§2).
2. **Hypotheses (i), (iii) and (iv) are refuted as the cause of the over-travel. (ii) is what changes the gait, and it is
   PhysX-like.** Evidence is in §2:
   - (iii) rows allocated but inert reproduce the base preset to 0.001–0.004 m;
   - (iv) the policy observes no contact quantity;
   - (i) removing the friction coupling makes the robot travel **further**, not less.
3. **The investigation found two real defects in the prototype, and both are fixed in the fork.** A third, minor one
   (rows for contacts at exactly dist == margin) was found by the fork suite and fixed in e636feb (§6).
   - (a) *Friction coupling (MuJoCo's soft elliptic cone).* A speculative row pushes and brakes a body that is only sliding
     past inside the gap, although its normal row alone would not act. A box skimming 2 mm above the plane at 1 m/s
     with no gravity gets 119 N, is lifted 9.6 mm and slowed to 0.90 m/s. MuJoCo C, stock MuJoCo Warp and PhysX give
     0 N. PhysX bounds friction by the contact's accumulated normal impulse, so a speculative contact that does not push
     has no friction (`DyTGSContactPrep.cpp:1643–1654`, `solverBlockTGS.cuh:337–338, 421–423`).
     Fix: `MJW_SPEC_FRICTION=live`. A speculative contact keeps its friction rows only when its normal row predicts a
     crossing within the step (pos + h v_n < 0); otherwise the friction rows are inert (J = 0).
   - (b) *Inconsistent cone impedance.* The prototype's friction rows kept imp(|pos|), which is dmax beyond `width`,
     while their normal row had d0. That makes the tangential regularisation 1110× stiffer than MuJoCo's cone
     construction. Against a MuJoCo C reference of the rule it diverges at 1 s (5e-3 rad) where the stock floor is
     4e-6. Fix: `MJW_SPEC_FRICIMP=d0`, which gives the friction rows the normal row's impedance. With it the rule tracks
     the C reference at the stock floor through 2 s.
4. **With both defects fixed ("final rule"), the protocol still does not support adoption** (§3):
   - What it fixes: the drop-torso 5 ms force is now within PhysX's bound (4976–5020 N vs ≤ 7854 N; prototype 8168 N).
     The landing and recovery are unchanged (1.56 mm, 0.08 mm at 55 ms).
   - Losses on τ 5 ms:
     - the slide match: −0.0111 vs Isaac −0.0127 (13 % low, 50× the seed noise; the prototype's −0.0137 came partly from
       defect (a));
     - closed-loop short contact phases come back to 16 % (prototype 0.4 %, default 36 %). The prototype's chatter
       reduction was partly the defect keeping the foot flag on while the foot hovered;
     - the hold-torso 20 ms force is 1.58×. This is a phase artefact of the 20 ms grid: the phase-robust 40 ms force is
       1.10×, the 5 ms force is within bounds, and the impulse ratio is 0.989;
     - the cap-probe floor rises 10–20× over the default's. The gate is a discontinuous switch, so float noise between
       two identical runs flips it;
     - the physics cost is 1.20×, because the gate kernel is unoptimised;
     - the transfer error (4-env protocol) is 0.167 m, because the stride is now right and the cadence is not.
   - On the τ 10 ms base (the default plus the gap and the final rule): foot peak 5.46 mm, slide −0.0136, drop-torso 5 ms
     force 7071 N within the bound. But the hold-torso 20 ms force is 1.78× (40 ms: 1.13×), short phases are 15.5 %, and
     the transfer error is 0.087 m.
5. **Recommendation: keep `recommended` as the default. The speculative rule stays archived** on
   `metalsim-spec-proto` (head e636feb), now with the corrections and a C reference. **No training run was queued.** The owner's
   condition was "over-travel explained **and fixed**, force bound holds". The over-travel is explained, but it cannot
   be fixed inside the contact rule: its cause is the cadence. The corrected rule also loses the slide match.
   What remains (§5):
   - the +4–6 % cadence gap to PhysX, which is present with every contact model and is not a contact property;
   - the lost slide match under the corrected τ 5 ms rule;
   - the gate's discontinuity (floor and cost).

## 1. What was built

- **Probe** (`scripts/diagnostics/speculative/overtravel_probe.py`). The transfer protocol of `newton_transfer.py`:
  Isaac's default state, command 0.5 m/s, mean action, 400 control steps, observation noise on, task seed 0, at 4 envs
  (the protocol) and at 64 envs (standard errors). A substep hook logs, per env and per foot:
  - the normal force and |friction| of all foot-ground contacts, and of the speculative ones (pos > 0);
  - the force of speculative contacts that do **not** predict a crossing (pos + h v_n ≥ 0), i.e. force PhysX's rule
    would not apply;
  - contact counts, world-x friction, nefc and Newton iterations.

  Per control step it also logs actions, root, joint positions (Isaac order), foot COM speed and height, and the task
  ContactSensor's forces and flags. Tables: `probe_table.py`, `gait_vs_isaac.py`.
- **Fork switches** (prototype only, active with `MJW_SPECULATIVE_GAP=1`, elliptic cones):
  - `MJW_SPEC_FRICTION=cone|none|live` (d99186b): a gate kernel after `_efc_contact_update` zeroes J, vel and aref of the
    selected rows, so they sit in the cone's top zone or add nothing;
  - `MJW_SPEC_NULL=1`: every row of a speculative contact is inert (the control for hypothesis (iii));
  - `MJW_SPEC_FRICIMP=d0` (27f1fcd).

  The defaults (`cone`, `pos`) reproduce bc83fb0.
- **MuJoCo C reference of the rule** (`c_oracle.py`, float64). It runs MuJoCo C's own pipeline on a model copy whose
  gap geoms carry margin = gap, so that C creates the rows. After `mj_fwdVelocity` every contact row is
  re-parameterised from the true distance: MuJoCo's imp / K / B / R for pos ≤ 0; the speculative a_ref and d0 for
  pos > 0; the friction gate. Then C's `mj_fwdConstraint` (Newton, ls_iterations 50) and the model's integrator run.
  Two details matter:
  - MuJoCo C's island solver copies efc_D / efc_R into per-island arrays during `mj_fwdPosition`, before the rows can
    be patched. The reference runs with `mjDSBL_ISLAND`. With islands on, C silently solved the unpatched R. An early
    run looked like "MuJoCo Warp beats C"; an independent L-BFGS minimiser located the bug.
  - Self-test: with the gap removed the reference equals `mj_step` to 5e-14 over a 600-substep G1 drop.
- **Analytic box checks** (`box_checks.py`, CPU device). The box is 0.2 × 0.06 × 0.02 m, 3 kg, elliptic impratio 10,
  τ 5 ms, gap 10 mm. Cases: flat drop; sliding landing; skim (gravity off, 2 mm above the plane, 1 m/s); edge landing
  (10° tilt).

## 2. Task 1: the over-travel, hypothesis by hypothesis

Transfer at 64 envs (x travelled in m, SE 0.005–0.006). Isaac's play recordings give 3.205 / 3.036 / 3.099; the protocol
quotes 3.19 / 3.02 / 3.11.

| setting | it 500 | it 1000 | it 1499 | cadence it 1000 [Hz] | stride it 1000 [m] | leg-gait distance to PhysX (500 / 1000 / 1499) [rad] |
|---|---|---|---|---|---|---|
| Isaac PhysX (4 envs) | 3.205 | 3.036 | 3.099 | 3.597 ± 0.009 | 0.1096 | 0 |
| recommended (τ 10 ms impact) | 3.181 | 2.911 | 3.006 | 3.791 ± 0.011 | 0.0998 | 0.0153 / 0.0170 / 0.0151 |
| ellip10_tau5 (stock) | 3.233 | 3.005 | 3.105 | 3.725 | 0.105 | 0.0134 / 0.0131 / 0.0118 |
| spec τ 5 ms, NULL rows | 3.229 | 3.009 | 3.106 | 3.738 | 0.104 | 0.0134 / 0.0131 / 0.0117 |
| spec τ 5 ms, prototype (cone) | 3.285 | 3.088 | 3.181 | 3.756 ± 0.006 | 0.1068 | 0.0116 / 0.0104 / 0.0094 |
| spec τ 5 ms, friction none | 3.318 | 3.146 | 3.229 | 3.733 | 0.109 | 0.0120 / 0.0100 / 0.0091 |
| spec τ 5 ms, friction live | 3.323 | 3.154 | 3.231 | 3.718 ± 0.008 | 0.1103 | 0.0118 / 0.0098 / 0.0088 |
| spec τ 5 ms, final (live + d0) | 3.326 | 3.155 | 3.230 | 3.725 | 0.110 | 0.0118 / 0.0098 / 0.0088 |
| spec τ 10 ms (default + gap), NULL rows | 3.182 | 2.911 | 3.006 | – | – | – |
| spec τ 10 ms, cone / final | 3.256 / 3.267 | 3.020 / 3.037 | 3.111 / 3.128 | 3.780 / 3.771 | 0.104 / 0.105 | 0.0127 / 0.0130 / 0.0116 (cone) |

Cadence and stride for all three checkpoints (± SE over envs):

| | Isaac | recommended | prototype | live |
|---|---|---|---|---|
| it 500 cadence [Hz] / stride [m] | 3.846 / 0.1082 | 4.013 / 0.1031 | 4.036 / 0.1055 | 3.996 / 0.1078 |
| it 1000 | 3.597 / 0.1096 | 3.791 / 0.0998 | 3.756 / 0.1068 | 3.718 / 0.1103 |
| it 1499 | 3.231 / 0.1230 | 3.420 / 0.1138 | 3.364 / 0.1220 | 3.352 / 0.1244 |

Hypothesis by hypothesis:

- **(iii) Constraint count and order.** Rejected. With the rows allocated but inert (`MJW_SPEC_NULL=1`: +0.6 rows per
  foot, nefc 13.7 → 16.9), transfer equals the base preset to 0.001–0.004 m (τ 10 ms: 3.182 / 2.911 / 3.006 vs
  3.181 / 2.911 / 3.006). The mean-action difference is 0.001–0.002 against 0.06–0.21 for the active variants. Newton
  iterations are unchanged.
- **(iv) Contact-sensor readings.** Refuted by construction for transfer. The policy observation is base velocities,
  gravity, command, joint positions, joint velocities and the last action (`g1_velocity.py:671`,
  `obs_dim = 12 + 3·nj`). No contact quantity reaches the policy; the sensor feeds only rewards and terminations, and
  transfer has neither.

  The sensor readings do change. The foot flag fraction is 0.525 → 0.539 (prototype) → 0.530 (final), and the
  prototype's flags count the coupling force of (i) as contact while the foot hovers. That matters for training
  (air-time reward, chatter), not for transfer.
- **(i) Friction on speculative rows.** It is a real defect, but it does not cause the over-travel.
  - Mechanism: MuJoCo's elliptic cone has a middle zone. When the normal slack N = jar_n·μ is below μ·T (T = |friction
    jar|), the cone produces **normal and friction force from tangential slip alone**
    (`solver.py _eval_elliptic_middle`). A swing foot inside the gap moving at ~1 m/s has jar_t ≈ b·v_t ≈ 400 m/s².
    For pos ≲ 2.5 mm, jar_n = pos/h² + v_n/h is smaller than that, so the foot is braked and lifted.
  - On the G1 the no-cross normal force averages 3.0–3.7 N per foot (2 % of the foot load; 1.8–2.2 N in swing) and the
    no-cross friction 2.7–3.3 N.
  - PhysX (5.6.1): friction anchors exist for every contact of the patch (`DyTGSContactPrep.cpp:71–94`), but each
    friction impulse is clamped by μ × the patch's accumulated normal impulse, solved first (CPU TGS
    `DyTGSContactPrep.cpp:1643–1654, 1748–1750`; GPU `solverBlockTGS.cuh:337–338, 421–423`). So friction never creates
    a normal impulse, and a speculative contact without a normal impulse has no friction.
  - Removing the coupling (none / live) makes the robot travel **2.5–6 cm further** (3.154 vs 3.088 at it 1000). The
    stride gets even closer to PhysX's, and the cadence is unchanged.
- **(ii) The normal impulse profile.** This is what changes the gait, and it moves it toward PhysX. The landing force per
  5 ms window (C_drop, env 0, N):

  | window start [ms] | 235 | 240 | 245 | 250 | 255 | 270 | 295 | impulse 225–300 ms |
  |---|---|---|---|---|---|---|---|---|
  | Isaac 5.1 PhysX | **6497** | 1376 | 1472 | 1278 | 1134 | 903 | 659 | 92.0 N s |
  | recommended | 1385 | **5009** | 2790 | 1725 | 1228 | 872 | 706 | 93.0 |
  | ellip10_tau5 | 2803 | **5902** | 1088 | 971 | 701 | 834 | 662 | 92.4 |
  | prototype / final | **5511** | 2798 | 1176 | 1096 | 1078 | 925–941 | 669–670 | 92.0 |

  - The speculative rows move the impact into PhysX's window: 5.5 kN in the 235 ms window where PhysX has 6.5 kN, where
    the stock contacts peak one window late.
  - The remaining split, 5.5 + 2.8 kN against 6.5 + 1.4 kN, comes from the look-ahead. Ours looks 2.5 ms ahead and
    PhysX 5 ms. The speculative step therefore leaves the approach velocity −pos/h to the next step, and the foot's
    second edge slaps down in the next window.
  - In walking, the rows catch each foot corner at the surface (no-cross forces excepted). The ankle-pitch action
    changes most (Isaac joints 15 / 16: 0.08–0.16 of mean action), and the stride lengthens to PhysX's.

**Conclusion.** The x-travel transfer error is a compensated metric. The default's 0.060 m is a too-short stride
cancelling a too-high cadence. The speculative rule fixes the stride and exposes the cadence. The cadence gap is present
under every contact model (3.72–3.79 Hz at it 1000 vs PhysX 3.60), so its cause lies outside the contact model: the
actuator / PD path, joint dynamics, or the control loop. It is not investigated here.

## 3. Task 2: the fix and the full protocol

The fix is what the explanation pointed at in the rule itself: defect (a), PhysX's friction semantics (`live`), and
defect (b), found by the C reference (`d0`). Open-loop protocol (PARITY §1.7, 4 envs, each run repeated; landing numbers
are deterministic across envs and repeats to 0.01 mm). Env ranges are in brackets. The repeat runs give the torso-impact
floor.

| setting | foot peak substep / 5 ms tick [mm] | torso peak drop [mm] | settle [mm] | 55 ms after peak [mm] | impulse ratio land / drop-torso / hold | 20 ms force ÷ PhysX land / drop-torso / hold | 40 ms force ÷ PhysX | chatter A+C 2.5 / 5 ms | limit exc. [rad] | max 5 ms force land / drop-torso / hold [N] (bounds 6522 / 7854 / 8045) |
|---|---|---|---|---|---|---|---|---|---|---|
| PhysX 5.1 | – / 0.51 | 0.10 | 0.013 | 0.08 | 1 | 1 | 1 | – | 0.004 | – |
| recommended (+ repeat) | 13.25 / 12.79 | 10.7–12.6 (8.9–16.5) | 0.25 | 1.53 | 0.994 / 0.969–0.970 / 0.988 | 1.35 / 1.04–1.12 / 1.07 | 1.11 / 1.06–1.09 / 1.07 | 120–122 / 58–60 | 0.0035 | 5009 / 4744–4862 / 4909 |
| ellip10_tau5 (+ repeat) | 8.97 / 8.23 | 5.1–5.2 | 0.064 | 0.10 | 0.994 / 0.971 / 0.988 | 1.08 / 1.21–1.22 / 1.15 | 1.00 / 1.10 / 1.06 | 116–118 / 42 | 0.0028 | 5902 / 10029–10043 / 9663 |
| prototype τ 5 ms, cone (+ repeat) | 1.56 / 1.56 | 0.33 | 0.064 | 0.08 | 0.993 / 0.977 / 0.989 | 0.76 / 0.98–0.99 / 1.11 | 0.91 / 0.98 / 1.04 | 12 / 12 | 0.0038–0.0039 | 5511 / **8168–8182** / 5838 |
| τ 5 ms, live (friction imp pos) (+ repeat) | 1.56 / 1.56 | 0.36 | 0.065 | 0.08 | 0.994 / 0.974 / 0.989 | 0.76 / 1.05 / 1.57 | 0.91 / 1.03 / 1.10 | 12 / 12 | 0.0031 | 5512 / 5143–5183 / 6579 |
| **τ 5 ms, final (live + d0)** (+ repeat) | **1.56 / 1.56** | 0.36 | 0.065 | 0.08 | 0.994 / 0.974 / 0.989 | 0.76 / 1.05 / **1.58** | 0.91 / 1.03 / 1.10 | 12 / 12 | 0.0032 | 5512 / **4976–5020** / 6587 |
| τ 10 ms prototype, cone | 5.46 / 5.46 | 0.93–1.05 | 0.25 | 0.18 | 0.994 / 0.974 / 0.988 | 1.02 / 1.63–1.67 / 1.10 | 0.94 / 1.11 / 1.03 | 12 / 12 | 0.0040 | 3824 / 8261 / 7892 |
| τ 10 ms final | 5.46 / 5.46 | 1.87 (1.87–9.35) | 0.24 | 0.18 | 0.994 / 0.972 / 0.987 | 1.02 / 1.01–1.23 / **1.78** | 0.94 / 1.02–1.05 / 1.13 | 12 / 8 | 0.0025 | 3824 / 7071 / 4472 |

**Hold-torso 20 ms force.** Per-20 ms-step torso forces in A_hold: Isaac 2171 / 2190 N; recommended 2331 / 2339; final
3454 / 1349. The final rule's torso impact lands earlier in the 20 ms grid, so the same impulse (ratio 0.989) splits
unevenly. The 40 ms force (1.10 vs 1.07) and the 5 ms force (within bounds) are the phase-robust measures, and they
agree with the default to 3 %.

Closed loop: Isaac's checkpoint 1000 in our sim, mean action, 1024 envs × 1000 steps, seeds 1 / 2. Isaac's log: air
0.0446, slide −0.0127. Noise floor: air ±0.0006, slide ±0.0002, short phases ±0.004, falls ±13.

| setting | air | slide (COM) | falls / episodes | contact / air phases < 20 ms | median contact phase | double stance / flight | dof_acc |
|---|---|---|---|---|---|---|---|
| recommended | 0.0373 / 0.0384 | −0.0130 / −0.0134 | 169 / 144 | 36.3 % / 32.0 % | 120 ms | 8.7 / 1.5 % | −0.0117 |
| ellip10_tau5 | 0.0390 | −0.0102 | 129 | 51 % | 17 ms | – | −0.0242 |
| prototype τ 5 ms, cone | 0.0390 / 0.0393 | −0.0136 / −0.0137 | 132 / 130 | 0.4 % / 0.7 % | 145 ms | 11.5 / 0.6 % | −0.0118 |
| τ 5 ms, live (imp pos) | 0.0409 / 0.0410 | −0.0112 / −0.0111 | 123 / 128 | 16 % / 15 % | 135 ms | 8.8 / 0.9 % | −0.0147 |
| **τ 5 ms, final** | 0.0407 / 0.0410 | **−0.0111 / −0.0111** | 132 / 129 | 16 % / 15.5 % | 135 ms | 8.8 / 0.9 % | −0.0146 |
| τ 10 ms prototype, cone | 0.0392 | −0.0152 | 139 | 2.9 % | 142 ms | 11.2 / 0.6 % | −0.0108 |
| τ 10 ms final | 0.0398 / 0.0400 | −0.0135 / −0.0137 | 139 / 135 | 15.7 % / 15.2 % | 135 ms | 9.5 / 0.8 % | −0.0117 |

Transfer by the 4-env protocol (`sweep.sh transfer`; Isaac 3.19 / 3.02 / 3.11; the 4-env spread is ±0.05 m and the
4-env mean sits 0.02–0.07 m above the 64-env mean in our sim):

| setting | x it 500 / 1000 / 1499 | error |
|---|---|---|
| recommended | 3.22 / 2.95 / 3.03 | 0.060 |
| prototype τ 5 ms | 3.32 / 3.16 / 3.19 | 0.117 |
| τ 5 ms final | 3.36 / 3.21 / 3.25 | 0.167 |
| τ 10 ms final | 3.32 / 3.10 / 3.16 | 0.087 |

Newton cap probe (4096 envs; random actions / the elliptic-trained policy). Worlds at cap 20 per step, cap 20 p99
|Δqvel| / worlds > 0.01, the floor (cap 100 vs cap 100, two instances), and mean / max iterations:

| setting | states | worlds at cap | cap 20 p99 / > 0.01 | floor p99 / > 0.01 | iterations mean / max |
|---|---|---|---|---|---|
| recommended | random / policy | 4.4 / 0.0 | 0.0016 / 6.1, 0.0026 / 14.4 | 0.0014 / 4.9, 0.0022 / 12.4 | 3.85 / 37, 3.62 / 17 |
| prototype | random / policy | 0 / 0 | 0.0090 / 12.6, 0.0078 / 13.7 | 0.0091 / 13.7, 0.0075 / 11.3 | 3.70 / 17, 3.40 / 13 |
| live (imp pos) | random / policy | 0.1 / 0 | 0.2035 / 63.4, 0.0232 / 55.8 | 0.1119 / 60.1, 0.0321 / 57.0 | 3.10 / 20, 3.31 / 14 |
| final | random / policy | 0 / 0 | 0.0786 / 57.4, 0.0272 / 54.5 | 0.0486 / 54.7, 0.0291 / 53.1 | 3.10 / 18, 3.31 / 14 |

Cap 20 sits at each setting's own floor. But the gated rules' floor is 10–20× the default's: identical runs diverge
faster, because the gate switches the friction rows on the sign of pos + h v_n, a discontinuity that float noise flips.

Physics-only cost (4096 envs, own caps, 3 interleaved repeats, timing class): recommended 46.70 ms; ellip10_tau5 45.71
(0.98×); final rule **55.99 (1.20×)**; live (imp pos) 56.02 (1.20×). The prototype measured 1.00× on 2026-09-26. The
gate kernel launches over naconmax × condim and zeroes a dense J row per inert row. It was not optimised, since the rule
is not adopted.

## 4. Task 3: correctness

**(a) Analytic box checks** (`runs/speculative/box/`). MuJoCo C and stock MuJoCo Warp agree to 4 digits, so they share
one row. "Final" is live + d0 in MuJoCo Warp. The C reference of the rule gives the identical numbers to 4 digits
(`oracle_live.jsonl`).

| case | quantity | analytic (rigid, restitution 0 = PhysX's model) | MuJoCo C / stock | prototype (cone) | friction none | final |
|---|---|---|---|---|---|---|
| A flat drop | first force at dist | – | −1.35 mm | +2.15 mm | +2.15 | +2.15 |
| | peak penetration | 0 | 1.35 mm | 0.087 | 0.087 | 0.087 |
| | impulse in 20 ms vs m\|v⁻\| + m g 20 ms | 4.709–4.782 N s | 4.847 (+1.3 %) | 4.715 (+0.13 %) | 4.715 | 4.715 |
| | v_z 20 ms after first force | 0 | +0.022 m/s | +0.002 | +0.002 | +0.002 |
| | rest penetration / force | – / 29.43 N | 0.0076 mm / 29.43 | 0.0076 / 29.43 | same | same |
| B sliding landing (1 m/s) | v_x 5 ms after first force (Coulomb: 1 − μ\|v_z⁻\|) | 0.36–0.39 | 0.109 | 0.606 | 0.471 | 0.208 |
| | stop time after impact | 37–39 ms | 42.5 ms | 45.0 | **127.5 (bounce: v_z +0.31 m/s)** | 42.5 |
| C skim, 2 mm, no gravity | max normal force / lift / v_x end | 0 / 0 / 1.0 | 0 / 0 / 1.0 | **118.8 N / 9.6 mm / 0.901** | 0 / 0 / 1.0 | 0 / 0 / 1.0 |
| D edge landing, 10° | peak penetration | 0 | 3.64 mm | 0.081 | 0.080 | 0.081 |
| | peak ω after the edge hit (rigid: 34.5 rad/s) | 34.5 | 40.7 | 31.9 | 38.9 | 31.9 |
| | ω / v_z 60 ms after | 0 / 0 | 0.225 rad/s / 0.0075 | 0.0007 / 0 | 0.0006 / 0 | 0.0006 / 0 |

- The rule has the same rest state as MuJoCo's own contact, exactly: the rows at pos ≤ 0 are MuJoCo's.
- Post-impact velocities agree within 0.02 m/s. The rule is closer to the rigid answer, because MuJoCo's contact is late
  and bounces slightly (C: +0.022 m/s, 0.23 rad/s after 60 ms).
- The impulse is m v⁻ to 0.13 %, the PhysX (rigid, restitution 0) value.
- No PhysX box-drop recording exists (`runs/parity*` holds only the G1 drop). The PhysX comparison is therefore the
  analytic rigid impulse and the G1 landing impulse: 92.0 N s vs PhysX 92.0 N s over 225–300 ms; impulse ratios
  0.993–0.994.
- "none" is rejected: the missing friction on the speculative step, followed by full cone coupling one step later,
  bounces a sliding box.

**(b) Inert at gap 0.** CPU device, G1, 2 worlds, drop plus random PD targets, 400 substeps (`cpu_traj.py`,
`runs/speculative/cpu/inactive.log`). Bitwise equal to the fork head f824af1 in qpos, qvel, nefc and Σ efc_force:
- `MJW_SPECULATIVE_GAP=1` with a gap-free preset (recommended), on d99186b;
- the same with `MJW_SPEC_FRICTION=live`, on d99186b and on 27f1fcd (`+ MJW_SPEC_FRICIMP=d0`);
- the gap preset with `MJW_SPECULATIVE_GAP=0`, on d99186b and 27f1fcd.

The rows of a contact with dist ≤ margin are untouched by construction: `write_contact` detects only dist < margin + gap,
and every change is gated on pos > 0.

**(c) Fork suite on Metal**: see §6. Without the switch the suite is at baseline. With the switch the only failures are
the two tests that assert stock MuJoCo's gap semantics, which the rule changes by design.

**(d) MuJoCo C reference checks** (`c_traj.py`, `c_oracle.py`). ls_iterations 50, elliptic impratio 10, Newton cap 20,
islands off in both the stock and the rule runs. Max |Δq| of MuJoCo Warp (CPU, float32) against C (float64), per world,
same protocol as (b):

| run | 0.25 s | 0.5 s | 1.0 s | 1.5 s | 2.0 s |
|---|---|---|---|---|---|
| stock ellip10_tau5 vs mj_step (the floor) | 6e-7 / 1e-6 | 1e-6 / 8e-7 | 3.6e-6 / 1.8e-6 | 2.7e-5 / 8.7e-6 | 1.7e-2 / 2.2e-2 (torso impact, chaotic) |
| recommended vs mj_step | 1e-6 / 3e-6 | 8e-7 / 1.7e-6 | 4.8e-6 / 3.1e-6 | – | – |
| prototype (cone, imp pos) vs C reference | 7e-7 / 6e-7 | 1.7e-6 / 4.3e-6 | **1.2e-3** / 1e-5 | – | – |
| live, imp pos vs C reference | 5.5e-7 / 7.3e-7 | 9e-7 / 1.2e-6 | **5.5e-3** / 1e-5 | 4.9e-2 / 1.1e-4 | 1.6e-1 / 2.6e-2 |
| cone, imp d0 vs C reference | 5.5e-7 / 6.8e-7 | 1.1e-6 / 1.7e-6 | 1.4e-5 / 3.0e-6 | – | – |
| **final (live, imp d0) vs C reference** | 4.5e-7 / 9.1e-7 | 8.1e-7 / 1.0e-6 | **4.2e-6 / 4.0e-6** | 1.7e-5 / 2.5e-5 | 1.6e-2 / 1.3e-2 |

- The final rule's MuJoCo Warp implementation matches its MuJoCo C reference at the stock floor through the landing and
  up to the chaotic torso impact.
- The prototype's friction impedance (imp(|pos|) against d0 on the normal) departs from the reference at the first
  foot-slap after the landing.

## 5. Task 4: decision and what remains

- **Default unchanged** (`recommended`). No training run was queued, per the owner's condition.
- The over-travel is explained. It is not fixable in the contact rule, because the cause is the +4–6 % cadence that
  every contact model shows against PhysX. The corrected rule also gives up the slide match (−0.0111 vs −0.0127) and
  brings back 16 % short contact phases.
- The corrected rule, the switches, the C reference and all tools are archived on the fork branch `metalsim-spec-proto`
  (e636feb; d99186b = gate and NULL switches, 27f1fcd = the friction impedance, e636feb = the dist == margin fix) and in `scripts/diagnostics/speculative/`.
- Before the rule can be reconsidered:
  1. Explain the cadence gap (it exists under the default too): actuator / PD implementation, armature, joint friction,
     action latency. Once the cadence matches, the x-travel test becomes a valid check of the stride, which the rule
     already matches.
  2. Recover the slide on the corrected rule. Candidates: τ 7.5 / 10 ms bases (τ 10 ms final: −0.0136), or friction
     bounded by the speculative normal force rather than switched.
  3. Replace the discontinuous gate by a continuous one to bring the floor back, and fuse it into the update kernel
     (cost).

## 6. Fork suite (task 3c)

Frozen worktree `upstream/mujoco_warp-spec-27f1fcd`, Warp `upstream/warp-innate` c200d46b, Metal, `pytest
mujoco_warp/_src` (`runs/speculative/suite.wrapper.log`):

| configuration | 27f1fcd (`suite.wrapper.log`) | e636feb (`suite2.wrapper.log`) |
|---|---|---|
| no switch | 1451 passed, 1 failed (pre-existing), 39 skipped | **1451 / 1 / 39** |
| `MJW_SPECULATIVE_GAP=1` | 1445 / 7 / 39 | **1449 / 3 / 39** |
| `+ MJW_SPEC_FRICTION=live MJW_SPEC_FRICIMP=d0` | 1445 / 7 / 39 | **1449 / 3 / 39** |

- On 27f1fcd, four of the six extra failures (`constraint_test::test_jdotv{2..5}`) exposed a third prototype bug. A
  contact at exactly dist == margin got rows: MuJoCo excludes it (C `exclude = 1`), and such contacts arrive through
  `put_data`, e.g. keyframes resting at dist 0. Fixed in e636feb (`active = active or pos > 0`).
- e636feb reproduces 27f1fcd bitwise on the G1 (CPU, final rule over 800 substeps, and gap 0), so every protocol number
  above holds for it.
- The two remaining failures under the switch, `constraint_test::test_efc_address_inactive_contacts` and
  `collision_driver_test::test_ccd_margin_dist` (nefc 4 != 0), assert that contacts inside the gap have **no** rows.
  Giving them rows is the rule itself. Both tests pass without the switch.

Baseline: 1451 passed, 1 failed (the pre-existing `io_test::test_put_data_nefc_zero_dense`), 39 skipped.

## Artifacts

- Scripts in `scripts/diagnostics/speculative/`: `overtravel_probe.py`, `probe_table.py`, `gait_vs_isaac.py`,
  `force_trace.py`, `box_checks.py`, `cpu_traj.py`, `c_traj.py`, `c_oracle.py`.
- `runs/speculative/`:
  - `probe/` and `probe2/` (npz per setting × {4, 64} envs), with `probe_table_n64.md`, `gait_n64.md`, `gait_n4.md`;
  - `protocol_table.md`, `cap_table.md`;
  - `box/`, `cpu/`;
  - job scripts `job_*.sh` and their wrapper logs.
- `runs/penetration/`: `rec/*_{live,fin}[_r2]_spec`, `air/*_{live,fin}_spec*.json` and `cap/*_{live,fin}_spec_*.json`.
