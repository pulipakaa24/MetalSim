"""Feasibility: Isaac's G1 USD loaded by Newton's UsdPhysics importer and simulated with Newton's XPBD
solver on the Metal Warp fork: PD hold at Isaac's gains, penetration, stability, and rough speed."""
import sys, re, time, numpy as np, warp as wp, newton
wp.config.quiet = True
# Findings (2026-09-24, M4 Max, Warp fork branch metalsim, Newton 1.7.0.dev): Newton's UsdPhysics importer
# loads g1_minimal.usd directly (44 bodies, 44 joints, 43 DoF); XPBD, Featherstone and SemiImplicit
# solvers run on Metal; VBD fails to compile; the GJK/MPR narrow phase needs Warp fixed-size arrays
# (not in the Metal codegen), so the three box-shaped mesh colliders are replaced by box primitives and
# self-collision is off. A box rests with 0.3 mm penetration under XPBD. With Isaac's drive gains the
# G1 does not yet stand under XPBD (collapses within 1 s at 4 XPBD iterations, 2.5 ms): drive/joint
# compliance tuning is the open item. Speed at 256 envs, eager launches: 283K physics-steps/s.
POS = newton.JointTargetMode.POSITION
GAINS = [(r".*_hip_yaw_joint", 150, 5), (r".*_hip_roll_joint", 150, 5), (r".*_hip_pitch_joint", 200, 5), (r".*_knee_joint", 200, 5), (r"torso_joint", 200, 5),
         (r".*_ankle_.*", 20, 2), (r".*_shoulder_.*", 40, 10), (r".*_elbow_.*", 40, 10), (r".*_(five|three|six|four|zero|one|two)_joint", 40, 10)]
INIT = [(r".*_hip_pitch_joint", -0.20), (r".*_knee_joint", 0.42), (r".*_ankle_pitch_joint", -0.23), (r".*_elbow_pitch_joint", 0.87), (r"left_shoulder_roll_joint", 0.16),
        (r".*_shoulder_pitch_joint", 0.35), (r"right_shoulder_roll_joint", -0.16), (r"left_one_joint", 1.0), (r"right_one_joint", -1.0), (r"left_two_joint", 0.52), (r"right_two_joint", -0.52)]


def robot_builder():
    b = newton.ModelBuilder()
    b.add_usd("assets/isaac/G1/g1_minimal.usd", floating=True, xform=wp.transform((0, 0, 0.78), wp.quat_identity()), enable_self_collisions=False, load_visual_shapes=False)
    b.gravity = -9.81                       # the USD authors gravityMagnitude 0 (PhysX default), the importer copies it
    types = list(b.shape_type); srcs = list(b.shape_source); scales = list(b.shape_scale); xf = list(b.shape_transform)
    for i, t in enumerate(types):
        if srcs[i] is not None and t in (newton.GeoType.MESH, getattr(newton.GeoType, "CONVEX_MESH", -1)):
            v = np.asarray(srcs[i].vertices, float) * np.asarray(scales[i], float); lo, hi = v.min(0), v.max(0); c = 0.5 * (lo + hi); he = 0.5 * (hi - lo)
            types[i] = newton.GeoType.BOX; scales[i] = wp.vec3(*he.tolist()); srcs[i] = None
            p0, q0 = wp.transform_get_translation(xf[i]), wp.transform_get_rotation(xf[i]); xf[i] = wp.transform(wp.vec3(*(np.array(p0) + np.array(wp.quat_rotate(q0, wp.vec3(*c.tolist())))).tolist()), q0)
    b.shape_type = types; b.shape_source = srcs; b.shape_scale = scales; b.shape_transform = xf
    labels = list(b.joint_label if hasattr(b, "joint_label") else b.joint_key)
    ke = list(b.joint_target_ke); kd = list(b.joint_target_kd); tq = list(b.joint_target_q); mode = list(b.joint_target_mode); q = list(b.joint_q)
    qd_start = list(b.joint_qd_start) + [b.joint_dof_count]; q_start = list(b.joint_q_start) + [b.joint_coord_count]
    for j, lab in enumerate(labels):
        name = lab.split("/")[-1]
        for pat, kp, kv in GAINS:
            if re.fullmatch(pat, name):
                val = next((v for ip, v in INIT if re.fullmatch(ip, name)), 0.0)     # Isaac: unlisted joints default to 0 (the USD's own drive targets are ignored)
                for d in range(qd_start[j], qd_start[j + 1]): ke[d] = kp; kd[d] = kv; mode[d] = POS; tq[d] = val
                for c in range(q_start[j], q_start[j + 1]): q[c] = val
    b.joint_target_ke, b.joint_target_kd, b.joint_target_q, b.joint_target_mode, b.joint_q = ke, kd, tq, mode, q
    return b


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4; iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4; h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0025
    with wp.ScopedDevice("metal:0"):
        rb = robot_builder(); s = newton.ModelBuilder(); s.gravity = -9.81; s.add_ground_plane(); s.replicate(rb, N, spacing=(2.5, 2.5, 0.0)); s.gravity = -9.81
        m = s.finalize(); solver = newton.solvers.SolverXPBD(m, iterations=iters); s0, s1 = m.state(), m.state(); ctrl = m.control(); newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
        contacts = None; wp.synchronize(); t0 = time.time()
        for k in range(int(2.0 / h)):
            s0.clear_forces(); contacts = m.collide(s0, contacts); solver.step(s0, s1, ctrl, contacts, h); s0, s1 = s1, s0
            if k % int(0.5 / h) == int(0.5 / h) - 1:
                wp.synchronize(); bq = s0.body_q.numpy().reshape(N, -1, 7); bqd = s0.body_qd.numpy().reshape(N, -1, 6)
                print(f"t={(k+1)*h:.2f}s pelvis z {np.round(bq[:4, 0, 2], 3)} | lowest body z {bq[0, :, 2].min():.3f} | max |v| {np.abs(bqd[..., 3:]).max():.2f} m/s", flush=True)
        wp.synchronize(); el = time.time() - t0
        print(f"N={N} XPBD it={iters} h={h}: {int(2.0/h)} steps in {el:.1f} s -> {N * int(2.0/h) / el:,.0f} physics-steps/s (eager)", flush=True)
