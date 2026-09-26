"""Where one G1 control step spends its GPU time on MuJoCo Warp (flat, training setting), 4096 envs.

Three instruments:
  1. segment graphs: each piece of benchmark_step's graph captured on its own and replayed K times,
     synchronized (graph mode, i.e. the timing that matters); plus the whole step graph for the sum check
  2. Warp Metal runtime counters: dispatches / command buffers / syncs per replay of each graph
  3. per-kernel GPU time (WP_METAL_PROFILE=1, eager launches, one command buffer per dispatch, so an
     attribution, not an absolute timing), per segment, top kernels ranked

usage: python scripts/diagnostics/g1_step_profile.py [N] [DT] [--kernels] [--reps K] [--terrain rough] [--reward_cfg rough_isaac]
  --kernels must be run in a process started with WP_METAL_PROFILE=1 (the flag is read at load time)."""
import os, sys, time, json
import numpy as np, torch, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from metalsim.interop import warp_metal as wm
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import RolloutBuffers, bump, zero_int

_opt = lambda k, dflt: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else dflt
TERRAIN = _opt("--terrain", "flat"); RCFG = _opt("--reward_cfg", None)
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1] in ("--reps", "--terrain", "--reward_cfg")]
N = int(args[0]) if args else 4096
DT = float(args[1]) if len(args) > 1 else 0.0025
KERNELS = "--kernels" in sys.argv
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
RESET_FRAC = 0.02          # share of envs flagged for reset when the reset segments are timed (early training ~2.5 %)

# optional configuration variant (keys as in g1_tp_variants.py), e.g. MJW_TP_VARIANT='{"m_dense_max":0}' or '{"contact_cfg":null}'
VAR = json.loads(os.environ.get("MJW_TP_VARIANT", "{}") or "{}")
if VAR:
    import metalsim.learn.g1_velocity as _g1v
    from metalsim.physics.batch import BatchSimOptions as _BSO

    def _opts(**kw):
        for k in ("njmax", "nconmax", "jacobian", "block_dim", "m_dense_max", "metal_register_cholesky_max", "solver_iterations", "ls_iterations"):
            if k in VAR:
                kw[k] = VAR[k]
        return _BSO(**kw)
    _g1v.BatchSimOptions = _opts
    print(f"variant {VAR}", flush=True)
task = G1VelocityTask(N, terrain=TERRAIN, physics_dt=DT, reward_cfg=RCFG, seed=0,
                      **{k: VAR[k] for k in ("contact_cfg", "solver_cfg") if k in VAR})   # task presets, as in g1_tp_variants.py
print(f"task: terrain {TERRAIN}, reward_cfg {task.reward_cfg}, obs_dim {task.obs_dim}, height scan {task.n_scan} rays", flush=True)
task.reset_all()
dev = task.device


class _Pol:
    step_idx = wp.zeros(1, dtype=int, device=dev)


pol = _Pol()
bufs = RolloutBuffers(1, N, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, N), dtype=float, device=dev); bufs.done = wp.zeros((1, N), dtype=float, device=dev)
action = wp.array(np.random.default_rng(0).uniform(-1, 1, (N, task.act_dim)).astype(np.float32), dtype=float, device=dev)
d, m = task.sim.d, task.sim.m


def seg_apply():
    wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=dev)
    wp.launch(bump, dim=1, inputs=[pol.step_idx], device=dev)
    task.launch_apply_action(action)


def seg_physics():
    task.sim.launch_step()


def seg_reward():
    # launch_reward_done_reset up to (not including) the MuJoCo reset
    from metalsim.learn import g1_velocity as g
    wp.launch(g.g1_foot_vel_mjwarp, dim=N, inputs=[d.cvel, d.xpos, d.subtree_com, task.foot_body, task.foot_root],
              outputs=[task.foot_vel], device=dev)
    if task.contact is not None:
        wp.launch(g.g1_contact_hist_max, dim=N, inputs=[task.contact._hist, task.foot_hist, task.torso_hist], device=dev)
    sens_air, sens_con = (task.contact._cur_air, task.contact._cur_con) if task.contact is not None else (task._dummy2, task._dummy2)
    wp.launch(g.g1_reward_done, dim=N, inputs=[
        d.qpos, d.qvel, d.qacc, d.qfrc_actuator, d.sensordata, d.site_xpos, task.foot_vel, task.cmd, task.last_action, task.prev_action,
        task.default_q, task.jnt_range, task.group, task.touch_adr, task.foot_site, task.foot_body, task.air_time,
        task.contact_time, g.CONTROL_DT, task.t, task.max_t, pol.step_idx, bufs.rew, bufs.done, task.sim._reset_mask,
        task.resample, int(10.0 / g.CONTROL_DT), task.ep_ret, task.ep_len, task.stats, task.stats_i, task.terms,
        task.curriculum, task.level, task.col, task.origin_table, task.n_levels, task.n_cols, task.cell_size, g.EPISODE_S,
        task.origins, task.seed, task.isaac_flat, task.torso_hist, task.use_sensor, sens_air, sens_con, task.foot_hist,
        task.root_com], device=dev)


def seg_reset_data():
    mjw.reset_data(m, d, reset=task.sim._reset_mask)


def seg_sensor_reset():
    if task.contact is not None:
        task.contact.launch_reset(task.sim._reset_mask)


def seg_g1_reset():
    from metalsim.learn import g1_velocity as g
    wp.launch(g.g1_reset, dim=N, inputs=[task.sim._reset_mask, task.default_q, task.origins, task.seed, pol.step_idx,
                                         d.qpos, d.qvel, task.last_action, task.prev_action], device=dev)


def seg_kinematics():
    mjw.kinematics(m, d)


def seg_obs():
    task.launch_obs(pol.step_idx)


def full_step():
    seg_apply(); seg_physics(); task.launch_reward_done_reset(pol, bufs); seg_obs()


SEGMENTS = [("apply_action", seg_apply), ("physics (substeps + contact-sensor hooks)", seg_physics),
            ("reward/done (foot vel, hist max, reward kernel)", seg_reward), ("mjw.reset_data (masked)", seg_reset_data),
            ("contact sensor reset", seg_sensor_reset), ("g1_reset (masked)", seg_g1_reset),
            ("mjw.kinematics (all worlds)", seg_kinematics), ("commands + obs (incl. height scan on rough)", seg_obs)]
if task.scanner is not None:
    SEGMENTS.append(("  of which height scan (187 rays)", lambda: task.scanner.launch(pol.step_idx)))


def set_mask():
    msk = np.zeros(N, bool); msk[np.random.default_rng(1).choice(N, int(RESET_FRAC * N), replace=False)] = True
    task.sim.synchronize(); task.sim._reset_mask.assign(msk); task.sim.synchronize()


def sub_physics():
    """physics split: one substep's MuJoCo Warp pieces, each captured (to see which part of step() dominates)"""
    import mujoco_warp._src.forward as F
    import mujoco_warp._src.solver as S
    import mujoco_warp._src.sensor as SE
    import mujoco_warp._src.collision_driver as CD
    parts = [("fwd_position (kinematics, com, crb, factor, collision, make_constraint, transmission)", lambda: F.fwd_position(m, d, factorize=False)),
             ("  of which collision", lambda: CD.collision(m, d)),
             ("sensor_pos + fwd_velocity + sensor_vel", lambda: (SE.sensor_pos(m, d), F.fwd_velocity(m, d), SE.sensor_vel(m, d))),
             ("fwd_actuation + fwd_acceleration(factorize)", lambda: (F.fwd_actuation(m, d), F.fwd_acceleration(m, d, factorize=True))),
             ("solver.solve", lambda: S.solve(m, d)),
             ("sensor_acc + integrate (implicitfast)", lambda: (SE.sensor_acc(m, d, skip_rne_postconstraint=True), F.implicit(m, d))),
             ("mjw.step (whole substep)", lambda: mjw.step(m, d))]
    return parts


def graph_of(fn):
    with wp.ScopedDevice(dev):
        with wp.ScopedCapture(device=dev) as cap:
            fn()
    return cap.graph


def time_graph(g, reps, before=None):
    with wp.ScopedDevice(dev):
        if before: before()
        wp.capture_launch(g); task.sim.synchronize()
        c0 = wm.counters(dev)
        wp.capture_launch(g); task.sim.synchronize()
        c1 = wm.counters(dev)
        if before: before()
        t0 = time.perf_counter()
        for _ in range(reps):
            wp.capture_launch(g)
        task.sim.synchronize()
        return (time.perf_counter() - t0) / reps, c1 - c0


def main_graphs():
    print(f"=== segment graphs, N={N}, dt {DT*1e3:g} ms, decimation {task.decimation}, {REPS} replays each, synchronized; "
          f"reset segments with {RESET_FRAC:.0%} of envs flagged", flush=True)
    # warm the step so contacts/constraints are populated
    g_full = graph_of(full_step)
    for _ in range(20):
        wp.capture_launch(g_full)
    task.sim.synchronize()
    rows = []
    for name, fn in SEGMENTS:
        g = graph_of(fn)
        before = set_mask if ("reset" in name) else None
        t, c = time_graph(g, REPS, before)
        rows.append((name, t, c))
        print(f"  {name:55s} {t*1e3:8.2f} ms   dispatches {c.dispatches:5d} cmdbufs {c.flushes:3d} syncs {c.syncs}", flush=True)
    tsum = sum(r[1] for r in rows if not r[0].startswith("  of which"))
    t, c = time_graph(g_full, REPS)
    print(f"  {'sum of segments':55s} {tsum*1e3:8.2f} ms")
    print(f"  {'whole step graph':55s} {t*1e3:8.2f} ms   dispatches {c.dispatches:5d} cmdbufs {c.flushes:3d} syncs {c.syncs}  -> {N/t:,.0f} env-steps/s", flush=True)
    phys = rows[1][1]
    print(f"  outside physics: {(t - phys)*1e3:.2f} ms ({(t - phys)/t:.1%} of the step)")
    print(f"=== one physics substep split (pieces captured separately; they overlap state, so not additive exactly)", flush=True)
    for name, fn in sub_physics():
        g = graph_of(fn)
        tt, cc = time_graph(g, REPS)
        print(f"  {name:90s} {tt*1e3:8.2f} ms  dispatches {cc.dispatches:4d}", flush=True)
    # restore a consistent state
    for _ in range(3):
        wp.capture_launch(g_full)
    task.sim.synchronize()


def main_kernels():
    core = wp._src.context.runtime.core
    core.wp_metal_profile_report()          # clear
    print(f"=== per-kernel GPU time (WP_METAL_PROFILE=1, eager, one command buffer per dispatch), N={N}; 3 steps per segment", flush=True)
    for _ in range(3):
        with wp.ScopedDevice(dev):
            full_step()
    task.sim.synchronize(); core.wp_metal_profile_report()
    allrows = []
    for name, fn in SEGMENTS + [("  substep: " + k, f) for k, f in sub_physics()[:-1] if "of which" not in k]:
        if "reset" in name: set_mask()
        with wp.ScopedDevice(dev):
            for _ in range(3):
                fn()
        task.sim.synchronize(); time.sleep(0.2)
        rep = core.wp_metal_profile_report().decode()
        tot = 0.0; lines = []
        for ln in rep.strip().splitlines():
            ms, cnt, kname = ln.split(None, 3)[0], ln.split(None, 3)[2], ln.split(None, 3)[3]
            ms = float(ms) / 3; tot += ms; lines.append((ms, int(cnt) // 3, kname))
            allrows.append((ms, name.strip(), kname, int(cnt) // 3))
        print(f"--- {name}: {tot:.2f} ms/step GPU (sum of per-dispatch times)", flush=True)
        for ms, cnt, kname in lines[:12]:
            print(f"   {ms:8.3f} ms  x{cnt:3d}  {kname}")
    print("=== top kernels overall (ms per control step; substep pieces are per substep, excluded)")
    allrows = [r for r in allrows if not r[1].startswith("substep:")]
    allrows.sort(reverse=True)
    for ms, seg, kname, cnt in allrows[:30]:
        print(f"   {ms:8.3f} ms  x{cnt:3d}  [{seg[:40]}] {kname}")


if KERNELS:
    main_kernels()
else:
    main_graphs()
