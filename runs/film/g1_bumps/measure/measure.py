# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# The CPU loop (run_cpu) is unitree_rl_gym's deploy/deploy_mujoco/deploy_mujoco.py (commit 276801e4) per world, through
# the ported observation of metalsim/render/film/unitree.py: Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD.
# ("Unitree Robotics"), BSD-3-Clause (licenses/BSD-3-Clause-Unitree.txt).
"""Measure Unitree's G1 walking policy on Isaac Lab's random_rough tile before choosing the g1_bumps take.

Per level: 16 robots for 8 s, command (0.5, 0, 0), one row of lanes at x = -2 m, y = -3 ... 3 m (0.4 m apart), each
start raised by the highest terrain under its soles (``unitree.unitree_start_qpos``). A level is the tile's
``noise_range``: Isaac's own is (0.02, 0.10) (ROUGH_TERRAINS_CFG; Isaac's random_uniform_terrain ignores difficulty, so
this is the tile at every difficulty level); (0.02, 0.04 / 0.06 / 0.08) are variants with a lower ceiling, NOT
Isaac's terrain, to see how the outcome depends on bump height. Same tile seed (0) for all.

  gpu: the film's adapter (``unitree.UnitreeG1`` on MuJoCo Warp BatchSim, 16 worlds; a GPU-queue job)
  cpu: Unitree's loop on CPU MuJoCo, one MjData per robot, the same model and starts (no GPU)

Usage (repository root): .venv/bin/python runs/film/g1_bumps/measure/measure.py gpu|cpu <out_dir> [ceilings in m]
Writes <out_dir>/<mode>_levels.json (per level: summary and per robot) and <mode>_<ceiling>.npz (qpos per frame).
First take (CPU MuJoCo film, 2026-10-31 afternoon): out_dir runs/film/g1_bumps/measure; the JSONs stay here, the .npz
arrays (12 MB) were moved to runs/readme_media/scratch/g1-bumps/measure_npz/ (not for the repository). Its gpu_* files
ran a stale fork checkout (07a51a6; docs/research/hfield_contacts_2026-10-31.md section 2).
MetalSim-physics take: ``gpu`` on the MuJoCo Warp fork at 787a5b3 (heightfield contacts fixed), out_dir
runs/film/g1_bumps/measure/mjw_787a5b3 (arrays in runs/readme_media/scratch/g1-bumps-metalsim/npz/).
Experiment switch (gpu only): MJW_HFIELD_PLANE_CONTACTS=0 sets MuJoCo Warp's ``collision_convex.HFIELD_PLANE_CONTACTS``
to False before any kernel is built; the outputs are then named gpu-upstreamhf_*. (On fork 07a51a6 that chose
upstream's GJK/EPA against each heightfield prism; since 93e5ce1 False is the default, so the switch changes nothing.)

Definitions: down / fall = pelvis below 0.4 m (``unitree.FALLEN_Z``; the robot is on the ground); first non-foot
contact = first frame where any robot geom other than the sole spheres touches the terrain (a knee, hand or pelvis);
gait = ``record.gait_metrics`` with each robot's window from 1 s to its fall (or 8 s), foot height above the highest
terrain under the sole (``unitree.sole_grid``), contact = a sole sphere touching the terrain; forward = pelvis x at
its fall (or at 8 s) minus its start x; sole penetration = in contact frames, how far the terrain rises above the sole
quadrilateral spanned by the four sphere bottoms (the bumps that the visual sole, which does not collide, would cut).
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

from metalsim.learn.terrain import collision_top
from metalsim.render.film import record as REC
from metalsim.render.film import tasks
from metalsim.render.film import unitree as U

N, SECONDS, SEED = 16, 8.0, 0
STARTS = dict(x0=-2.0, y=[-3.0, 3.0])
CMD = (0.5, 0.0, 0.0)
SPHERE_R = 0.005


def sole_penetration(m, d_fk, qpos, foot_geoms, terrain, contact):
    """(F, N, 2) terrain height above the sole quadrilateral (sphere bottoms) where the foot is in contact, else nan."""
    import mujoco

    F, n = qpos.shape[:2]
    out = np.full((F, n, 2), np.nan)
    for k in range(F):
        for e in range(n):
            if not contact[k, e].any():
                continue
            d_fk.qpos[:] = qpos[k, e]
            mujoco.mj_kinematics(m, d_fk)
            for f, gs in enumerate(foot_geoms):
                if not contact[k, e, f]:
                    continue
                P = d_fk.geom_xpos[gs]                                       # (4, 3) sphere centres
                xy = U.sole_grid(P[None, :, :2])[0]                          # (K, 2)
                zs = U.sole_grid(np.stack([P[:, 2] - SPHERE_R, np.zeros(4)], 1)[None])[0][:, 0]
                zt = collision_top(terrain, xy[:, 0], xy[:, 1])
                out[k, e, f] = float((zt - zs).max())
    return out


def summarise(m, terrain, foot_geoms, qpos, contact, nonfoot, frame_dt, start, lanes, cmd):
    """Per-level summary and per-robot rows from (F, N, nq) qpos, (F, N, 2) sole contact, (F, N) non-foot contact."""
    import mujoco

    F, n = qpos.shape[:2]
    down = qpos[:, :, 2] < U.FALLEN_Z
    fall = np.array([int(np.argmax(down[:, e])) if down[:, e].any() else -1 for e in range(n)])
    nf = np.array([int(np.argmax(nonfoot[:, e])) if nonfoot[:, e].any() else -1 for e in range(n)])
    d_fk = mujoco.MjData(m)
    snaps = []
    for k in range(F):
        feet, fp, _ = U.unitree_feet(m, d_fk, qpos[k], foot_geoms)
        snaps.append(dict(qpos=qpos[k], feet=feet, feet_footprint=U.sole_grid(fp), feet_contact=contact[k],
                          cmd=np.tile(np.asarray(cmd, np.float64), (n, 1))))
    ground = lambda x, y: collision_top(terrain, np.ravel(x), np.ravel(y)).reshape(np.shape(x))
    g = REC.gait_metrics(snaps, fall, fall >= 0, frame_dt, ground)
    pen = sole_penetration(m, d_fk, qpos, foot_geoms, terrain, contact)
    k0 = int(round(REC.GAIT_T0 / frame_dt))
    rows = []
    for e in range(n):
        kend = fall[e] if fall[e] >= 0 else F - 1
        r = g["per_env"][str(e)]
        p = pen[k0:kend + 1, e]
        rows.append(dict(lane_y=round(float(lanes[e]), 3), fell=bool(fall[e] >= 0),
                         fall_s=round((fall[e] + 1) * frame_dt, 2) if fall[e] >= 0 else None,
                         first_nonfoot_contact_s=round((nf[e] + 1) * frame_dt, 2) if nf[e] >= 0 else None,
                         forward_m=round(float(qpos[kend, e, 0] - start[e, 0]), 3),
                         lateral_m=round(float(qpos[kend, e, 1] - start[e, 1]), 3),
                         gait=r, sole_penetration_m=dict(p50=round(float(np.nanmedian(p)), 4) if np.isfinite(p).any() else None,
                                                        p95=round(float(np.nanpercentile(p, 95)), 4) if np.isfinite(p).any() else None,
                                                        max=round(float(np.nanmax(p)), 4) if np.isfinite(p).any() else None)))
    med = lambda v: None if not len(v) else round(float(np.median(v)), 4)
    win = [r for r in rows if r["gait"]]                          # robots with >= 2 s of walking before a fall or 8 s
    falls = [r["fall_s"] for r in rows if r["fell"]]
    pen_all = pen[k0:][np.isfinite(pen[k0:])]
    summary = dict(robots=n, falls=len(falls), fall_s_first=min(falls) if falls else None, fall_s_median=med(falls),
                   fall_s_all=sorted(falls), nonfoot_contacts=sum(r["first_nonfoot_contact_s"] is not None for r in rows),
                   forward_m_median=med([r["forward_m"] for r in rows]),
                   forward_m_median_standing=med([r["forward_m"] for r in rows if not r["fell"]]),
                   forward_m_range=[min(r["forward_m"] for r in rows), max(r["forward_m"] for r in rows)],
                   gait_robots=len(win),
                   cadence_hz=[med([r["gait"]["cadence_hz"][f] for r in win]) for f in range(2)],
                   touchdowns_hz=[med([r["gait"]["touchdowns_hz"][f] for r in win]) for f in range(2)],
                   apex_m=[med([r["gait"]["apex_m"][f] for r in win]) for f in range(2)],
                   slip_mps=[med([r["gait"]["slip_mps"][f] for r in win if r["gait"]["slip_mps"][f] is not None]) for f in range(2)],
                   speed_mps=med([r["gait"]["speed_mps"] for r in win]),
                   sole_penetration_m=dict(p50=med(pen_all), p95=round(float(np.percentile(pen_all, 95)), 4) if len(pen_all) else None,
                                           max=round(float(pen_all.max()), 4) if len(pen_all) else None))
    return summary, rows, g


def nonfoot_contacts(pairs, worlds, n, foot_geoms, floor, ngeom_robot):
    """(n,) bool: a terrain contact with a robot geom that is not a sole sphere."""
    feet = set(int(g) for gs in foot_geoms for g in gs)
    out = np.zeros(n, bool)
    for (a, b), w in zip(np.asarray(pairs).reshape(-1, 2), np.asarray(worlds).reshape(-1)):
        o = b if a == floor else a if b == floor else None
        if o is not None and o not in feet:
            out[w] = True
    return out


def run_gpu(ceiling):
    from metalsim.render.film.cfg import FilmCfg

    if os.environ.get("MJW_HFIELD_PLANE_CONTACTS") == "0":
        from mujoco_warp._src import collision_convex
        collision_convex.HFIELD_PLANE_CONTACTS = False

    task = SimpleNamespace(adapter_kw=dict(cmd=CMD, terrain=dict(noise_range=(0.02, ceiling), seed=SEED), starts=STARTS))
    A = U.UnitreeG1(task, FilmCfg(num_envs=N), None)
    A.reset()
    F = int(round(SECONDS / A.frame_dt))
    Q, C, NF = [], [], []
    t0 = time.perf_counter()
    for k in range(F):
        snap, done, info, _ = A.step()
        d = A.sim.d
        nac = int(d.nacon.numpy()[0])
        pairs, worlds = d.contact.geom.numpy()[:nac], d.contact.worldid.numpy()[:nac]
        Q.append(snap["qpos"]); C.append(snap["feet_contact"])
        NF.append(nonfoot_contacts(pairs, worlds, N, A.foot_geoms, A.floor, A.model.ngeom))
    ov = A.sim.check_overflow()
    return A.model, A.terrain, A.foot_geoms, np.array(Q), np.array(C), np.array(NF), A.frame_dt, A.start, \
        A.start_xy()[:, 1], dict(engine="MuJoCo Warp BatchSim (film adapter UnitreeG1)", overflow=ov,
                                 hfield_plane_contacts=__import__("mujoco_warp._src.collision_convex",
                                                                  fromlist=["x"]).HFIELD_PLANE_CONTACTS,
                                 rollout_s=round(time.perf_counter() - t0, 1))


def run_cpu(ceiling):
    import mujoco
    import torch

    P = U.unitree_paths(tasks.UNITREE_RL_GYM)
    cfg = U.load_unitree_config(P["config"])
    terrain = U.random_rough_tile((0.02, ceiling), SEED)
    m = U.load_unitree_model(P["scene"], cfg, terrain)
    fg = U.unitree_foot_geoms(m)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, U.UNITREE_FLOOR)
    xy = np.stack([np.full(N, STARTS["x0"]), np.linspace(*STARTS["y"], N)], 1)
    Q0 = U.unitree_start_qpos(m, fg, terrain, xy)
    dec, dt = int(cfg["control_decimation"]), float(cfg["simulation_dt"])
    F = int(round(SECONDS / (dt * dec)))
    kps, kds, dflt = cfg["kps"], cfg["kds"], cfg["default_angles"]
    cmd = np.array(CMD, np.float32)
    Q = np.zeros((F, N, m.nq)); C = np.zeros((F, N, 2), bool); NF = np.zeros((F, N), bool)
    t0 = time.perf_counter()
    for e in range(N):
        d = mujoco.MjData(m)
        d.qpos[:] = Q0[e]
        mujoco.mj_forward(m, d)
        policy = torch.jit.load(P["policy"])
        action = np.zeros(int(cfg["num_actions"]), np.float32)
        target = dflt.copy()
        counter = 0
        for k in range(F):
            for _ in range(dec):
                d.ctrl[:] = (target - d.qpos[7:]) * kps + (0 - d.qvel[6:]) * kds
                mujoco.mj_step(m, d)
                counter += 1
            obs = U.unitree_obs(cfg, d.qpos[None].astype(np.float64), d.qvel[None].astype(np.float64), action[None], cmd,
                                counter)
            action = policy(torch.from_numpy(obs)).detach().numpy().squeeze().astype(np.float32)
            target = action * cfg["action_scale"] + dflt
            Q[k, e] = d.qpos
            pairs = d.contact.geom[:d.ncon]
            C[k, e] = U.unitree_contacts(pairs, np.zeros(d.ncon, int), 1, fg, floor)[0]
            NF[k, e] = nonfoot_contacts(pairs, np.zeros(d.ncon, int), 1, fg, floor, m.ngeom)[0]
    return m, terrain, fg, Q, C, NF, dt * dec, Q0[:, :3], xy[:, 1], \
        dict(engine=f"CPU MuJoCo {mujoco.__version__} (Unitree's loop per robot)", rollout_s=round(time.perf_counter() - t0, 1))


def main():
    mode, out = sys.argv[1], sys.argv[2]
    tag = mode + ("-upstreamhf" if mode == "gpu" and os.environ.get("MJW_HFIELD_PLANE_CONTACTS") == "0" else "")
    ceilings = [float(c) for c in sys.argv[3:]] or [0.04, 0.06, 0.08, 0.10]
    os.makedirs(out, exist_ok=True)
    res = dict(mode=mode, robots=N, seconds=SECONDS, cmd=list(CMD), starts=STARTS, tile_seed=SEED, levels={})
    for c in ceilings:
        m, terrain, fg, Q, C, NF, frame_dt, start, lanes, info = (run_gpu if mode == "gpu" else run_cpu)(c)
        summary, rows, g = summarise(m, terrain, fg, Q, C, NF, frame_dt, start, lanes, CMD)
        key = f"{c:.2f}"
        res["levels"][key] = dict(noise_range=[0.02, c], exact_isaac=abs(c - 0.10) < 1e-9, terrain=terrain["config"],
                                  terrain_stats=terrain["stats"], info=info, summary=summary, robots=rows,
                                  gait_definition=g["definition"])
        np.savez(os.path.join(out, f"{tag}_{key}.npz"), qpos=Q, contact=C, nonfoot=NF, lanes=lanes)
        s = summary
        print(f"[{tag}] ceiling {c:.2f} m: falls {s['falls']}/{N} (first {s['fall_s_first']} s, median {s['fall_s_median']} s); "
              f"non-foot contacts {s['nonfoot_contacts']}; forward median {s['forward_m_median']} m "
              f"(standing robots {s['forward_m_median_standing']}); gait robots {s['gait_robots']}: cadence {s['cadence_hz']} Hz, "
              f"touch-downs {s['touchdowns_hz']}/s, apex {s['apex_m']} m, speed {s['speed_mps']} m/s; "
              f"sole penetration {s['sole_penetration_m']}; {info}", flush=True)
        json.dump(res, open(os.path.join(out, f"{tag}_levels.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
