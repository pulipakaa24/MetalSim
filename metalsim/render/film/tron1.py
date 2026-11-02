# SPDX-License-Identifier: Apache-2.0
"""LimX's TRON1 wheel-legged robot (WF_TRON1A) driven by LimX's own RL policy, in MetalSim's TRON1 simulation, for the
filming tool. The task entry (``TRON1``) lives here with its adapter; ``tasks.py`` registers it.

The rollout is the TRON1 evaluation's own path (``metalsim/tron1/evaluate.py``, ``episode``): per env one
``metalsim.tron1.sim.Tron1Sim`` (LimX's MJCF on CPU MuJoCo, 0.5 ms physics step, with MetalSim's hardware-realism
layer: transport latency and packet loss, the drives' mode-0 law on quantized encoders, joint friction, IMU / AHRS
errors, the sensor pack, the tyre model) at one realism level of ``metalsim.tron1.realism.SimParams`` ("nominal" by
default: the best single guess of the real robot, docs/TRON1.md "Realism layer"), and one
``metalsim.tron1.limx_policy.LimxPolicyController`` (LimX's ONNX encoder and policy, ported from LimX's
``tron1-rl-deploy-python``), stepped by ``Tron1Sim.run`` itself: the controller at 500 Hz, the policy every 10th call
(50 Hz). One film frame is one policy step (20 ms), the state after its 20 1-ms SDK ticks. Each env gets its own noise
seed (``film.seed + env``), as the evaluation gives each episode its own.

The velocity command is a fixed schedule (``SCHEDULE``, within the evaluation's ranges: vx in [-1, 1] m/s, yaw rate in
[-0.6, 0.6] rad/s), the same for every env. No pushes. The procedure is a real robot's (``adapter_kw`` ``pre_roll_s``,
``ramp_s``): ``reset`` places each robot on its wheels with straight legs (``LimxPolicyController.initial_q``, LimX's
default joint angles, which is not the policy's stance) and lets the policy stand it up for ``PRE_ROLL_S`` at the
schedule's first command before the film's t = 0, as a robot is stood up before it drives; every command change then
ramps linearly over ``RAMP_S`` from the previous command, as an operator's stick moves, instead of stepping. The film's
clock starts after the pre-roll (frame 0 is 20 ms after it), and the schedule's times are film times.

Done mask: ``Tron1Sim``'s own fall check (base tilted more than 70 deg, or the base, sensor pack or a leg link touching
the ground), reported as ``info["terminated"]``; a fallen env is frozen where it fell. No time-out inside a film.

Tracking (``meta()["tracking"]``, written to ``rec.json`` ``adapter.tracking``): the evaluation's error definitions
(forward speed in the heading frame minus the commanded vx; yaw rate minus the commanded yaw rate; RMS over frames at
least 0.5 s after a command change, before a fall; peak |pitch| and |roll|), sampled once per frame (every 20 ms; the
evaluation samples every 10 ms), against the command the robots were sent (with ``ramp_s`` <= ``SETTLE_S``, the
schedule's own value on every frame the RMS counts). Per env summaries for every env, per-frame series for the filmed
env(s). The pre-roll is not part of any metric.
"""
from __future__ import annotations

import os
import time

import numpy as np

from metalsim.render.film.cfg import AssetRef, FilmCfg, FilmTask, LookCfg, ShotCfg, ViewerCfg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# (start time [s], (vx [m/s], vy, yaw rate [rad/s])): stand, drive, turn left while driving, slow down, stop
SCHEDULE = ((0.0, (0.0, 0.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0)),
            (2.0, (0.8, 0.0, 0.6)),
            (3.5, (0.4, 0.0, 0.0)),
            (4.5, (0.0, 0.0, 0.0)))
SETTLE_S = 0.5           # s after a command change left out of the tracking RMS (evaluate.py: ``(t % seg) > 0.5``)
CONTROL_HZ = 500         # LimX's loop_frequency (params.yaml), evaluate.py's Tron1Sim.run default
PRE_ROLL_S = 3.0         # s standing under the policy at the first command before the film's t = 0 (Tron1Limx.reset)
RAMP_S = 0.5             # s over which each command change ramps linearly from the previous command

LIMX_DEPLOY = "https://github.com/limxdynamics/tron1-rl-deploy-python"
LIMX_COMMIT = "035e4c9e2f24e916fa5e8363ac6f0f788a2e1e8c"
LIMX_POLICY = dict(
    repo=LIMX_DEPLOY, commit=LIMX_COMMIT, licence="Apache-2.0 (assets/tron1/LICENSE-rl-deploy-python)",
    upstream_dir="controllers/model/WF_TRON1A", local_dir="assets/tron1/limx_policy/WF_TRON1A",
    policy="policy/isaacgym/policy.onnx", encoder="policy/isaacgym/encoder.onnx", params="params.yaml",
    # sha256 of the files in this repository; byte-identical to LimX's at LIMX_COMMIT (fetched 2026-10-31)
    sha256={"policy/isaacgym/policy.onnx": "bfc4617f65cef4b1c952156352dd2fee0deffd68458959a20b4bc0bb0285d85b",
            "policy/isaacgym/encoder.onnx": "f79e29d12b670aa9078b4a825f3b30e75018efb15c8108487a0bf2c222e78ba6",
            "params.yaml": "85b5431ae2562d754e836668c1046aba3525aeac29d508a99c23a184a278512b"})


def command_at(schedule, t: float) -> tuple:
    """The schedule's (vx, vy, wz) at time ``t``."""
    cur = schedule[0][1]
    for t0, c in schedule:
        if t >= t0 - 1e-9:
            cur = c
    return tuple(float(x) for x in cur)


def command_ramped(schedule, t: float, ramp_s: float) -> tuple:
    """The schedule's (vx, vy, wz) at time ``t`` with each change ramped linearly over ``ramp_s`` seconds from the
    previous segment's command (``ramp_s`` 0: ``command_at``). Segments must be at least ``ramp_s`` long."""
    i = max([j for j, (t0, _) in enumerate(schedule) if t >= t0 - 1e-9] or [0])
    cur = np.asarray(schedule[i][1], float)
    prev = np.asarray(schedule[i - 1][1], float) if i > 0 else cur
    a = (t - schedule[i][0]) / ramp_s if ramp_s > 0 else 1.0
    return tuple(float(x) for x in (cur if a >= 1.0 else prev + a * (cur - prev)))


def since_change(schedule, t: float) -> float:
    """Seconds since the last command change at or before ``t`` (the start counts as a change)."""
    last = 0.0
    for t0, _ in schedule:
        if t >= t0 - 1e-9:
            last = t0
    return t - last


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


class Tron1Limx:
    """``num_envs`` TRON1 robots (``Tron1Sim``, one per env) with LimX's policy, all on the same command schedule.
    ``adapter_kw``: ``params`` ("nominal" | "ideal": ``SimParams``' levels; "random" is not offered, the film shows one
    level), ``schedule`` (default ``SCHEDULE``), ``pre_roll_s`` (default ``PRE_ROLL_S``; 0: the film starts from
    ``initial_q``), ``ramp_s`` (default ``RAMP_S``; 0: stepped commands)."""

    backend = "mujoco"

    def __init__(self, task, film, checkpoint, viewer=None):
        from metalsim.tron1.limx_policy import POLICY_DIR, LimxPolicyController
        from metalsim.tron1.realism import SimParams
        from metalsim.tron1.sim import Tron1Sim

        akw = dict(getattr(task, "adapter_kw", None) or {})
        self.level = akw.get("params", "nominal")
        if self.level not in ("nominal", "ideal"):
            raise ValueError(f"params {self.level!r}: the film adapter offers 'nominal' or 'ideal'")
        self.schedule = tuple((float(t0), tuple(float(x) for x in c)) for t0, c in akw.get("schedule", SCHEDULE))
        self.pre_roll_s = float(akw.get("pre_roll_s", PRE_ROLL_S))
        self.ramp_s = float(akw.get("ramp_s", RAMP_S))
        if self.pre_roll_s < 0 or abs(self.pre_roll_s * 1000 - round(self.pre_roll_s * 1000)) > 1e-6:
            raise ValueError(f"pre_roll_s {self.pre_roll_s}: a whole number of ms >= 0")
        seg = min((b[0] - a[0] for a, b in zip(self.schedule, self.schedule[1:])), default=np.inf)
        if not 0 <= self.ramp_s <= seg:
            raise ValueError(f"ramp_s {self.ramp_s}: between 0 and the shortest command segment ({seg} s)")
        self.off_us = int(round(self.pre_roll_s * 1e6))       # Tron1Sim's clock at the film's t = 0
        pol = os.path.join(str(POLICY_DIR), "isaacgym", "policy.onnx")
        if checkpoint and os.path.abspath(checkpoint) != os.path.abspath(pol):
            raise ValueError(f"{checkpoint}: LimxPolicyController plays LimX's isaacgym export ({pol}) only")
        self.checkpoint = pol
        self.make_params = SimParams.nominal if self.level == "nominal" else SimParams.ideal
        n = self.num_envs = int(film.num_envs)
        self.seeds = [int(film.seed) + e for e in range(n)]
        t0 = time.perf_counter()
        self.sims = [Tron1Sim(self.make_params(), seed=s) for s in self.seeds]
        self.ctrls = [LimxPolicyController() for _ in range(n)]
        self.build_s = time.perf_counter() - t0
        dec = self.ctrls[0].decimation
        self.ms_per_frame = dec * int(round(1000 / CONTROL_HZ))            # 10 controller calls x 2 ms = 20 ms
        self.frame_dt = self.ms_per_frame * 1e-3
        self.film_envs = [int(e) for e in film.envs] if isinstance(film.envs, (list, tuple)) else None
        files = {k: os.path.join(ROOT, LIMX_POLICY["local_dir"], k) for k in LIMX_POLICY["sha256"]}
        self.files_sha256 = {k: _sha256(p) for k, p in files.items()}
        bad = [k for k, h in self.files_sha256.items() if h != LIMX_POLICY["sha256"][k]]
        if bad:
            raise ValueError(f"{LIMX_POLICY['local_dir']}: {bad} differ from LimX's files at {LIMX_COMMIT[:7]}")
        self.ckpt_format = "onnx (LimX isaacgym export: encoder.onnx + policy.onnx)"
        self.ckpt_iter = None
        self.ckpt_source = dict(source="LimX Dynamics tron1-rl-deploy-python, WF_TRON1A isaacgym policy",
                                source_url=f"{LIMX_DEPLOY}/blob/{LIMX_COMMIT}/{LIMX_POLICY['upstream_dir']}/"
                                           f"{LIMX_POLICY['policy']}",
                                published_sha256=LIMX_POLICY["sha256"]["policy/isaacgym/policy.onnx"],
                                commit=LIMX_COMMIT, licence=LIMX_POLICY["licence"])

    def reset(self):
        """Each robot placed on its wheels with straight legs (``initial_q``: LimX's default joint angles, q = 0), then
        ``pre_roll_s`` of standing under the policy at the schedule's first command through ``Tron1Sim.run`` itself
        (same loop, controller phase and noise streams as the film that follows). The film's t = 0 is the end of it."""
        n = self.num_envs
        self.fallen = np.zeros(n, bool)
        self.fall_t = np.full(n, np.nan)
        first = self.schedule[0][1]
        for e, (sim, c) in enumerate(zip(self.sims, self.ctrls)):
            c.reset()
            sim.reset(c.initial_q())
            if self.off_us:
                sim.run(c, self.off_us * 1e-6, control_hz=CONTROL_HZ, commands=lambda t: first,
                        log_every_ms=self.ms_per_frame)
            if sim.fallen:                                  # fell while standing: a fall before the film's t = 0
                self.fallen[e] = True
                self.fall_t[e] = (sim.t_us - self.off_us) * 1e-6
            elif sim.t_us != self.off_us:
                raise RuntimeError(f"env {e}: Tron1Sim clock {sim.t_us} us after the pre-roll, not {self.off_us}")
        self.start = np.array([s.d.qpos[:3].copy() for s in self.sims])
        self._xy = self.start[:, :2].copy()
        self.dist = np.zeros(n)
        self.rows = []                     # per frame: t, cmd (3), per env vx, wz, pitch, roll, base z

    def command(self, t):
        """The (vx, vy, wz) sent at film time ``t``: the schedule, each change ramped over ``ramp_s``."""
        return command_ramped(self.schedule, t, self.ramp_s)

    def _command(self, t_sim):
        """``Tron1Sim.run``'s ``commands(t)``: Tron1Sim's clock (``t_us`` x 1e-6) to film time, then ``command``."""
        return self.command((int(round(t_sim * 1e6)) - self.off_us) * 1e-6)

    def step(self):
        t0 = time.perf_counter()
        n = self.num_envs
        new = np.zeros(n, bool)
        for e, (sim, c) in enumerate(zip(self.sims, self.ctrls)):
            if self.fallen[e]:
                continue
            # the evaluation's loop itself: controller at CONTROL_HZ, Tron1Sim's fall check after every 1 ms tick
            sim.run(c, self.ms_per_frame * 1e-3, control_hz=CONTROL_HZ, commands=self._command,
                    log_every_ms=self.ms_per_frame)
            if sim.fallen:
                self.fallen[e] = new[e] = True
                self.fall_t[e] = (sim.t_us - self.off_us) * 1e-6
        sim_s = time.perf_counter() - t0
        qpos = np.array([s.d.qpos.copy() for s in self.sims])
        tr = [s.truth() for s in self.sims]
        t = (max(s.t_us for s in self.sims) - self.off_us) * 1e-6        # film time
        cmd = np.array(self.command(t - 1e-3))            # the command of the frame's last 1 ms tick
        self.rows.append(dict(t=t, cmd=cmd, vx=np.array([r["v_head"][0] for r in tr]),
                              wz=np.array([r["yaw_rate"] for r in tr]), pitch=np.array([r["pitch"] for r in tr]),
                              roll=np.array([r["roll"] for r in tr]), z=qpos[:, 2].copy(),
                              fallen=self.fallen.copy()))
        al = ~self.fallen | new
        self.dist[al] += np.linalg.norm(qpos[al, :2] - self._xy[al], axis=1)
        self._xy = qpos[:, :2].copy()
        snap = dict(qpos=qpos)
        info = dict(terminated=self.fallen.copy(), base_z=qpos[:, 2].copy())
        return snap, self.fallen.copy(), info, dict(sim_s=sim_s)

    # ------------------------------------------------------------------ tracking
    def tracking(self, per_frame_envs=()):
        """The evaluation's metrics per env (``evaluate.episode``) over this rollout, plus per-frame series for
        ``per_frame_envs``."""
        R = self.rows
        if not R:
            return None
        t = np.array([r["t"] for r in R])
        cmd = np.stack([r["cmd"] for r in R])
        vx, wz = np.stack([r["vx"] for r in R]), np.stack([r["wz"] for r in R])
        pitch, roll = np.stack([r["pitch"] for r in R]), np.stack([r["roll"] for r in R])
        fallen = np.stack([r["fallen"] for r in R])
        settled = np.array([since_change(self.schedule, x) > SETTLE_S for x in t])
        per = {}
        for e in range(self.num_envs):
            ok = ~fallen[:, e]
            s = settled & ok
            ex, ew = vx[s, e] - cmd[s, 0], wz[s, e] - cmd[s, 2]
            per[str(e)] = dict(
                seed=self.seeds[e], fallen=bool(self.fallen[e]),
                fall_s=None if np.isnan(self.fall_t[e]) else round(float(self.fall_t[e]), 3),
                vx_rmse=round(float(np.sqrt(np.mean(ex ** 2))), 4) if len(ex) else None,
                wz_rmse=round(float(np.sqrt(np.mean(ew ** 2))), 4) if len(ew) else None,
                pitch_max_deg=round(float(np.degrees(np.abs(pitch[ok, e]).max())), 2),
                roll_max_deg=round(float(np.degrees(np.abs(roll[ok, e]).max())), 2),
                path_m=round(float(self.dist[e]), 3))
        good = [p for p in per.values() if not p["fallen"]]
        med = lambda k: round(float(np.median([p[k] for p in good])), 4) if good else None
        p95 = lambda k: round(float(np.percentile([p[k] for p in good], 95)), 2) if good else None
        summary = dict(envs=self.num_envs, falls=int(self.fallen.sum()), vx_rmse_median=med("vx_rmse"),
                       wz_rmse_median=med("wz_rmse"), vx_rmse_max=max((p["vx_rmse"] for p in good), default=None),
                       wz_rmse_max=max((p["wz_rmse"] for p in good), default=None),
                       pitch_max_p95_deg=p95("pitch_max_deg"), roll_max_p95_deg=p95("roll_max_deg"),
                       seconds=round(float(t[-1]), 3))
        frames = {}
        for e in per_frame_envs:
            e = int(e)
            frames[str(e)] = dict(vx=np.round(vx[:, e], 4).tolist(), wz=np.round(wz[:, e], 4).tolist(),
                                  vx_err=np.round(vx[:, e] - cmd[:, 0], 4).tolist(),
                                  wz_err=np.round(wz[:, e] - cmd[:, 2], 4).tolist(),
                                  pitch_deg=np.round(np.degrees(pitch[:, e]), 2).tolist(),
                                  roll_deg=np.round(np.degrees(roll[:, e]), 2).tolist(),
                                  fallen=fallen[:, e].astype(int).tolist())
        return dict(
            definition=("evaluate.episode's metrics at one sample per film frame (20 ms; the evaluation samples every "
                        "10 ms): vx = base forward speed in the heading frame, wz = base yaw rate (Tron1Sim.truth); "
                        "errors are measured minus the command sent (the schedule, each change ramped over ramp_s; "
                        "complete on every frame the RMS counts); RMS over frames at least 0.5 s after the last command "
                        "change and before a fall; pitch / roll peaks of |angle| before a fall; fall = Tron1Sim's fall "
                        "check (tilt > 70 deg, or base / sensor pack / leg link on the ground). Times are film times, "
                        "after the pre-roll (pre_roll_s), which no metric includes. Summary: medians, maxima and 95th "
                        "percentiles over envs that did not fall."),
            frame_dt=self.frame_dt, settle_s=SETTLE_S, pre_roll_s=self.pre_roll_s, ramp_s=self.ramp_s,
            schedule=[[t0, list(c)] for t0, c in self.schedule],
            per_frame=dict(t=np.round(t, 3).tolist(), cmd_vx=np.round(cmd[:, 0], 3).tolist(),
                           cmd_wz=np.round(cmd[:, 2], 3).tolist(), settled=settled.astype(int).tolist(),
                           falls=fallen.sum(1).astype(int).tolist(), envs=frames),
            per_env=per, summary=summary)

    def summary(self, ends) -> str:
        tr = self.tracking()
        rows = [f"{e}:{p['vx_rmse']}/{p['wz_rmse']}" + (f"@{p['fall_s']}s" if p["fallen"] else "")
                for e, p in tr["per_env"].items()]
        s = tr["summary"]
        return (f"[record] tracking, env:vx rmse m/s/wz rmse rad/s[@fall]: {' '.join(rows)}\n[record] falls "
                f"{s['falls']}/{s['envs']}; median vx rmse {s['vx_rmse_median']} m/s, wz rmse {s['wz_rmse_median']} "
                f"rad/s; pitch p95 {s['pitch_max_p95_deg']} deg, roll p95 {s['roll_max_p95_deg']} deg")

    def rank(self, first):
        """Envs that did not fall first; among them the most typical robot first: the smallest distance of its
        (vx RMSE, yaw-rate RMSE), each divided by its median, from the medians (1, 1). Not the best tracker."""
        tr = self.tracking()
        per = tr["per_env"]
        good = [e for e in range(self.num_envs) if not per[str(e)]["fallen"]]
        mv = np.median([per[str(e)]["vx_rmse"] for e in good]) if good else 1.0
        mw = np.median([per[str(e)]["wz_rmse"] for e in good]) if good else 1.0
        dist = {e: float(np.hypot(per[str(e)]["vx_rmse"] / mv - 1, per[str(e)]["wz_rmse"] / mw - 1)) if e in good
                else np.inf for e in range(self.num_envs)}
        order = sorted(range(self.num_envs), key=lambda e: (per[str(e)]["fallen"], dist[e], e))
        if self.film_envs is None:
            self.film_envs = order[:1]
        return order

    def twin_spec(self):
        """The simulated model itself (``metalsim.tron1.sim.build_spec`` at this film's realism level): LimX's MJCF
        with LimX's STL visual meshes and colours, the realism layer's tyre colliders (hidden) and the sensor pack
        (an enclosure box and a lidar cylinder, drawn: they are colliding geoms of the simulated robot); the mesh
        directory made absolute so the exported twin finds the meshes."""
        from metalsim.tron1.sim import WF_XML, build_spec

        spec = build_spec(self.make_params())
        spec.meshdir = os.path.normpath(os.path.join(os.path.dirname(str(WF_XML)), spec.meshdir or ""))
        return spec

    def meta(self):
        import mujoco

        from metalsim.render.film import provenance

        p = self.make_params()
        m = self.sims[0].m
        out = dict(policy=provenance.relpath(self.checkpoint), policy_format=self.ckpt_format,
                   policy_source=self.ckpt_source, policy_files_sha256=self.files_sha256,
                   limx_deploy=LIMX_POLICY,
                   policy_io=("metalsim/tron1/limx_policy.py (port of LimX's controllers/WheelfootController.py at "
                              f"{LIMX_COMMIT[:7]}): 28 observations, 10-frame history -> encoder.onnx (3), then "
                              "policy.onnx on [encoder 3, obs 28, commands x (2, 2, 0.25)] -> 8 actions at 50 Hz; legs "
                              "q = 0.25 a, Kp 42 / Kd 2.5; wheels dq = 0.8 a via Kd 0.8; LimX's torque-bound action "
                              "clips (80 / 40 N m)"),
                   physics=(f"metalsim.tron1.sim.Tron1Sim on CPU MuJoCo {mujoco.__version__} (LimX's WF_TRON1A MJCF, "
                            f"physics step {m.opt.timestep * 1e3:.1f} ms) with MetalSim's TRON1 realism layer at "
                            f"'{self.level}' (metalsim.tron1.realism.SimParams.{self.level}()); controller "
                            f"{CONTROL_HZ} Hz, policy 50 Hz; one Tron1Sim per env, seeds {self.seeds[0]}.."
                            f"{self.seeds[-1]}"),
                   realism_level=self.level, sim_params={k: _plain(v) for k, v in vars(p).items()},
                   schedule=[[t0, list(c)] for t0, c in self.schedule], control_hz=CONTROL_HZ,
                   pre_roll_s=self.pre_roll_s, ramp_s=self.ramp_s,
                   procedure=(f"each robot placed on its wheels with straight legs (LimxPolicyController.initial_q, "
                              f"q = 0), then {self.pre_roll_s:g} s standing under the policy at the schedule's first "
                              f"command before the film's t = 0; every command change ramped linearly over "
                              f"{self.ramp_s:g} s from the previous command"),
                   env_build_s=self.build_s, seeds=self.seeds,
                   actions="LimX's policy output (deterministic); noise from each env's Tron1Sim seed",
                   done="Tron1Sim's fall check (tilt > 70 deg, or base / sensor pack / leg link touching the ground); "
                        "no time-out")
        if hasattr(self, "rows"):
            out["tracking"] = self.tracking(self.film_envs or [])
            out["first_episode"] = dict(start=self.start.round(4).tolist(), path_m=self.dist.round(3).tolist())
        return out


# ---------------------------------------------------------------------------------------------- the film entry
PH_CC0 = "CC0 1.0 (Poly Haven, polyhaven.com/license; fetched, not committed)"
FETCH_PH_HELDOUT = "scripts/fetch_polyhaven_assets.sh heldout"
KIT_FOVY_16_9 = 36.0     # tasks.KIT_FOVY_16_9: Kit's default perspective lens
LIMX_BLOB = f"{LIMX_DEPLOY}/blob/{LIMX_COMMIT}"
LIMX_DESC = "https://github.com/limxdynamics/robot-description"

# Camera: follows base_Link (origin about 0.79 m up); looks at 0.35 m below it, the middle of the 0.95 m tall robot
# (wheels to sensor pack). The G1 films' azimuth and Kit's lens, pitched down 12 deg (the G1 films: 21 deg) so the top
# sixth of the frame shows the machine shop above the horizon; hero 2.57 m away (the robot fills ~0.55 of the frame
# height, as the G1 hero), readme 0.75 x closer (~3/4) with the 10 deg/s orbit (tasks.G1_README_VIEWER).
TRON1_VIEWER = ViewerCfg(eye=(1.99, -1.53, 0.18), lookat=(0.0, 0.0, -0.35), origin_type="asset_body",
                         body_name="base_Link", fovy=KIT_FOVY_16_9, resolution=(1280, 720), smooth_s=1.0)
TRON1_README_VIEWER = ViewerCfg(eye=(1.49, -1.15, 0.05), lookat=(0.0, 0.0, -0.35), origin_type="asset_body",
                                body_name="base_Link", fovy=KIT_FOVY_16_9, resolution=(640, 480), smooth_s=1.0,
                                orbit_deg_s=10.0)
# Fixed exposure, a grey card's: the automatic rule (frame median luminance to 0.2 before ACES) exposes this scene's
# first frame at 1.68 (hero) / 1.75 (readme), because the frame is mostly a dark floor (concrete_floor_worn_001: median
# linear albedo 0.095; the G1 films' laminate: 0.237). 0.9 = that rule with the floor taken as mid grey
# (1.68 x 0.095 / 0.18 = 0.89; GPU preview of hero frame 0, 2026-11-01). The earlier 1.4 came from the same rule on
# frames whose floor the film tool drew darker (times LimX's floor rgba, 0.5 / 0.6 / 0.7; render.add_ground_material).
TRON1_EXPOSURE = 0.9
# concrete_floor_worn_001: 3000 x 3000 mm of floor per image (Poly Haven "dimensions"; LookCfg.ground_tile_m)
CONCRETE_TILE_M = 3.0

# Colours. The real TRON1 is black: charcoal-black satin shells (body, hip actuators, thigh and shank links) with thin
# orange stripes, black treaded tyres on dark hubs (LimX's product images, e.g. its TRON 1 launch news
# limxdynamics.com/en/news/BK000040, and robotsguide.com/robots/tron-1). LimX's MJCF `rgba` (body 0.79 / 0.82 / 0.93,
# thighs white, wheels 0.27) and its URDF's (wheels 0.70) are display colours: LimX's own simulator renders the robot
# white. Linear albedo, the omnipbr base colour (specular 0.5: F0 0.04, a dielectric's):
# - shells 0.03: black ABS 0.015 to 0.03 (USGS spectral library splib06, GDS341 black ABS pipe, "overall reflectance
#   1.5 to 3 %"); blackboard paint 0.039 (physicallybased.info "Blackboard", spectraldb.com measurement 00079); charcoal
#   and fresh asphalt 0.04 measured with their specular, so a diffuse colour "a little less" (Lagarde, "Feeding a
#   physically based shading model", 2011). Roughness 0.5: the photos show a satin sheen (soft highlights along the
#   shell and leg edges), not a gloss; from the photos, not measured. Painted or anodised aluminium is a dielectric
#   coating, so metallic 0.
# - tyres 0.023, roughness 0.7: physicallybased.info "Tire". LimX's wheel is one mesh, so the hub is drawn in the
#   tyre's rubber (the real hub is dark too).
# - the sensor pack's enclosure (generic, not LimX's; metalsim/tron1/sim.py): black, as the shells (assumed: no source
#   gives its colour); the MID-360 lidar: bare aluminium, metallic, colour 0.916 / 0.923 / 0.924 (physicallybased.info
#   "Aluminum"), roughness 0.45 (bead-blasted, as on Livox's product image livoxtech.com/mid-360; its dark dome window,
#   the top third, is not modelled).
# LimX's visual geoms are unnamed in its MJCF: the record stage names them _film_geom_<index in the spec>
# (record.name_geoms); the index -> LimX mesh table below is twin.xml's (checked against runs/film/tron1/rec/twin.xml).
TRON1_VISUAL_GEOMS = {1: "base_Link", 3: "abad_L_Link", 5: "hip_L_Link", 7: "knee_L_Link", 9: "wheel_L_Link",
                      11: "abad_R_Link", 13: "hip_R_Link", 15: "knee_R_Link", 17: "wheel_R_Link"}
TRON1_SHELL = dict(rgba=(0.03, 0.03, 0.03, 1.0), roughness=0.5, metallic=0.0, specular=0.5)
TRON1_TYRE = dict(rgba=(0.023, 0.023, 0.023, 1.0), roughness=0.7, metallic=0.0, specular=0.5)
TRON1_LIDAR = dict(rgba=(0.916, 0.923, 0.924, 1.0), roughness=0.45, metallic=1.0, specular=0.5)


def _visual_geoms(*links) -> str:
    """Name regex of the twin's LimX visual geoms whose mesh is one of ``links`` (``TRON1_VISUAL_GEOMS``)."""
    return "^_film_geom_(" + "|".join(str(i) for i, m in TRON1_VISUAL_GEOMS.items() if m.split("_")[0] in links) + ")$"


TRON1_MATERIALS = {_visual_geoms("base", "abad", "hip", "knee"): TRON1_SHELL, _visual_geoms("wheel"): TRON1_TYRE,
                   r"^pack_box$": TRON1_SHELL, r"^pack_lidar$": TRON1_LIDAR}

LIMX_SHA = LIMX_POLICY["sha256"]
TRON1_STORY = f"""
**The policy.** LimX's RL policy for the wheel-foot TRON1 (WF_TRON1A), from LimX's `tron1-rl-deploy-python`
(<{LIMX_DEPLOY}>, Apache-2.0, commit
[`{LIMX_COMMIT[:7]}`]({LIMX_DEPLOY}/commit/{LIMX_COMMIT})):
`controllers/model/WF_TRON1A/policy/isaacgym/encoder.onnx` and `policy.onnx` with `params.yaml`. The copies in
`assets/tron1/limx_policy/WF_TRON1A/` are byte-identical to LimX's at that commit (fetched and compared on 2026-10-31;
sha256 `{LIMX_SHA['policy/isaacgym/policy.onnx']}` for the policy,
`{LIMX_SHA['policy/isaacgym/encoder.onnx']}` for the encoder), and the adapter checks the hashes
before it runs. That repository is LimX's Python controller for its robots: `main.py` connects through LimX's SDK to
LimX's MuJoCo simulator by default, or to the robot when given its IP address (`main.py` lines 27-39), and loads
`policy/$RL_TYPE/` with `RL_TYPE` isaacgym or isaaclab (the README's example is isaacgym; the isaaclab export uses
another joint order, which the port does not handle). It is LimX's open policy, not the closed controller of the robot's
Remote Control Mode, which is not documented (docs/TRON1.md, "What runs the robot today"). It is played by
`metalsim/tron1/limx_policy.py`, a port of LimX's `controllers/WheelfootController.py` at the same commit: 28
observations (gyro x 0.25, projected gravity, six leg joint angles, eight joint velocities x 0.05, last actions), a
10-frame history into the encoder (3 outputs), then the policy on [encoder, observation, command x (2, 2, 0.25)] to 8
actions at 50 Hz; legs to `q = 0.25 a` with Kp 42 and Kd 2.5, wheels to `dq = 0.8 a` through Kd 0.8, with LimX's
torque-bound action clips (80 / 40 N m). Two differences from LimX's script, both in the port's docstring: commands are
scaled as in LimX's training rather than its joystick mapping, and LimX's stand mode (leg targets blended toward a fixed
pose before the policy takes over) is skipped, so the policy itself stands the robot up (see "Commands"). Nothing is
tuned for the film.

**MetalSim's TRON1 simulation, nominal settings.** The rollout is the TRON1 evaluation's own path
(`metalsim/tron1/evaluate.py`, `episode`): one `Tron1Sim` per robot (`metalsim/tron1/sim.py`), LimX's WF_TRON1A MJCF
on CPU [MuJoCo](https://github.com/google-deepmind/mujoco) with a 0.5 ms physics step, stepped by `Tron1Sim.run` with
the controller at 500 Hz. Between the policy and the physics is MetalSim's realism layer at its `nominal` level
(`SimParams.nominal()`, `metalsim/tron1/realism.py`; docs/TRON1.md, "Realism layer"), unchanged: commands reach the
drives 4 ms late plus 0.5 ms mean jitter, with 0.2 % packet loss; sensor messages 1 ms late; drives run LimX's mode-0
law on 17-bit encoder angles and 1 kHz finite-difference velocities, with a torque-speed envelope and a 0.5 ms current
lag; joint Coulomb friction 0.3 N m (legs) and 0.1 N m (wheels), viscous damping and rotor armature; IMU white noise,
bias random walk and AHRS tilt wander and yaw drift; a 1.5 kg sensor pack 8 cm above the base (drawn: the black box and
the aluminium lidar cylinder on top); ground friction 0.8; and the tyre as a smooth fit to the real 5 cm rounded tread
instead of the 1 cm disc of LimX's MuJoCo file. The full parameter set is in `rec.json` `adapter.sim_params`. Each robot
gets its own noise seed (0 to 15), as each evaluation episode does; no pushes.

**Commands.** The same schedule for every robot, within the evaluation's ranges (vx in [-1, 1] m/s, yaw rate in
[-0.6, 0.6] rad/s): stand until 0.5 s, then vx 1.0 m/s; at 2.0 s vx 0.8 m/s with a 0.6 rad/s left turn; at 3.5 s vx
0.4 m/s straight; at 4.5 s stop (`tron1.SCHEDULE`). The film follows a real robot's procedure: a robot is stood up
before it drives, and a joystick does not step. The simulation places each robot on its wheels with straight legs
(q = 0, LimX's default joint angles, which is not the policy's stance: started there, the policy drops the base from
0.91 to 0.68 m in about a quarter second and rocks for seconds), so before the film's t = 0 the policy
stands it up for 3 s at zero command (`tron1.PRE_ROLL_S`, in `Tron1Limx.reset`, through `Tron1Sim.run` itself); and
every command change ramps linearly over 0.5 s from the previous command (`tron1.RAMP_S`) instead of stepping. The
pre-roll is not filmed and no metric includes it. At the film's t = 0 the filmed robot stands at its own stance
height, 0.77 m: over the 0.5 s before the drive its base moves 6.9 mm peak to peak and its pitch stays within 3.1 to
4.8 deg (`rec.npz` `track`, `rec.json` per-frame pitch), as still as it stands after the stop.

**Measured before filming.** 16 robots for 8 s with this schedule and procedure (`Tron1Limx` itself, nominal
settings): **0 of 16 fell**. Forward-speed tracking error 0.269 m/s RMS (median over the robots; 0.260 to 0.291),
yaw-rate error 0.138 rad/s (0.135 to 0.140), peak pitch 10.3 deg and peak roll 1.4 deg (95th percentiles), with the
evaluation's definitions (`evaluate.episode`: RMS from 0.5 s after each command change, here against the command the
robot was sent, which has finished its ramp by then; sampled every 20 ms). For comparison, the evaluation's nominal row
(100 episodes x 20 s of random stepped commands and two shoves each) is 0 / 100 falls, 0.27 m/s, 0.17 rad/s, 15 deg,
5 deg (docs/TRON1.md, "Results"). How it tracks, medians over the 16 robots: the policy accelerates slowly, 0.58 m/s
from 1.0 to 1.5 s after the drive command starts to rise and 0.8 m/s first at about 2.15 s; in the turn it yaws at
0.91 rad/s against the 0.6 commanded (76 deg of heading from 2.0 to 4.5 s against the 52 deg commanded) at 0.71 m/s; it
holds 0.45 m/s against 0.4; stopped, it creeps at 0.04 m/s. The 16 robots differ only in their sensor and transport
noise, and their errors agree to within 0.031 m/s and 0.004 rad/s. The procedure trades some speed tracking for a
calmer robot: the same measurement started from straight legs with stepped commands (`adapter_kw` `pre_roll_s=0,
ramp_s=0`, how this film's earlier take was made) gives 0.240 m/s and 0.130 rad/s, but 14.3 deg peak pitch and
3.4 deg peak roll. The 7 s take is the first 7 s of the same rollout (frame for frame: the per-frame speeds of the
filmed robot are identical): 0 of 16 fell, 0.296 m/s and 0.152 rad/s RMS (`rec.json` `adapter.tracking`; the shorter
window has less of the stop, where the error is small). `rec.json` `adapter.tracking` also holds the filmed robot's
per-frame command, speed, yaw rate, errors, pitch and roll, and each robot's fall time (none).

**How the robot was chosen.** `envs="auto"` ranks robots that did not fall first, then the most typical tracker:
the smallest distance of its (forward-speed, yaw-rate) RMS errors, each divided by the median over the robots, from
(1, 1) (`Tron1Limx.rank`). That was robot 9 (0.296 m/s, 0.152 rad/s in the take). Both shots film it.

**Camera.** Both shots track `base_Link` (Isaac's `origin_type="asset_body"`), the tracked point smoothed by a centred
1 s moving average, with Kit's default lens (36 deg vertical field of view) and the G1 films' azimuth, looking at
0.35 m below the base origin (the middle of the 0.95 m robot, tyres to lidar). `hero` (1280x720; the video and
stills on this page): eye (1.99, -1.53, 0.18) m from the base, 2.57 m from the look-at point, pitched down 12 deg (the
G1 films: 21 deg) so the top sixth of the frame shows the machine shop above the horizon; the robot fills about 0.55
of the frame height. `readme` (640x480, 4:3, for the README grid): the same direction at 0.75 times the distance
(eye (1.49, -1.15, 0.05) m), so the robot fills about three quarters of the frame height, plus the slow 10 deg/s orbit
of the G1 films' README shot (symmetric about the middle of the take, as MuJoCo Warp's README clips orbit).
"""


def tron1_notes(take) -> str:
    """TRON1's notes: the story, then rendering, README clip, the eye check and this take, whose floor size, day,
    machine, commits, fork pins and power are read from the take's stage logs (``doc.Take``)."""
    t = take.ground_tile_m           # the tile of this take's frames (None: the logs do not show it)
    dims = f"{CONCRETE_TILE_M:g} m x {CONCRETE_TILE_M:g} m"
    laid = "" if t is None else (
        f" and is laid at {'that size, ' if abs(t - CONCRETE_TILE_M) < 1e-9 else ''}one image per {t:g} m "
        f"(`ground_tile_m` {t!r}; the film tool\nsets MuJoCo's `texrepeat` to 2 / {t:g}, as MuJoCo lays a `texuniform` "
        "texture once per 2 / `texrepeat` m: an infinite\nplane's texture coordinates are half its world position, "
        "MuJoCo's `makePlane`, mirrored in\n`metalsim/render/builtin_geoms.py`; `metalsim/render/film/render.py`, "
        "`ground_texrepeat`)")
    mjw, warp = take.fork_at("mujoco_warp"), take.fork_at("warp", n=8)
    rendered_at = take.commit_md("metalsim", "hero") + (
        " plus uncommitted files, this film's colours and exposure among them" if take.dirty("hero") else "")
    this_take = take.fill(f"""
**This take, and where its timings come from.** Recorded, rendered and encoded on {take.day} on the {take.machine}; the
record stage ran at metalsim {take.metalsim_at("this film's uncommitted pre-roll and ramps and other films' uncommitted film-tool "
                                "work, whose new settings (`after_end`, `rank_keeps_terminated`) stay at their defaults "
                                "here")}, with the main venv's checkout of {mjw}
({warp}; the physics here is CPU MuJoCo, which neither checkout touches). This was not a timing run: quote
nothing on this page as MetalSim's speed (MetalSim's throughput figures are in `docs/PARITY.md` §1.4 and §1.8). The
record stage simulates on the CPU (one CPU MuJoCo `Tron1Sim` per robot, run one after another) and runs LimX's ONNX
files with [onnxruntime](https://github.com/microsoft/onnxruntime) 1.31.0 (CPU, one thread; the `tron1` extra of `pyproject.toml`, which the record stage needs:
`pip install -e '.[tron1]'`; the pins below list a fixed set of modules and omit it). Power, as the stages stamped it
("Hardware and pins" below): {take.power_prose()}. The record ran with a first render as one GPU-queue job (`scripts/gpu_run.sh`) of kind
render, into `runs/film/tron1`: `--stage record`, then a check that the recording was new and made with the pre-roll and
ramps, then `--stage render`. That render drew the robot in the MJCF's colours, white. A second `--stage render` of
the same recording drew it in the robot's real colours, once the film tool wrote a `look.materials` colour into the
geom's own `rgba` as well (`render.build_scene`; MuJoCo draws a geom's own `rgba` over its material's), but it drew
the concrete times LimX's floor `rgba` (0.5 / 0.6 / 0.7), at exposure 1.4. The frames on this page are a third (both shots,
a GPU-queue job of kind render of its own), at metalsim {rendered_at}: the film tool now also writes the ground tint
into the floor's `rgba`, so the concrete is in its own colours, and the exposure is the grey card's 0.9 ("Exposure"
above); the render cache key records the `rgba` writes (`render.geoms_recoloured`), so every frame was redrawn. Then
`--stage encode --shots hero`, the README WebP command above and `--stage doc` (CPU). The recording replaces the
film's earlier take (`rec.json` written 2026-10-31 at 16:05, at [`ed66d8ef`](https://github.com/pulipakaa24/MetalSim/commit/ed66d8ef3ecd7f17aa16207231000b1d22ff7812) plus the uncommitted TRON1 film code),
which started each robot from straight legs with stepped commands.""")
    return TRON1_STORY + f"""
**Rendering.** Tier 2 (the path tracer), 64 samples per pixel (16 passes x 4), 3 bounces,
[Open Image Denoise](https://www.openimagedenoise.org), ACES tone map, sRGB; lit by Poly Haven's `machine_shop_02` (an
industrial hall, 2k) as an importance-sampled dome, with Poly Haven's `concrete_floor_worn_001` (1k diffuse) on
LimX's `floor` plane in place of its checker, in the texture's own colours (the film tool writes the look's ground tint,
white here, into the plane's `rgba`, which LimX sets to 0.5 / 0.6 / 0.7 and MuJoCo would otherwise multiply into the
texture). The texture is {dims}
of real floor (Poly Haven's dimensions){laid}. The render twin is the
simulated model itself (`build_spec` at the nominal level), posed per frame from the recorded state: LimX's STL meshes,
the sensor pack's two geoms, and the colliders hidden (`rec.json` `twin.visibility`).

**Colours.** The TRON1 is black. LimX's own product images (its TRON 1 launch news,
<https://www.limxdynamics.com/en/news/BK000040>, and the photos at <https://robotsguide.com/robots/tron-1>) show
charcoal-black satin shells on the body, the hip actuators and the leg links, with thin orange stripes, and black
treaded tyres on dark hubs. LimX's description files carry other colours: the MJCF's `rgba` (the body 0.79 / 0.82 /
0.93, the thighs white, the wheels 0.27) and the URDF's (the wheels 0.70) are display colours, and LimX's own simulator
draws the robot white. This film's earlier renders used the MJCF's `rgba` as linear albedo, so the robot was white.
Here the film's look sets each part (`look.materials`, `metalsim/render/film/tron1.py`), as linear albedo: the base
colour of the renderer's OmniPBR-style material, under a dielectric coat that reflects 4 % at normal incidence.
- The shells: 0.03, roughness 0.5, not metallic. Black ABS measures 1.5 to 3 %
  ([USGS spectral library splib06](https://doi.org/10.3133/ds231), GDS341, black ABS pipe), blackboard paint 0.039
  (<https://physicallybased.info>, "Blackboard"). The satin roughness is read from the photos, not measured. A painted
  or anodised metal part is a dielectric coat, so not metallic either.
- The wheels: 0.023, roughness 0.7 (physicallybased.info, "Tire"). LimX's wheel is one mesh, so the hub is drawn in
  the tyre's rubber.
- The sensor pack's enclosure: as the shells. Its colour is assumed: the enclosure is a generic estimate, not LimX's.
- The MID-360 lidar: bare aluminium, metallic, colour 0.92 (physicallybased.info, "Aluminum"), roughness 0.45. Livox's
  product image (<https://www.livoxtech.com/mid-360>) shows a silver housing; its dark dome window is not modelled.
- Not drawn: the orange stripes and LimX's lettering. An STL mesh carries no texture coordinates, so each part is one
  colour.

**Exposure** is fixed at 0.9, a grey card's: the films' automatic rule (frame median luminance to 0.2 before ACES)
gives 1.68 for this scene's first frame (1.75 for the README shot's), because the frame is mostly a dark floor (the
concrete's median linear albedo is 0.095, the G1 films' laminate 0.237); 0.9 is that rule with the floor taken as a mid
grey (1.68 x 0.095 / 0.18). In hero frame 175 the body's side shows at about a quarter of the floor's display luminance
(sRGB about 54 against 104), with its sheen and edges visible. It reads charcoal rather than ink black, because its
coat reflects about as much of the bright hall as its 0.03 base colour adds. The earlier renders used 1.4: the same
rule, on frames whose concrete the film tool drew times LimX's floor `rgba`, darker and bluer than it is.

**README clip.** [tron1.webp](tron1.webp) is the `readme` shot from 0.5 s to 5.5 s (frames 25 to 274: drive, turn, slow
down, stop), resampled from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos), lossy
animated WebP from [Pillow](https://github.com/python-pillow/Pillow) at quality 70, 33 ms per frame, looping; 150
frames. Command, run from the repository root after the render stage: `python -m metalsim.render.film.encode webp
runs/film/tron1/readme/frames docs/renders/tron1.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30`.

**Checked by eye.** The robot starts standing: in hero frames 0 to 11 it holds one crouched stance with both tyres on
their contact shadows, and in hero frames every 0.2 s from 0 to 3 s the top of the sensor pack stays on one line. The
wheels turn visibly: their three-spoke hubs rotate a few degrees per frame while driving (`rec.npz` wheel-mesh poses
relative to the knee: 3.8 deg per frame from 1 to 2 s; in the turn, from 2.5 to 3.5 s, 4.8 deg for the inner wheel and
7.8 deg for the outer; 0.3 deg per frame once stopped).
Scale: each wheel's 0.1298 m rim, projected from its recorded pose through the film camera, lies on the rendered tyre
outline (hero frames 88 and 175, readme frame 150). The colours, on these frames:
- Hero crops every 1 s (frames 0 to 350), and frames 0 and 175 at native resolution, show the body's vents and carry
  handle, the hip actuators, the thigh and shank links, the knee joints and the hub spokes. The robot is not a
  silhouette.
- The README WebP, every 15th frame: a dark robot against the floor through the drive, the turn and the stop.
- The floor is the concrete's own warm grey (sRGB about 105 / 103 / 96 in two boxes either side of the robot in hero
  frame 175; the second render's, times LimX's floor `rgba`, was bluer: 98 / 105 / 106).
- Against the earlier renders of the same recording, every logged render setting is the same (camera, look, samples
  and twin geoms) except the exposure.

{this_take}
"""


TRON1 = FilmTask(
    name="tron1",
    adapter="metalsim.render.film.tron1:Tron1Limx",
    venv="main",
    checkpoint=f"{LIMX_POLICY['local_dir']}/{LIMX_POLICY['policy']}",
    adapter_kw={"params": "nominal", "pre_roll_s": PRE_ROLL_S, "ramp_s": RAMP_S},
    title="TRON1 wheel-legged drive, LimX's policy",
    summary=("LimX's TRON1 wheel-legged robot (WF_TRON1A) driven by LimX's own RL policy (the ONNX policy of LimX's "
             "[`tron1-rl-deploy-python`](https://github.com/limxdynamics/tron1-rl-deploy-python), Apache-2.0) in MetalSim's TRON1 simulation at its nominal realism settings "
             "(latency, drive model, sensor noise, joint friction, sensor pack), on LimX's own model and meshes: "
             "drive forward, turn, slow down, stop. Rendered on the 14-inch MacBook Pro (Mac16,6), not in a timing run, "
             "so its Performance table is not a timing figure."),
    notes=tron1_notes,
    viewer=TRON1_VIEWER,
    film=FilmCfg(num_envs=16, envs="auto", grid=1, seed=0, max_seconds=7.0, hold_s=0.0, passes=16, spp=4, bounces=3,
                 output_filename_prefix="tron1"),
    look=LookCfg(hdri="hdri/heldout/machine_shop_02.hdr", sun=False,
                 ground_texture="tex/heldout/concrete_floor_worn_001.png", ground_geoms=("floor",),
                 ground_tile_m=CONCRETE_TILE_M, materials=TRON1_MATERIALS,
                 exposure=TRON1_EXPOSURE),
    assets=[
        AssetRef("LimX TRON1 wheel-foot model WF_TRON1A (MJCF robot.xml and STL meshes)", "LimX Dynamics "
                 "robot-description", LIMX_DESC, "Apache-2.0 (assets/tron1/LICENSE-robot-description)",
                 "in this repository: assets/tron1/WF_TRON1A"),
        AssetRef("Policy: LimX's WF_TRON1A RL policy, isaacgym export (encoder.onnx + policy.onnx, params.yaml; "
                 f"sha256 {LIMX_POLICY['sha256']['policy/isaacgym/policy.onnx'][:8]}...)", "LimX Dynamics "
                 "tron1-rl-deploy-python", f"{LIMX_BLOB}/{LIMX_POLICY['upstream_dir']}",
                 "Apache-2.0 (assets/tron1/LICENSE-rl-deploy-python)",
                 "in this repository: assets/tron1/limx_policy/WF_TRON1A"),
        AssetRef("Controller port (WheelfootController.py -> metalsim/tron1/limx_policy.py)", "LimX Dynamics "
                 "tron1-rl-deploy-python", f"{LIMX_BLOB}/controllers/WheelfootController.py",
                 "Apache-2.0 (assets/tron1/LICENSE-rl-deploy-python)", ""),
        AssetRef("Industrial HDRI (machine_shop_02, 2k)", "Poly Haven", "https://polyhaven.com/a/machine_shop_02",
                 PH_CC0, FETCH_PH_HELDOUT),
        AssetRef("Floor texture (concrete_floor_worn_001, 1k diffuse)", "Poly Haven",
                 "https://polyhaven.com/a/concrete_floor_worn_001", PH_CC0, FETCH_PH_HELDOUT),
    ],
    shots=[ShotCfg("hero", 1), ShotCfg("readme", 1, TRON1_README_VIEWER)],
    doc="docs/renders/TRON1.md",
)
