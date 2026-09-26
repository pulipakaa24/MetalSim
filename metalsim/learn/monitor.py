"""Generic anomaly monitor for RL training on MetalSim: task-agnostic checks on the learning signals
and on the physics state, evaluated at every logging interval and written as `[anomaly]` lines and a
JSON trail. The rules are the standard RL-debugging ones (Schulman, Nuts and Bolts of Deep RL; Andy
Jones, Debugging RL Systems; Huang et al., 37 implementation details of PPO) and the physics
invariants used to validate simulators (energy/momentum drift, penetration depth, contact-force
plausibility). None of them encodes a specific past bug; each states what a healthy run looks like.

Learning-signal rules (per update):
  * KL per update should sit near the target: flag < desired/10 (no learning) or > desired x 10 (stale data)
  * explained variance of the value function should become positive and stay there after warm-up
  * policy entropy must not collapse (< 20 % of its initial value) nor explode (> 3x)
  * the learning rate must not stay pinned at its bound for many iterations
  * per-step reward decomposition: if the terminal-step reward dominates the episode
    (|terminal| > 5 x |sum of non-terminal rewards| once episodes exceed 50 steps), survival is being
    punished, and returns must not fall while episode length rises (negative return/length correlation)
  * termination mix: fraction of episodes ended by time-out vs failure vs blow-up
  * non-finite values anywhere in the rollout, rows dropped from the update
Physics rules (from the sim buffers at the log instant):
  * joint speeds vs the configured actuator velocity limits (flag > 3x the limit)
  * joint positions beyond their ranges by more than 0.05 rad (soft limits are penalties, not walls)
  * contact penetration deeper than 2 cm; contact forces above 50 x the robot's weight
  * mechanical energy (MuJoCo's energy flag) rising by more than the actuator work bound in one step
  * MuJoCo Warp overflow flags (constraint / contact capacity) and non-finite state
"""
from __future__ import annotations

import json, math, time
import numpy as np
import torch
import warp as wp


class AnomalyMonitor:
    def __init__(self, task, cfg, log=print, path=None, vel_limit_rad_s=37.0, ent_ref=None):
        self.task, self.cfg, self.log, self.path = task, cfg, log, path
        self.vel_limit = vel_limit_rad_s
        self.ent0 = ent_ref
        self.hist = []
        self.pinned = 0
        m = task.model
        self.mass = float(m.body_subtreemass[1]) if m.nbody > 1 else 1.0
        self.jnt_range = np.array([m.jnt_range[j] for j in range(1, m.njnt)]) if m.njnt > 1 else None
        self.limited = np.array([m.jnt_limited[j] for j in range(1, m.njnt)], bool) if m.njnt > 1 else None
        self.prev_energy = None

    # -- learning signals --------------------------------------------------------------------------
    def learning(self, it, algo, stats, ep_len_now=0.0):
        cfg = self.cfg; flags = []
        kl = stats.get("kl", float("nan"))
        if cfg.desired_kl:
            if kl > 10 * cfg.desired_kl: flags.append(f"KL {kl:.3f} > 10x target: stale data or LR too high")
            if it > 5 and 0 <= kl < cfg.desired_kl / 10: flags.append(f"KL {kl:.4f} < target/10: policy not moving")
        b = algo.bufs
        with torch.no_grad():
            v = b.t_value.float().reshape(-1); r = b.t_rew.float(); d = b.t_done.float()
            ret = torch.zeros_like(r); acc = torch.zeros(r.shape[1], device=r.device)
            for t in reversed(range(r.shape[0])):
                acc = r[t] + cfg.gamma * acc * (1 - d[t]); ret[t] = acc
            ret = ret.reshape(-1)
            ev = float(1 - (ret - v).var() / (ret.var() + 1e-8))
            ent = float(algo.net.log_std.exp().mean() if hasattr(algo.net, "log_std") else float("nan"))
            fin = bool(torch.isfinite(b.t_obs).all() and torch.isfinite(r).all())
            # reward decomposition per episode: terminal-step vs the rest
            term_mask = (d > 0)
            term_r = r[term_mask]; nonterm_sum = float(r[~term_mask].abs().sum()); n_term = int(term_mask.sum())
            survival_pen = (n_term > 0 and float(term_r.abs().sum()) > 5 * nonterm_sum / max(1, (~term_mask).sum().item() / 50))
        if it > 20 and ev < 0: flags.append(f"explained variance {ev:.2f} < 0 after warm-up: value function not predicting returns")
        if self.ent0 is None and math.isfinite(ent): self.ent0 = ent
        if self.ent0 and ent < 0.2 * self.ent0: flags.append(f"action std {ent:.3f} collapsed below 20% of initial")
        if self.ent0 and ent > 3 * self.ent0: flags.append(f"action std {ent:.3f} exploded above 3x initial")
        lr = getattr(algo, "lr", None)
        if lr is not None and (lr <= 1.05e-5 or lr >= 0.95e-2):
            self.pinned += 1
            if self.pinned >= 10: flags.append(f"learning rate pinned at bound ({lr:.1e}) for {self.pinned} iterations")
        else:
            self.pinned = 0
        if not fin: flags.append("non-finite observations or rewards in the rollout")
        if getattr(algo, "dropped_rows", 0): flags.append(f"{algo.dropped_rows} non-finite rows dropped from the update")
        ret_m, len_m, cnt = self.hist[-1]["ep"] if self.hist else (None, None, None)
        entry = {"it": it, "kl": kl, "explained_variance": ev, "action_std": ent, "lr": lr, "terminal_reward_mean": float(term_r.mean()) if n_term else None,
                 "nonterminal_reward_mean": float(r[~term_mask].mean()) if int((~term_mask).sum()) else None}
        if survival_pen and ep_len_now > 50: flags.append("terminal-step reward dominates the episode: survival is being punished")
        return entry, flags

    def curve(self, it, ep_ret, ep_len, count):
        """Return-vs-length direction over the last 10 log points once episodes are lengthening."""
        self.hist.append({"it": it, "ep": (ep_ret, ep_len, count)})
        flags = []
        pts = [h["ep"] for h in self.hist[-100:] if h["ep"][2]]      # long window: the trend, not the noise
        if len(pts) >= 30:
            lens = np.array([p[1] for p in pts]); rets = np.array([p[0] for p in pts])
            if lens[-1] > lens[0] * 1.3 and np.corrcoef(lens, rets)[0, 1] < -0.6:
                flags.append(f"returns fall as episodes lengthen (corr {np.corrcoef(lens, rets)[0, 1]:.2f}): the reward punishes survival")
        return flags

    # -- physics invariants ------------------------------------------------------------------------
    def physics(self):
        sim = self.task.sim; d = sim.d; flags = []
        sim.synchronize()
        qpos = d.qpos.numpy(); qvel = d.qvel.numpy()
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()): flags.append("non-finite qpos/qvel")
        ok = np.isfinite(qvel).all(1)
        if ok.any():
            jv = np.abs(qvel[ok][:, 6:]).max() if qvel.shape[1] > 6 else 0.0
            if jv > 3 * self.vel_limit: flags.append(f"joint speed {jv:.0f} rad/s > 3x actuator velocity limit ({self.vel_limit})")
        if self.jnt_range is not None and qpos.shape[1] > 7:
            jp = qpos[np.isfinite(qpos).all(1)][:, 7:7 + len(self.jnt_range) - 0]
            lim = self.limited[1:] if len(self.limited) == jp.shape[1] + 1 else self.limited[:jp.shape[1]]
            rng = self.jnt_range[1:] if len(self.jnt_range) == jp.shape[1] + 1 else self.jnt_range[:jp.shape[1]]
            over = np.maximum(rng[:, 0] - jp, jp - rng[:, 1]); over = over[:, lim]
            if over.size and over.max() > 0.05: flags.append(f"joint limit violated by {over.max():.2f} rad")
        if hasattr(d, "nacon"):
            na = int(d.nacon.numpy()[0])
            if na:
                dist = d.contact.dist.numpy()[:na]
                if dist.min() < -0.02: flags.append(f"contact penetration {-dist.min() * 100:.1f} cm")
        sd = d.sensordata.numpy() if d.sensordata.shape[1] else None
        if sd is not None and sd.size:
            peak = float(np.nanmax(sd[np.isfinite(sd).all(1)])) if np.isfinite(sd).any() else 0.0
            if peak > 50 * self.mass * 9.81: flags.append(f"contact force {peak:.0f} N > 50x robot weight")
        ov = sim.overflow_flags() if hasattr(sim, "overflow_flags") else {}
        # capacity overflows drop constraints or contacts silently (MuJoCo Warp OverflowType names; ITERATIONS /
        # LS_ITERATIONS are solver caps, not capacity, and are reported separately)
        for k in ("NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD", "EPA_HORIZON", "CONTACT_MATCH"):
            if k in ov: flags.append(f"MuJoCo Warp {k} capacity overflow in {ov[k]} worlds")
        if hasattr(d, "energy"):
            e = d.energy.numpy(); tot = e.sum(1) if e.ndim == 2 else e
            if self.prev_energy is not None and np.isfinite(tot).all():
                jump = float((tot - self.prev_energy).max())
                if jump > 500.0: flags.append(f"mechanical energy rose by {jump:.0f} J between log points")
            self.prev_energy = tot
        return {"blown": getattr(self.task, "blown_up_episodes", 0), "overflow": ov}, flags

    # -- callback ------------------------------------------------------------------------------------
    def __call__(self, it, algo, stats, ep):
        entry, flags = self.learning(it, algo, stats, ep_len_now=ep[1])
        flags += self.curve(it, *ep)
        phys, pflags = self.physics(); entry.update(phys); flags += pflags
        entry["flags"] = flags
        if self.path:
            with open(self.path, "a") as f: f.write(json.dumps(entry) + "\n")
        for fl in flags:
            self.log(f"[anomaly] it {it}: {fl}")
        return flags
