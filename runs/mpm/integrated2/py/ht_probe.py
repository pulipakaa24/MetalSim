"""Quantify the Metal HashTable race in the rough G1 Newton-engine task (contact-reduction hash table statistics).

record: free-run the task (seeded actions), save the state before every collide call
replay: load those states, run pipeline.collide on each (--repeat times), record statistics
run:    free-run, record per-collide statistics
"""
import argparse, json
import numpy as np
import warp as wp

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["record", "replay", "run"])
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--states", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--contact-max", type=int, default=0)
a = ap.parse_args()

import newton
if a.contact_max:
    _init = newton.CollisionPipeline.__init__
    def _patched(self, model, *args, **kw):
        kw.setdefault("rigid_contact_max", a.contact_max)
        _init(self, model, *args, **kw)
    newton.CollisionPipeline.__init__ = _patched
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import RolloutBuffers, bump, zero_int

dev = "metal:0"
task = G1VelocityTask(a.n, terrain="rough", seed=0, engine="newton", newton_iterations=4, newton_dt=0.00125)
sim = task.sim
red = sim.pipeline.narrow_phase.global_contact_reducer
ht = red.hashtable
print("warp", wp.__file__, "newton", newton.__file__, "table", ht.capacity, "rigid_contact_max", sim.contacts.rigid_contact_max, flush=True)
FIELDS = ("contacts", "active", "dup_listed", "distinct_keys", "dup_keys", "unlisted")

def stats():
    keys = ht.keys.numpy(); act = ht.active_slots.numpy()
    cnt = int(act[ht.capacity]); listed = np.unique(act[: min(cnt, ht.capacity)])
    nonempty = np.flatnonzero(keys != np.uint64(0xFFFFFFFFFFFFFFFF)); nk = keys[nonempty]
    return dict(contacts=int(sim.contacts.rigid_contact_count.numpy()[0]), active=cnt, dup_listed=cnt - len(listed),
                distinct_keys=len(np.unique(nk)), dup_keys=len(nk) - len(np.unique(nk)),
                unlisted=len(np.setdiff1d(nonempty, listed)))

orig = sim.pipeline.collide
log, states = [], []
if a.mode == "replay":
    st = np.load(a.states)
    for r in range(a.repeat):
        for bq, bqd in zip(st["body_q"], st["body_qd"]):
            sim.s0.body_q.assign(bq); sim.s0.body_qd.assign(bqd)
            orig(sim.s0, sim.contacts); wp.synchronize_device(dev)
            log.append(stats())
else:
    def hooked(state, contacts, *args, **kw):
        if a.mode == "record":
            wp.synchronize_device(dev)
            states.append((state.body_q.numpy().copy(), state.body_qd.numpy().copy()))
        orig(state, contacts, *args, **kw); wp.synchronize_device(dev)
        log.append(stats())
    sim.pipeline.collide = hooked
    n = task.n
    class _Pol:
        step_idx = wp.zeros(1, dtype=int, device=dev)
    pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device=dev); bufs.done = wp.zeros((1, n), dtype=float, device=dev)
    action = wp.zeros((n, task.act_dim), dtype=float, device=dev)
    rng = np.random.default_rng(1)
    task.reset_all()
    for t in range(a.steps):
        action.assign(rng.uniform(-1, 1, (n, task.act_dim)).astype(np.float32))
        wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=dev); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=dev)
        task.launch_apply_action(action); sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
        sim.synchronize()
    if a.mode == "record":
        np.savez_compressed(a.out + ".states.npz", body_q=np.array([s[0] for s in states]), body_qd=np.array([s[1] for s in states]))
L = np.array([[d[f] for f in FIELDS] for d in log])
summary = {f: float(L[:, j].mean()) for j, f in enumerate(FIELDS)}
summary["calls"] = len(L)
json.dump(summary, open(a.out + ".json", "w"), indent=1)
print("SUMMARY per collide:", {k: round(v, 2) for k, v in summary.items()}, flush=True)
