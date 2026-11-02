"""so101Sim's SO101LiftEnv (unchanged, imported from so101Sim) with a MetalSim servo option on CPU MuJoCo.

``attach(env, servo, **over)`` applies ``metalsim.render.film.so101.apply_servo`` to the env's MjModel and, for the
"_real" options, runs ``servo_layer`` (command delay + firmware setpoint profile) between the env's command and
``d.ctrl`` at every 5 ms physics step: ``env.step`` is wrapped to plan the control period's targets from the command
it writes, and the ``mujoco`` module seen by so101_lift.py is a proxy whose ``mj_step`` writes the planned target into
``d.ctrl`` first. Nothing in so101Sim is modified on disk. Returns the applied-servo dict.
"""
import types

import mujoco
import numpy as np

from metalsim.render.film.so101 import ARM_JOINTS, apply_servo, servo_layer

_PLANS = {}                     # id(MjData) -> dict(plan, k): every attached env in this process


def _mj_step(m_, d_, *args):
    st = _PLANS.get(id(d_))
    if st is not None and st["plan"] is not None:
        d_.ctrl[:] = st["plan"][min(st["k"], len(st["plan"]) - 1)]
        st["k"] += 1
    return mujoco.mj_step(m_, d_, *args)


def attach(env, servo: str, **over) -> dict:
    import so101_lift

    m, d = env.model, env.data
    applied = apply_servo(m, servo, **over)
    layer = servo_layer(applied, 1)
    if layer is None:
        return applied
    act = np.array(applied["actuators"])
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
    st = _PLANS.setdefault(id(d), dict(plan=None, k=0, data=d))

    def q_act():
        q = np.zeros(m.nu)
        q[act] = d.qpos[qadr]
        return q

    orig_reset, orig_step = env.reset, env.step

    def reset(*a, **kw):
        out = orig_reset(*a, **kw)
        layer.reset(q_act())
        st["plan"], st["k"] = None, 0
        return out

    def step(action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cmd = env._ctrl_lo + 0.5 * (a + 1.0) * (env._ctrl_hi - env._ctrl_lo)     # so101_lift.py step's mapping
        st["plan"], st["k"] = layer.plan(cmd[None], env._n_sub)[0], 0
        return orig_step(action)

    if not isinstance(so101_lift.mujoco, types.SimpleNamespace):
        proxy = types.SimpleNamespace(**{k: getattr(mujoco, k) for k in dir(mujoco) if not k.startswith("__")})
        proxy.mj_step = _mj_step
        so101_lift.mujoco = proxy
    env.reset, env.step = reset, step
    env._servo_layer = layer
    return applied
