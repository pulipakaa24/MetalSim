"""Cross-world consistency checker: finds racy or nondeterministic kernels.

Runs a MuJoCo Warp step on N identical worlds with every ``wp.launch`` / ``wp.launch_tiled``
followed by a synchronize and a check that all per-world output arrays are identical across
worlds. Any kernel after which worlds differ is nondeterministic (a race, uninitialized memory,
or an order-dependent atomic). Usage: python -m metalsim.tools.race_finder scene.xml [state.npz]
"""
import sys

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw


def check_outputs(name, outputs, nworld, seen, names=None):
    for a in outputs:
        if not isinstance(a, wp.array) or a.size == 0 or not a.shape or a.shape[0] != nworld:
            continue
        v = a.numpy()
        if v.dtype.kind not in "fiub":
            continue
        ref = v[0]
        diff = ~np.isclose(v, ref[None], equal_nan=True) if v.dtype.kind == "f" else v != ref[None]
        bad = np.nonzero(diff.reshape(nworld, -1).any(1))[0]
        if len(bad):
            key = (name, id(a))
            if key not in seen:
                seen.add(key)
                label = (names or {}).get(a.ptr, "?")
                print(f"  DIVERGENCE after {name}: {label} shape {a.shape} dtype {a.dtype}, {len(bad)} worlds differ "
                      f"(e.g. {bad[:6]}), nan in output: {bool(np.isnan(v).any()) if v.dtype.kind == 'f' else False}")
                if v.ndim == 1:
                    print(f"    values world0={v[0]} others={v[bad[:6]]}")
            return True
    return False


def run(scene, state=None, nworld=64, device="metal:0", reps=3):
    m = mujoco.MjModel.from_xml_path(scene)
    orig_launch, orig_tiled = wp.launch, wp.launch_tiled
    log = []

    def make_hook(orig):
        def hook(kernel, *args, **kwargs):
            r = orig(kernel, *args, **kwargs)
            wp.synchronize_device(device)
            name = getattr(kernel, "key", getattr(kernel, "__name__", str(kernel)))
            hook.count += 1
            inputs = kwargs.get("inputs") or (args[1] if len(args) > 1 else None) or []
            outputs = kwargs.get("outputs") or (args[2] if len(args) > 2 else None) or []
            arrays = list(outputs) + [x for x in inputs if isinstance(x, wp.array)]
            if check_outputs(f"#{hook.count} {name}", arrays, nworld, hook.seen, hook.names):
                hook.first = hook.first or name
            return r
        hook.count, hook.seen, hook.first = 0, set(), None
        hook.names = names
        return hook

    with wp.ScopedDevice(device):
        for rep in range(reps):
            mw = mjw.put_model(m); mw.opt.graph_conditional = False; mw.opt.warn_overflow = 0
            d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
            dw = mjw.put_data(m, d0, nworld=nworld)
            if state is not None:
                st = np.load(state); e = int(st["e"])
                for k in ("qpos", "qvel", "act", "qacc_warmstart", "ctrl"):
                    getattr(dw, k).assign(np.ascontiguousarray(np.repeat(st[k][e:e + 1], nworld, axis=0)))
            mjw.forward(mw, dw)
            wp.synchronize_device(device)
            names = {}
            import dataclasses
            def walk(obj, prefix):
                for f in dataclasses.fields(obj):
                    v = getattr(obj, f.name)
                    if isinstance(v, wp.array):
                        names[v.ptr] = prefix + f.name
                    elif dataclasses.is_dataclass(v):
                        walk(v, prefix + f.name + ".")
            walk(dw, "d."); walk(mw, "m.")
            print(f"rep {rep}: stepping {nworld} identical worlds with per-launch checks")
            hl, ht = make_hook(orig_launch), make_hook(orig_tiled)
            wp.launch, wp.launch_tiled = hl, ht
            try:
                for _ in range(4):
                    mjw.step(mw, dw)
            finally:
                wp.launch, wp.launch_tiled = orig_launch, orig_tiled
            wp.synchronize_device(device)
            q = dw.qpos.numpy()
            print(f"  launches: {hl.count + ht.count}; first divergence: {hl.first or ht.first}; "
                  f"NaN worlds at end: {int((~np.isfinite(q).all(1)).sum())}")


if __name__ == "__main__":
    wp.config.quiet = True
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
