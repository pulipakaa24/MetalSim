"""Throughput of the PhysX-protocol scenes at the fitted settings (one 5 ms frame = one env-step), graph replay on Metal.
usage: bench_protocol.py REC_DIR [N ...]"""
import sys, os, json, time, numpy as np, mujoco, warp as wp
wp.config.quiet = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physx_protocol as pp
from metalsim.physics import deformable as dfm

rec = sys.argv[1]; Ns = [int(x) for x in sys.argv[2:]] or [256, 1024, 4096]
_, meta = pp.physx_metrics(rec)
FIT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "physx_fit_5p1.json")))
DEV = "metal:0"

def timed(step, sync, n=10):
    step(); sync()
    t = time.perf_counter()
    for _ in range(n): step()
    sync()
    return (time.perf_counter() - t) / n

for (backend, obj), params in [((k.split("/")[0], k.split("/")[1]), v) for k, v in FIT.items()]:
    for N in Ns:
        try:
            if backend == "flex":
                p = pp._flex_params(obj, meta, params); dt = float(p["dt"])
                m = mujoco.MjModel.from_xml_string({"cloth": pp.cloth_xml, "rope": pp.rope_xml, "cube": pp.cube_xml}[obj](meta, p, dt))
                sub = int(round(1.0 / pp.HZ / dt))
                sim = dfm.DeformableSim(m, N, device=DEV, substeps=sub)
                t = timed(sim.step, sim.synchronize)
                extra = {"substeps_per_frame": sub, "nv": int(m.nv)}
            else:
                p = dict(pp.XPBD_DEFAULTS[obj]); p.update(params)
                fp = pp._flex_params(obj, meta, {})
                xml = pp.cloth_xml(meta, fp, 1.0 / pp.HZ) if obj == "cloth" else f"""<mujoco><option timestep="{1.0/pp.HZ}"/><worldbody><geom type="plane" size="3 3 0.1"/>
                  <flexcomp name="rope" type="grid" count="11 1 1" spacing="0.05 0.02 0.02" pos="0.25 0 1.0" dim="1" radius="0.01" mass="0.2"><pin id="0"/><edge equality="true"/></flexcomp></worldbody></mujoco>"""
                m = mujoco.MjModel.from_xml_string(xml)
                cfg = dfm.XPBDCfg(substeps=int(p["substeps"]), stretch_compliance=p["stretch_compliance"], bend_compliance=p["bend_compliance"],
                                  stretch_damping=p["stretch_damping"], bend_damping=p["bend_damping"], damping=p["damping"])
                sim = dfm.XPBDSim(m, N, device=DEV, cfg=cfg)
                t = timed(sim.step, sim.synchronize)
                extra = {"substeps_per_frame": int(p["substeps"]), "nvert": int(sim.nvert)}
            print(json.dumps({"backend": backend, "object": obj, "num_envs": N, "ms_per_frame": t * 1e3, "env_steps_per_s": N / t, **extra}), flush=True)
            del sim
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"backend": backend, "object": obj, "num_envs": N, "error": repr(e)[:200]}), flush=True)
