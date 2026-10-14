"""Cost of the 2026-10-13 render additions (docs/RENDER_COVERAGE.md section 3d), each against its switch, GPU path.

  haze       Menagerie Go2 scene (skybox + infinite plane): haze=True (default) vs False; T0 1024, T1 256, T2 64
  skirts     a raised 64 x 64-cell heightfield over a floor: skirt_shadows True vs False; T0 1024
  tendons    tests/test_render_tendons.TENDONS: tendons="mujoco" vs "straight"; T0 1024, T2 64
  textures   tests/test_render_env_textures.XML: none / texture pool on every env's box / per-world geom_matid; T0 1024, T2 64
  rolling    cloth scene (deformable_render_bench): global shutter vs 8 bands vs one band per row; T0 1024, T2 256

    scripts/gpu_run.sh rg2_cost timing 30 -- .venv/bin/python scripts/diagnostics/render_gaps2_cost.py --item haze --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

MENAGERIE = os.environ.get("MENAGERIE", os.path.join(ROOT, "upstream", "mujoco_menagerie"))


def frame_ms(r, sim, frames=10):
    for _ in range(2):
        r.render(sim)
    r.ctx.synchronize()
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(frames):
            r.render(sim)
        r.ctx.synchronize()
        best.append((time.perf_counter() - t0) / frames * 1e3)
    return round(min(best), 3)


def batch(m, n, fields=()):
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    sim = BatchSim(m, n, options=BatchSimOptions(per_world_fields=tuple(fields), njmax=64) if fields else None)
    sim.forward(); sim.synchronize()
    return sim


def T0(m, n, res, cam, **kw):
    from metalsim.render.tier0 import Tier0Renderer
    return Tier0Renderer(m, n, width=res, height=res, camera=cam, outputs=("rgb", "depth"), **kw)


def T2(m, n, res, cam, **kw):
    from metalsim.render.tier2 import Tier2Renderer
    return Tier2Renderer(m, n, width=res, height=res, camera=cam, spp=1, max_bounces=1, **kw)


def rows_haze():
    spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "unitree_go2", "scene.xml"))
    spec.worldbody.add_camera(name="c", pos=[1.2, -1.2, 0.5], xyaxes=[0.707, 0.707, 0, -0.1, 0.1, 0.99])
    m = spec.compile()
    m.geom_margin[:] = 0.0          # MuJoCo Warp's MULTICCD refuses margins; render-only benchmark
    out = []
    for name, mk, n, kw in (("T0", T0, 1024, {}), ("T1", T0, 256, dict(tier=1)), ("T2", T2, 64, {})):
        sim = batch(m, n)
        for on in (True, False):
            out.append(dict(item="haze", renderer=name, envs=n, haze=on, ms=frame_ms(mk(m, n, 128, "c", haze=on, **kw), sim)))
    return out


HFIELD = """<mujoco><visual><quality shadowsize="4096"/></visual>
<asset><hfield name="h" nrow="65" ncol="65" size="1.5 1.5 0.2 0.3"/></asset>
<worldbody><light directional="true" dir="0.6 0 -0.8" diffuse="0.8 0.8 0.8" castshadow="true"/>
<geom type="plane" size="4 4 0.1" rgba="0.8 0.8 0.8 1"/><geom type="hfield" hfield="h" pos="0 0 0.7" rgba="0.3 0.6 0.3 1"/>
<camera name="c" pos="0.4 -4.5 3.0" xyaxes="1 0 0 0 0.63 0.77"/></worldbody></mujoco>"""


def rows_skirts():
    """A raised 64 x 64-cell heightfield over a floor (its sides cast the floor shadow): skirt_shadows on / off."""
    m = mujoco.MjModel.from_xml_string(HFIELD)
    rng = np.random.default_rng(0)
    m.hfield_data[:] = rng.uniform(0.3, 0.7, m.hfield_data.shape)
    sim = batch(m, 1024)
    return [dict(item="skirt_shadows", renderer="T0", envs=1024, skirt_shadows=on,
                 ms=frame_ms(T0(m, 1024, 128, "c", skirt_shadows=on), sim)) for on in (True, False)]


def rows_tendons():
    from test_render_tendons import TENDONS
    # plus a floor: the GPU path binds geom_xpos, which MuJoCo Warp leaves unallocated for a model without geoms
    m = mujoco.MjModel.from_xml_string(TENDONS.replace("<worldbody>", '<worldbody><geom type="plane" size="2 2 0.1"/>', 1))
    out = []
    for name, mk, n in (("T0", T0, 1024), ("T2", T2, 64)):
        sim = batch(m, n)
        for mode in ("mujoco", "straight"):
            out.append(dict(item="tendons", renderer=name, envs=n, tendons=mode, ms=frame_ms(mk(m, n, 128, "cam", tendons=mode), sim)))
    return out


def rows_textures():
    import warp as wp
    from test_render_env_textures import XML, _pool_images
    m = mujoco.MjModel.from_xml_string(XML)
    out = []
    for name, mk, n in (("T0", T0, 1024), ("T2", T2, 64)):
        sim = batch(m, n)
        out.append(dict(item="textures", renderer=name, envs=n, mode="none", ms=frame_ms(mk(m, n, 128, "cam"), sim)))
        r = mk(m, n, 128, "cam")
        ids = r.set_texture_pool(_pool_images())
        sel = np.full((n, r.G), -1, np.int32)
        box = list(r.tables.geoms).index(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box"))
        sel[:, box] = ids[np.arange(n) % len(ids), 0]
        r.set_material_ids(sel)
        out.append(dict(item="textures", renderer=name, envs=n, mode="texture_pool", ms=frame_ms(r, sim)))
        simw = batch(m, n, ("geom_matid",))
        mid = simw.m.geom_matid.numpy()
        mid[1::2, box] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "B")
        simw.m.geom_matid.assign(mid); wp.synchronize_device("metal:0")
        out.append(dict(item="textures", renderer=name, envs=n, mode="per_world_geom_matid", ms=frame_ms(mk(m, n, 128, "cam"), simw)))
    return out


def rows_rolling():
    from deformable_render_bench import cloth_scene, make_sim
    from metalsim.render.rolling import RollingShutter
    m, kind = cloth_scene()
    out = []
    for name, mk, n, res in (("T0", T0, 1024, 64), ("T2", T2, 256, 64)):
        sim = make_sim(m, kind, n)
        for label, rs in (("global", None), ("8_bands", RollingShutter.top_to_bottom(res, 0.5, bands=8)),
                          ("per_row", RollingShutter.top_to_bottom(res, 0.5))):
            out.append(dict(item="rolling", renderer=name, envs=n, res=res, mode=label, ms=frame_ms(mk(m, n, res, "cam", rolling_shutter=rs), sim)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", nargs="+", default=["haze", "skirts", "tendons", "textures", "rolling"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import warp as wp
    wp.config.quiet = True
    for it in a.item:
        for row in globals()["rows_" + it]():
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
