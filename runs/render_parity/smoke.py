# smoke: new renderer (legacy defaults) vs the original code on one hero frame; every preset runs; OIDN works
import sys, json, importlib.util, os, time, numpy as np, mujoco
import warp as wp; wp.config.quiet = True
from metalsim.render.rtx_parity import build_scene, preset_kwargs, isaac_states
from metalsim.render.tier2 import Tier2Renderer
d = 'runs/parity/isaac/parity_out2/rt'; meta = json.load(open(d + '/meta.json')); m = build_scene(meta)
names = [str(s) for s in np.load(d + '/action_sequence_B.npz')['joint_names']]
st = isaac_states(d, m, names); dd = mujoco.MjData(m); dd.qpos[:] = st['A_hold'][25]; mujoco.mj_forward(m, dd)
W, H = 256, 144
# original code, loaded from the saved copy
orig = '/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/render_orig'
import metalsim.render.metal_context as mc
sd = mc.SHADER_DIR; mc.SHADER_DIR = type(sd)(orig + '/shaders')
spec = importlib.util.spec_from_file_location('tier2_orig', orig + '/tier2.py'); t2o = importlib.util.module_from_spec(spec); spec.loader.exec_module(t2o)
a = t2o.Tier2Renderer(m, 1, width=W, height=H, camera='hero', spp=8, max_bounces=3, ctx=mc.MetalContext()).render_host([dd], passes=2)
mc.SHADER_DIR = sd
b = Tier2Renderer(m, 1, width=W, height=H, camera='hero', spp=8, max_bounces=3, ctx=mc.MetalContext()).render_host([dd], passes=2)
print('legacy new vs original: max |rgb diff|', np.abs(a['rgb'].astype(int) - b['rgb'].astype(int)).max(), 'hdr max diff', np.abs(a['hdr'] - b['hdr']).max())
for p in ['brdf', 'lights', 'tonemap', 'oidn', 'atrous']:
    r = Tier2Renderer(m, 2, width=W, height=H, camera='hero', spp=8, max_bounces=3, **preset_kwargs(p, meta))
    t0 = time.time(); o = r.render_host([dd, dd], passes=2); t = time.time() - t0
    sky = o['rgb'][0][5, W // 2]
    print(p, 'ok', f'{t*1e3:.0f} ms', 'sky', sky, 'mean', o['rgb'].mean().round(1), 'finite', np.isfinite(o['hdr']).all(), 'dn' if 'hdr_denoised' in o else '',
          (np.abs(o['hdr_denoised'] - o['hdr_mean']).mean() if 'hdr_denoised' in o else ''), flush=True)
    import imageio.v2 as iio; iio.imwrite(f'runs/render_parity/smoke_{p}.png', o['rgb'][0])
