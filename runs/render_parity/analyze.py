import json, os, sys, glob, numpy as np, imageio.v2 as iio
from metalsim.parity.compare import robot_mask
R = 'runs/render_parity'
def summ(d):
    s = json.load(open(f'{R}/{d}/summary.json'))
    w, r, rb = s['whole_frame_raw'], s['robot_raw'], s['robot_brightness_matched']
    f = lambda x: '–' if x is None else f'{x:.3f}' if x < 1 else f'{x:.2f}'
    return (f"| {d} | {s['frames']} | {f(w['psnr_db'])} / {f(w['ssim'])} / {f(w.get('lpips_alex'))} / {f(w['flip'])} | "
            f"{f(r['psnr_db'])} / {f(r['ssim'])} / {f(r.get('lpips_alex_crop'))} / {f(r['flip'])} | {f(rb['psnr_db'])} / {f(rb['ssim'])} (gain {f(s['robot_gain_mean'] or 0)}) | "
            f"{s['robot_frames']} | {s['sky_ours']} | {s['render_ms_per_frame_host_path']:.0f} |")
def vs_gt(d, gt):
    ps, pr = [], []
    for f in sorted(glob.glob(f'{R}/{d}/*_rgb.png')):
        b = os.path.basename(f); a = iio.imread(f).astype(float) / 255; g = iio.imread(f'{R}/{gt}/{b}').astype(float) / 255
        m, _ = robot_mask(np.load(f'{R}/{gt}/' + b.replace('_rgb.png', '_depth.npy')))
        ps.append(10 * np.log10(1 / ((a - g) ** 2).mean())); pr.append(10 * np.log10(1 / ((a - g)[m] ** 2).mean()))
    return np.mean(ps), np.mean(pr), len(ps)
if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    print('| run | frames | whole frame PSNR / SSIM / LPIPS / FLIP | robot raw PSNR / SSIM / LPIPS(crop) / FLIP | robot brightness-matched PSNR / SSIM | robot frames | sky | ms/frame |')
    for d in sys.argv[1:]:
        if d.startswith('gt:'):
            _, d, gt = d.split(':'); p = vs_gt(d, gt); print(f'| {d} vs {gt} | {p[2]} | whole {p[0]:.2f} dB | robot {p[1]:.2f} dB |')
        elif os.path.exists(f'{R}/{d}/summary.json'): print(summ(d))
