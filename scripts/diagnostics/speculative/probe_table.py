"""Tables from overtravel_probe.py outputs: python scripts/diagnostics/speculative/probe_table.py DIR TAG[,TAG...] [--n 64]"""
import sys, os, numpy as np
D = sys.argv[1]; TAGS = sys.argv[2].split(","); N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 64
H = 0.0025; CT = 0.02
ISAAC = {500: 3.19, 1000: 3.02, 1499: 3.11}


def stats(z, it):
    g = lambda k: z[f"it{it}_{k}"]
    sub = g("sub")          # (S, 8, n, 2, K)
    S = sub.shape[0]; T = S * CT
    x = g("x")[-1]
    fn, ft, fns, fts, fnu, ftu = (sub[..., k] for k in range(6))
    ftx, ftxs = sub[..., 10], sub[..., 11]
    imp = lambda a: a.sum((0, 1)).mean() * H / T        # mean over envs and feet of the time-integrated force / s = mean force
    flag = g("sens_flag")                                # (S, n, 2)
    trans = (np.diff(flag.astype(int), axis=0) != 0).sum() / flag.shape[1] / T
    slide = g("slide"); stance_slide = (slide * flag).sum() / max(flag.sum(), 1)
    act = g("act")                                       # (S, n, 37)
    subw = g("subw")
    # touch-down events (sensor flag 0 -> 1): foot COM horizontal speed and vertical speed of the previous step
    td = (flag[1:] & ~flag[:-1])
    vz = g("vz_foot"); td_vz = -vz[:-1][td].mean() if td.any() else np.nan; td_slide = slide[:-1][td].mean() if td.any() else np.nan
    # swing: substeps where the foot has no penetrating contact; spec rows present in swing
    swing = sub[..., 6] - sub[..., 7] == 0
    fnu_sw = (fnu * swing).sum((0, 1)).mean() * H / T; fts_sw = (fts * swing).sum((0, 1)).mean() * H / T
    return dict(x=x.mean(), xse=x.std(ddof=1) / np.sqrt(len(x)), err=x.mean() - ISAAC[it],
                Fn=imp(fn), Ft=imp(ft), Fn_s=imp(fns), Ft_s=imp(fts), Fn_u=imp(fnu), Ft_u=imp(ftu), Ftx=imp(ftx), Ftx_s=imp(ftxs),
                Fnu_sw=fnu_sw, Fts_sw=fts_sw,
                n=sub[..., 6].mean(), n_s=sub[..., 7].mean(), n_sF=sub[..., 8].mean(),
                flag=flag.mean(), trans=trans, stance_slide=stance_slide, td_vz=td_vz, td_slide=td_slide,
                act_abs=np.abs(act).mean(), act_std=act.std(0).mean(), act=act.mean((0, 1)),
                nefc=subw[..., 0].mean(), niter=subw[..., 1].mean(), vx=g("vx")[S // 4:].mean())


rows = {}
for t in TAGS:
    f = os.path.join(D, f"{t}_n{N}.npz")
    if not os.path.exists(f): continue
    z = np.load(f); rows[t] = {it: stats(z, it) for it in (500, 1000, 1499)}
cols = [("x", "x [m] (SE)"), ("Fn", "Fn"), ("Ft", "|Ft|"), ("Ftx", "Ft_x"), ("Fn_s", "Fn spec"), ("Ft_s", "|Ft| spec"), ("Fn_u", "Fn spec no-cross"),
        ("Ft_u", "|Ft| spec no-cross"), ("Fnu_sw", "Fn no-cross in swing"), ("n", "contacts/foot"), ("n_s", "spec/foot"), ("flag", "sensor flag"),
        ("trans", "flag transitions/s"), ("stance_slide", "stance slide m/s"), ("td_vz", "touch-down v_z"), ("td_slide", "touch-down slide"),
        ("act_abs", "mean |a|"), ("act_std", "a std"), ("nefc", "nefc"), ("niter", "Newton it")]
print(f"forces are time-averaged per foot [N]; n = {N} envs, 400 control steps")
for it in (500, 1000, 1499):
    print(f"\n#### Isaac checkpoint {it} (Isaac {ISAAC[it]} m)\n")
    print("| setting | " + " | ".join(c[1] for c in cols) + " |"); print("|---|" + "---|" * len(cols))
    for t, r in rows.items():
        s = r[it]; cells = []
        for k, _ in cols:
            if k == "x": cells.append(f"{s['x']:.3f} ({s['xse']:.3f})")
            elif k in ("flag", "stance_slide", "td_vz", "td_slide", "act_abs", "act_std", "n", "n_s"): cells.append(f"{s[k]:.3f}")
            else: cells.append(f"{s[k]:.1f}")
        print(f"| {t} | " + " | ".join(cells) + " |")
if len(rows) > 1:
    base = list(rows)[0]
    print(f"\nmean-action difference vs {base} (L2 over 37 joints, Isaac order) / largest joints:")
    for t in rows:
        for it in (500, 1000, 1499):
            dA = rows[t][it]["act"] - rows[base][it]["act"]
            print(f"  {t} it{it}: |dA| {np.linalg.norm(dA):.4f}, top joints {np.argsort(-np.abs(dA))[:4].tolist()} ({np.sort(np.abs(dA))[::-1][:4].round(4).tolist()})")
