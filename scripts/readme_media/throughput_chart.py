"""README speed chart: MetalSim vs Isaac Lab, like-for-like pairs only, as a dark and a light SVG.

    .venv/bin/python scripts/readme_media/throughput_chart.py [--out DIR]

Writes throughput_dark.svg and throughput_light.svg to runs/readme_media/assets/ (or DIR) for a GitHub
<picture> element (<source media="(prefers-color-scheme: dark)">). CPU only, standard library only.

Rules the data below follows:
* Every number is a measurement already in the repo; the source (doc section and raw log) is the comment beside it.
  No number here was measured for this chart.
* Each pair compares the same quantity at matching settings. Isaac Lab 3.0's MuJoCo Warp (Newton) G1 settings use a
  pyramidal friction cone, so MetalSim's partner for them is a pyramidal run; MetalSim's default for the G1 (elliptic
  cone, impratio 10) is its PhysX-parity setting ("the exact cone PhysX uses", docs/research/elliptic_cones_2026-09-25.md
  section 7.1), so it is paired with PhysX. A reference with no matching MetalSim measurement is not drawn; the panel
  note names it instead.
* MetalSim timings are the 16-inch M4 Max's timing-queue runs (idle GPU at the start of each job); the Isaac Lab
  numbers were measured by this project on an NVIDIA L4 (Isaac Lab 3.0-EA, Isaac Sim 6.1).

Not drawn, and why:
* NVIDIA's published RTX 4090 figures (Isaac-Velocity-Rough-G1-v0, 4096 envs: env step 94K, with training 82K;
  docs/research/competitors_2026-09-25.md section 5, docs/RESEARCH.md "Published standards"): rough terrain only, and
  MetalSim has no rough-terrain timing in a PhysX-matched (elliptic) configuration. The README's G1 *flat* row quotes
  the 82K, which is a with-training number for the *rough* task.
* Isaac Lab 3.0 PhysX, rough, on the L4 (35.7K env step, PARITY section 1.8): same reason.
* MPM coupled scenes (metalsim/physics/mpm/MPM.md 8j / 8k): milliseconds per frame of one world, not env-steps/s.
"""
from __future__ import annotations

import argparse
import os
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "runs", "readme_media", "assets")

# --------------------------------------------------------------------------------------------------------------------
# Data. Series keys pick the colour (the system), never the rank.
#   metalsim  MetalSim on the Apple M4 Max (16-inch MacBook Pro)
#   newton    Isaac Lab 3.0-EA with its default Newton / MuJoCo Warp backend, NVIDIA L4
#   physx     Isaac Lab 3.0-EA with PhysX, NVIDIA L4
# --------------------------------------------------------------------------------------------------------------------

G1_FLAT = dict(
    title="G1 humanoid, flat terrain · 4,096 envs · env step alone",
    domain=(0, 80_000), ticks=(0, 20_000, 40_000, 60_000, 80_000),
    groups=[
        dict(heading="Isaac Lab 3.0's own MuJoCo Warp settings (pyramidal friction cone)", bars=[
            dict(series="metalsim", name="MetalSim",
                 detail="M4 Max 16-inch · Isaac's settings, iteration cap 20",
                 # docs/PARITY.md section 1.4, capacity-bound row ("Isaac Lab 3.0 cap-20 preset 56,301 -> 58,287");
                 # docs/research/throughput_2026-09-26.md section 4b table; raw: runs/tp26/cap.wrapper.log line
                 # "[J1_il3_njmax144] full env step 70.3 ms -> 58,287 env-steps/s" (2026-09-26 04:11, AC power, GPU 0 %
                 # at start; solver_cfg isaaclab3_every_substep_cap20, njmax 144 / nconmax 24, one run, 2.3.2 task
                 # terms: 3.0's push / mass events not on). Measured before that day's later solver fusions.
                 value=58_287),
            dict(series="newton", name="Isaac Lab 3.0, Newton / MuJoCo Warp",
                 detail="NVIDIA L4 · task default",
                 # docs/PARITY.md section 1.8 "Runtime benchmark, 4096 envs" (flat Newton/MuJoCo-Warp 72,025);
                 # raw: runs/parity3/isaac/bench/Isaac-Velocity-Flat-G1_newton_mjwarp_a/run.log ("total_fps 72,025",
                 # isaaclab benchmark runtime, 1,000 iterations, 2026-09-25). Its synchronized run (_b) gave 67,130;
                 # the higher, README-quoted figure is used.
                 value=72_025),
        ]),
        dict(heading="PhysX's friction cone (elliptic): MetalSim's default against PhysX", bars=[
            dict(series="metalsim", name="MetalSim",
                 detail="M4 Max 16-inch · elliptic default, impratio 10",
                 # docs/PARITY.md section 1.4, row "G1 flat, elliptic default ... 2026-09-26 12:03" ("env step
                 # 41.8 / 42.0 K"); raw: runs/ellip_ws/final.wrapper.log, [E0_1_ellip10_cap20] 41,779 and
                 # [E0_2_ellip10_cap20] 42,010 env-steps/s (AC power, GPU 0 %); value = mean of the two repeats.
                 # README.md line 54 says 41.0 K citing the same log; the log says 41.8 / 42.0 K.
                 value=(41_779 + 42_010) / 2),
            dict(series="physx", name="Isaac Lab 3.0, PhysX",
                 detail="NVIDIA L4 · task default",
                 # docs/PARITY.md section 1.8 "Runtime benchmark" (flat PhysX 45,114); raw:
                 # runs/parity3/isaac/bench/Isaac-Velocity-Flat-G1_isaacsim_physx_a/run.log ("total_fps 45,114").
                 value=45_114),
        ]),
    ],
    notes=[],
)

G1_ROUGH = dict(
    title="G1 humanoid, rough terrain · 4,096 envs · env step alone",
    domain=(0, 80_000), ticks=(0, 20_000, 40_000, 60_000, 80_000),
    groups=[
        dict(heading="Pyramidal friction cone (MetalSim: its own pyramidal preset; Isaac's exact settings not timed on rough)",
             bars=[
            dict(series="metalsim", name="MetalSim",
                 detail="M4 Max 16-inch · pyramidal preset",
                 # docs/PARITY.md section 1.4, "Throughput without fidelity cost" row "rough terrain with everything
                 # landed" (full env step 49,013-49,100); README.md line 54 "Rough, pyramidal: env step 49.1 K";
                 # raw: runs/tp26/rough2.wrapper.log [rough1_landed] 49,013 / [rough2_landed] 49,100 (2026-09-26
                 # 08:33, AC power, GPU 0 %; contact_cfg "recommended", which was the pyramidal preset until 11:00
                 # that day, now "recommended_pyramidal"). Mean of the two repeats. On flat, in one session, Isaac's
                 # settings timed 4.6 % faster than this preset (runs/tp26/cap.wrapper.log J1 vs A1/A2), so this
                 # stand-in does not flatter MetalSim.
                 value=(49_013 + 49_100) / 2),
            dict(series="newton", name="Isaac Lab 3.0, Newton / MuJoCo Warp",
                 detail="NVIDIA L4 · task default",
                 # docs/PARITY.md section 1.8 "Runtime benchmark" (rough Newton/MuJoCo-Warp 51,181); raw:
                 # runs/parity3/isaac/bench/Isaac-Velocity-Rough-G1_newton_mjwarp_a/run.log ("total_fps 51,181").
                 value=51_181),
        ]),
    ],
    notes=["Not drawn: Isaac Lab 3.0 PhysX on the L4 (35.7K) and NVIDIA's published RTX 4090 figure (94K, not "
           "measured by us). MetalSim has no rough-terrain timing in a PhysX-matched configuration to pair them with."],
)

FRANKA_POUR = dict(
    title="Franka-Pour, granular MPM manipulation · 1,024 envs · full PPO loop",
    domain=(0, 1_200), ticks=(0, 400, 800, 1_200),
    groups=[
        dict(heading="Isaac's coupled solver (MuJoCo Warp arm + implicit MPM), same settings on both", bars=[
            dict(series="metalsim", name="MetalSim",
                 detail="M4 Max 16-inch",
                 # docs/PARITY.md section 1b.4, row "1,024 envs, step only / full PPO loop" (992); README.md line 57;
                 # raw: runs/franka_pour/bench_mac2_1024_ppo.json "env_steps_per_s": 992.37 (one 96-step iteration
                 # plus update after one warm-up, metal_fast_math "off", capture_unit "sim"); GPU 0 % at start
                 # (runs/franka_pour/bench_mac2_1024_ppo.log).
                 value=992),
            dict(series="newton", name="Isaac Lab 3.0, Newton",
                 detail="NVIDIA L4 \u00b7 MuJoCo Warp + MPM",
                 # docs/PARITY.md section 1b.4, same row (317: rsl_rl collection 309-314 s per 96 steps, update
                 # 0.2 s); README.md line 57; raw: runs/parity/isaaclab3_pour/train_1024.log iterations 1-2,
                 # 98,304 steps per 309.50 s / 310.48 s iteration = 317.6 / 316.6 env-steps/s (2026-10-12).
                 value=317),
        ]),
    ],
    notes=["The task's own 2,048 envs do not fit in the L4's 24 GB, so the pair is at 1,024."],
)

PANELS = [G1_FLAT, G1_ROUGH, FRANKA_POUR]

LEGEND = [
    ("metalsim", "MetalSim · Apple M4 Max"),
    ("newton", "Isaac Lab 3.0, Newton / MuJoCo Warp · NVIDIA L4"),
    ("physx", "Isaac Lab 3.0, PhysX · NVIDIA L4"),
]

TITLE = "Throughput: MetalSim vs Isaac Lab"
SUBTITLE = ("Environment steps per second, higher is better. Each pair: same quantity, matching settings. "
            "Each panel has its own scale.")
FOOTER = [  # one line each; no wrapping
    "Measured Sept.\u2013Oct. 2026. MetalSim: Apple M4 Max, 16-inch MacBook Pro, timing queue, idle GPU.",
    "Isaac Lab 3.0-EA (Isaac Sim 6.1): NVIDIA L4, run by this project; no vendor figures are drawn.",
    "Sources: docs/PARITY.md \u00a71.4, \u00a71.8, \u00a71b.4; every number with its log in "
    "scripts/readme_media/throughput_chart.py.",
]

# --------------------------------------------------------------------------------------------------------------------
# Themes. Categorical slots 1-3 of the dataviz reference palette (blue, orange, aqua), validated all-pairs against
# GitHub's page surfaces (#ffffff light, #0d1117 dark) with the dataviz skill's validate_palette.js. Light aqua is
# below 3:1 on white, so every bar carries a direct text label (the palette's relief rule).
# --------------------------------------------------------------------------------------------------------------------

THEMES = {
    "light": dict(surface="#ffffff", primary="#0b0b0b", secondary="#52514e", muted="#6e6d68", grid="#e1e0d9",
                  baseline="#c3c2b7", rule="#e1e0d9",
                  series=dict(metalsim="#2a78d6", newton="#eb6834", physx="#1baf7a")),
    "dark": dict(surface="#0d1117", primary="#f0efec", secondary="#c3c2b7", muted="#9a9890", grid="#262a30",
                 baseline="#3d4148", rule="#262a30",
                 series=dict(metalsim="#3987e5", newton="#d95926", physx="#199e70")),
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

# Layout (px). GitHub scales the image to its column; at 900 wide it is drawn about 1:1 in the 1000 px README column.
W = 900
PAD = 28                     # outer margin
LABEL_X = PAD + 14           # bar labels, indented under their group heading
BAR_X0 = 372                 # bar baseline
BAR_X1 = 800                 # end of each panel's domain
ROW_H = 42                   # one bar row (two label lines)
BAR_H = 20                   # bar thickness (dataviz cap: 24)
RADIUS = 4                   # rounded data end, square at the baseline
AVG_EM = 0.52                # average glyph width / font size, for wrapping and legend spacing only: measured
                             # 0.41-0.47 (SF), 0.43-0.50 (Helvetica, Arial); the rest is margin for wider fallbacks


def _fmt(v: float) -> str:
    """58287 -> '58.3K'; 992 -> '992'; 0 -> '0'."""
    if v >= 10_000:
        return f"{v / 1000:.1f}K" if v % 1000 else f"{v / 1000:.0f}K"
    return f"{v:,.0f}"


def _wrap(text: str, size: float, max_px: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and len(trial) * size * AVG_EM > max_px:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def _text(x, y, s, size, fill, weight=None, anchor=None) -> str:
    attrs = f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
    if weight:
        attrs += f' font-weight="{weight}"'
    if anchor:
        attrs += f' text-anchor="{anchor}"'
    return f"<text {attrs}>{escape(s)}</text>"


def _bar(x0, y, w, h, r, fill) -> str:
    """Horizontal bar: square at the baseline, rounded (r) at the data end."""
    if w <= r:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h}" fill="{fill}"/>'
    return (f'<path d="M{x0:.1f} {y:.1f}h{w - r:.1f}a{r} {r} 0 0 1 {r} {r}v{h - 2 * r}a{r} {r} 0 0 1 -{r} {r}'
            f'h-{w - r:.1f}z" fill="{fill}"/>')


def render(theme: str) -> str:
    t = THEMES[theme]
    out: list[str] = []
    y = PAD + 22
    out.append(_text(PAD, y, TITLE, 21, t["primary"], weight="600"))
    for line in _wrap(SUBTITLE, 13, W - 2 * PAD):
        y += 21
        out.append(_text(PAD, y, line, 13, t["secondary"]))

    # Legend: one row, swatch beside the name (text stays in ink colours).
    y += 28
    x = PAD
    for key, label in LEGEND:
        out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="2" fill="{t["series"][key]}"/>')
        out.append(_text(x + 18, y, label, 12.5, t["primary"]))
        x += 18 + len(label) * 12.5 * AVG_EM + 32

    for panel in PANELS:
        y += 24
        out.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" stroke="{t["rule"]}" stroke-width="1"/>')
        y += 26
        out.append(_text(PAD, y, panel["title"], 15.5, t["primary"], weight="600"))
        lo, hi = panel["domain"]
        sx = lambda v: BAR_X0 + (v - lo) / (hi - lo) * (BAR_X1 - BAR_X0)  # noqa: E731
        rows: list[str] = []
        grid: list[str] = []
        for group in panel["groups"]:
            y += 24
            rows.append(_text(PAD, y, group["heading"], 12.5, t["secondary"], weight="500"))
            y += 6
            seg_top = y + 4
            for bar in group["bars"]:
                ry = y
                rows.append(_text(LABEL_X, ry + 17, bar["name"], 13.5, t["primary"], weight="600"))
                rows.append(_text(LABEL_X, ry + 33, bar["detail"], 12, t["secondary"]))
                by = ry + (ROW_H - BAR_H) / 2 + 1
                x1 = sx(bar["value"])
                rows.append(_bar(BAR_X0, by, x1 - BAR_X0, BAR_H, RADIUS, t["series"][bar["series"]]))
                rows.append(_text(x1 + 8, by + 15, _fmt(bar["value"]), 14, t["primary"], weight="600"))
                y += ROW_H
            seg_bottom = y
            # Recessive vertical gridlines across this group's bars only; the baseline on top of them.
            for v in panel["ticks"][1:]:
                grid.append(f'<line x1="{sx(v):.1f}" y1="{seg_top:.1f}" x2="{sx(v):.1f}" y2="{seg_bottom:.1f}" '
                            f'stroke="{t["grid"]}" stroke-width="1"/>')
            grid.append(f'<line x1="{BAR_X0}" y1="{seg_top:.1f}" x2="{BAR_X0}" y2="{seg_bottom:.1f}" '
                        f'stroke="{t["baseline"]}" stroke-width="1"/>')
        bottom = y
        for v in panel["ticks"]:
            grid.append(_text(sx(v), bottom + 14, _fmt(v), 11.5, t["muted"], anchor="middle"))
        out.extend(grid)
        out.extend(rows)
        y = bottom + 14
        for note in panel["notes"]:
            for line in _wrap(note, 12, W - 2 * PAD):
                y += 19
                out.append(_text(PAD, y, line, 12, t["secondary"]))

    y += 26
    out.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" stroke="{t["rule"]}" stroke-width="1"/>')
    y += 4
    for line in FOOTER:
        y += 18
        out.append(_text(PAD, y, line, 11.5, t["muted"]))
    h = int(y + PAD - 6)

    desc = "; ".join(
        f'{p["title"]}: ' + ", ".join(f'{b["name"]} ({b["detail"]}) {_fmt(b["value"])}'
                                      for g in p["groups"] for b in g["bars"])
        for p in PANELS)
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" role="img" '
            f'aria-labelledby="title desc" font-family="{FONT}">'
            f'<title id="title">{escape(TITLE)}</title><desc id="desc">{escape(desc)}</desc>'
            f'<rect width="{W}" height="{h}" fill="{t["surface"]}"/>')
    return head + "\n" + "\n".join(out) + "\n</svg>\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=ASSETS, help="output directory (default: runs/readme_media/assets)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for theme in ("dark", "light"):
        path = os.path.join(args.out, f"throughput_{theme}.svg")
        svg = render(theme)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"{path}: {len(svg.encode()):,} bytes")


if __name__ == "__main__":
    main()
