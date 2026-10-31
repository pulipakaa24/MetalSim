# Hand-off: chart (README throughput bar chart), 2026-10-31

## Status
Done. Both SVGs are generated, rendered to PNG and inspected. They are not yet in the README (that is the assembly stream's job).

## Changed
- `scripts/readme_media/throughput_chart.py` (new): standard-library generator. Every number is inline with its doc section and raw log beside it, and the docstring lists what is not drawn and why.
- `runs/readme_media/assets/throughput_dark.svg` (new, 10,028 bytes): GitHub dark surface `#0d1117`.
- `runs/readme_media/assets/throughput_light.svg` (new, 10,028 bytes): white surface.
- `runs/readme_media/scratch/chart/final_{light,dark}.png` (scratch): headless-Chrome renders at 2x.

## The chart
Three panels, each with its own linear scale. Within a panel, bars are grouped into like-for-like pairs. Colour marks the system: MetalSim is blue, Isaac Lab 3.0 Newton/MuJoCo Warp is orange, Isaac Lab 3.0 PhysX is aqua. Each bar is labelled with its name, its machine and its config, and its value sits at the bar's end.

| task | quantity | config | machine | value (drawn) | source line |
|---|---|---|---|---|---|
| G1 flat, 4096 envs | env step alone | MetalSim, `solver_cfg="isaaclab3_every_substep_cap20"` (Isaac Lab 3.0's MuJoCo Warp settings, pyramidal, cap 20), njmax 144 / nconmax 24 | M4 Max 16-inch | 58,287 (58.3K) | PARITY §1.4, capacity-bound row ("Isaac Lab 3.0 cap-20 preset 56,301 -> 58,287"); `runs/tp26/cap.wrapper.log` `[J1_il3_njmax144] full env step ... 58,287` (2026-09-26 04:11, AC, GPU 0 %) |
| G1 flat, 4096 envs | env step alone | Isaac Lab 3.0-EA, Newton/MuJoCo Warp, task default (pyramidal), `isaaclab benchmark runtime`, 1,000 iterations | NVIDIA L4 | 72,025 (72.0K) | PARITY §1.8 "Runtime benchmark"; `runs/parity3/isaac/bench/Isaac-Velocity-Flat-G1_newton_mjwarp_a/run.log` `total_fps 72,025` |
| G1 flat, 4096 envs | env step alone | MetalSim elliptic default (`recommended`: impratio 10, cap 20), which is PhysX's cone | M4 Max 16-inch | 41,779 / 42,010, mean 41.9K | PARITY §1.4, elliptic row (2026-09-26 12:03, "env step 41.8 / 42.0 K"); `runs/ellip_ws/final.wrapper.log` E0_1 / E0_2 |
| G1 flat, 4096 envs | env step alone | Isaac Lab 3.0-EA, PhysX, task default | NVIDIA L4 | 45,114 (45.1K) | PARITY §1.8; `.../Isaac-Velocity-Flat-G1_isaacsim_physx_a/run.log` |
| G1 rough, 4096 envs | env step alone | MetalSim pyramidal preset (`recommended` before 11:00 on 2026-09-26, now `recommended_pyramidal`) | M4 Max 16-inch | 49,013 / 49,100, mean 49.1K | PARITY §1.4, row "rough terrain with everything landed"; README line 54 ("Rough, pyramidal: env step 49.1 K"); `runs/tp26/rough2.wrapper.log` |
| G1 rough, 4096 envs | env step alone | Isaac Lab 3.0-EA, Newton/MuJoCo Warp, task default | NVIDIA L4 | 51,181 (51.2K) | PARITY §1.8; `.../Isaac-Velocity-Rough-G1_newton_mjwarp_a/run.log` |
| Franka-Pour, 1,024 envs | full PPO loop (96-step rollout + update) | MetalSim port, the same coupled solver, `metal_fast_math="off"` | M4 Max 16-inch | 992 | PARITY §1b.4, row "1,024 envs"; README line 57; `runs/franka_pour/bench_mac2_1024_ppo.json` (992.37) |
| Franka-Pour, 1,024 envs | full PPO loop | Isaac's own task with the asset patch, rsl_rl | NVIDIA L4 | 317 | PARITY §1b.4; `runs/parity/isaaclab3_pour/train_1024.log` (98,304 steps per 309.50 s / 310.48 s iteration) |

The pairing rule follows the repo's own position. Isaac Lab 3.0's Newton numbers impose a pyramidal cone (README, "Default contact model"), so they are paired with MetalSim runs that also use a pyramidal cone. MetalSim's elliptic default is its PhysX-parity setting, "the exact cone PhysX uses" (`docs/research/elliptic_cones_2026-09-25.md` §7.1), so it is paired with PhysX.

MetalSim is slower in every G1 pair: 0.81x, 0.93x and 0.96x. The chart shows that. It is 3.1x faster on Franka-Pour.

**Not drawn, and stated in the chart's rough-panel note:**
- **NVIDIA's published RTX 4090 figure.** It is 94K env step / 82K with training, and it exists only for Rough-G1 (`docs/research/competitors_2026-09-25.md` §5).
- **Isaac Lab 3.0 PhysX rough on the L4 (35.7K).**

Both are left out for the same reason: MetalSim has no rough-terrain timing in a PhysX-matched (elliptic) configuration. Dropping them removes one pair that favours MetalSim (49.1K vs 35.7K) and one that does not (94K), so this is not cherry-picking.

**Not drawn either:** the MPM coupled scenes in MPM.md 8j/8k. Their unit is ms per frame of a single world, not env-steps/s.

## `<picture>` snippet (replace `ASSET_BASE/`)
```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="ASSET_BASE/throughput_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="ASSET_BASE/throughput_light.svg">
  <img src="ASSET_BASE/throughput_light.svg" width="900" alt="Throughput in env-steps per second, higher is better. G1 flat, 4,096 envs, env step: MetalSim on an M4 Max 58.3K vs Isaac Lab 3.0 Newton/MuJoCo Warp on an NVIDIA L4 72.0K (Isaac's MuJoCo Warp settings); MetalSim elliptic default 41.9K vs Isaac Lab 3.0 PhysX on the L4 45.1K. G1 rough, env step: MetalSim pyramidal preset 49.1K vs Isaac Lab 3.0 Newton on the L4 51.2K. Franka-Pour, 1,024 envs, full PPO loop: MetalSim 992 vs Isaac Lab 3.0 on the L4 317.">
</picture>
```

## Verified
- **Generator run:** `.venv/bin/python scripts/readme_media/throughput_chart.py` printed 10,028 bytes for each SVG. md5: dark `0f371f83ce638281b053b5357827fb31`, light `204053787e514b2fcff123cd72992c3d`.
- **XML and content checks:** `xmllint --noout` passes on both files. `grep -c -iE "href|url\(|@import|<script|<style|<image|<foreignObject"` returns 0 for both. Each file has no external references, uses system fonts only, carries `font-family` on the root, and contains `<title>` and `<desc>` (with all values) and an opaque background rect.
- **Renders inspected (Read tool), whole page, both themes:** `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars --force-device-scale-factor=2 --window-size=900,910 --screenshot=... file://.../throughput_{light,dark}.svg` produced `runs/readme_media/scratch/chart/final_{light,dark}.png`. There are no collisions or clipping, and every label and value is legible.
  - `qlmanage -t` was also tried. It crops to a square, so it is not usable for the full page.
- **Palette:** the dataviz skill's `scripts/validate_palette.js` passes on `#2a78d6,#eb6834,#1baf7a` (light, `--surface #ffffff --pairs all`) and on `#3987e5,#d95926,#199e70` (dark, `--surface #0d1117 --pairs all`).
  - Light aqua scores 2.82:1 against white, which triggers the WARN. The palette's relief rule is met because every bar carries a direct text label.
  - Text contrast (computed): light muted 5.18 and secondary 7.94; dark muted 6.55, secondary 10.56 and primary 16.46.
- **Text widths:** measured with Pillow at 0.41-0.47 em per character (SF) and 0.43-0.50 (Helvetica, Arial). Wrapping and the legend use 0.52 for margin.
- **Every number was re-read from its raw log** (the table above), not only from the docs.
- **GitHub behaviour, researched rather than tried:**
  - **`<picture>` support:** docs.github.com "Basic writing and formatting syntax" says "The `<picture>` HTML element is supported" (fetched 2026-10-31 via the docs API). The theme-switching docs section is gone from the current page.
  - **Theme switching:** the GitHub changelog 2022-08-15 "Specify theme context for images in Markdown (GA)" describes `<source media="(prefers-color-scheme: dark)">` with an `<img>` fallback.
  - **The media query follows the OS/browser appearance, not GitHub's theme toggle:** reported in community discussion #210983 (https://github.com/orgs/community/discussions/210983), not confirmed by staff. Because of this, each SVG has an opaque background, so a mismatched pick stays legible.
  - **Raw file headers:** raw.githubusercontent.com serves `.svg` as `image/svg+xml` with `content-security-policy: default-src 'none'; style-src 'unsafe-inline'; sandbox`, so no external fonts or resources load (checked with `curl -sI`).
  - **SVG as an image:** scripts are disabled and external resources do not load (MDN, "SVG as an image").
  - **README sanitising:** the README's HTML has script tags, inline styles and class/id attributes stripped (github/markup README, step 1). The snippet uses only `media`, `srcset`, `src`, `width` and `alt`.

## Open
- **README inconsistencies the chart exposed** (README is not my file):
  - **Elliptic and pyramidal throughput figures:** line 54 quotes "env step 41.0 K, physics 84.0 K; pyramidal env step 61.3 K, physics 100.5 K" and cites `runs/ellip_ws/final.wrapper.log`. That log and PARITY §1.4 say 41.8 / 42.0 K, 85.7 / 85.9 K, 62.7 K and 102.7 / 103.2 K. None of 41.0, 84.0, 61.3 or 100.5 appears in any log under `runs/ellip_ws` or `runs/tp26`.
  - **RTX 4090 figure:** line 54 puts "RTX 4090: 82 K (published)" in the G1 flat env-step row, but NVIDIA published it for Rough-G1, as a with-training number.
  - **MPM ratios:** line 57's "1.29x / 1.76x" are MPM.md 8j. Section 8k later measured 1.21x captured for both scenes.
- **Machine attribution is not in the logs.** None of the MetalSim logs stamps the Mac model. The "16-inch" label rests on PARITY §1b.4 ("this Mac", with training runs on "the second M4 Max") and §1.4 (the 14-inch measured "within 2-4 % of this machine"), plus the brief.
- **The flat measurement under Isaac's settings is stale.** It is one run, taken before 2026-09-26's later solver fusions and the Metal early exit. On current main it would probably be higher, but that has not been measured.
- **Two rough timings were never taken:** rough under Isaac's exact settings and rough with the elliptic cone. They are what would let the chart add the rough PhysX bar and the 4090 bar.
- **The 4090's backend is ambiguous in the repo:** PARITY §1.4 says PhysX; RESEARCH.md says "Isaac Sim 6.1".
- **Only macOS fonts were rendered.** Windows and Linux fallbacks (Segoe UI, Noto Sans, DejaVu Sans) were not measured. DejaVu Sans is about 15 % wider than Arial, so the longest legend item may come within a few pixels of the next on Linux machines without Noto Sans.
- **Not seen on GitHub itself**, since pushing was not allowed.
- **Placement:** plan §4a's 2x3 grid has the chart as one cell. At 1/2 or 1/3 of the column (300-450 px) the 11.5-13.5 px text drops to 4-7 px and becomes unreadable. Put it full width under the grid.

## Next
1. **README assembly stream:**
   - Publish `runs/readme_media/assets/throughput_{dark,light}.svg` alongside the other media on the `readme-assets` branch.
   - Paste the snippet above with `ASSET_BASE/` set to that branch's raw URL, full width below the media grid, with a caption pointing to `scripts/readme_media/throughput_chart.py`.
   - Reconcile README line 54 with the logs (Open, first bullet), or decide that the README keeps its own wording.
   - After merging, check the chart on github.com in both themes.
2. **Optional timing increment** (16-inch, AC power, one job at a time, through the queue), to add the rough PhysX pair and the 4090 bar:
   - `scripts/gpu_run.sh chart-tp timing 15 -- bash -c 'source .venv/bin/activate; for v in "{\"solver_cfg\":\"isaaclab3_every_substep_cap20\",\"label\":\"il3\"}" "{\"contact_cfg\":\"recommended\",\"label\":\"ellip\"}"; do for t in flat rough; do MJW_TP_TERRAIN=$t MJW_TP_VARIANT="$v" python scripts/diagnostics/g1_tp_variants.py 4096; done; done'`. This is the pattern of `runs/tp26/job_rough2.sh`. Run it twice interleaved.
   - Rough elliptic at 2.5 ms needs the MuJoCo Warp fork's CCD scratch cache (`docs/research/rough_ellip_oom_2026-09-26.md`).
   - Then update the values and sources in `throughput_chart.py`, rerun it, and look at the renders.
