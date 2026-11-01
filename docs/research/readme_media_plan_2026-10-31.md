# Front-page media: audit, survey and plan (2026-10-31)

Question: what should the README show instead of pointing at `docs/renders` and `docs/gallery`, and what has to be fixed
before new renders can be made. Status: verified the same day (section 0) and in progress (pipeline in section 4a).
Nothing here is committed media yet.

## 0. Verification of this plan (2026-10-31, later the same day)

Each claim was checked against the code and the files on disk. Corrections are made in place below; this list says
what changed and why.

- **Bug 1 holds; the replacement checkpoints have provenance after all.** `G1Flat.step()` returns
  `np.zeros(n, bool)` (adapters.py). `flat_main_s0.pt` / `rough_main_s0.pt` are documented in
  `docs/research/ppowarp_checkpoint_load_2026-10-13.md`: trained on main with the documented settings, curves match
  the 2026-09-26 reference, replays reach the time-out in 1,012 / 1,024 (flat, 500 it) and 1,016 / 1,024 (rough,
  1000 it) episodes; their logs are tracked in `runs/g1regress/ckpt14/`. `g1_rough_ppowarp_fixed_it1500.pt` walks
  only with its own pyramidal-era settings (`contact_cfg=None`, `ij` scan, ...), so it is not a film candidate on
  today's task. The film adapter runs its own torch module, so the PPOWarp load bug of that note does not apply to it.
- **Bug 2: mechanism narrower than stated.** The G1 floor is an infinite plane (`build_g1_model`: size 0 0), which
  the compact env grid is meant to continue analytically past the cell (`metalsim/sensors/env_grid.py`); so this is
  a defect in that continuation or in a camera outside the cell, not planes being clipped by design. The suggested
  workaround `RayTracer(env_margin=...)` is not reachable from the film: `Tier2Renderer` passes `env_layout` but not
  `env_margin`.
- **Bug 3 is worse than stated: it already corrupted provenance.** Besides the frames, `render()` carries `exposure`,
  `probe` and `hardware` over from an old `render.json` and appends to `render_frames.jsonl`. The working-tree
  `runs/film/g1_flat/hero/render.json` stamps `Mac16,5` (16-inch) for a render made on this Mac16,6 (14-inch), and
  its `render_frames.jsonl` has 552 rows for a 301-frame film. Do not commit the working-tree G1 re-film as is.
- **Bug 4 overstated.** README.md has no `runs/policies/` path; GUIDE's are the *output* paths of training commands,
  which is correct. Broken defaults are the film's `G1_FLAT.checkpoint`, `docs/gallery/scripts/14_g1_policy_video.py`
  and an example in `metalsim/parity/side_by_side.py`. Fix chosen: track the two G1 checkpoints (0.7 MB, 2.6 MB) next
  to their tracked logs in `runs/g1regress/ckpt14/`, as `runs/franka_pour/logs/m414_2048_model_299.pt` and
  `runs/lift_baselinecfg_seed0.pt` already are; release assets stay optional.
- **Bug 5 confirmed as far as the geoms go:** the twin has mesh geoms `Cube`, `torso_link_Cube`,
  `right_ankle_roll_link_Cube`; whether they are colliders or USD visuals is open.
- **Sand clip confirmed:** 1920x1080, 30 fps, 240 frames, 8.0 s, 94.6 MB, with its 241 PNG frames beside it.
- **WebP tooling:** Homebrew's ffmpeg has no libwebp (encoder list checked); the main venv's Pillow 12.3 writes WebP.
- **Render cost confirmed:** the re-film's per-frame log gives a median 0.12 s per frame (1280x720, 16 x 4 spp,
  one panel, OIDN) on the 14-inch on battery in Low Power Mode; 250 frames is about 30 s plus about 4 s set-up.
- **Section 3, MuJoCo Warp row: checked and adopted as the format to copy exactly** (it should have been checked first;
  the first encoder pass ran a byte-budget search instead). From its repo (README.md lines 35-79, `mujoco_warp/record.py`,
  CONTRIBUTING.md "Contributing Benchmarks"), with four of its files measured:
  - a 3-column HTML `<table>`, cells `<td align="center" width="33%">`, `<img width="320" src="<relative path>">`, then
    `<br><b>reproduce command</b>`;
  - lossy animated WebP (VP8 frames with ALPH sub-rectangles), 320x240 (4:3), rendered at that size;
  - written by Pillow in one call: `frames[0].save(path, save_all=True, append_images=frames[1:],
    duration=int(render_every * timestep * 1000), loop=0, quality=70)`, no size budget;
  - 30 fps default (stored as 30 ms frames), 159-167 frames (about 5 s), 0.47-2.0 MB (the heightfield G1 is 2.0 MB);
  - the camera orbits slowly (0.05 deg per step, symmetric about mid-clip);
  - files sit in the repo next to the benchmark that makes them, linked by relative path, although the package is on
    PyPI. They autoplay and loop on GitHub, which settles that question.
  The rest of section 3 was not re-checked.
- **README numbers the chart work found wrong** (to fix in the README pass):
  - line 54 quotes 41.0 K / 84.0 K / 61.3 K / 100.5 K and cites `runs/ellip_ws/final.wrapper.log`, but that log and
    PARITY §1.4 say 41.8 / 42.0 K, 85.7 / 85.9 K, 62.7 K and 102.7 / 103.2 K;
  - line 54 puts NVIDIA's RTX 4090 82 K in the flat env-step row, but it is a Rough-G1 with-training figure
    (`docs/research/competitors_2026-09-25.md` §5);
  - line 57's MPM ratios (1.29x / 1.76x) come from MPM.md 8j, and 8k later measured 1.21x for both scenes.
- **The time estimates were serial human time.** Measured compute for the whole minimum plan is under 10 GPU
  minutes (two G1 takes of about 1 min each, one lidar clip, checkpoint checks of about 10 s each; sand, Franka-Pour,
  WebP and chart are CPU). The work is now split into five parallel streams; wall time is set by the longest code
  stream, not by compute. Compute only dominates for optional work: retraining the G1 policies from a script
  (flat 500 it about 28 min, rough 1000 it about 65 min on the 14-inch, from the checkpoint timestamps), 1080p or
  higher sample counts, and the deferred race clip (needs the L4).

## 1. Audit of the existing media

Every tracked video was sampled frame by frame (contact strips) and every still was looked at.

| asset | verdict | reason |
|---|---|---|
| `docs/renders/franka_pour.mp4`, `franka_pour_grid.mp4` | keep | the one front-page-ready clip: trained policy, clean studio look, particles visible |
| `runs/scenes/go2_boxes/film/go2_boxes.mp4` (scene YAML film) | good look, no action | `policy: hold`, the Go2 only stands |
| `~/metalsim-fresh/runs_render/sand25m/demo/anymal_sand_25m_twolevel.mp4` (ANYmal on 25 M grains, 1920x1080, 8 s, 240 frames, 90 MB; PNG frames beside it and in `runs_render/demo/frames`) | keep, front-page ready | only needs a trimmed WebP and a small mp4; not in the repo, on the 14-inch only |
| `docs/renders/g1_flat.mp4` (as committed) | reshoot | the robot falls within 2 s (section 2, bug 1) |
| `docs/gallery/g1_tier2_policy.mp4` | reshoot | same checkpoint class, robots tumble, tiny in a 2x2 at 2048x1152 |
| `docs/gallery/cartpole_tier2_policy.mp4` | reshoot | poles are a few pixels tall |
| `docs/gallery/g1_stage_*.mp4` (Isaac-side playback) | not front page | tiny robot in a white void; fine as parity evidence |
| `docs/gallery/rollout.gif`, tier-0 atlases, `throughput.png`, `isaac_comparison.png` | retire from the front page | 2026-09 test renders; the charts quote superseded numbers |
| `docs/gallery/g1_tier2_policy.mp4` is 68 MB; `docs/gallery/` is 109 MB | move out of the repo | see section 5 |

The inventory covered every checkout on this Mac (`~/MetalSim`, `~/metalsim-fresh/**`, `~/metalsim-splat`,
`~/MetalSim-purge`) and every video named in the docs and logs. A first pass searched only this checkout and wrongly
listed the sand clip as missing; corrected the same day.

Contact strips of each video are reproducible with
`ffmpeg -i <video> -vf "fps=1,scale=480:-1,tile=6x1" -frames:v 1 strip.jpg`.

## 2. Defects found on the way (all in the film tool unless noted)

Verified:

1. **The film's G1 checkpoint never walked.** `runs/policies/g1_flat_1500.pt` (the path in `tasks.py`) falls in 16 of
   16 envs within 8 s under `recommended`, `recommended_pyramidal` and `default` contacts (16 envs, seed 3, 400 control
   steps). The fresh-install check logged `g1_record: ok (rc 0)`, which is all it tested. The G1 adapter returns an
   all-false done mask (`np.zeros(n, bool)`), so the record log says "episodes ended 0/4" with every robot on the
   floor. The checkpoint's provenance is unknown: it sits in `~/metalsim-fresh/fix1-20261014-0325/` with no training log.
   A checkpoint that walks exists: `~/metalsim-fresh/agents/g1regress/out/flat_main_s0.pt` (500 iterations, ep_len
   996, 16 of 16 upright for 8 s); rough: `~/metalsim-fresh/agents/g1regress/out/rough_main_s0.pt` (terrain level
   5.86 at iteration 1000) and `~/metalsim-fresh/agents/fixfail/g1_rough_ppowarp_fixed_it1500.pt` (15 of 16 upright).
   Fix: the adapter should derive `done` from base height / the task's termination, the record stage should refuse a
   film in which the filmed env terminates early unless asked, and the checkpoint should be fetched or trained by a
   script, not assumed to be on disk.
2. **Ground plane clipped to the ray tracer env cell** (MuJoCo film path). With 4 envs and with 1 env the laminate
   plane is drawn as a wedge with the HDRI's own floor showing through, and the plane edge moves as the cell grows.
   The render log warns `ray tracer env grid: geometry neared its env's cell boundary` twice per run. Franka-Pour
   (Newton path, table) and Go2 (scene-layer floor, not a plane) do not show it. Fix: planes must not be clipped to
   the cell, or the film path sizes the cell up front (`RayTracer(env_margin=...)`) to the camera's far distance.
3. **Render stage silently reuses stale frames.** Re-recording into the same `--out` with different settings, the
   render stage reports "0 frames rendered" (`skip_existing=True`) and encode produces a video labelled as the new
   run. Fix: key the frame cache on the recording's hash, or clear frames when `rec.json` changes.
4. **Default checkpoint paths do not exist on a fresh clone.** The film's `G1_FLAT` entry, the gallery's
   `14_g1_policy_video.py` and an example in `parity/side_by_side.py` default to `runs/policies/...`, which is
   gitignored (GUIDE's `runs/policies/` paths are training outputs, which is fine; corrected in section 0). Fix: a `scripts/fetch_policies.sh` (release assets) or a
   `scripts/train_demo_policies.sh` with the measured wall time, and the film doc stage records the checkpoint's hash.

Plausible, not fully diagnosed:

5. **A collider box is drawn in the G1 twin.** The recording's geom list has a bare `Cube` (and
   `torso_link_Cube`, `right_ankle_roll_link_Cube`); a grey cube appears beside the robot. The rule "a body with visual
   meshes hides its colliders" misses this case on the USD-converted G1.

Known already (NOW.md): the main venv lacked `pyoidn` (installed on 2026-10-31 on the 14-inch).

Why they were not caught: verification is numbers-first (returns, PSNR, env-steps/s, bitwise parity) and the smoke
tests check exit codes. No step looks at a rendered frame, and the adapter's done mask removed the one metric that
would have flagged a fallen robot. The demo tag was cut on exit codes.

## 3. What comparable projects do (survey of 2026-10-31)

- **MuJoCo Warp**: 3x3 HTML `<table>`, `<img width="320">` animated WebP per cell (320x240, ~30 fps, ~5 s, 0.5 to
  2 MB), no burned-in text; the caption under each cell is the exact reproduce command; numbers live on a nightly
  benchmark dashboard linked by a badge.
- **Genesis**: same WebP catalogue (480x270, 15 fps, 5 s, ~300 KB each) hosted on an orphan `readme-assets` branch via
  absolute raw URLs; the famous speed claim lived in a bullet and a static teaser, not an animation.
- **Newton, NVIDIA Warp**: tables of static thumbnails linked to the example source; tidy, lifeless, no numbers.
- **Isaac Lab, Isaac Sim**: one static hero JPG, badges, prose; no animation, no evidence.
- **ManiSkill 3**: static collage plus numbers in bullets; the honest side-by-side video against Isaac Lab (same task,
  theirs on top, Isaac below) and FPS-vs-envs plots are in the docs.
- **Brax, Taichi**: uncaptioned inline GIFs, several MB each.
- **uv, hyperfine, instant-ngp, tiny-cuda-nn** (outside robotics): one speed artefact above the fold (a 10 KB SVG bar
  chart in dark/light `<picture>`, a 260 KB terminal clip, a convergence GIF with a time claim, a
  steps-vs-reference table).
- Nobody in robotics shows speed or latency as an animation, and nobody shows a renderer comparison with the
  quality number attached. That gap is the opportunity.

GitHub README constraints (confirmed against the html-pipeline sanitizer allowlist and GitHub docs):
animated GIF and WebP autoplay inside `<img>`; `<video>` pointing into the repo is stripped; video only plays from
files uploaded through the GitHub UI (`github.com/user-attachments/...`), never autoplays, 10 MB on free plans;
`width`, `height`, `align`, `loading="lazy"` survive, `style` and `class` are removed; `<picture>` with
`prefers-color-scheme` works; content column is about 1000 px, so 3x320 px or 4x240 px fits; relative image paths
break on PyPI. Keep each animation at 2 to 3 MB or less.

## 4. The plan

Order on the page: evidence first, install second, the fork history far below.

### 4a. Minimum

The effort column is the original serial estimate, kept for reference. As run (section 0), five streams run at
once with file ownership split so they do not collide: film record side (bugs 1, 4, 5, G1 rough task), film render
side (bugs 2, 3, in a worktree), WebP encoder plus the sand and Franka-Pour clips (CPU), throughput chart (CPU),
lidar through glass. The G1 renders start when the two film streams land; the README is assembled last. Hand-offs:
`runs/readme_media/handoff/`.

| step | effort | depends on |
|---|---|---|
| Fix bugs 2 and 5 (plane clipping, stray collider) in the MuJoCo film path; add a done mask to the G1 adapter (bug 1) and a stale-frame guard (bug 3) | 1 to 2 h | |
| G1 flat and G1 rough clips with the g1regress checkpoints, 5 s each, 1280x720, tier 2, OIDN | 30 min (mostly render) | the fixes |
| ANYmal on sand: encode the existing clip to WebP and a small mp4 | 15 min | nothing; no re-render |
| Lidar-through-glass 3-up (RGB, depth, point cloud) from the transparency diagnostic scene | 1 to 2 h | |
| WebP encode step in `metalsim/render/film/encode.py` (Pillow writes animated WebP; the local ffmpeg has no libwebp), 320 or 480 px, 30 fps, 5 s, 2 MB budget | 1 h | |
| Throughput bar chart as dark/light SVG from the numbers already in the README table (G1 flat and rough, Franka-Pour; MetalSim vs Isaac Lab Newton and PhysX on the L4, published 4090 marked) | 1 h | |
| README: the 3x3 table copied from MuJoCo Warp (section 0), 320x240 WebP cells, caption = reproduce command; the chart full width below the table (in a 320 px cell its text would be 4 to 7 px); media in `docs/renders/` by relative path as MuJoCo Warp does (the orphan branch is dropped); `docs/gallery` trimmed. Cells, decided 2026-10-31: row 1 G1 flat, G1 rough, ANYmal on sand; row 2 Franka-Pour, then the same Isaac Lab G1 policy (iteration 1499) in Isaac Sim 5.1 and in MetalSim, two 4:3 crops of the existing side-by-side video `docs/gallery/g1_stage_isaac_it1499.mp4` (0.5 to 4.5 s, 25 fps; PARITY §1.5); an earlier draft used stills, superseded the same day; row 3 lidar through glass as RGB, depth, two-return lidar | 1 h | all above |

Render cost is not the bottleneck: tier 2 at 1280x720 and 64 spp measured a median 0.12 s per frame during the re-film
(14-inch, Mac16,6, on battery in Low Power Mode, so an upper bound and not a timing figure), so a 5 s clip renders in
under a minute. The bottlenecks were the film-tool fixes and a camera move per cell; the policies exist (section 0).

### 4b. Deferred, add one at a time only if the six-cell page feels thin

- **Better comparison cells (next L4 session).** The current cells are crops of a 2048x576 video, 4 s, with a grey
  look. A new pair would be rendered at 640x480 for these cells, 5 s at 30 fps, tracking the robot, with a policy
  trained here (the g1regress checkpoint) as well as Isaac's, and optionally a rendering-only pair replayed from
  Isaac's recorded poses (`metalsim/render/rtx_parity.py`, the section 1.7 protocol). Only `g1_stage_isaac_*` qualify
  today; `g1_stage_metalsim_*` use pre-fix MetalSim checkpoints that fall.

- Hero race clip: split screen, same G1 scene, MetalSim on the M4 Max vs Isaac Lab on the L4, shared wall clock and
  env-steps odometer, frozen at the finish. Needs an L4 session to record the Isaac side.
- Render fidelity strip: same frame at tier 2 and the RTX references, PSNR in the caption row. The frames exist
  (`docs/gallery/fidelity_A_hold_0025.png`, `fidelity_C_drop_0050.png`, `fidelity_crop_A_hold_0025.png`), so this is
  cropping and captions, not rendering. A progressive-convergence WebP with a time counter would be new work.
- Learning curves on a wall-clock axis, MetalSim vs Isaac Lab, seeds shaded (data in `runs/il3/`, PARITY 1.5, 1.8).
- Physics-parity ghost: the PhysX drop trajectory as a translucent overlay on MetalSim's, joint error below.
- More grid cells: TRON1 on LimX's ONNX policy (`assets/tron1/limx_policy`, a known-good walker; needs a film
  adapter around `Tron1Sim`), Go2 through boxes with a walking policy, SO-101 lift (`runs/lift_baselinecfg_seed0.pt`),
  Gaussian-splat background with the robot composited (camera save path only; the path tracer does not draw splats).
- A nightly benchmark page plus badge, MuJoCo Warp style, so the numbers never go stale in the media.

### 4c. Timing runs

The minimum plan needs no timing run. The chart and any number in a caption come from measurements already in the
README and PARITY (16-inch, on power). Re-films are render-class jobs. One catch: the film tool's doc stage writes
per-stage timings into `docs/renders/<TASK>.md` and stamps the machine. Films made on the 14-inch, or on battery,
must have their doc stage re-run from a 16-inch render before those timings are committed, or the timing table
must say where it came from. The deferred race clip and a nightly benchmark page are timing work: 16-inch, on
power, `scripts/gpu_run.sh ... timing`.

## 5. Asset hygiene

- Animations in the repo, linked by relative path, as MuJoCo Warp does (it is on PyPI and accepts this). The
  orphan-branch option (Genesis) was dropped for the simpler precedent: about 1 MB per clip, about ten clips. Long side-by-side videos uploaded through the GitHub UI, linked, not committed.
- WebP, not GIF; no burned-in numbers except in the race clip; `loading="lazy"` on grid images; at most about ten
  animations and under 6 MB above the fold.
- `docs/renders/<TASK>.md` stays the provenance record per clip (assets, licences, timings, checkpoint hash).

## 6. State on 2026-10-31

- Done in the working tree, not committed: bugs 1 to 5 fixed (film tool, ray tracer env grid); G1 flat and rough
  re-filmed with the tracked g1regress checkpoints (`runs/film/g1_{flat,rough}`, docs `docs/renders/G1_*.md`); the
  README's 3x3 table, chart and number corrections; `docs/gallery` trimmed (7 files). Render and sensor tests: 194
  passed, 0 failed, 8 skipped. Agent hand-offs: `runs/readme_media/handoff/`.
- Merged as PR #83 (2026-10-31 03:26). Follow-up the same day: the G1 clips shuffled (4.0 Hz steps, 5-7 cm foot
  lift; our checkpoints were 500 and 1000 iterations). NVIDIA's published Isaac Lab 2.3 checkpoints (iterations 1499
  and 2999) now drive both films: 3.78 Hz and 7 cm lift on flat, a real stair climb on rough. The quick steps are
  Isaac's own gait (3.2-3.8 Hz in PhysX, `docs/research/speculative_contacts_2026-09-27.md`). The film tool now stamps
  the power source and pmset's power-mode setting per stage. The first G1 docs said "on battery in Low Power Mode"
  for renders made on AC; that came from a session-start reading never re-checked, and is corrected.
- Open: the doc lists the top four ranked envs as "filmed" where one is; `docs/gallery/scripts/14_g1_policy_video.py` still writes the
  removed `g1_tier2_policy.mp4`; nothing seen on github.com yet (checked through GitHub's markdown API only).
