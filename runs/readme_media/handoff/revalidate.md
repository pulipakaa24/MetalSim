# Hand-off: revalidate (2026-10-31)

## Status
**Done.** I re-ran all five items with the main venv's MuJoCo Warp checkout at the pinned `8eb51fe`. Nothing is
committed. README.md and the plan doc are untouched.

One finding is outside the brief and changes how to read every number below.
- **The Warp fork checkout is stale too.** `upstream/warp-metalsim` is at `4127c481` (2026-09-25).
- Its pin is `52c09cdd`, set in `scripts/setup_warp.sh:13` and in `docs/STATUS.md:57`.
  `gh api repos/pulipakaa24/warp/compare/4127c481...52c09cdd` gives `ahead_by` 70. The commit is not even fetched
  locally: `git cat-file` fails, and `git fetch --dry-run` shows `4127c481..52c09cdd metalsim`.
- What it costs: MuJoCo Warp `8eb51fe` checks for the fork's `metal_capture_range_begin` and quietly disables its ICB
  early exit when that is missing (`upstream/mujoco_warp/mujoco_warp/_src/solver.py:5221-5226`). `4127c481` does not
  have it (grep of `warp/_src/context.py`).
  - The early exit is "bitwise the fixed-count loop" (`solver.py:5214`), so that part changes speed, not results.
  - The other 69 Warp commits were not checked.
- So every "new" number here comes from MuJoCo Warp `8eb51fe` on Warp `4127c481`, not from the fully pinned stack. Every
  film's pins line says warp `4127c48183`.
- I changed neither checkout: they are shared, and Warp needs a native rebuild (`scripts/setup_warp.sh`, or
  `scripts/fork_sync.sh` as that script's note says).

The fork itself:
- `git -C upstream/mujoco_warp log -1` gives `8eb51fe6`.
- `.venv/bin/python -c "import mujoco_warp ..."` imports `upstream/mujoco_warp/mujoco_warp/__init__.py`, and HEAD there
  is `8eb51fe637437108185f5a4750b7f5f64ef26624`.
- The reflog records the fast-forward at 15:47:53. All my runs started after 15:49.

## Old against new
"Old" means the hand-offs' numbers, from the stale checkout `07a51a6`. "New" means `8eb51fe` (on Warp `4127c481`).

| item | measure | old (07a51a6) | new (8eb51fe) | material? |
|---|---|---|---|---|
| 1 G1 Unitree sim2sim | pelvis distance to Unitree's CPU run at 5 s, worlds 0-3 | 1.7 / 1.3 / 2.2 / 1.5 µm | 1.8 / 1.4 / 1.8 / 1.7 µm | no |
| | at 8 s | 3.6 / 2.8 / 3.7 / 3.0 µm | 3.8 / 4.4 / 4.3 / 4.0 µm | no (doc table updated) |
| | cadence L / R (FFT), touch-downs per s | 1.251 / 1.251 Hz; 1.143 / 1.286 | identical | no |
| | foot contact per frame; max policy-output difference; max joint difference | 400 / 400; 3.0e-5; 7e-7 rad | 400 / 400; 2.7e-5; 7e-7 rad | no |
| | spread between the 4 GPU worlds at 8 s | 1.2 µm | 2.8 µm | no |
| 1 G1 Unitree take | filmed world 2, gait from 1 to 6.02 s | 1.245 Hz, 1.20 / 1.39 td/s, apex 4.2 / 4.4 cm, 0.48 m/s | identical | no |
| | any body position, old rec against new, 301 frames x 4 worlds | | max 1.9 µm | no |
| 2 TRON1, 16 robots x 8 s | falls; vx / yaw-rate RMSE, median (max); pitch / roll p95 | 0/16; 0.2404 (0.2472) / 0.1299 (0.1325); 14.32 / 3.36 deg | identical, per robot too | no |
| 2 TRON1 take | tracking; README WebP; mp4; stills | 0/16, 0.264 / 0.143 | identical; WebP, mp4 and stills byte-identical (md5) | no |
| 3 SO-101, 12 V | successes / 100 (box still up at 6 s) | 98 (94); a second old run gave 98 (94) | 98 (92), 97 (94), 98 (94): three runs | no |
| | time to lift, mean | 0.91 s; 0.97 s | 0.94, 0.90, 0.93 s | no |
| | seeds that differ from CPU | 10021, 10027; 10027, 10098 | 10021, 10027 (twice); 10021, 10027, 10098 | no |
| 3 SO-101, 7.4 V | successes / 100 (box still up at 6 s) | 78 (65) | **73 (66), 78 (71)**: two runs | central value no; the spread is new |
| | seeds with the same outcome as CPU | 81 | 86, 81 | |
| 4 G1 on Isaac's bumps (0.02-0.10), MetalSim default | falls / 16 | **1** | **0, 2, 2**: three runs | **yes**, see below |
| | the same, all heightfield pairs on GJK/EPA (`MJW_HFIELD_PLANE_CONTACTS=0`, the old "upstream MJW" set-up) | **6** | **5, 1**: two runs | **yes** |
| | variants 0.04 / 0.06 / 0.08, default | 0 / 0 / 0 | 0 / 0 / 2 (one run) | |
| | CPU MuJoCo (not re-run; the fork does not touch it) | 11 | (11) | |
| 4 contact check, 3000 spheres | MetalSim default: count equal to the exact set | **81.9 %** (543 fewer, 31 none, 0.06 % of normals > 10 deg) | **88.8 %** (126 fewer, 211 more, 0 none, 6.1 % > 10 deg, normal p95 21.8 deg, depth p95 8.0 mm) | **yes** |
| | old "upstream MJW" column against the new default | 88.8 % | new default contacts are **bitwise** the old upstream-prism contacts (`np.array_equal` on every array) | |

### What the bumps result means
The contact claim moves exactly as the hfield-contacts hand-off predicted:
- `b6305308` sends primitives to upstream's GJK/EPA prism path by default (`HFIELD_PLANE_CONTACTS_PRIMITIVES = False`,
  `collision_convex.py:42-48, 479-481`);
- the "plane" and "prism" modes of `contact_check.py` now give identical arrays.

The fall count does **not** move as predicted. It is not "6/16 like the upstream column":
- The default set-up gave 0, 2 and 2 falls.
- The old upstream set-up, re-run, gave 5 and 1.

Two things explain this:
1. **The GPU rollout is not repeatable run to run on Metal by default.** Float atomics are applied in arrival order:
   `docs/GAPS.md:144`. Repeatability is opt-in (`BatchSimOptions(deterministic=True)`, `docs/GUIDE.md:275`), and that mode
   refuses heightfield GJK/EPA (`collision_convex.py:1795-1805`). On this chaotic task, one run's fall count is one draw
   from a spread of at least 0-2 (default) or 1-5 (all GJK/EPA).
   - Old single numbers like "1 of 16" and "6 of 16" are draws of the same kind.
2. **The default is not the old "upstream" set-up.**
   - Unitree's 12-dof model also has collision meshes: for example `pelvis_contour_link` and `torso_link` in
     `assets/unitree_rl_gym/.../g1_12dof.xml:45,47`.
   - With the default, meshes of 256 vertices or fewer keep the fork's plane path. `MJW_HFIELD_PLANE_CONTACTS=0` puts
     them on GJK/EPA too.
   - This rests on code reading. I did not separate it from point 1 by experiment.

What holds in every run:
- MetalSim's MuJoCo Warp falls far less than CPU MuJoCo on Isaac's tile: 0-5 against 11 of 16.
- The new default is not exact: its contact count matches the exact set 88.8 % of the time, against CPU MuJoCo's 99.6 %.
  6.1 % of its normals are more than 10 deg off, and the depth error p95 is 8 mm.

## Claims this affects

**Fixed, in files I own:**
- `docs/renders/G1_UNITREE.md`:
  - the comparison table: 5 s "1.3 to 2.2 um" becomes "1.4 to 1.8 um"; 8 s "2.8 to 3.7 um" becomes "3.8 to 4.4 um";
  - the "This take" paragraph: the commit becomes `ed66d8ef` plus uncommitted changes, it names the pinned fork, and it
    says which take this one replaces;
  - the power prose: it said AC, 36 %, 13:55. It now says battery, 86/86/83 %, 15:52/15:52/15:53.
- `docs/renders/TRON1.md`:
  - the power prose: it said AC, 14:38-14:46. It now says battery, 69/69/66 %, 16:05/16:05/16:07;
  - the "This take" sentence now says the physics is CPU MuJoCo and the take is identical.
- Both docs' "This take" sentences also say that Warp is at `4127c481`, not its pin `52c09cdd`.
- `runs/film/g1_unitree/sim2sim_check/compare.json`: replaced by the `8eb51fe` run. It is the same at its printed mm
  precision; only float noise in the swing p95 differs.

**Stale, not mine, for whoever owns them:**
1. **The generators of those doc lines.** The next `--stage doc` will undo my doc edits unless these change too:
   - `metalsim/render/film/tasks.py:460-461` (the G1_UNITREE table);
   - `tasks.py:416` (`G1_UNITREE_POWER`);
   - the `G1_UNITREE_NOTES` "This take" sentence near `tasks.py:505` ("at metalsim `dfec9ebf` plus the uncommitted
     Unitree port");
   - `metalsim/render/film/tron1.py:330` (`TRON1_POWER`).

   Copy the replacement text from the two docs.
2. **`docs/renders/G1_BUMPS.md`, lines 92-112** (physics paragraph and contact table):
   - "11 fall in CPU MuJoCo and 1 in MetalSim's MuJoCo Warp";
   - "MetalSim's MuJoCo Warp (the fork's per-triangle plane contact) | 81.9 % | 543 | 31 | 0.06 %";
   - "in a crease it keeps one face and drops the other";
   - "With upstream MuJoCo Warp's contacts the policy falls in 6 of 16".

   On the pinned fork, MetalSim's row is the upstream row: 88.8 %, bitwise. The falls are 0-2 of 16 (3 runs) and 1-5
   with all pairs on GJK/EPA (2 runs). The "MetalSim MJW falls" column of its level table (0/0/0/1) is one stale draw.
   Its generator is `G1_BUMPS_NOTES` in `tasks.py`.
3. **`docs/renders/README.md:12`** (bumps row): "MuJoCo Warp, whose contacts for the G1's sphere feet on a heightfield
   drop true contacts (1 of 16 falls there)" is stale. Suggested wording: "...whose heightfield contacts for the G1's
   sphere feet (upstream's GJK/EPA in the pinned fork) are not exact, and which keeps the policy walking (0-2 of 16
   falls over three runs)".
4. **The g1-bumps hand-off's proposed README sentence** ("drops contacts ... keeps it walking (1 fall in 16)") is stale in
   the same way. The engine decision for the film still holds: CPU MuJoCo matches the exact contact set, and MuJoCo Warp
   still falls far less.
5. **The hfield-contacts hand-off's "baseline to beat"**: the 88.8 % and 6.1 % hold, bitwise. "6/16 falls" does not:
   the new default gives 0-2. A fix should be judged by `contact_check.py` and by falls over several runs, against CPU
   MuJoCo's 11. One run is not enough.
6. **`docs/renders/SO101_LIFT.md`** (lines 104, 257, 320) and **`docs/renders/README.md:13`**: "98/100" and "78 on
   MuJoCo Warp" are single GPU draws.
   - At 12 V, 97-98 in 3 runs; the claim stands.
   - At 7.4 V, 73-78 in 2 runs. Better: "73-78 of 100 over two runs (77 on CPU)".
   - Its pins line still says `07a51a63bc`.
   - The so101-smooth stream owns `so101.py`; its md5 was `dd51e416...` before and after my evals, unchanged.
7. **Every doc filmed today names `07a51a63bc` in its pins**:
   - G1_FLAT.md and G1_ROUGH.md: not re-recorded by me;
   - SO101_LIFT.md;
   - G1_BUMPS.md: its physics is CPU MuJoCo, but its "MetalSim MJW" numbers are not.

   All of them, mine included, name Warp `4127c48183`, which is not its pin `52c09cdd`.
8. **README.md:63-64** ("within 3 µm ... after 5 s", "1.25 gait cycles per second"): still true (max 1.8 µm at 5 s).
   No change needed. TRON1, SO-101 and bumps are not in README.md yet.

## Test sweep
- Command: one queue job, `scripts/gpu_run.sh revalidate-test-sweep render 60 --
  runs/readme_media/scratch/revalidate/tests/run_sweep.sh`.
- Files: the 28 `tests/test_*.py` files whose names match film, render, tier, sensor, lidar, depth, raytrace, env_grid,
  plane, terrain, splat, particle or tron1. The list is in `scratch/revalidate/tests/files.txt`.
  - That is provenance-docs' 26 plus `test_terrain.py` and `test_tron1.py`.
  - `tests/isaac_terrain_ref.py` matches too, but it is a helper module, not a test file.
- Result: **231 passed, 0 failed, 8 skipped**, in 151.9 s (`sweep.log`, `junit.xml`).
- The 8 skips are the same as provenance-docs':
  - Newton is not installed: `test_render_newton_bridge`, `test_render_newton_perworld`, `test_film.py:102`;
  - 5 in `test_render_splat_public`, whose fetched scenes are absent.
- Nothing failed, so no HEAD comparison was needed.

## Changed
- `runs/film/g1_unitree/`:
  - `rec/rec.json`, `rec.npz` and `twin*` (the last two ignored);
  - `{hero,readme}/render.json`, `render_frames.jsonl`, `frames/`;
  - `hero/encode.json`;
  - `sim2sim_check/compare.json`;
  - old frames moved by the render stage to `{hero,readme}/stale/` (ignored).
- `docs/renders/g1_unitree.webp` (new bytes, 0.70 MB, 150 frames, PSNR 36.73 dB), `g1_unitree.mp4`,
  `g1_unitree_still{1,2,3}.jpg` (new bytes).
- `docs/renders/G1_UNITREE.md`: doc stage, then the 4 edits above.
- `runs/film/tron1/`:
  - `rec/rec.json`, `rec.npz`, `twin*`;
  - `{hero,readme}/render.json`, `render_frames.jsonl`, `frames/`;
  - `hero/encode.json`.
  - Before this, the main tree had only the JSON files; the frames were in the tron1 worktree.
- `docs/renders/tron1.webp`, `tron1.mp4`, `tron1_still{1,2,3}.jpg`: rewritten, byte-identical to before.
- `docs/renders/TRON1.md`: doc stage, then the 2 edits above.
- WebP sidecars: the encoder wrote `docs/renders/{g1_unitree,tron1}.webp.json`. I moved them to
  `runs/readme_media/assets/` as the streams did, overwriting the old ones; copies of the old ones are in
  `scratch/revalidate/old/`.
- Scratch, all in `runs/readme_media/scratch/revalidate/`:
  - `old/`: the previous take JSONs, both docs, rec.npz clones, media md5s;
  - `g1u/`: `cmp/`, `fullprec.json`;
  - `tron1/`: `measure.py`, `measure_16x8s_nominal.json`;
  - `so101/`: `run_evals.sh`, 5 run dirs;
  - `bumps/`: `measure/`, `measure_r2`, `measure_r3`, `measure_allgjk_r{1,2}`, `repeat.sh`, `contact_check/`;
  - `tests/`;
  - `look/`, `strips.py`, `g1u_fullprec.py`, `logs/`.

## Verified
**Commands** (repository root; every GPU job through `scripts/gpu_run.sh <name> render <min> -- ...`, one at a time)
- 1, G1 Unitree sim2sim (GPU): `runs/film/g1_unitree/sim2sim_check/compare.py runs/readme_media/scratch/g1-unitree/ref
  runs/readme_media/scratch/revalidate/g1u/cmp 4 8`.
  - Full precision came from `scratch/revalidate/g1u_fullprec.py`, run against the old and the new `metalsim.npz`, giving
    `g1u/fullprec.json`.
  - It reproduces the old hand-off's figures exactly from the old npz.
- 1, film:
  - `-m metalsim.render.film g1_unitree --stage record`, then `--stage render` (GPU). Cache keys: hero
    `b7d497f8ffa31ffd5cf7`, readme `ce7034c91bc5ca1dd7e9`; 0 frame corrections.
  - Then `--stage encode --shots hero` and
    `python -m metalsim.render.film.encode webp runs/film/g1_unitree/readme/frames docs/renders/g1_unitree.webp --start
    0.5 --seconds 5 --src-fps 50 --fps 30`, then `--stage doc`.
- 2, TRON1 measurement (CPU, as the tron1 stream ran it): `scratch/revalidate/tron1/measure.py`. It runs `Tron1Limx`
  with TRON1's film settings, 16 robots, 400 frames, then `.tracking()`.
- 2, film:
  - one queue job ran `tron1 --stage record`, then `--stage render`. Cache keys: hero `e380feb638b0e45fd9f2`, readme
    `f43293eac6050257a3fe`; 4 frame corrections, as before.
  - The keys differ from the worktree take's; the frames are byte-identical.
  - Then encode, the WebP with the same flags (`runs/film/tron1/readme/frames docs/renders/tron1.webp`), then doc.
- 3, SO-101 (one GPU job, `scratch/revalidate/so101/run_evals.sh`):
  - `runs/film/so101_lift/port/compare.py runs/readme_media/scratch/so101/ref <out>` twice;
  - `runs/film/so101_lift_74v/port/compare_servo.py 7v4 runs/readme_media/scratch/so101-74v/ref_7v4 <out>` twice;
  - `compare_servo.py 12v` once.
  - Capacity: ITERATIONS cap hit in 8-10 worlds at 12 V and 20 at 7.4 V; LS_ITERATIONS in 100. No overflow.
- 4, bumps:
  - GPU: `runs/film/g1_bumps/measure/measure.py gpu runs/readme_media/scratch/revalidate/bumps/measure`, all 4 levels;
  - `scratch/revalidate/bumps/repeat.sh`: level 0.10 twice more with the default, and twice with
    `MJW_HFIELD_PLANE_CONTACTS=0`;
  - CPU: `contact_check.py plane|prism|compare scratch/revalidate/bumps/contact_check`, with the old `contacts_exact.npz`
    and `contacts_c.npz` copied in. Neither depends on the fork.
  - The engine labels inside the new `contact_check.json` still read "fork plane contacts". That is the script's fixed
    text; the contacts are GJK/EPA.
- Power, read live:
  - 15:49: battery, 87 %, discharging, `powermode 2`;
  - later readings from the stage stamps: 86 % (15:52) to 66 % (16:07), all on battery.
  - Mac16,6. No timing number is quoted.

**Frames I looked at** (Read tool; sheets made by `scratch/revalidate/strips.py`, in `scratch/revalidate/look/`):
- `g1u_webp_strip.jpg`: every 15th frame of the new WebP.
  - The robot walks upright, legs alternating, and the orbit turns the planks.
  - It fills about 3/4 of the height; the floor is complete; no stray boxes.
- `g1u_hero_150_161.jpg`: 12 consecutive hero frames. One foot finishes its swing and lands flat; the stance foot stays
  planted.
- `g1u_stills.jpg`: the three new stills, mid-stride.
- Numeric check: hero frame 150, old (in `stale/`) against new, PSNR 64.9 dB, max pixel difference 3/255. The look is
  unchanged.
- `tron1_webp_strip.jpg`:
  - the drive, leaning forward, then the turn with the orbit, then the stop;
  - the machine shop is across the top;
  - the wheels sit on their shadows.
- `tron1_hero_75_86.jpg`: 12 consecutive hero frames, robot centred, driving.
- `tron1_stills.jpg`.

## Open
- **The Warp fork is at `4127c481`, not its pin `52c09cdd`** (Status). None of today's numbers, mine included, are on
  the fully pinned stack.
- **Is the bumps change "default config" or "noise"?** Not separated:
  - 3 + 2 runs show the spread overlaps (default 0-2, all GJK/EPA 1-5);
  - the old single values, 1 and 6, are each inside one of those spreads.
- **GPU run-to-run spread on these tasks** is now measured only roughly: SO-101 7.4 V 73-78, bumps 0-5. Any single-run
  figure in the docs should say so, or come from several runs or from deterministic mode where it applies.
- I did not re-run G1_FLAT / G1_ROUGH (not in the brief). They were recorded on `07a51a6`.
  - The G1 velocity task's default `contact_cfg` uses elliptic cone rows (`metalsim/learn/g1_velocity.py:646`), and the
    films use `contact_cfg="recommended"` (`tasks.py:276`).
  - Elliptic cones got a new default Newton path in this fork range (`b895bff5`, "mode 2 (default)").
  - Whether "recommended" is elliptic, and whether the takes move, is not checked.

## Next
1. (Orchestrator) Decide whether to bring `upstream/warp-metalsim` to `52c09cdd` and rebuild (`scripts/setup_warp.sh`).
   Do it while no GPU jobs run; every agent shares the checkout. Then re-run items 1, 3 and 4 with this brief's commands
   (`scratch/revalidate/so101/run_evals.sh` and `bumps/repeat.sh` are one queue job each, about 5 min).
2. Apply the generator edits in Claims, point 1, so `--stage doc` keeps the corrected text.
3. Fix `G1_BUMPS.md` / `G1_BUMPS_NOTES`, `docs/renders/README.md:12-13` and `SO101_LIFT.md` as in Claims, points 2-6.
4. Optional: re-measure G1_FLAT / G1_ROUGH on the pinned stack. The G1_ROUGH take is the one that could move, because of
   the elliptic default.
