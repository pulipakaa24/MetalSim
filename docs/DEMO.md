# Demo: what works at `demo-2026-10`

This page covers each demo that runs on this tag: the command to run it, the numbers the shipped docs measured for it,
and the limits that are still open. Every number below is quoted from the doc named next to it, and none was measured
for this page.

Timings come from the 16-inch M4 Max (40-core GPU, 64 GB, macOS 26) unless a line says otherwise. The 14-inch M4 Max
was used for correctness only.

Install first, as `docs/GUIDE.md` §1 describes (`pip install -e ".[full]"`, the Warp / MuJoCo Warp forks,
`scripts/setup_newton152.sh`, the asset fetch scripts). Run every command from the repository root.

**Verified:** `scripts/fresh_install_check.sh` ran on commit `168637f0` from a brand-new clone on both Macs.
- 23 of 24 steps passed on each Mac, including all six demo smokes (steps 20–25 below).
- The one failure, `test_go2_boxes_gpu` in step 11, was fixed in two steps:
  - `bf8ee2ba` compares depth with each pixel ray's first hit.
  - `ff6b63f8` makes the lidar keep its configured name.
- `tests/test_compose.py` then passed 15/15 on the 16-inch at `ff6b63f8` (`runs/checks/fix3/test_compose_16in.log`).
- The fresh-install results table is in PR #51.

## Demos

### G1 walking (MuJoCo Warp, Isaac Lab's G1 velocity task)

```bash
python -m metalsim.learn.g1_velocity 64 flat                       # environment smoke (fresh-install step 20)
python -m metalsim.learn.g1_velocity 64 flat train 2 runs/fresh/g1_flat.log runs/fresh/g1_flat.pt 0.0025 --seed 0
python -m metalsim.render.film g1_flat                             # film: record, path-trace, encode
```

Measured (README "Measured state", `docs/PARITY.md` §1.4–1.5, §1.8):
- **Flat-terrain training, return at iteration 1000:** +26.8 ± 0.2 over three seeds, against Isaac Lab 2.3.2 + PhysX's
  +27.3 (one seed).
- **Isaac Lab 3.0-EA's G1 flat task:** +27.0 ± 1.4 at iteration 1499, against Isaac's +27.5.
- **Throughput at 4,096 envs, full PPO loop:**
  - elliptic default: 37.3–37.7 K env-steps/s;
  - pyramidal: 56.9–57.5 K env-steps/s;
  - reference: Isaac Lab 3.0's env step alone on the L4 is 72.0 K (Newton / MuJoCo Warp).
- **Film:** `docs/renders/g1_flat.mp4`. Its record / render / encode stages ran on the 14-inch from a `.[full]` install
  (`runs/checks/fix1/`).

### Franka-Pour (Isaac Lab 3.0-EA `IsaacContrib-Franka-Pour`: granular MPM media poured between cups)

```bash
.venv-newton152/bin/python -m metalsim.learn.franka_pour_ppo bench --num-envs 16 --mode step --steps 5      # step 21
.venv-newton152/bin/python -m metalsim.learn.franka_pour_ppo train --num-envs 16 --iterations 1 --save-interval 1 --out runs/fresh/franka_pour
python -m metalsim.render.film franka_pour                          # film + doc: docs/renders/FRANKA_POUR.md
```

Measured:
- **Full PPO loop at 1,024 envs:** 992 env-steps/s, about 3.1× Isaac Lab 3.0-EA on the L4, which runs 317 (README;
  `docs/PARITY.md` §1b).
- **Trained policy, 2,048 envs, 300 iterations:** 47.7 % success over 2,048 first episodes
  (`runs/franka_pour/eval/eval_2048_isaac_budget.json`).
- **Film:** 1920×1080 at 256 spp, 3.96 s per frame, 13.8 min end to end for both shots (`docs/renders/FRANKA_POUR.md`).

### ANYmal C on MPM sand (Newton 1.5.2's `example_mpm_anymal`, path-traced)

```bash
.venv-newton152/bin/python scripts/particle_demo/sim_anymal.py --out runs/demo/anymal_small --seconds 1.5 --bed -0.3 -0.3 0 0.3 1.0 0.1   # step 23
python scripts/particle_demo/render_anymal.py --sim runs/demo/anymal_small --out runs/demo/anymal_small_frames --frames 0:2 --width 480 --height 270 --passes 4 --grains 8 --decor 0 0 0 0   # step 24
```

Measured (`metalsim/physics/mpm/MPM.md` §8k, `metalsim/render/PARTICLES.md` "Demo"):
- **Captured coupled step:** 29.9–31.2 ms per frame against the L4's 24.2, that is 1.23–1.29×.
- **Full demo (14-inch):** 669,372 sand particles at 46 ms per simulated frame.

### Box on sand (`MPMCoupling`: MuJoCo Warp `BatchSim` + the vendored implicit MPM, two-way)

```bash
python metalsim/physics/mpm/probes/box_on_sand.py --engine metalsim --steps 60 --out runs/fresh/box_on_sand.npz   # step 22
```

Measured (`metalsim/physics/mpm/MPM.md` §11):
- **Against Isaac's coupling path:** every particle is bitwise equal until the box reaches the sand. After that the box
  stays within 1.7e-7 m.
- **Captured coupled step:**

  | worlds | ms per step |
  |---|---|
  | 1 | 13.7 |
  | 64 | 24.3 |
  | 1,024 | 223.6 |

- **Resting box, lagged exchange (converged solve):** settles 10.2 mm.

### The 25 M-grain render (ANYmal on sand at demo scale)

```bash
.venv-newton152/bin/python scripts/particle_demo/sim_anymal.py --out runs/demo/anymal_sim --seconds 9
python scripts/particle_demo/render_anymal.py --sim runs/demo/anymal_sim --out runs/demo/frames --frames 0:240 \
    --width 1920 --height 1080 --passes 32 --grains 27
```

Measured (`metalsim/render/PARTICLES.md`; logs in `runs/sand25m_16in/`):
- **Grains:** 25.12 M, structure built in 654 ms.
- **1080p frame at 128 spp:** 5.1 s, using 10.5 GB of Metal memory.
- **Earlier 240-frame clip (14-inch):** 10.4 s per frame.

The fresh-install check runs the small version only (step 24). Run the full scale on a machine whose other work is
saved: a very large acceleration-structure build once wedged a GPU on macOS 26. That was fixed by the two-level
structure in `PARTICLES.md`.

### Transparency and lidar (ray-traced sensors through glass)

```bash
python scripts/diagnostics/sensor_transparency_cost.py --n 64 --out runs/fresh/sensor_transparency_64.json   # step 25
```

Measured (`docs/RENDER_COVERAGE.md` §3e):
- **Accuracy:** depth and lidar match `mj_ray` within 0.2 mm, and partial returns through glass match the analytic
  intensities within 1e-4.
- **Cost, lidar of 16×360 beams:**

  | scene | 1,024 envs | 4,096 envs |
  |---|---|---|
  | plain | 1.64 ms | 6.86 ms |
  | glass, default `"transmit"` mode | 5.77 ms | 23.34 ms |

### Gaussian-splat render

```bash
scripts/fetch_splat_scenes.sh dxgl          # CC0 object splat (Poly Haven model), open
scripts/fetch_splat_scenes.sh interiorgs    # InteriorGS 0204_840158: gated on Hugging Face (accept the terms, then HF_TOKEN or a CLI login)
python -m pytest -q tests/test_render_splat_public.py
python scripts/splat_bench.py verify && python scripts/splat_bench.py bench
```

Measured on InteriorGS 0204_840158 (972,596 splats, 640×360; `metalsim/render/SPLAT.md` §4–5):
- **Against an independent float64 reference:** PSNR 58.9 dB.
- **Speed:** 0.68 ms for one camera, 0.43 ms per camera at 8 (2,309 cameras/s).

Licences:
- InteriorGS is non-commercial and not redistributable. It is fetched by the script and never committed.
- The dxgl chair is CC0.

The fresh-install check's render suite (step 12) ran the synthetic splat tests. The real-data tests skip there
because the data is not fetched.

### Scenes from a YAML file, and the filming tool

```bash
python -m metalsim.run configs/scenes/go2_boxes.yaml --check
python -m metalsim.run configs/scenes/go2_boxes.yaml --steps 100 --save-every 5 --save runs/scenes/go2_boxes
python -m metalsim.run configs/scenes/go2_boxes.yaml --film
python -m metalsim.render.film --list
```

Measured (`runs/scenes/go2_boxes/run/run.json`, 16-inch, 4 envs):
- **Control step:** 6.0 ms median.
- **Tier-1 head camera at 640×360:** 1.7 ms.
- **Lidar of 16×360 beams:** 0.6 ms.

Where these commands have run:
- `tests/test_compose.py` runs in the fresh-install check's rigid suite.
- The headless run and `--film` also ran on the 14-inch from a `.[full]` install (`runs/checks/fix1/`).

The scene file uses Isaac Lab 3.0 names; see `docs/SCENES.md`. Filmed tasks are listed in `docs/renders/README.md`.

## Known limits

- **Sand physics** (`metalsim/physics/mpm/MPM.md` §11). This is the same solver as Isaac Lab 3.0.
  - Newton 1.5.2's implicit-MPM sand piles 7–9° below atan μ: 14.7° / 25–27° / 32.6° against 21.8° / 34.2° / 42.0°.
  - A body resting on sand settles 10 mm with the lagged exchange. With Isaac's staggered exchange it sinks through.
  - Drops penetrate 30–39 mm, where 10–25 mm is expected.
- **ANYmal on sand is one-way coupled by design.** The robot is kinematic for the sand (`docs/NOW.md`).
- **Coupled scenes are slower than the L4.** ANYmal takes about 30 ms per frame against the L4's 24.2
  (`metalsim/physics/mpm/MPM.md` §8k).
- **BatchSim is not run-to-run bitwise repeatable by default.** The opt-in deterministic mode (PR #29) still fails on
  the captured G1 at 1,024 worlds.
- **Franka-Pour learning against Isaac:** equal through iteration 49. At 50–59 Isaac's mean success is 0.035 higher,
  which is not significant (p = 0.09) but not ruled out (`docs/PARITY.md` §1b.5).
- **Particles are not in the ray-traced sensors.** The ray-traced depth camera and lidar do not see MPM particles, and
  tier 1 draws them without ray-traced shadows (`docs/RENDER_COVERAGE.md` §1c).
- **Filming:**
  - The path tracer does not draw Gaussian splats yet, so a splat-background scene is recorded through its cameras
    (`--save`), not filmed.
  - The Franka-Pour hero shot is 3.5 s, the length of the successful episode it shows.
