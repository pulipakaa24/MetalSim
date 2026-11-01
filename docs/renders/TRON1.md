# TRON1 wheel-legged drive, LimX's policy: a metalsim render

LimX's TRON1 wheel-legged robot (WF_TRON1A) driven by LimX's own RL policy (the ONNX policy of LimX's `tron1-rl-deploy-python`, Apache-2.0) in MetalSim's TRON1 simulation at its nominal realism settings (latency, drive model, sensor noise, joint friction, sensor pack), on LimX's own model and meshes: drive forward, turn, slow down, stop. Rendered on the 14-inch MacBook Pro (Mac16,6), not in a timing run, so its Performance table is not a timing figure.

**hero** (351 frames, 7.0 s at 50 fps): [video](tron1.mp4) (1.2 MB, H.264)

![hero, frame 88](tron1_still1.jpg)
![hero, frame 175](tron1_still2.jpg)
![hero, frame 280](tron1_still3.jpg)

## What is shown

- 16 envs simulated, filmed: 8, 6, 14, 13 (selection: automatic, the adapter ranking).
  - env 8: terminated False, base_z 0.000, end_frame -1
  - env 6: terminated False, base_z 0.000, end_frame -1
  - env 14: terminated False, base_z 0.000, end_frame -1
  - env 13: terminated False, base_z 0.000, end_frame -1
- Policy: `assets/tron1/limx_policy/WF_TRON1A/policy/isaacgym/policy.onnx` (onnx (LimX isaacgym export: encoder.onnx + policy.onnx)); physics: metalsim.tron1.sim.Tron1Sim on CPU MuJoCo 3.14.0 (LimX's WF_TRON1A MJCF, physics step 0.5 ms) with MetalSim's TRON1 realism layer at 'nominal' (metalsim.tron1.realism.SimParams.nominal()); controller 500 Hz, policy 50 Hz; one Tron1Sim per env, seeds 0..15.

## Performance

Measured on the machine below; one frame = one policy step (Isaac Lab's `render_fps` = 1 / step_dt).

| stage | measurement |
|---|---|
| simulate (record) | 351 frames x 16 envs: 30 ms/frame mean (30 median, first 30 incl. graph capture), 10.4 s total; env build 0.3 s; stage wall 10.9 s |
| render twin (CPU) | 21 geoms, 9 meshes, 2 textures; build 0.0 s, poses 0.0 s |
| render: hero | 1280x720 (1x1 panels of 1280x720), tier 2 path tracing, 64 spp (16 passes x 4), 3 bounces, denoiser oidn; 0.34 s/frame mean (0.34 median, 0.37 max), 117.7 s for 351 frames; first frame: one pass 51 ms, 9 ms per further pass; set-up 1.0 s |
| encode: hero | 3.4 s (master CRF 16 + docs copy CRF 22, libx264 slow) |
| render: readme | 640x480 (1x1 panels of 640x480), tier 2 path tracing, 64 spp (16 passes x 4), 3 bounces, denoiser oidn; 0.14 s/frame mean (0.14 median, 0.15 max), 47.6 s for 351 frames; first frame: one pass 21 ms, 5 ms per further pass; set-up 0.9 s |
| end to end | 3.0 min (record + render + encode, sum of stage walls) |

## Hardware and pins

- Mac16,6 (Apple M4 Max, 40 GPU cores, 64 GB), macOS 27.0.1 (26A434).
- Power as each stage stamped it (`pmset -g batt` and `pmset -g`; `hardware.power`: the record stage's when it wrote rec.json, a render shot's at its first frame): record: on battery (69 %, discharging), power-mode setting High Power (pmset `powermode 2`), 2026-10-31 16:05 (-0700); render hero: on battery (69 %, discharging), power-mode setting High Power (pmset `powermode 2`), 2026-10-31 16:05 (-0700); render readme: on battery (66 %, discharging), power-mode setting High Power (pmset `powermode 2`), 2026-10-31 16:07 (-0700).
- metalsim `ed66d8ef3e+dirty`; record stage (main venv): warp 1.18.0.dev20260917 (`4127c48183`), mujoco 3.14.0, mujoco_warp 3.14.0 (`8eb51fe637`), torch 2.14.0, numpy 2.5.3.
- render stage (main venv): warp 1.18.0.dev20260917 (`4127c48183`), mujoco 3.14.0, numpy 2.5.3.

## Assets

| asset | source | licence | where from |
|---|---|---|---|
| LimX TRON1 wheel-foot model WF_TRON1A (MJCF robot.xml and STL meshes) | LimX Dynamics robot-description | Apache-2.0 (assets/tron1/LICENSE-robot-description) | https://github.com/limxdynamics/robot-description (`in this repository: assets/tron1/WF_TRON1A`) |
| Policy: LimX's WF_TRON1A RL policy, isaacgym export (encoder.onnx + policy.onnx, params.yaml; sha256 bfc4617f...) | LimX Dynamics tron1-rl-deploy-python | Apache-2.0 (assets/tron1/LICENSE-rl-deploy-python) | https://github.com/limxdynamics/tron1-rl-deploy-python/blob/035e4c9e2f24e916fa5e8363ac6f0f788a2e1e8c/controllers/model/WF_TRON1A (`in this repository: assets/tron1/limx_policy/WF_TRON1A`) |
| Controller port (WheelfootController.py -> metalsim/tron1/limx_policy.py) | LimX Dynamics tron1-rl-deploy-python | Apache-2.0 (assets/tron1/LICENSE-rl-deploy-python) | https://github.com/limxdynamics/tron1-rl-deploy-python/blob/035e4c9e2f24e916fa5e8363ac6f0f788a2e1e8c/controllers/WheelfootController.py |
| Industrial HDRI (machine_shop_02, 2k) | Poly Haven | CC0 1.0 (Poly Haven, polyhaven.com/license; fetched, not committed) | https://polyhaven.com/a/machine_shop_02 (`scripts/fetch_polyhaven_assets.sh heldout`) |
| Floor texture (concrete_floor_worn_001, 1k diffuse) | Poly Haven | CC0 1.0 (Poly Haven, polyhaven.com/license; fetched, not committed) | https://polyhaven.com/a/concrete_floor_worn_001 (`scripts/fetch_polyhaven_assets.sh heldout`) |

## How it was made

**The policy.** LimX's RL policy for the wheel-foot TRON1 (WF_TRON1A), from LimX's
`tron1-rl-deploy-python` (<https://github.com/limxdynamics/tron1-rl-deploy-python>, Apache-2.0, commit `035e4c9e2f24e916fa5e8363ac6f0f788a2e1e8c`):
`controllers/model/WF_TRON1A/policy/isaacgym/encoder.onnx` and `policy.onnx` with `params.yaml`. The copies in
`assets/tron1/limx_policy/WF_TRON1A/` are byte-identical to LimX's at that commit (fetched and compared on
2026-10-31; sha256 `bfc4617f65cef4b1c952156352dd2fee0deffd68458959a20b4bc0bb0285d85b` for the policy,
`f79e29d12b670aa9078b4a825f3b30e75018efb15c8108487a0bf2c222e78ba6` for the encoder), and the adapter checks the hashes before it runs. That
repository is LimX's Python controller for its robots: `main.py` connects through LimX's SDK to LimX's MuJoCo
simulator by default, or to the robot when given its IP address (`main.py` lines 27-39), and loads
`policy/$RL_TYPE/` with `RL_TYPE` isaacgym or isaaclab (the README's example is isaacgym; the isaaclab export uses
another joint order, which the port does not handle). It is LimX's open policy, not the closed controller of the
robot's Remote Control Mode, which is not documented (docs/TRON1.md, "What runs the robot today"). It is played by
`metalsim/tron1/limx_policy.py`, a port of LimX's `controllers/WheelfootController.py` at the same commit: 28
observations (gyro x 0.25, projected gravity, six leg joint angles, eight joint velocities x 0.05, last actions), a
10-frame history into the encoder (3 outputs), then the policy on [encoder, observation, command x (2, 2, 0.25)] to 8
actions at 50 Hz; legs to `q = 0.25 a` with Kp 42 and Kd 2.5, wheels to `dq = 0.8 a` through Kd 0.8, with LimX's
torque-bound action clips (80 / 40 N m). Two differences from LimX's script, both in the port's docstring: commands
are scaled as in LimX's training rather than its joystick mapping, and the 1 s stand-up blend is skipped because the
robot starts standing in the policy's default pose. Nothing is tuned for the film.

**MetalSim's TRON1 simulation, nominal settings.** The rollout is the TRON1 evaluation's own path
(`metalsim/tron1/evaluate.py`, `episode`): one `Tron1Sim` per robot (`metalsim/tron1/sim.py`), LimX's WF_TRON1A MJCF
on CPU MuJoCo with a 0.5 ms physics step, stepped by `Tron1Sim.run` with the controller at 500 Hz. Between the policy
and the physics is MetalSim's realism layer at its `nominal` level (`SimParams.nominal()`,
`metalsim/tron1/realism.py`; docs/TRON1.md, "Realism layer"), unchanged: commands reach the drives 4 ms late plus
0.5 ms mean jitter, with 0.2 % packet loss; sensor messages 1 ms late; drives run LimX's mode-0 law on 17-bit encoder
angles and 1 kHz finite-difference velocities, with a torque-speed envelope and a 0.5 ms current lag; joint Coulomb
friction 0.3 N m (legs) and 0.1 N m (wheels), viscous damping and rotor armature; IMU white noise, bias random walk
and AHRS tilt wander and yaw drift; a 1.5 kg sensor pack 8 cm above the base (drawn: the grey box and the lidar
cylinder on top); ground friction 0.8; and the tyre as a smooth fit to the real 5 cm rounded tread instead of the
1 cm disc of LimX's MuJoCo file. The full parameter set is in `rec.json` `adapter.sim_params`. Each robot gets its
own noise seed (0 to 15), as each evaluation episode does; no pushes.

**Commands.** The same schedule for every robot, within the evaluation's ranges (vx in [-1, 1] m/s, yaw rate in
[-0.6, 0.6] rad/s): stand until 0.5 s, then vx 1.0 m/s; at 2.0 s vx 0.8 m/s with a 0.6 rad/s left turn; at 3.5 s vx
0.4 m/s straight; at 4.5 s stop (`tron1.SCHEDULE`).

**Measured before filming.** 16 robots for 8 s with this schedule (`Tron1Limx` itself, nominal settings): **0 of 16
fell**. Forward-speed tracking error 0.240 m/s RMS (median over the robots; 0.235 to 0.247), yaw-rate error 0.130 rad/s
(0.127 to 0.133), peak pitch 14.3 deg and peak roll 3.4 deg (95th percentiles), with the evaluation's definitions
(`evaluate.episode`: RMS from 0.5 s after each command change; here sampled every 20 ms). For comparison, the
evaluation's nominal row (100 episodes x 20 s of random commands and two shoves each) is 0 / 100 falls, 0.27 m/s,
0.17 rad/s, 15 deg, 5 deg (docs/TRON1.md, "Results"). How it tracks, medians over the 16 robots: the policy
accelerates slowly, 0.63 m/s from 1.0 to 1.5 s after the 1.0 m/s command and 0.8 m/s first at about 2.0 s; in the turn
it yaws at 0.87 rad/s against the 0.6 commanded (74 deg of heading over the 1.5 s turn against 52 deg) and slows to
0.64 m/s; it holds 0.38 m/s against 0.4; stopped, it creeps at 0.03 m/s. The 16 robots differ only in their sensor
and transport noise, and their errors agree to within 0.013 m/s and 0.006 rad/s. The 7 s take is the first 7 s of the
same rollout (frame for frame: the per-frame speeds of the filmed robot are identical): 0 of 16 fell, 0.264 m/s and
0.143 rad/s RMS (`rec.json` `adapter.tracking`; the shorter window has less of the stop, where the error is small).
`rec.json` `adapter.tracking` also holds the filmed robot's per-frame speed, yaw rate, errors, pitch and roll, and
each robot's fall time (none).

**How the robot was chosen.** `envs="auto"` ranks robots that did not fall first, then the most typical tracker:
the smallest distance of its (forward-speed, yaw-rate) RMS errors, each divided by the median over the robots, from
(1, 1) (`Tron1Limx.rank`). That was robot 8 (0.265 m/s, 0.143 rad/s in the take). Both shots film it.

**Camera.** Both shots track `base_Link` (Isaac's `origin_type="asset_body"`), the tracked point smoothed by a centred
1 s moving average, with Kit's default lens (36 deg vertical field of view) and the G1 films' azimuth, looking at
0.35 m below the base origin (the middle of the 0.95 m robot, tyres to lidar). `hero` (1280x720; the video and
stills on this page): eye (1.99, -1.53, 0.18) m from the base, 2.57 m from the look-at point, pitched down 12 deg (the
G1 films: 21 deg) so the top sixth of the frame shows the machine shop above the horizon; the robot fills about 0.55
of the frame height. `readme` (640x480, 4:3, for the README grid): the same direction at 0.75 times the distance
(eye (1.49, -1.15, 0.05) m), so the robot fills about three quarters of the frame height, plus the slow 10 deg/s orbit
of the G1 films' README shot (symmetric about the middle of the take, as MuJoCo Warp's README clips orbit).

**Rendering.** Tier 2 (the path tracer), 64 samples per pixel (16 passes x 4), 3 bounces, Open Image Denoise, ACES tone
map, sRGB; lit by Poly Haven's `machine_shop_02` (an industrial hall, 2k) as an importance-sampled dome, with Poly
Haven's `concrete_floor_worn_001` (1k diffuse) on LimX's `floor` plane in place of its checker. The texture is 3 m x 3 m
of real floor (Poly Haven's dimensions) and is laid at that size, one image per 3 m (`ground_tile_m` 3.0; the film tool
sets MuJoCo's `texrepeat` to 2 / 3, as MuJoCo lays a `texuniform` texture once per 2 / `texrepeat` m: an infinite
plane's texture coordinates are half its world position, MuJoCo's `makePlane`, mirrored in
`metalsim/render/builtin_geoms.py`; `metalsim/render/film/render.py`, `ground_texrepeat`). The render twin is the
simulated model itself (`build_spec` at the nominal level), posed per frame from the recorded state: LimX's STL meshes
in LimX's colours as authored (`rgba`, read as linear albedo like every MuJoCo model in these films: the body 0.79 /
0.82 / 0.93, the hip links white, the wheels 0.27 grey), the sensor pack's two geoms, and the colliders hidden
(`rec.json` `twin.visibility`). Exposure is fixed at 1.4: the films' automatic rule (frame median luminance to 0.2
before ACES) chose 2.6 for this scene's first frame, because the frame is mostly a dark floor (the concrete's median
linear albedo is 0.095, the G1 films' laminate 0.237), and washed the robot out; 1.4 is that rule with the floor taken
as a mid grey (x 0.095 / 0.18).

**README clip.** [tron1.webp](tron1.webp) is the `readme` shot from 0.5 s to 5.5 s (frames 25 to 274: drive, turn,
slow down, stop), resampled from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos),
lossy animated WebP from Pillow at quality 70, 33 ms per frame, looping; 150 frames. Command, run from the repository
root after the render stage: `python -m metalsim.render.film.encode webp runs/film/tron1/readme/frames
docs/renders/tron1.webp --start 0.5 --seconds 5 --src-fps 50 --fps 30`.

**Checked by eye.** The wheels turn visibly: their three-spoke hubs rotate about 5 deg per frame while driving
(`rec.npz` wheel-mesh poses relative to the knee: 5.4 deg per frame from 1 to 2 s, 0.7 deg per frame once stopped).
Scale: each wheel's 0.1298 m rim, projected from its recorded pose through the film camera, lies on the rendered tyre
outline (hero frames 88 and 175, readme frame 150).

**This take, and where its timings come from.** Recorded, rendered and encoded on 2026-10-31 on the 14-inch MacBook Pro
(Mac16,6), at metalsim `ed66d8ef` plus the uncommitted TRON1 film code (hence `+dirty` in the pins), with the main
venv's MuJoCo Warp checkout at its pinned commit `8eb51fe` (Warp at `4127c481`, not its pin `52c09cdd`; the physics here
is CPU MuJoCo, which neither checkout touches: this take is frame for frame the 14:38 take recorded on the stale
`07a51a6` checkout, and the README WebP is byte-identical). This was not a timing run: quote nothing on this page as
MetalSim's speed (MetalSim's throughput figures are in `docs/PARITY.md` §1.4 and §1.8). The record stage simulates on
the CPU (one CPU MuJoCo `Tron1Sim` per robot, run one after another) and runs LimX's ONNX files with onnxruntime 1.31.0
(CPU, one thread; the `tron1` extra of `pyproject.toml`, which the record stage needs: `pip install -e '.[tron1]'`; the
pins below list a fixed set of modules and omit it). Power, as the stages stamped it ("Hardware and pins" below): on
battery and discharging at every stamp (69 % when the record stage wrote `rec.json` at 16:05, 69 % at the hero shot's
first frame at 16:05 and 66 % at the readme shot's at 16:07, local time), with pmset's power-mode setting at High Power
(`powermode 2`: the configured setting, which does not show that the machine ran in High Power Mode); the encode and doc
stages stamp nothing. Record and render ran as GPU-queue jobs of kind render, one at a time, into `runs/film/tron1`:
`--stage record`, then `--stage render` (both shots); then `--stage encode --shots hero`, the README WebP command above
and `--stage doc` (CPU).

## Reproduce

```bash
scripts/fetch_polyhaven_assets.sh heldout    # CC0 HDRI and textures
python -m metalsim.render.film tron1   # record, render, encode, doc (main venv)
# or stage by stage (resumable):
python -m metalsim.render.film tron1 --stage record   # runs in the .venv venv
python -m metalsim.render.film tron1 --stage render
python -m metalsim.render.film tron1 --stage encode
python -m metalsim.render.film tron1 --stage doc
```

Generated by `python -m metalsim.render.film tron1 --stage doc` from the stage logs (`runs/film/tron1/rec/rec.json`, `render.json`, `render_frames.jsonl`, `encode.json` per shot).
