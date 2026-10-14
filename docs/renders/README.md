# Renders

Tasks filmed with metalsim's own renderer, each with a doc that says what is shown, how long every stage took on
which machine, every asset with its source and licence, and the commands to reproduce it.

| film | what | doc |
|---|---|---|
| Franka-Pour | trained policy pours granular MPM media between two cups (Isaac Lab 3.0-EA task); one env and a 2 x 2 grid | [FRANKA_POUR.md](FRANKA_POUR.md) |
| G1 flat walk | the tool's MuJoCo-path smoke test: Isaac Lab's G1 flat velocity policy, 5 s, 1280x720, 128 spp | [g1_flat.mp4](g1_flat.mp4) ([still](g1_flat_still2.jpg)) |

## Making one

Needs the documented install (`pip install -e ".[full]"`, docs/GUIDE.md): Open Image Denoise comes with the core
dependency `pyoidn` (a macOS arm64 wheel that bundles OIDN; MIT binding, OIDN Apache-2.0), ffmpeg with the `film`
extra (`imageio-ffmpeg`). The tool checks both before it records anything and prints the fix; `--no-denoise` renders
without OIDN (said so in the doc).

```bash
python -m metalsim.render.film --list                 # filmable tasks
python -m metalsim.render.film franka_pour            # record, render, encode, doc
python -m metalsim.render.film franka_pour --stage render --frames 0:30:10 --passes 8   # quick preview frames
```

The tool (`metalsim/render/film/`) runs in four file-based stages, so the path-traced render happens offline, after
the rollout, in the venv that has the renderer:

1. **record** (the task's venv): the task's recorder adapter rolls out the policy; per frame the bodies (and MPM
   particles) are captured and turned into the world poses of a render twin: an MJCF of the scene's visual geometry
   (Newton scenes through `newton_bridge.NewtonRenderSource`, MuJoCo scenes by forward kinematics).
2. **render** (main venv): tier-2 path tracing per shot (one env, or a 2 x 2 grid), particles as render grains, OIDN.
3. **encode**: H.264 master next to the frames, a 1280-wide copy and JPEG stills here.
4. **doc**: `docs/renders/<TASK>.md` generated from the stage JSONs (`runs/film/<task>/`).

Camera and recording settings use Isaac Lab's names (`ViewerCfg`: `eye`, `lookat`, `origin_type` world / env /
asset_root / asset_body, `env_index`, `body_name`, `resolution`; `VideoRecorderCfg`: `fps` = 1 / step_dt,
`video_length`). A new task needs a `FilmTask` entry in `metalsim/render/film/tasks.py` (camera, envs, look, assets)
and a recorder adapter in `adapters.py` (build the env, load the policy, step); see the Franka-Pour (Newton) and G1
(MuJoCo Warp) adapters.

Fits but not yet ported: the ANYmal-on-sand demo (`scripts/particle_demo/`) is the same record-then-render shape; its
extras (a sand bed larger than the sim's, decoration grains, a surface-depth cull of the 25 M grains, a tracking orbit)
would become look options and an adapter around Newton's `example_mpm_anymal`.
