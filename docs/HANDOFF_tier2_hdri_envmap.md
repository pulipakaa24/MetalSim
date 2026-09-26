# Handoff: tier 2 HDR environment-map lighting (port onto main)

**Status: ported on `302bab5` (main, 2026-09-26)**. The port's files entered the tree in that commit (its subject is another agent's elliptic-cone run: the shared index swept the staged port in); the follow-up commit that carries the port's own message and this line is the record to cite. What landed: `set_environment` / `env_sampling_table` / `set_environment_pose` with MIS against main's diffuse/GGX mixture pdf, the denoiser guide on a primary miss, the non-finite guard kept, bindings flag 32 / buffer 26 / texture 1 / sampler 1 (`PTConsts` 28 words), both env-map tests plus a USD DomeLight import test and a replicator test (13 passed vs 9 before), `set_fovy`, `set_materials`, opt-in `firefly_clamp` (DECISIONS), the RGBE reader `metalsim/render/hdr.py` (no OpenCV), USD import of DomeLights (`usd_dome` custom text), `Randomizer.environment`, gallery `docs/gallery/g1_hdri_tier2.png`, costs and the dome-orientation research in `docs/research/rendering_vs_rtx_2026-09-25.md` §6 / PARITY §1.7. Convention change from this patch: the dome is oriented as Kit/RTX renders a USD DomeLight on a z-up stage (u = 0.5 faces −y, u = 0.25 faces +x), not `atan2(y, x)`. The superseded items below were not ported.

Branch `tier2-hdri-envmap`, based on `main` at `fbaab25`. It contains no renderer changes yet. It carries
uncommitted work from an old checkout (base `4e6d663`, 2026-09-23) that `main` does not have, as a
reference patch plus the pieces that apply as-is. The task is to port the feature onto `main`'s tier 2
path tracer, which was restructured after `4e6d663`, so the patch does not apply directly.

## Files on this branch

| file | what |
|---|---|
| `patches/handoff/tier2-hdri-envmap-on-4e6d663.patch` | the old diff of `metalsim/render/tier2.py`, `metalsim/render/shaders/pathtrace.metal` and `tests/test_render_tier2.py` against `4e6d663`. Reference only: read it, don't `git apply` it |
| `scripts/fetch_polyhaven_assets.sh` | fetches CC0 Poly Haven 2k equirectangular HDRIs (10 train and 3 held-out rooms) and 5 wood/laminate textures into `assets/polyhaven/{hdri,tex}/{train,heldout}` |
| `.gitignore` | ignores `assets/polyhaven/` |

## What to port (worth keeping)

### 1. Equirectangular HDR environment map, importance-sampled with MIS (the main item)

`main` only has a uniform environment: the `dome=(intensity, color)` kwarg sets one radiance
(`env_rad`) for every ray that misses. A textured DomeLight is what Isaac scenes use for backgrounds
and image-based lighting. `docs/research/replicator_2026-09-25.md` lists "dome-light HDR randomization"
as a renderer gap, and this closes it.

What the old patch implements (see its `pathtrace.metal` and `tier2.py` hunks):

- **Python:**
  - `env_sampling_table(hdr, grid=(512, 256))` builds the importance-sampling table: pixel weight
    luminance × sin θ, floored so it is never zero (MIS needs full support). It returns one float buffer
    `[marginal CDF (gh) | per-row conditional CDFs (gh*gw) | density over (u, v) (gh*gw)]` plus `gw, gh`.
  - `set_environment(hdr, yaw=0, intensity=1, grid=..., key=None)` uploads an RGBA32F texture and the
    table. `key` caches the GPU resources per map, for per-episode randomization without re-upload.
    `hdr=None` goes back to the sky colour.
  - `set_environment_pose(yaw, intensity)` only changes the constants.
- **Shader:**
  - Convention: z up; `u = atan2(y, x)/2π + 0.5 + yaw/2π`, `v = acos(z)/π`.
  - `env_pdf`: solid-angle pdf = density / (2π² sin θ).
  - `env_sample`: two binary searches (`upper_bound`) over the CDFs.
  - On a miss: add `env_radiance(d)`, weighted by the power heuristic against the previous bounce's
    BRDF-sampling pdf (`prev_pdf`; the camera ray has pdf 0, so weight 1).
  - At each hit: next-event estimation on the map (sample a direction, cast a shadow ray, then
    `brdf · N·L · L_env · w / p_env`, where `w` is the power heuristic against the BRDF pdf of that
    direction).
- **Constants and bindings:** it used `PTConsts.flags` bit 1 (env map on), extra constants `env_yaw`,
  `env_intensity`, `env_w`, `env_h`, `buffer(22)` for the CDF and `texture(1)` for the map.

**Porting notes for `main`:**

- **Bindings.** On `main`, `PTConsts` is 16 words with `env_scale` and `clip_far` in floats 14–15,
  flags 1/2/4/8/16 are taken (`FLAG_RESET`, `FLAG_MATMODEL`, `FLAG_TONEMAP_RTX`, `FLAG_AUX`,
  `FLAG_CENTER0`), and buffers 22–25 are the AUX outputs. Use a new flag bit (32), grow `PTConsts`
  (Python `self.consts` is `np.zeros(16, np.uint32)` in `Tier2Renderer.__init__`), and bind the CDF at
  `buffer(26)` and the map at `texture(1)`.
- **Scale.** Multiply the map by `c.env_scale` so the USD `dome` intensity keeps its meaning. A
  textured `DomeLight` is `dome=(intensity, color)` × texture.
- **The MIS pdf must match `main`'s BSDF sampling.** `main` evaluates `brdf_any(model, ...)` with
  per-material models (UsdPreviewSurface/OmniPBR layering), but it still samples the same
  diffuse/GGX mixture with `pd = clamp((1-metallic)*0.5 + 0.5*(1-specw), 0.1, 0.95)`. The mixture pdf
  used by the old patch is therefore still right. Compute it once and reuse it for the env-NEE weight
  and `prev_pdf`, as the old patch did by moving `pd` above the NEE block.
- **Denoiser guide.** On a primary miss, `main` writes a normalised `env_rad` into the albedo sum.
  With a map, use the normalised `env_radiance(d)` instead, or the OIDN/a-trous guide sees a flat
  background.
- **Non-finite guard.** Keep `main`'s guard against non-finite radiance after the env term.
- **USD import.** It could pass a DomeLight's `inputs:texture:file` through to `set_environment` (the
  latlong format). Check `metalsim/scene` for where dome lights are read.
- **Replicator.** A randomizer that picks a map from a list by `key` per episode is the natural next
  step (`metalsim/replicator`).

**Tests to port** (from the patch's `tests/test_render_tier2.py` hunk; pytest's cache on the old checkout recorded them run on 2026-09-23 with no failures; not re-run since):

- `test_env_map_furnace`: a constant map of 0.5 on a white Lambertian plane gives mean 0.5 ± 0.01.
- `test_env_map_importance_sampling_unbiased`: a map with a small, very bright "window" matches the
  analytic plane radiance `_env_irradiance_radiance` within 2 % at yaw 0 and 1.3. A uniform sampling
  table stays unbiased (5 %) but is more than 3× noisier.

On `main`, construct these with `headlight=False` in place of the old `flags` bit 3.

### 2. Small `Tier2Renderer` helpers

- `set_fovy(fovy_deg, env=None)` changes vertical FOV per env or for all, by rewriting
  `cam_spec[:, 0:2]` (fx = fy) and copying it to `t_cam_spec`. Check that `main` still keeps
  `cam_spec` / `t_cam_spec` under those names.
- `set_materials(materials)` replaces the (G, 16) material table and rewrites the GPU buffer. On
  `main`, write to `self.mat_buf` when `material_model != "legacy"`, or to the ray tracer's material
  buffer otherwise.
- The optional per-sample luminance clamp against fireflies is a biased option, off by default.
  Add it only as an explicit opt-in knob, with the bias documented, per the fidelity-first rule.

## Superseded, do not port

- The old a-trous denoiser (`denoise=True`, old `shaders/denoise.metal`): `main` has
  `denoise='atrous' | 'oidn'`.
- The old ACES + sRGB flag: `main` has `tonemap='rtx'`.
- The headlight-off flag bit: `main` has `headlight=False`.
- The albedo output: `main` has the `FLAG_AUX` outputs.
- `test_denoiser_against_reference`: `main` has `test_rtx_tonemap_and_denoisers_on_furnace`.

## Dropped

`metalsim/learn/yam_act/` (ACT imitation learning on so101Sim's YAM scene, path-traced) was
abandoned: the best held-out result was 4/56 successes. Its code, demos, gallery video and stash were
deleted on 2026-09-25 and are not recoverable from this branch.

## Done when

- Both env-map tests pass on `main`'s renderer.
- `pytest tests/test_render_tier2.py -q` is otherwise unchanged.
- The parity presets (`usd_scene_kwargs`, `mujoco_scene_kwargs`) render bit-for-bit as before when no
  map is set.
- A gallery frame of the G1 under a Poly Haven HDRI exists, and the GAPS / DECISIONS / CHANGELOG rows
  are added.
