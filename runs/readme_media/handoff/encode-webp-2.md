# Hand-off: encode-webp-2 (2026-10-31)

## Status
Done. `encode_webp` now defaults to MuJoCo Warp's recipe, and both clips are remade at 320x240 and checked by eye.
One number for the orchestrator to decide on: with the exact recipe, sand is 3.34 MB, above MuJoCo Warp's largest clip (2.0 MB). See Open.

## Changed
- `metalsim/render/film/encode.py`:
  - New `parse_crop_x` and `crop_box`.
  - `read_clip` gains `height` and `crop_x`: ffmpeg `crop` before the Lanczos `scale`; `height=None` keeps the old aspect-preserving path.
  - `_webp_bytes` is now the recipe: `save_all, duration=<int ms>, loop=0, quality=q`, with no method/kmin/kmax/allow_mixed.
  - `encode_webp` defaults: 320x240, 30 fps, 5 s, q70, centred crop, no budget. The search runs only when `max_bytes` is set.
  - `encode_mp4_clip` defaults to 640x480 with the same crop.
  - CLI: `--height` (0 = keep aspect), `--crop-x` (fraction in [0,1], or px if > 1 or `Npx`), and `--budget MB` for webp (replaces `--max-mb`; mp4 keeps `--max-mb`).
  - The `.json` gains `crop`, `frame_ms` and `recipe`.
  - Kept: `-fps_mode passthrough`, the frame-count check, and `im.load()` on readback.
  - Readback now maps timestamps by `frame_ms`. The old `stamp*fps/1000` was off by about 1 frame at the clip end with integer 33 ms.
- `metalsim/render/film/__main__.py`:
  - `--webp [WxH]` is now a bare flag, default 320x240, writing `docs/renders/<task>[_<shot>].webp`. It was `--webp W[,W]`, which wrote `_<w>px.webp`.
  - New `--webp-crop-x`. `--webp-window` is unchanged.
  - The WebP rate is `min(30, film fps)`.
- `runs/readme_media/assets/`: new `anymal_sand.webp`, `anymal_sand.mp4`, `franka_pour.webp`, `franka_pour.mp4` (+ `.json` each).
- `runs/readme_media/scratch/encode-webp-2/old/`: the previous 480 px webps, 320x180 `_320` webps and 960x540 mp4s, with their json.
- `runs/readme_media/scratch/encode-webp-2/`:
  - `make_assets.sh` reproduces the 4 assets.
  - Scripts: `crop_sheet.py`, `kf_test.py`, `kf_zoom.py`, `check.py`.
  - Their sheets, plus `filmtest/`.

## Outputs (runs/readme_media/assets/)
| file | bytes | WxH | frames (in) | per frame | duration | crop (source px) | settings |
|---|---|---|---|---|---|---|---|
| anymal_sand.webp | 3,342,500 | 320x240 | 150 (150) | 33 ms | 4.950 s | 1440x1080 at x=240 of 1920 (centred) | q70, Pillow defaults, loop 0 |
| franka_pour.webp | 294,242 | 320x240 | 83 (105) | 33 ms | 3.465 s | 960x720 at x=0 of 1280 (left-aligned) | q70, Pillow defaults, loop 0 |
| anymal_sand.mp4 | 2,553,207 | 640x480 | 150 | 30 fps | 5.0 s | same as webp | crf 22 (budget 3 MB) |
| franka_pour.mp4 | 248,052 | 640x480 | 105 | 30 fps | 3.5 s | same as webp | crf 20 |

- Sources:
  - Sand: window 3.0–8.0 s = PNG 0090–0239 of `~/metalsim-fresh/runs_render/sand25m/demo/frames`.
  - Franka: the whole 3.5 s of `docs/renders/franka_pour.mp4`.
- Franka has 83 frames because libwebp merges near-identical consecutive frames into one longer frame. Durations still sum to 105 x 33 ms.
- PSNR against the cropped and scaled source: sand webp 33.21 dB, Franka webp 39.13 dB, sand mp4 31.87 dB, Franka mp4 44.14 dB.
- Mean RGB is in each `.json`.
- With integer 33 ms frames, playback runs at 30.3 fps (5 s of source plays in 4.95 s). This is the recipe as briefed and is what MuJoCo Warp gets too.

## Verified
- **Recipe sources.**
  - `mjw_src/mujoco_warp/record.py:39-40` (fps 30, quality 70), `:151-159` (`save(..., save_all, append_images, duration=int(...), loop=0, quality=...)`), `:173-174` (320x240).
  - Pillow's animated-writer defaults in `.venv/lib/python3.13/site-packages/PIL/WebPImagePlugin.py`: `:195` quality 80, `:197` method 0, `:207-210` kmin 3 / kmax 5 for lossy. So dropping the explicit `kmin=3, kmax=5` changes nothing. Dropping `method=4` changes method 4 to method 0.
- **Grey-block test (Franka).** `python runs/readme_media/scratch/encode-webp-2/kf_test.py runs/readme_media/scratch/encode-webp-2`, at 320x240, crop x=0, q70, 33 ms:

  | variant | bytes | PSNR min | PSNR mean |
  |---|---|---|---|
  | defaults (method 0, kmin 3 / kmax 5) | 294,242 | 38.04 (frame 19) | 38.92 |
  | method=4 | 191,744 | 38.02 | 38.95 |
  | kmax=0 (positive control) | 269,992 | 33.82 (frame 84) | 35.55 |

  - I looked at `kf_compare.png` (frames 69, 95) and `kf_zoom.png` (cup region at 4x, frames 19, 69, 84).
  - kmax=0 shows the grey 8x8 squares on the gripper, cup and table.
  - The defaults and method=4 show none.
  - Conclusion: Pillow's defaults do not bring the artefact back, so the explicit settings are dropped. This rests on that experiment plus the mechanism cited in the encode-webp hand-off: key frames every 3–5 frames stop the block carry-over.
- **Crop choice, by looking.**
  - Contact sheets with the candidate crop edges drawn: `crop_sheet.py` → `franka_crops.png` (12 frames, x=0/80/160) and `sand_crops.png` (9 frames 90–239, x=0/120/240). I also used the old strips in `scratch/encode-webp/*_strip.jpg`.
  - Sand: the robot spans about 0.32–0.70 of the width and its shadow reaches about 0.81 in every sampled frame. The centred crop keeps both, with the robot centred, and keeps most of the footprint trail on the left.
  - Franka: the action spans the source's left edge (gripper and box at x≈0 in frames 55–57) to about 0.77 (the wrist in frame 0). No 4:3 box holds both.
    - Left-aligned (x=0) keeps both cups, the gripper and the whole pour stream in every frame.
    - The cost: in frames 0 to about 3, the wrist housing is clipped by ≤30 source px (about 10 px at 320). From frame 5 on, the hand is fully inside (`franka_edges.png`).
    - A centred crop would cut the frame-57 gripper and box.
- **Decoded frames vs source.** `python runs/readme_media/scratch/encode-webp-2/check.py runs/readme_media/scratch/encode-webp-2` writes `check_anymal_sand.png` and `check_franka_pour.png`: the source at 16:9 with the crop box on top, the decoded WebP frame below.
  - Sand webp frames 0/50/99/149 (src 90/140/189/239): robot and shadow in frame, colours and sand texture match. No blocks.
  - Franka webp frames 0/27/55/82 (src 0/27/55/104): approach, lift, carry and pour all in frame, stream visible. No blocks.
  - Both read back with `loop=0`.
  - mp4s: `mp4_strips.jpg` (4 frames each), same framing, nothing wrong seen.
- **Size comparison, for information only (not the default).** Sand q70 at 320x240: Pillow defaults 3,342,500 B; `method=4` 2,599,076 B. For Franka, method 4 is 35% smaller (above). No visible difference at equal PSNR in the zooms.
- **Film-tool path.**
  - Command: `python -m metalsim.render.film franka_pour --stage encode --shots hero --out runs/readme_media/scratch/encode-webp-2/filmtest --doc runs/readme_media/scratch/encode-webp-2/filmtest/DOC.md --skip-preflight --webp --webp-window 0.5:1` on 45 symlinked sand PNGs (stub `rec/rec.json` with fps 30).
  - Result: `filmtest/franka_pour.webp` at 320x240, centred crop 1440x1080+240, 30 frames, 0.990 s, q70, 638 KB, first frame 15. Exit 0.
  - Repeated with `--webp-crop-x 0px`: crop at x=0.
  - `git status docs/renders` was unchanged before and after.
- **Other CLI checks.**
  - `encode webp ... --budget 0.2` searched q70/60/50 at 30 fps, then 24 fps, and settled on 24 fps q50, 0.18 MB. At 24 fps the duration is 42 ms per frame (23.8 fps), as `round(1000/fps)` implies.
  - `--crop-x 400` on a 1280-wide source raises "crop x offset 400 px is outside 0..320".

## Open
- **Sand is 3.34 MB**, against 0.47–2.0 MB for MuJoCo Warp's clips. That is the exact recipe at q70; the orchestrator decides. Options, all measured:
  - `method=4` (one kwarg, not the recipe): 2.60 MB.
  - `--budget 2`: drops quality and then fps.
  - Accept it.
- **Franka frames 0 to about 3** clip the wrist by about 10 output px (see Verified). Fixing it would mean a re-film with a wider or re-aimed camera.
- **Franka is 3.5 s, not 5 s**, and is a second-generation encode from the crf-22 docs mp4 (unchanged from the encode-webp hand-off).
- **Not checked in a browser or on GitHub.**
- **The `--webp` flag changed shape.** The encode-webp hand-off's `--webp 480,320` now fails to parse as `WxH`. Use `--webp` or `--webp 320x240`.

## Next
- Decide on the sand size. To switch both clips to method 4, add `method=4` to `ims[0].save(...)` in `_webp_bytes` (`metalsim/render/film/encode.py`) and run `runs/readme_media/scratch/encode-webp-2/make_assets.sh`.
- Then lay out the README grid as MuJoCo Warp does: a 3-column table of `<img width="320">` with `runs/readme_media/assets/anymal_sand.webp` and `franka_pour.webp`, linking the 640x480 mp4s.
- Publish to the `readme-assets` branch (plan section 5) and confirm autoplay on a pushed branch.
- For G1 flat and rough after their re-films, run `python -m metalsim.render.film g1_flat --stage encode --webp --webp-window 0:5 [--webp-crop-x X]`. This writes into `docs/renders`, so mind the protocol.
