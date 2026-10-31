# Hand-off: encode-webp (2026-10-31)

## Status
Done. WebP + small-mp4 export are in the film tool, and the six README assets exist and have been checked by eye and by number.
Two caveats are under Open: sand at 480 px fits 2 MB only at 15 fps, and the Franka clip is only 3.5 s long.

## Changed
- `metalsim/render/film/encode.py`: new `probe_source`, `read_clip`, `encode_webp`, `encode_mp4_clip`, and the CLI subcommands `webp` and `mp4`. The old `--shot/--out/--fps` CLI and `encode()` are unchanged.
- `metalsim/render/film/__main__.py`: two new flags, `--webp W[,W]` and `--webp-window START:SECONDS` (default `0:5`), used only by the encode stage. They write `docs/renders/<task>[_<shot>]_<w>px.webp` (+ `.json`) next to the docs mp4. Without `--webp`, behaviour is unchanged. I did not touch `cli.py` or `cfg.py`.
- `runs/readme_media/assets/`: 6 media files, each with a `<file>.json` (settings, budget search, source window, colour check, command).
- `runs/readme_media/scratch/encode-webp/`: `make_assets.sh` (reproduces all 6), `check.py` (verification), contact strips and check sheets.

## Outputs (runs/readme_media/assets/)
| file | bytes | WxH | frames | fps | source window | settings |
|---|---|---|---|---|---|---|
| anymal_sand.webp | 1,921,582 | 480x270 | 75 | 15 | 3.0-8.0 s (PNG 0090-0239) | q60 |
| anymal_sand_320.webp | 1,775,200 | 320x180 | 150 | 30 | 3.0-8.0 s | q70 |
| anymal_sand.mp4 | 2,203,554 | 960x540 | 150 | 30 | 3.0-8.0 s | crf 26 |
| franka_pour.webp | 384,716 | 480x270 | 105 | 30 | 0-3.5 s (whole clip) | q80 |
| franka_pour_320.webp | 217,158 | 320x180 | 98 (105 in) | 30 | 0-3.5 s | q80 |
| franka_pour.mp4 | 347,591 | 960x540 | 105 | 30 | 0-3.5 s | crf 20 |

Every WebP loops forever (loop=0) and reads back as exactly 5.000 s (sand) or 3.500 s (Franka). All six fit their budgets (WebP 2 MB, mp4 3 MB).

Reproduce commands. These are the captions; each is also stored in the asset's `.json` under `command`:
```
python -m metalsim.render.film.encode webp ~/metalsim-fresh/runs_render/sand25m/demo/frames runs/readme_media/assets/anymal_sand.webp --start 3 --seconds 5 --width 480
python -m metalsim.render.film.encode webp ~/metalsim-fresh/runs_render/sand25m/demo/frames runs/readme_media/assets/anymal_sand_320.webp --start 3 --seconds 5 --width 320
python -m metalsim.render.film.encode mp4 ~/metalsim-fresh/runs_render/sand25m/demo/frames runs/readme_media/assets/anymal_sand.mp4 --start 3 --seconds 5 --width 960
python -m metalsim.render.film.encode webp docs/renders/franka_pour.mp4 runs/readme_media/assets/franka_pour.webp --start 0 --seconds 0 --width 480
python -m metalsim.render.film.encode webp docs/renders/franka_pour.mp4 runs/readme_media/assets/franka_pour_320.webp --start 0 --seconds 0 --width 320
python -m metalsim.render.film.encode mp4 docs/renders/franka_pour.mp4 runs/readme_media/assets/franka_pour.mp4 --start 0 --seconds 0 --width 960
```
(`--seconds 0` means "to the end of the source".) For a README caption, the upstream command that made the source clip may be more useful than the encode command. For Franka that is `python -m metalsim.render.film franka_pour`; the sand clip's render command is in `~/metalsim-fresh/runs_render/sand25m/demo/log.txt`.

## Verified
- **Encoder choice.** The bundled ffmpeg (`imageio_ffmpeg`, 7.1) does have `libwebp_anim`. I still used Pillow 12.3's libwebp writer: ffmpeg decodes, trims, resamples and Lanczos-scales the clip to RGB24, and Pillow encodes it, so libwebp does its own RGB to YUV conversion.
- **Sand window.** I made a 4x4 contact strip of the 240 PNGs (`scratch/encode-webp/sand_strip.jpg`) and measured mean frame-difference energy per second: 5.45, 5.81, 6.07, 6.23, 6.51, 6.61, 6.65, 6.91. Every 5 s window was scored, and the best starts at frame 90, so I used 3.0-8.0 s. That window shows a steady trot with the full trail of footprints in the sand.
- **Franka source.** No PNG frames or `master.mp4` survive anywhere. I searched `runs/film/franka_pour`, `~/metalsim-fresh` (every franka_pour mp4 there has the same md5, 5c51ef51...) and `.-purge`. The only source is the 3.5 s, 105-frame `docs/renders/franka_pour.mp4`. Its strip (`franka_strip.jpg`) shows approach, grasp, lift, carry and pour, so I used the whole clip and cut nothing.
- **Readback.** `python scratch/encode-webp/check.py` decodes every frame of every WebP with Pillow and maps each frame back to its source frame by its timestamp. It reports frame count, total duration and size (table above), PSNR against the Lanczos-scaled source, and mean RGB:
  - anymal_sand: PSNR min 31.90 / mean 31.99 dB; mean RGB source (130.96, 117.17, 100.31) vs output (131.03, 117.37, 100.30).
  - anymal_sand_320: 32.76 / 32.97 dB.
  - franka_pour: 40.48 / 41.16 dB; mean RGB (132.43, 127.15, 126.56) vs (132.64, 127.26, 126.47).
  - franka_pour_320: 39.32 / 39.88 dB.
  - mp4s, decoded as BT.709: sand 27.7-31.4 dB, Franka 44.2-46.0 dB, mean RGB within 1 level (sand mp4 blue +1.0).
- **Frames I looked at.**
  - WebP frames 0, 25, 49, 74 (sand 480); 0, 50, 99, 149 (sand 320); 0, 35, 69, 104 (Franka 480); 0, 32, 65, 97 (Franka 320). Each was viewed above its source frame in `scratch/encode-webp/check_*.png`.
  - The worst-PSNR frame of each WebP (`worst_*.png`).
  - mp4 frames 0 and 50 (sand) and 0 and 35 (Franka) in `check_*_mp4.png`.
  - Colours and gamma match the source. The 16:9 framing is complete, with no crop or stretch. I saw no banding.
- **Bugs found and fixed while looking at output.**
  1. ffmpeg dropped frames: 26 of 30 came out of `trim,setpts` into rawvideo. After `setpts` the rawvideo output defaults to 25 fps constant frame rate and drops frames. Fixed with `-fps_mode passthrough`. `read_clip` now raises an error if the frame count is off by more than 1. Found by experiment.
  2. Grey 8x8 squares on the white cup in the Franka WebP when key frames were off (`kmax=0`). Cause: libwebp `anim_encode.c` `FlattenSimilarBlocks` carries 8x8 blocks that are within `QualityToMaxDiff(q)` (about 4 levels at q80) over from older frames. With `kmax <= 0`, `SanitizeEncoderOptions` turns key frames off for the whole clip (`/opt/homebrew/include/webp/mux.h:436-447`, https://chromium.googlesource.com/webm/libwebp/+/refs/heads/main/src/mux/anim_encode.c). The fix is gif2webp's lossy defaults, kmin=3 / kmax=5, which are also Pillow's (`.venv/.../PIL/WebPImagePlugin.py:205-209`). Smallest test: Franka frame 69 at q80 with kmax=0 vs 1 vs 3/5. Only kmax=0 showed blocks. Minimum PSNR went from 37.76 to 40.48 dB, at a size cost of 324 KB to 385 KB (`scratch/encode-webp/franka_kf_compare.png`). I found this by experiment first and confirmed the mechanism in the source afterwards.
  3. Pillow's animated writer defaults to `method=0` (`WebPImagePlugin.py:197`), not the documented 4 (https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp), so `method=4` is passed explicitly. method 6 was 30x slower for 3% smaller output (12.0 s vs 0.38 s for 30 frames; measured on battery, not a timing figure).
  4. Reading frame durations with Pillow needs `im.load()` after `seek()`; before that, `info["duration"]` is absent (same Pillow doc page, "duration ... only available after the current frame is loaded").
  5. The mp4 colour check first read about 2 levels dark. The encode was correct: decoding with swscale's default flags is inaccurate. Decoding with `in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int` matches the source to within 0.2 levels. mp4s are now tagged BT.709 primaries, transfer and matrix, TV range (via `setparams`; ffprobe confirms). Rests on experiment, plus the ffmpeg scaler docs (https://ffmpeg.org/ffmpeg-scaler.html) for the flags.
- **Film-tool path.** I ran `python -m metalsim.render.film franka_pour --stage encode --shots hero --out runs/readme_media/scratch/encode-webp/filmtest --doc runs/readme_media/scratch/encode-webp/filmtest/DOC.md --skip-preflight --webp 480,320 --webp-window 0.5:1` on 45 symlinked sand frames. It wrote `franka_pour_480px.webp` and `franka_pour_320px.webp` next to the scratch mp4, each 30 frames and 1.000 s from first frame 15. `docs/renders` was not touched.
- **GitHub limits.** These come from plan section 3, which was checked against the sanitizer allowlist and the GitHub docs. GitHub's attaching-files doc (https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files) caps images at 10 MB but does not list WebP among uploadable types. That doc covers UI uploads only, not images stored in the repo.

## Open
- **Sand at 480 px does not fit 2 MB at 30 fps.** The 25 M-grain texture costs about 34 KB per frame even at q50 (3.4 MB at 30 fps). The budget search (quality 80 down to 50, then fps 24/20/15, then width) settled on 15 fps q60. Measured alternatives at 480 px:
  - 20 fps q40: 1.95 MB
  - 30 fps q30: 2.41 MB (sand grain visibly softened; `scratch/encode-webp/sand_q_compare.png`)
  - The 320 px version is 30 fps q70, 1.78 MB.
  The orchestrator should choose. If the grid is 320 px, use `anymal_sand_320.webp`.
- **Franka is 3.5 s, not 5 s.** The source clip ends there. A longer cut needs a re-film (GPU, queue).
- **Franka is a second-generation encode.** The Franka assets come from the 1280x720 crf-22 docs mp4, not from the original PNGs.
- **Not checked in a browser.** No asset has been viewed on GitHub or in a browser, and I have no first-hand check that WebP in a repo-path `<img>` autoplays on GitHub. That rests on plan section 3 (MuJoCo Warp and Genesis do it).
- **Seam.** The loop jumps from source frame 239 back to frame 90 (the robot pose and trail jump). Every window cut from a longer clip has this.

## Next
Lay the assets out in the README grid. Pick 480 or 320 px for every cell, and settle the sand 15 fps question (re-run with `--fps 20 --min-quality 40` if 20 fps at 480 px is preferred). Publish the files to the `readme-assets` orphan branch per plan section 5, and check on a pushed branch that each `<img>` autoplays.

For G1 flat and rough, after their re-films, run `python -m metalsim.render.film g1_flat --stage encode --webp 480,320 --webp-window 0:5`. Mind the protocol: that writes into `docs/renders`.
