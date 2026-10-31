# Hand-off: lidar-glass (2026-10-31)

## Status
Done for this increment. The clip was rebuilt for the coordinator's format change: three separate 4:3 views (RGB,
depth, lidar), each 150 frames at 640x480, with no labels. The WebP encode (320x240, quality 70) is left to the
orchestrator and the WebP stream.

## Changed
- `scripts/readme_media/lidar_glass.py` (new): builds the scene, renders RGB with tier 2, traces the depth camera and lidar, draws the point cloud, and encodes the mp4s.
- `runs/readme_media/assets/lidar_glass_{rgb,depth,lidar}_frames/frame_0000..0149.png` (new): 150 frames per view at 640x480.
- `runs/readme_media/assets/lidar_glass_{rgb,depth,lidar}.mp4` (new): H.264, yuv420p, CRF 20, 640x480, 30 fps, 5.000 s, 150 frames (203 KB / 25 KB / 205 KB).
- `runs/readme_media/scratch/lidar-glass/`: `lidar_glass.json` (settings, exposure 0.4744, per-frame timings, provenance), `preview_320.mp4` and `contact_320.jpg` (the three views side by side at 320x240), `with_vs_without_frame_0075_320.png`, `no_panes_frame_0075_{320,rgb,depth,lidar}.png`, `full_00{24,75,110}.png`, `glass_tints.png`.
- I removed my earlier 3-up strip outputs (`assets/lidar_glass_frames/`, `assets/lidar_glass.mp4`), which the format change replaced.

## Verified
- **Reproduce command (README caption):** `scripts/gpu_run.sh lidar-glass render 10 -- .venv/bin/python scripts/readme_media/lidar_glass.py --compare 75`. Without `--compare 75` it renders only the clip.
- **GPU time:** the final job was one queue job of kind render, taking 146 s of wall time. It ran 150 frames in 127.4 s, a median of 0.83 s per frame (the slowest was 1.37 s). Per frame, RGB took 0.58 s (640x480 at 32 passes x 4 spp, plus OIDN) and the sensors took 2 ms; the rest is CPU drawing. Set-up took 1.4 s. The machine is the Mac16,6 on battery in Low Power Mode, so these times are not performance figures. All jobs together, including about 10 previews of 3 to 15 s each, used about 7 GPU minutes.
- **Reproducible:** frame 0 re-rendered with the delivered exposure is byte-identical in all three views (max abs diff 0). Tier-2 noise depends on the frame counter (`tier2.py:449,500`), so a frame matches the full run only when it is rendered in the same order.
- **Frames I looked at:** full-resolution rows of frames 24, 75 and 110 (`full_00{24,75,110}.png`); a 1 fps contact sheet of `preview_320.mp4` covering t = 0..4 s; and the with/without-panes pair at frame 75.
  - The glass is visible in RGB: a tinted, reflecting sheet with clear top and side edges, and the box and ball show through it.
  - Depth: the pane is one flat region. The ball is visible on open floor (frame 0), cut in half at the pane edge (frame 24) and gone behind the pane (frame 75).
  - Lidar: the pane's echoes are cyan, and the ball's echoes stay visible behind them on every frame.
  - The loop is seamless: frames 149 and 0 look the same.
- **With versus without panes, frame 75:**
  - Lidar with panes: 30,933 echoes, 12,072 of them on glass (all first returns), and 8,438 beams with two returns.
  - Lidar without panes: 18,886 echoes, none on glass, and no beam with two returns.
  - Depth without panes shows the ball and box where the pane was (`with_vs_without_frame_0075_320.png`).
- **Lidar output checks** (on dumped arrays from a preview of the same pane layout):
  - Points are in env-0 frame, which is the world frame here: `|p - sensor| - range` < 7e-7.
  - The ball's surface gets 255 first-return points on open floor and 227 second-return points behind the pane.
- **Sources for the setup:**
  - `max_returns=2` and reading back `range`, `n_returns`, `points` and `slot`: `tests/test_sensor_transparency.py` (`_beam`, `test_lidar_modes_analytic`).
  - Transmit mode, and depth "surface" equal to `mj_ray`, MuJoCo and Isaac glass depth: `research_notes/Render sensor fidelity/transparency_area_lights.md` section 2, and `docs/RENDER_COVERAGE.md` section 3e and the transparency row.
  - `hits_per_ray=1`: with the default (`= max_returns`), an opaque hit is re-cast at `transmission` = 1.0 and gives a phantom second echo. Pane passes do not count against `hits_per_ray` (`rt.metal:305-320`, `tr_step` ~`:400-410`).
  - `min_echo_sep=0.1`: the default 0.4 m merges echoes closer than that into one range-averaged echo (`rt.metal:309-311`).
  - A separate sensor RayTracer: the sensors leave sites out by default, while `Tier2Renderer` would draw them (`raytrace.py:96-103`). The RGB renderer is given a `sitegroup` of 0.
  - Glass returns are found by mapping the slot to the geom through `rt.tables.geoms`.
- **What rests on experiment only:** the pane colour (0.35, 0.48, 0.55) and the lidar viewpoint, glass-dot opacity and dot size were chosen by looking at candidates at 320x240.

## Open
- The scene departs from the diagnostic in look only, not in sensor parameters:
  - The pane colour is (0.35, 0.48, 0.55) instead of (0.6, 0.8, 0.9). Alpha stays 0.3; the sensors read alpha and reflectance, not colour.
  - The objects are rearranged for a camera, and the sphere moves.
- At alpha 0.3, tier 2's presence model makes the glass look hazy or frosted rather than clear.
- The lidar view's colouring needs the caption: cyan = echoes on glass; other echoes coloured by range, with near points bright, on the same inferno scale (0.4-2.8 m) as the depth view. The white ring is the sensor.
- The bottom quarter of the lidar view is the lidar's empty blind zone near the sensor.
- No WebPs were made; that is the orchestrator's job.
- Nothing is committed.

## Next
1. Make the 320x240 WebPs from `runs/readme_media/assets/lidar_glass_{rgb,depth,lidar}_frames/` with the WebP stream's encoder: lanczos downscale, quality 70, loop 0, 30 fps.
2. Check them side by side against `scratch/lidar-glass/preview_320.mp4`.
3. Suggested caption: "Glass in front of a lidar (`scripts/readme_media/lidar_glass.py`): tier-2 RGB | ray-traced depth (stops at the pane, as MuJoCo and Isaac do) | 2-return lidar (cyan = echoes on glass, the rest coloured by range)".
4. If the lidar view needs to read better at 320 px, tune these without GPU work: rerun a preview with `--dump` on frames 0, 24 and 75, then tune `--eye`, `--target`, `--view-fovy`, `--dot` and `--glass-alpha` on the dumped arrays with `lidar_glass.cloud_panel`.
