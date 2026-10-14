## What

Coupled MPM scenes on Metal move closer to the L4. Details are in MPM.md section 8k; the research is in `research_notes/coupled_speed/NOTE.md`; logs are in `runs/mpm/coupled_speed/`.

**Pins**
- Warp `metalsim` 95f6f121 → **31cf79d7**.
- Newton `metalsim-1.5.2` 99843d09 → **addd10cb** (three commits).
- The vendored `implicit_mpm` is regenerated and `--check`ed.
- `patches/` fallbacks are regenerated.

**Changes, all Metal-only and each with an off switch**
1. **Warp: `bsr_mv` makes CUDA's tiled-kernel choice for long rows** (`warp.config.metal_bsr_mv_tiles`, `WP_METAL_BSR_MV_TILES=0` to disable). On Metal it summed every row serially in one thread. Metal's `tile_sum` is CUDA's shuffle tree, so the summation order is the L4's.
2. **Newton: the long-row rigidity product (IJtm) goes through `bsr_mv`.** It was the cause of two-way's growing frame cost: 12 rows of up to about 1,170 blocks; 29-40 → 12-13 ms per 50 calls.
3. **Newton: the proxy-wrench harvest compacts the proxy-body nodes first** (`NEWTON_METAL_HARVEST_COMPACT=0` to disable). This is the path Franka-Pour and MPMCoupling use.
4. **Newton example: the two-way body-force sum compacts the body nodes first** (`NEWTON_METAL_BODY_FORCE_COMPACT=0` to disable).

**Results** (16-inch; exclusive timing jobs; 3 interleaved reps; ms/frame on the L4's protocol, Metal / L4)

| scene | before | after | L4 |
|---|---|---|---|
| two-way, captured | 54.8 (1.77x) | **37.6 (1.21x)** | 31.0 |
| two-way, eager | 78.8 (1.08x) | **61.3 (0.84x)** | 73.0 |
| ANYmal, captured | 29.9 (1.23x) | 29.4 (1.21x) | 24.2 |
| MPMCoupling `multibody_sand`, 16 bodies (coupled step) | 49.3 ms | **34.2 ms** | — |
| Franka-Pour, 2,048 envs (policy step) | 1.80 s | 1.80 s | does not fit |

**Bitwise and parity**
- The pure-MPM floor scenes and ANYmal are bitwise unchanged.
- Two-way and Franka-Pour change only in rounding (summation order). They repeat bitwise.
- Two-way stays inside the L4 floor ensemble (0.83).
- Each switch off restores the previous results bitwise.
- `tests/test_mpm.py`: 15 passed, 1 xfailed. `tests/test_mpm_main.py`: 15 passed.

**ANYmal** (measured, no change): the colour-launch threadgroup count and Newton's `gs-soa` solver are bitwise equal to the current solver and no faster. What remains is the MuJoCo Warp robot graph, 8.3 ms against about 1.4 ms of L4 kernel time.

**Docs**: MPM.md 8k (8j item 1 marked done; 8j's history line restored to 99843d09), STATUS, GAPS, PARITY §1b.4, README pins, CHANGELOG.

## STATUS
Done. What is left on coupled speed is in MPM.md 8k:
- the Gauss-Seidel latency;
- the tiled `bsr_mv` on Metal at 1.3 ms vs the L4's 0.12 ms (a bigger tile would leave CUDA's order);
- the MuJoCo Warp robot graph, which is rigid-stack work.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
