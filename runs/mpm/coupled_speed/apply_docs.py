"""Doc updates for the coupled-speed PR (run once in the worktree; each anchor must match exactly once)."""
import pathlib

R = pathlib.Path(__file__).resolve().parents[3]
k8 = (R / "runs/mpm/coupled_speed/docs_8k.md").read_text()


def edit(path, old, new):
    p = R / path
    s = p.read_text()
    assert s.count(old) == 1, (path, old[:70], s.count(old))
    p.write_text(s.replace(old, new))


# MPM.md: 8j item 1 done, 8k inserted, history line restored (pin_bump replaced the GS fix commit)
edit("metalsim/physics/mpm/MPM.md",
     "**What would close it** (not started when this section was written; item 1 is in progress per `docs/NOW.md`):",
     "**What would close it** (as written on 2026-10-13; item 1 and the body-force sum are done, section 8k):")
edit("metalsim/physics/mpm/MPM.md", "\n## 9. Running\n", "\n" + k8 + "\n## 9. Running\n")
edit("metalsim/physics/mpm/MPM.md",
     "fork (`addd10cb`, blocks from each cell's partition node index;",
     "fork (`99843d09`, blocks from each cell's partition node index;")

# STATUS: speed line, MPM bullet (speed, pins, open item 2)
edit("docs/STATUS.md",
     "  coupled scenes: ANYmal eager 0.67x, two-way eager 1.09x, ANYmal captured 1.29x, two-way captured 1.76x.",
     "  coupled scenes (MPM.md 8k): two-way captured 1.21x (was 1.76x), two-way eager 0.84x, ANYmal captured 1.21x,\n"
     "  ANYmal eager 0.67x.")
edit("docs/STATUS.md",
     "coupled scenes re-timed 2026-10-13 on the L4's protocol (MPM.md 8j): ANYmal eager 0.67x (faster), two-way eager 1.09x, ANYmal captured 1.29x, two-way captured 1.76x (slower).",
     "coupled scenes on the L4's protocol (MPM.md 8j, 8k): two-way captured 1.76x -> 1.21x and eager 1.09x -> 0.84x after the tiled IJtm product and the body-force compaction; ANYmal eager 0.67x, captured 1.21x (the rest is the MuJoCo Warp robot graph: 8.3 ms vs about 1.4 ms of L4 kernel time).")
edit("docs/STATUS.md",
     "Pins: Warp `metalsim` 95f6f121, Newton `metalsim-1.5.2` 3fd1f052 (`scripts/setup_newton152.sh`)",
     "Pins: Warp `metalsim` 31cf79d7, Newton `metalsim-1.5.2` addd10cb (`scripts/setup_newton152.sh`)")
edit("docs/STATUS.md",
     "(2) captured coupled-scene speed at one world (two-way 1.76x, ANYmal 1.29x the L4; MPM.md 8j: the two-way excess is mostly one Metal-only rigidity product that grows during the run; candidate fix there, in progress per `docs/NOW.md`)",
     "(2) captured coupled-scene speed at one world, now two-way 1.21x and ANYmal 1.21x the L4 (MPM.md 8k: what is left is the latency-bound Gauss-Seidel passes and, for ANYmal, the MuJoCo Warp robot graph; no MPM-side lever measured that keeps results)")

# GAPS row
edit("docs/GAPS.md",
     "Fork commits pinned in the setup scripts and on GitHub (Warp `metalsim` 95f6f121, Newton `metalsim-1.5.2` 3fd1f052;",
     "Fork commits pinned in the setup scripts and on GitHub (Warp `metalsim` 31cf79d7, Newton `metalsim-1.5.2` addd10cb;")
edit("docs/GAPS.md",
     "coupled scenes captured at one world slower than the L4 (two-way 1.76x, ANYmal 1.29x; eager two-way 1.09x, ANYmal 0.67x; MPM.md 8j) |",
     "coupled scenes captured at one world slower than the L4: two-way 1.21x (was 1.76x), ANYmal 1.21x; eager two-way 0.84x, ANYmal 0.67x (MPM.md 8k: left are the Gauss-Seidel latency and, for ANYmal, the MuJoCo Warp robot graph) |")

# README: Warp pin date, Newton pin history
edit("README.md",
     "[`31cf79d7`](https://github.com/pulipakaa24/warp/commit/31cf79d781e4b10d8ff85c8cbd98fc0e40f75953) (2026-10-12, pinned by",
     "[`31cf79d7`](https://github.com/pulipakaa24/warp/commit/31cf79d781e4b10d8ff85c8cbd98fc0e40f75953) (2026-10-13, pinned by")
edit("README.md",
     "by `scripts/setup_newton152.sh`; `3fd1f052` before the 2026-10-13 Gauss-Seidel coloring fix,",
     "by `scripts/setup_newton152.sh`; `99843d09` before the 2026-10-13 coupled-speed commits, `3fd1f052` before the 2026-10-13 Gauss-Seidel coloring fix,")

# CHANGELOG: new section before the first existing one
p = R / "CHANGELOG.md"
c = p.read_text()
entry = ("## MPM: coupled scenes closer to the L4 (2026-10-13, `metalsim/physics/mpm/MPM.md` 8k)\n\n"
         "- 10-13 Warp pin 95f6f121 -> 31cf79d7: `bsr_mv` takes CUDA's tiled-kernel choice on Metal "
         "(`warp.config.metal_bsr_mv_tiles`). Newton pin 99843d09 -> addd10cb: the long-row rigidity product goes "
         "through `bsr_mv`; the proxy-wrench harvest and the two-way example's body-force sum compact their nodes first. "
         "Two-way captured 54.8 -> 37.6 ms/frame (1.77x -> 1.21x the L4), eager 0.84x; MPMCoupling coupled step "
         "49.3 -> 34.2 ms (16 bodies); Franka-Pour unchanged (1.80 s per 2,048-env step). Pure-MPM floors and ANYmal "
         "bitwise unchanged; two-way and Franka-Pour repeat bitwise, two-way inside the L4 floor. Vendored implicit_mpm "
         "regenerated. Logs `runs/mpm/coupled_speed/`, research `research_notes/coupled_speed/NOTE.md`.\n\n")
k = c.index("\n## ") + 1
p.write_text(c[:k] + entry + c[k:])
print("docs updated")
