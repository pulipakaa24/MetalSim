# MuJoCo Warp and MuJoCo upstream: filed issues, prepared PR (2026-09-26)

Filed under the owner's account (issues need no CLA). Bodies in `issues/` (`*.md`, `*.title`, `*.url`). Every number
below was re-measured today against upstream MuJoCo Warp `main` `cc97eea` (MuJoCo 3.14.0 pip, warp-lang 1.15.0,
Warp CPU device) unless marked as fork-measured. Reproducers: `flex_issue_harness.py` (flex issues),
`mujoco_c_linesearch_repro.py` (+ `_scan.py`, `_boxscan.py`) for the MuJoCo issue.

## Filed

| repo | # | title | state | UPSTREAM.md item |
|---|---|---|---|---|
| mujoco_warp | [1704](https://github.com/google-deepmind/mujoco_warp/issues/1704) | Elliptic cones off CUDA: the cone Hessian term is launched with one thread per contact slot (dim_block = naconmax) | open | elliptic note §1-2 |
| mujoco_warp | [1705](https://github.com/google-deepmind/mujoco_warp/issues/1705) | Flex element contacts of any flex after the first store the global element id | open | 1 |
| mujoco_warp | [1706](https://github.com/google-deepmind/mujoco_warp/issues/1706) | Box-triangle flex contacts: at most 2 per element, MuJoCo C generates up to 11 | open | 4 |
| mujoco_warp | [1707](https://github.com/google-deepmind/mujoco_warp/issues/1707) | 1D flexes (cables) collide as vertex spheres; MuJoCo C collides their segments as capsules | open | 5 |
| mujoco_warp | [1708](https://github.com/google-deepmind/mujoco_warp/issues/1708) | Volume flexes: interior elements collide with geoms; MuJoCo C only collides active (surface) layers | open | 6 |
| mujoco_warp | [1709](https://github.com/google-deepmind/mujoco_warp/issues/1709) | Flex-geom contacts: per-pair limit differs from MuJoCo C's mjMAXCONPAIR (parity data for the configurable cap in #1668) | open | 3 + 7 |
| mujoco | [3628](https://github.com/google-deepmind/mujoco/issues/3628) | Newton solver: with elliptic cones at impratio 10 and ls_iterations 20, an exhausted line search ends the solve at a non-converged qacc | open | elliptic cost review (A) |

## Not filed, and why

- **Flex item 2 (equality rows reserved by an atomic counter)**: covered by upstream's determinism work: issue #562,
  maintainer PR #1687 (`DeterminismType.CONSTRAINT`, "canonical constraint row sorting" as future work) and community
  PR #1422 (deterministic `_equality_flex_count`). Every constraint type reserves rows with `atomic_add`, not only flex.
  Only observable on a device whose thread order differs from edge order (Metal); on CPU/CUDA the order coincides.
- **Items 3 and 7 merged into #1709**: maintainer PR #1668 deliberately replaces the geom-flex mjMAXCONPAIR cap with
  proximity dedup and has a TODO for a configurable cap; one issue with parity data for both was more useful than two.
  Item 7's large-cloth numbers are fork-measured (2026-09-25, MuJoCo C rebuilt with a raised cap), stated as such.
- **Line-search budget (elliptic cost review finding B)**: does NOT reproduce on upstream main. Same 256 x 9 G1 states,
  8 substeps, elliptic impratio 10 (`competitors/elliptic_review/ls_by_iter.py`): line searches at the 20-iteration
  budget 0.4 % on `main` cc97eea (pyramidal imp1 0.3 %) vs 10.0 % on its parent before #1700 (04f10e2). Upstream fixed
  it on 2026-09-25 with #1700 "Accept converged linesearch candidates regardless of derivative sign" (a noise-floored
  `gtol_accept`, the MJX analogue being google-deepmind/mujoco#3619). MetalSim's fork (v3.14.0 base) lacks #1700:
  adopting it is a MetalSim follow-up (expected to remove most of the ~10 % budget-hitting searches).

## Pull request (prepared, NOT opened)

`elliptic-jtcj-offcuda` on github.com/pulipakaa24/mujoco_warp, commit 6d8e29a (on cc97eea = upstream head; amended
today only for the message: unwrapped per upstream AGENTS.md, CPU numbers against main). No co-author trailer.
Final text and checks: `DRAFT_elliptic_launch.md`. Blocked on the owner's Google CLA: the owner has no PR on any
Google repository, so no `cla/google` check exists to verify it.

Google CLA (owner): sign at https://cla.developers.google.com/ with a Google account whose primary or alternate
email is adipu@utexas.edu (the commit author), or whose linked GitHub username is pulipakaa24. Individual CLA if the
owner owns the code; if an employer may own it, a Corporate CLA by the employer (and the email added to its group).
Then open the PR (command in `DRAFT_elliptic_launch.md`) and check that `cla/google` is green.

## Still pending upstream for the flex fixes

A PR per flex fix would need the flex commits re-cut without the Claude co-author trailer (UPSTREAM.md pre-filing
step), plus the Google CLA. Not done: the maintainers may prefer their own fixes (and #1668 is reshaping item 3).

## 2026-10-03 follow-up on #1704

Maintainer thowell asked (2026-09-29) whether upstream PR #1715 (`dim_block = 1`, `nblocks_perblock = naconmax` in the
non-CUDA branch) addresses the issue. Measured on the Warp CPU device, Menagerie Go2 (own elliptic cone, impratio 100),
32 worlds, Newton 10 / 20, 3 x 200 steps, two interleaved repeats, M4 Max on AC in high-power mode:
`main` fb8c7b0 19.0-19.3 / 29.2-30.4 ms/step (first block / later blocks, the robot falls and contacts grow),
PR #1715 e2a9d76 7.3-7.4 / 9.9-10.2, our branch `elliptic-jtcj-offcuda` 6d8e29a 7.3-7.4 / 10.0-10.4.
The two are equal on the CPU, so #1715 covers the upstream case and our PR is not needed there. On a GPU device without
CUDA (our Metal fork) #1715's form would run every contact through `ntri` threads; the fork keeps its per-world kernel.
No PR was opened from this account (blocked on the CLA, and now superseded for this issue).
