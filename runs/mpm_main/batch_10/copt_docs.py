"""Docs of the coupling-opt PR (#33) from the measured JSONs (batch_07 split / rigid, batch_10 policy-step A/B).

    python3 copt_docs.py REPO B07_DIR B09_DIR
"""
import json
import pathlib
import sys

repo, b07, b09 = (pathlib.Path(p) for p in sys.argv[1:4])
missing = []


def load(p):
    try:
        return json.load(open(p))
    except Exception as e:  # noqa: BLE001
        missing.append(f"{p}: {e}")
        return None


split = {n: load(b07 / f"split_{n}.json") for n in (1, 64, 256, 1024)}
rigid = {n: load(b07 / f"rigid_{n}.json") for n in (1, 64, 256, 1024)}
pol = {n: load(b09 / f"policy_{n}.json") for n in (1, 64, 1024)}
one = load(b09 / "one_world_reset.json")
if missing:
    print("MISSING:\n  " + "\n  ".join(missing))
    sys.exit(1)

rows_split = "\n".join(
    f"| {n:,} | {s['full_ms_median']:.2f} | {s['rigid_ms_median']:.2f} | {s['coupling_ms_median']:.2f} "
    f"({100 * s['coupling_share_of_full']:.1f} %) | {s['mpm_ms_median']:.2f} | {rigid[n]['ms_per_step_median']:.2f} |"
    for n, s in split.items())
rows_pol = "\n".join(
    f"| {n:,} | {p['A_ms_median']:.2f} | {p['B_ms_median']:.2f} | {p['saving_ms_per_policy_step']:.2f} "
    f"({p['speedup']:.3f}x) | {p['resets_per_run']} / {p['steps']} | {'yes' if p['bitwise_equal'] else 'NO'} |"
    for n, p in pol.items())
if not all(p["bitwise_equal"] for p in pol.values()) or not one["ok"]:
    print("a bitwise check failed: not writing the docs")
    sys.exit(1)

section = f"""### Coupling-layer cost and the policy-step graph (2026-10-13, 16-inch)

Research: `research_notes/MPM in main simulator/coupling_opt.md`. The coupling layer is everything in the coupled step
except the MuJoCo Warp substeps and the MPM step: the caller's `xfrc_applied` saved and restored, the sand wrench added,
the proxies' begin / end state, the end-of-step kinematics and the lagged rewind. Box on a 2,592-particle bed per world,
each section captured as its own graph and replayed round-robin, median of 100 (40 at 1,024 worlds) after 10 warm-up
steps (`probes/scale_coupled.py --split`; `--rigid-only --time` is the same scene without sand; runs
`runs/mpm_main/batch_07/`), ms:

| worlds | coupled step | MuJoCo Warp substeps | coupling layer (share) | MPM + harvest | rigid-only scene |
|---|---|---|---|---|---|
{rows_split}

The layer costs about 0.4 ms whatever the world count (its kernels are at the dispatch floor), the MPM step is the
rest. Fusing its launches (the research note's items 3-4) could save at most part of that 0.4 ms, so it was not done.
What a policy loop does pay is the host side: `MPMCoupling.launch_step()` and the capturable reset put resets and the
coupled step into one graph per policy step. A/B on a policy-step-shaped loop (`probes/policy_step_ab.py`: each step,
2 % of the worlds reset by a done mask on the GPU, the coupled step, an observation; A = the eager API as before this
change, including the host read of the mask its numpy-only `reset` needed; B = one captured graph; every run in its
own process; median of 3 interleaved rounds; runs `runs/mpm_main/batch_10/`), ms per policy step:

| worlds | A (eager API) | B (one graph) | saving | steps with resets | bitwise A = B |
|---|---|---|---|---|---|
{rows_pol}

With one world, Newton 1.5.2's solver has no separate FEM environments, and its masked reset refused grid warm starts
(`mpm.reset(mask)` raised before this change); `MPMCoupling` now launches the same reset steps with the mask read on
the device: bitwise the solver's full reset when the world is selected, nothing when it is not
(`probes/one_world_reset.py`, `test_one_world_masked_reset`).

"""

p = repo / "metalsim/physics/mpm/MPM.md"
s = p.read_text()
anchor = "### Re-vendoring when the fork pin moves"
assert s.count(anchor) == 1
s = s.replace(anchor, section + anchor)
old_use = "mpm.wrenches(); mpm.particles(); mpm.reset(mask); mpm.check_status()\n"
assert s.count(old_use) == 1
s = s.replace(old_use, old_use + (
    "mpm.reset(done)                                      # (N,) bool, numpy or torch (no host sync); a graph after the first\n"
    "# one graph per policy step (after one eager mpm.step() and mpm.reset(...)); write sim.reset_mask on torch's queue first:\n"
    "with wp.ScopedCapture() as cap:\n"
    "    sim.launch_reset(); mpm.launch_reset(from_sim_mask=True); mpm.launch_step()   # + the policy's own launches\n"))
p.write_text(s)
print("MPM.md: section 11 updated")

p = repo / "docs/GUIDE.md"
s = p.read_text()
old = "`mpm.wrenches()` returns the sand wrench per coupled body, `mpm.reset(mask)` resets\nworlds."
assert s.count(old) == 1, "GUIDE anchor"
s = s.replace(old, "`mpm.wrenches()` returns the sand wrench per coupled body, `mpm.reset(mask)` resets\nworlds "
              "(numpy or torch mask); `sim.launch_reset()`, `mpm.launch_reset(from_sim_mask=True)` and `mpm.launch_step()` "
              "put resets and the coupled step into one captured graph per policy step (MPM.md section 11).")
p.write_text(s)
print("GUIDE.md updated")

p = repo / "research_notes/MPM in main simulator/coupling_opt.md"
s = p.read_text()
s += f"""
## Results (2026-10-13, 16-inch; batch_07, batch_10)

Done: items 1 (`MPMCoupling.launch_step()`) and 2 (capturable per-world reset: persistent device mask, `reset_mask`
torch view, `launch_reset(from_sim_mask=True)`, `BatchSim.launch_reset()`, a reset graph after the first `reset`).
Items 3-4 not done: measured, the whole layer is about 0.4 ms per coupled step (2.9 % at 1 world, 0.2 % at 1,024),
so trimming or fusing its ~17 launches can save at most a fraction of that. Bitwise: the coupled trajectory with
resets is identical on main and on this branch (`hash_trajectory.py`, 8 worlds, 60 steps), and the one-graph policy
step equals the eager API at every world count measured.

Section split, ms (`scale_coupled.py --split`):

| worlds | coupled step | MuJoCo Warp substeps | coupling layer (share) | MPM + harvest | rigid-only scene |
|---|---|---|---|---|---|
{rows_split}

Policy-step A/B, ms per policy step (`policy_step_ab.py`, 2 % of the worlds reset per step):

| worlds | A (eager API) | B (one graph) | saving | steps with resets | bitwise A = B |
|---|---|---|---|---|---|
{rows_pol}

Found on the way: with one world the solver's masked reset refused grid warm starts (no separate FEM environments
below two worlds), so `mpm.reset(mask)` raised at one world on main; fixed in `MPMCoupling._solver_reset`
(`one_world_reset.py`: selected = the solver's full reset bitwise, unselected = no reset). batch_07's 1,024-world A/B
ran out of memory because every run of the probe stayed in one process; each run is its own process now.
"""
p.write_text(s)
print("coupling_opt.md updated")

p = repo / "CHANGELOG.md"
s = p.read_text()
i = s.index("\n## ")
s = s[:i] + ("\n## MPM: coupled step and resets in one policy-step graph (2026-10-13, `metalsim/physics/mpm/MPM.md` 11)\n\n"
             "- 10-13 `MPMCoupling.launch_step()`, a capturable per-world reset (`reset_mask`, `launch_reset(from_sim_mask=True)`, "
             "`reset(mask)` with numpy or torch masks and a reset graph) and `BatchSim.launch_reset()`; `scale_coupled.py --split` "
             "per-section timings; `policy_step_ab.py`. The coupling layer is about 0.4 ms per coupled step; one graph per "
             "policy step saves " + ", ".join(f"{p_['saving_ms_per_policy_step']:.2f} ms at {n:,}" for n, p_ in pol.items())
             + " worlds, bitwise. One-world masked reset fixed (it raised).\n") + s[i:]
p.write_text(s)
print("CHANGELOG.md updated")
print("copt_docs: ok")
