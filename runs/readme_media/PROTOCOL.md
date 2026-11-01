# README media work: rules for every agent (2026-10-31)

Plan: docs/research/readme_media_plan_2026-10-31.md (sections 2 and 4a). Repo: ., branch main.

## Hand-off (required)
- Do ONE increment (your brief), then stop. Write `runs/readme_media/handoff/<stream>.md` before you finish, with
  these sections: **Status** (done / partial / blocked), **Changed** (each file, one line), **Verified** (exact
  commands, numbers, output paths, which frames you looked at), **Open** (what is not done or not proven),
  **Next** (the next increment, concrete enough for a fresh agent with no other context).
- Your final message: at most 150 words plus the hand-off path. The orchestrator reads the hand-off, not your transcript.

## Ground rules
- Touch only the files your brief says you own. Do not commit, push, branch, or edit README.md, docs/renders/*,
  runs/film/g1_flat/* or docs/research/readme_media_plan_2026-10-31.md.
- Python: `.venv/bin/python` (main venv: Metal, Warp, MuJoCo, pyoidn, imageio, Pillow 12.3 with WebP).
- GPU: every GPU job goes through the queue: `scripts/gpu_run.sh <name> render <minutes> -- <command>` (run from the
  directory whose code you want). Never run two GPU jobs at once. This Mac is the 14-inch (Mac16,6): no timing number from it goes into a doc
  as a performance figure. Read its power state live (`pmset -g batt`, `pmset -g | grep powermode`) when it matters;
  never repeat an earlier reading.
- Trained checkpoints, renders and old outputs live outside the repo: search ~/metalsim-fresh (agents/*/out,
  runs_render/, fix1-*/, film-*/) before calling anything missing.
- Look at what you render: open frames or a contact strip
  (`ffmpeg -i <video> -vf "fps=1,scale=480:-1,tile=6x1" -frames:v 1 strip.jpg`) with the Read tool before calling
  output good. Exit codes and numbers alone are how the bugs in plan section 2 shipped.
- Media for the README go to `runs/readme_media/assets/`; scratch goes to `runs/readme_media/scratch/<stream>/`.

## Research before trial and error (added after launch)
- Before changing code to make a symptom go away, find the existing resource that explains the behaviour:
  1. this repo's own record: `docs/research/`, `research_notes/`, `reports/`, `docs/*.md`, module docstrings, tests,
     `git log -S` / `git blame` on the lines involved (most subsystems here already have a note on their edge cases);
  2. upstream source vendored or pinned here (`upstream/`, the Warp / MuJoCo Warp / Newton forks) and the installed
     packages' own source in `.venv`;
  3. upstream docs and issue trackers (web search) for third-party behaviour: MuJoCo, Warp, Pillow, ffmpeg, GitHub's
     README rendering.
- Form a hypothesis from what you read, then run the smallest experiment that confirms or refutes it. Don't iterate
  on blind parameter changes; if two experiments in a row don't narrow the cause, stop and read more.
- Cite what you used (file:line, doc section, URL) in the hand-off's **Verified** section, and mark anything that rests
  only on an experiment.
