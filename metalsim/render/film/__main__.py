"""Film a task: roll out its policy, render offline with metalsim's path tracer, encode, and write the doc.

    python -m metalsim.render.film --list
    python -m metalsim.render.film franka_pour                       # every stage
    python -m metalsim.render.film franka_pour --stage record        # one stage (record | render | encode | doc)
    python -m metalsim.render.film g1_flat --seconds 5 --eye 2.6 -2 1.3 --stage all

Stages and outputs (``--out``, default ``runs/film/<task>``):
  record  -> rec/ (twin.xml, rec.npz, rec.json); runs in the task's venv (``FilmTask.venv``; started as a subprocess
             with that interpreter when the current one is not it)
  render  -> <shot>/frames/*.png, render.json, render_frames.jsonl per shot (``FilmTask.shots``); resumable
  encode  -> <shot>/master.mp4 and docs/renders/<task>[_<shot>].mp4 + stills
  doc     -> ``FilmTask.doc`` (docs/renders/<TASK>.md)
Each stage reads only files, so stages can run on different machines or venvs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace

from metalsim.render.film import provenance, tasks
from metalsim.render.film.cli import add_override_args, apply_overrides

VENVS = {"main": ".venv", "newton152": ".venv-newton152"}


def venv_python(name: str) -> str:
    """The interpreter of a named venv: in this checkout, else in the main checkout of a git worktree."""
    rel = os.path.join(VENVS[name], "bin", "python")
    roots = [os.environ.get("METALSIM_VENV_ROOT"), provenance.ROOT]
    common = provenance._run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=provenance.ROOT)
    if common:
        roots.append(os.path.dirname(common))
    for r in roots:
        if r and os.path.exists(os.path.join(r, rel)):
            return os.path.join(r, rel)
    raise FileNotFoundError(f"no {rel} under {roots}")


def shot_task(task, shot):
    film = replace(task.film, grid=shot.grid, passes=shot.passes or task.film.passes)
    return replace(task, film=film, viewer=shot.viewer or task.viewer)


def media_path(task, shot):
    base = os.path.splitext(task.doc or f"docs/renders/{task.name.upper()}.md")[0]
    d = os.path.dirname(base)
    stem = task.film.output_filename_prefix + ("" if shot.name == "hero" else f"_{shot.name}")
    return os.path.join(provenance.ROOT, d, stem + ".mp4")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stage", default="all", choices=("all", "record", "render", "encode", "doc"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--shots", default=None, help="comma-separated shot names (default: all of the task's)")
    ap.add_argument("--frames", default=None, help="render only start:stop[:step] (previews)")
    ap.add_argument("--doc", default=None, help="doc path (default: the task entry's)")
    add_override_args(ap)
    a = ap.parse_args(argv)
    if a.list or not a.task:
        for t in tasks.TASKS.values():
            print(f"{t.name:14s} venv {t.venv:10s} {t.title}: {t.summary[:90]}")
        return 0
    task = apply_overrides(tasks.get(a.task), a)
    if a.doc:
        task = replace(task, doc=a.doc)
    out = a.out or os.path.join(provenance.ROOT, "runs", "film", task.name)
    rec_dir = os.path.join(out, "rec")
    shots = [s for s in task.shots if not a.shots or s.name in a.shots.split(",")]
    stages = ["record", "render", "encode", "doc"] if a.stage == "all" else [a.stage]
    if "record" in stages:
        py = venv_python(task.venv)
        # venvs share one base interpreter: compare the venv (sys.prefix), not the resolved executable
        if os.path.realpath(sys.prefix) == os.path.realpath(os.path.dirname(os.path.dirname(py))):
            from metalsim.render.film.record import record
            ck = task.checkpoint
            if ck and not os.path.isabs(ck):
                ck = os.path.join(provenance.ROOT, ck)
            record(task, task.film, task.viewer, rec_dir, ck)
        else:
            args = [py, "-m", "metalsim.render.film", task.name, "--stage", "record", "--out", out]
            args += [x for x in (argv if argv is not None else sys.argv[1:])[1:] if x not in ("--stage", a.stage)]
            subprocess.run(args, check=True, cwd=provenance.ROOT)
    if "render" in stages:
        from metalsim.render.film.render import render
        for s in shots:
            render(shot_task(task, s), rec_dir, os.path.join(out, s.name), a.frames)
    if "encode" in stages:
        from metalsim.render.film.encode import encode
        import json
        fps = json.load(open(os.path.join(rec_dir, "rec.json")))["fps"]
        for s in shots:
            mp4 = media_path(task, s)
            encode(os.path.join(out, s.name), mp4, fps, task.film.crf, 1280, task.film.stills if s.name == "hero" else
                   task.film.stills[1:2])
    if "doc" in stages and task.doc:
        from metalsim.render.film.doc import write_doc
        cmds = ["scripts/fetch_isaac_assets.sh                # NVIDIA assets (not redistributed)" if any(
                    "fetch_isaac" in x.fetch for x in task.assets) else None,
                "scripts/fetch_polyhaven_assets.sh train      # CC0 HDRI and textures",
                f"python -m metalsim.render.film {task.name}   # record, render, encode, doc (main venv)",
                "# or stage by stage (resumable):",
                f"python -m metalsim.render.film {task.name} --stage record   # runs in the {VENVS[task.venv]} venv",
                f"python -m metalsim.render.film {task.name} --stage render",
                f"python -m metalsim.render.film {task.name} --stage encode",
                f"python -m metalsim.render.film {task.name} --stage doc"]
        write_doc(task, rec_dir, [(s.name, os.path.join(out, s.name)) for s in shots],
                  os.path.join(provenance.ROOT, task.doc), [c for c in cmds if c])
    return 0


if __name__ == "__main__":
    sys.exit(main())
