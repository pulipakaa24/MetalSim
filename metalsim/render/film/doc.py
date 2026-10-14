"""Doc stage: write ``docs/renders/<TASK>.md`` from the task entry and the stage JSONs (record, render, encode), so
every number in the doc is the logged one. Paths in the doc are repo-relative."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from metalsim.render.film import provenance


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _frames(shot):
    p = os.path.join(shot, "render_frames.jsonl")
    if not os.path.exists(p):
        return []
    rows = {}
    for line in open(p):
        line = line.strip()
        if line:
            d = json.loads(line); rows[d["frame"]] = d
    return [rows[k] for k in sorted(rows)]


def _fmt_s(s):
    if s is None:
        return "n/a"
    return f"{s / 60:.1f} min" if s >= 120 else f"{s:.1f} s"


def _rel(path, doc_dir):
    return os.path.relpath(os.path.join(provenance.ROOT, path), os.path.join(provenance.ROOT, doc_dir))


def write_doc(task, rec_dir, shots, doc_path, commands, extra_md=""):
    rec = _load(os.path.join(rec_dir, "rec.json"))
    doc_dir = os.path.dirname(doc_path)
    L = [f"# {task.title}: a metalsim render", "", task.summary, ""]
    shot_rows, media = [], []
    total_render = 0.0
    for label, shot in shots:
        r, e, fr = _load(os.path.join(shot, "render.json")), _load(os.path.join(shot, "encode.json")), _frames(shot)
        if r is None:
            continue
        secs = np.array([f["seconds"] for f in fr]) if fr else np.zeros(1)
        total_render += float(secs.sum()) + float(r.get("setup_s") or 0)
        enc = e["encodes"] if e else {}
        docmp4 = next((v for k, v in enc.items() if k != "master.mp4"), None)
        shot_rows.append((label, r, e, secs, docmp4))
        if e:
            for s in e["stills"]:
                media.append((label, s))
        if docmp4:
            L.append(f"**{label}** ({r['frames']} frames, {r['frames'] / r['fps']:.1f} s at {r['fps']:.0f} fps): "
                     f"[video]({_rel(docmp4['path'], doc_dir)}) ({docmp4['bytes'] / 1e6:.1f} MB, H.264)")
            L.append("")
    for label, s in media:
        L.append(f"![{label}, frame {s['frame']}]({_rel(s['path'], doc_dir)})")
    L += ["", "## What is shown", ""]
    if rec:
        eps = rec["episodes"]
        L.append(f"- {rec['num_envs']} envs simulated, filmed: {', '.join(str(e) for e in rec['envs'][:4])} "
                 f"(selection: {'given' if isinstance(task.film.envs, (list, tuple)) else 'automatic, the adapter ranking'}).")
        for e in rec["envs"][:4]:
            ep = eps[str(e)]
            bits = [f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in ep.items()]
            L.append(f"  - env {e}: " + ", ".join(bits))
        ad = rec.get("adapter", {})
        L.append(f"- Policy: `{ad.get('policy')}` (iteration {ad.get('policy_iteration')}); physics: {ad.get('physics')}.")
        if ad.get("note"):
            L.append(f"- Note: {ad['note']}")
    if extra_md:
        L += ["", extra_md.strip()]
    L += ["", "## Performance", "",
          "Measured on the machine below; one frame = one policy step (Isaac Lab's `render_fps` = 1 / step_dt).", "",
          "| stage | measurement |", "|---|---|"]
    if rec:
        t = rec["timing"]
        L.append(f"| simulate (record) | {rec['frames']} frames x {rec['num_envs']} envs: {t['sim_ms_mean']:.0f} ms/frame mean "
                 f"({t['sim_ms_median']:.0f} median, first {t['sim_ms_first']:.0f} incl. graph capture), "
                 f"{_fmt_s(t['sim_s_total'])} total; env build {_fmt_s(t.get('env_build_s'))}; stage wall "
                 f"{_fmt_s(t['record_wall_s'])} |")
        tw = rec["twin"]
        L.append(f"| render twin (CPU) | {tw['geoms']} geoms, {tw['meshes']} meshes, {tw['textures']} textures; build "
                 f"{tw['build_s']:.1f} s, poses {tw['poses_s']:.1f} s |")
        if rec.get("particles_per_env"):
            L.append(f"| particles | {rec['particles_per_env']} per env, radius {rec['particle_radius'] * 1e3:.2f} mm |")
    for label, r, e, secs, docmp4 in shot_rows:
        pr = r.get("probe") or {}
        L.append(f"| render: {label} | {r['width']}x{r['height']} ({r['grid']}x{r['grid']} panels of {r['panel'][0]}x"
                 f"{r['panel'][1]}), tier {r['tier']} path tracing, {r['samples_per_pixel']} spp ({r['passes']} passes x "
                 f"{r['spp']}), {r['bounces']} bounces, denoiser {r['denoise'] or 'none (--no-denoise)'}; {secs.mean():.2f} s/frame mean "
                 f"({np.median(secs):.2f} median, {secs.max():.2f} max), {_fmt_s(float(secs.sum()))} for {len(secs)} frames; "
                 f"first frame: one pass {pr.get('one_pass_s', float('nan')) * 1e3:.0f} ms, "
                 f"{pr.get('per_pass_s', float('nan')) * 1e3:.0f} ms per further pass; set-up {r['setup_s']:.1f} s |")
        if e:
            L.append(f"| encode: {label} | {_fmt_s(e['encode_s'])} (master CRF 16 + docs copy CRF "
                     f"{docmp4['crf'] if docmp4 else '?'}, libx264 slow) |")
    if rec:
        enc_total = sum((e or {}).get("encode_s", 0) for _, _, e, _, _ in shot_rows)
        L.append(f"| end to end | {_fmt_s(rec['timing']['record_wall_s'] + total_render + enc_total)} (record + render + "
                 f"encode, sum of stage walls) |")
        L += ["", "## Hardware and pins", ""]
        hw = rec["hardware"]
        L.append(f"- {hw.get('model', '')} ({hw.get('cpu', '')}, {hw.get('gpu_cores', '?')} GPU cores, "
                 f"{hw.get('memory_gb', '?')} GB), macOS {hw.get('macos', '')} ({hw.get('macos_build', '')}).")
        sw = rec["software"]
        L.append(f"- metalsim `{sw.get('metalsim')}`; record stage ({rec['venv']} venv): "
                 + ", ".join(f"{k} {v.get('version')}" + (f" (`{v['git']}`)" if v.get("git") else "")
                             for k, v in sw.items() if isinstance(v, dict)) + ".")
        r0 = shot_rows[0][1] if shot_rows else None
        if r0:
            sw2 = r0.get("software", {})
            L.append("- render stage (main venv): " + ", ".join(
                f"{k} {v.get('version')}" + (f" (`{v['git']}`)" if v.get("git") else "")
                for k, v in sw2.items() if isinstance(v, dict)) + ".")
    L += ["", "## Assets", "", "| asset | source | licence | where from |", "|---|---|---|---|"]
    for a in task.assets:
        url = a.url if a.url.startswith("http") else f"`{a.url}`"
        L.append(f"| {a.name} | {a.source} | {a.licence} | {url}" + (f" (`{a.fetch}`)" if a.fetch else "") + " |")
    if task.notes:
        L += ["", "## How it was made", "", task.notes.strip()]
    L += ["", "## Reproduce", "", "```bash"] + list(commands) + ["```", "",
          f"Generated by `python -m metalsim.render.film {task.name} --stage doc` from the stage logs "
          f"(`{provenance.relpath(rec_dir)}/rec.json`, `render.json`, `render_frames.jsonl`, `encode.json` per shot).", ""]
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    text = "\n".join(L)
    with open(doc_path, "w") as f:
        f.write(text)
    print(f"[doc] wrote {provenance.relpath(doc_path)} ({len(text)} chars)", flush=True)
    return text


def main(argv=None):
    from metalsim.render.film import tasks

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--shot", action="append", nargs=2, metavar=("LABEL", "DIR"), default=[])
    ap.add_argument("--doc", required=True)
    a = ap.parse_args(argv)
    write_doc(tasks.get(a.task), a.rec, a.shot, a.doc, [])


if __name__ == "__main__":
    sys.exit(main())
