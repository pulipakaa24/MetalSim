"""Doc stage: write ``docs/renders/<TASK>.md`` from the task entry and the stage JSONs (record, render, encode), so
every number in the doc is the logged one. Paths in the doc are repo-relative. A task's notes may be a callable
``notes(take) -> str`` (``Take``): notes that state a take's power, times, commits or floor size read them from the
stage logs, so a later ``--stage doc`` writes them again instead of reverting a corrected page."""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import textwrap

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


def _power_line(stamps) -> str:
    """The power state each stage stamped (``provenance.power`` in ``hardware.power``: rec.json's when the record stage
    wrote it, at its end; each render.json's at the shot's first frame, kept on resume); "not recorded" for stage logs
    written before 2026-10-31."""
    parts = []
    for name, hw in stamps:
        pw = (hw or {}).get("power")
        at = pw.get("at") if pw else None
        parts.append(f"{name}: {provenance.power_text(pw)}" + (f", {at[:10]} {at[11:16]} ({at[19:]})" if at else ""))
    return ("- Power as each stage stamped it (`pmset -g batt` and `pmset -g`; `hardware.power`: the record stage's "
            "when it wrote rec.json, a render shot's at its first frame): "
            + "; ".join(parts) + ".")


# Apple's model identifiers (rec.json ``hardware.model``) of the machines the films are made on: the 14-inch M4 Max
# (runs/readme_media/PROTOCOL.md) and the 16-inch (franka_pour's rec.json)
MACHINES = {"Mac16,6": "14-inch MacBook Pro", "Mac16,5": "16-inch MacBook Pro"}


def fork_pins() -> dict:
    """Full SHAs of the fork commits this tree pins (None when the source is missing), from the files that hold the
    pins: MuJoCo Warp = README.md's ``pip install ... mujoco_warp@<sha>`` line, Warp = ``PIN`` in
    scripts/setup_warp.sh."""
    out = {}
    for name, path, pat in (("mujoco_warp", "README.md", r"mujoco_warp@([0-9a-f]{40})"),
                            ("warp", os.path.join("scripts", "setup_warp.sh"), r"^PIN=([0-9a-f]{40})")):
        try:
            m = re.search(pat, open(os.path.join(provenance.ROOT, path)).read(), re.M)
        except OSError:
            m = None
        out[name] = m.group(1) if m else None
    return out


def _join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


class Take:
    """One take's stage logs (``rec``: rec.json; ``renders``: [(shot label, render.json)] in shot order), for notes
    that state the take's facts: the doc stage passes it to a callable ``FilmTask.notes``."""

    def __init__(self, task, rec_dir, rec, renders):
        self.task, self.rec_dir, self.rec, self.renders = task, rec_dir, rec or {}, renders

    def _log(self, stage):
        return self.rec if stage == "record" else (dict(self.renders).get(stage) or {})

    def commit(self, module="metalsim", stage="record", n=8) -> str:
        """The git SHA (``n`` characters, no "+dirty") a stage logged for a module: ``stage`` "record" (rec.json) or a
        shot label (its render.json); "" when not logged."""
        v = (self._log(stage).get("software") or {}).get(module)
        v = v.get("git") if isinstance(v, dict) else v
        return (v or "").split("+")[0][:n]

    def dirty(self, stage="record") -> bool:
        """metalsim's checkout had uncommitted changes when the stage ran ("+dirty" in its pin)."""
        return "+dirty" in ((self._log(stage).get("software") or {}).get("metalsim") or "")

    def metalsim_at(self, uncommitted: str) -> str:
        """"`ed66d8ef` plus <uncommitted> (hence `+dirty` in the pins)": metalsim's commit as the record stage logged
        it, with ``uncommitted`` (what the uncommitted changes were) when its checkout was dirty."""
        return f"`{self.commit()}`" + (f" plus {uncommitted} (hence `+dirty` in the pins)" if self.dirty() else "")

    def at_pin(self, module, stage="record") -> bool | None:
        """The stage ran ``module`` (mujoco_warp or warp) at the commit this tree pins (``fork_pins``); None if
        unknown."""
        sha, pin = self.commit(module, stage, 40), fork_pins().get(module)
        return (len(sha) >= 7 and pin.startswith(sha)) if sha and pin else None

    @staticmethod
    def pin(module, n=8) -> str:
        return (fork_pins().get(module) or "")[:n]

    @staticmethod
    def fill(paragraph: str, width: int = 120) -> str:
        """One markdown paragraph wrapped at ``width`` columns as the notes are (for a paragraph with values from the
        logs in it, whose widths a hand-wrapped template cannot know); a soft line break renders as a space. A line
        that would start like a list item, heading or quote ("- ", "2. ", "#", ">") stays on the line before."""
        out = []
        for line in textwrap.wrap(" ".join(paragraph.split()), width, break_long_words=False, break_on_hyphens=False):
            if out and re.match(r"([-+*]|\d{1,9}[.)])( |$)|#|>", line):
                out[-1] += " " + line
            else:
                out.append(line)
        return "\n".join(out)

    def at(self, stage="record") -> str:
        """When the stage stamped its power state, ISO local time ("" if it did not)."""
        return ((self._log(stage).get("hardware") or {}).get("power") or {}).get("at") or ""

    @property
    def day(self) -> str:
        """The day the record and render stages stamped ("2026-10-31"; "2026-10-30 to 2026-10-31" when they differ)."""
        days = sorted({a[:10] for a in [self.at()] + [self.at(lb) for lb, _ in self.renders] if a})
        return " to ".join(dict.fromkeys([days[0], days[-1]])) if days else "a day the logs do not record"

    @property
    def machine(self) -> str:
        """"14-inch MacBook Pro (Mac16,6)": the record stage's machine."""
        model = (self.rec.get("hardware") or {}).get("model", "")
        return f"{MACHINES[model]} ({model})" if model in MACHINES else f"a Mac ({model or 'model not recorded'})"

    @functools.cached_property
    def ground_tile_m(self) -> float | None:
        """Metres of floor per ground-texture image in this take's frames, from the render logs: each shot's render.json
        ``look.ground_tile_m`` (logged since 2026-10-31); for a shot rendered before that, the task's
        ``look.ground_tile_m`` when the shot's ``cache_key`` is the key the task's settings give now (the key hashes
        the ground's texrepeat, ``render._key_look``), so its frames were drawn with that tile. None when a shot was
        drawn with another tile, the logs cannot tell, or the shots disagree."""
        from metalsim.render.film.__main__ import shot_task
        from metalsim.render.film.render import cache_key

        tiles = set()
        for label, r in self.renders:
            tile = (r.get("look") or {}).get("ground_tile_m")
            if tile is None:
                sc = next((s for s in self.task.shots if s.name == label), None)
                if sc is None or not r.get("cache_key"):
                    return None
                st = shot_task(self.task, sc)
                try:
                    key = cache_key(self.rec_dir, st.film, st.look, st.viewer, r["grid"], r["slots"])
                except (OSError, KeyError, ValueError):
                    return None
                if key != r["cache_key"]:
                    return None
                tile = st.look.ground_tile_m
            tiles.add(float(tile))
        return tiles.pop() if len(tiles) == 1 else None

    def power_prose(self, note="") -> str:
        """The power state each stage stamped, as a clause for the notes: "on battery and discharging at every stamp
        (66 % when the record stage wrote `rec.json` at 04:51, 66 % at the hero shot's first frame at 04:51 and 62 % at
        the readme shot's at 04:53, local time), with pmset's power-mode setting at High Power (...); the encode and doc
        stages stamp nothing", one clause per stamp when source or battery state differ. ``note`` follows "local
        time"."""
        st = [("when the record stage wrote `rec.json`", (self.rec.get("hardware") or {}).get("power"))]
        st += [(f"at the {lb} shot's" + (" first frame" if i == 0 else ""), (r.get("hardware") or {}).get("power"))
               for i, (lb, r) in enumerate(self.renders)]
        if not all(pw and pw.get("source") and pw.get("battery_percent") is not None and pw.get("at")
                   for _, pw in st):
            return "; ".join(f"{what}: {provenance.power_text(pw)}" for what, pw in st)
        src = {"AC Power": "on AC power", "Battery Power": "on battery", "UPS Power": "on UPS power"}
        tail = "local time" + (f"; {note}" if note else "")
        same = len({(pw["source"], pw.get("battery_state")) for _, pw in st}) == 1
        if same:
            pw0 = st[0][1]
            body = (f"{src.get(pw0['source'], pw0['source'])} and {pw0['battery_state']} at every stamp ("
                    + _join([f"{pw['battery_percent']} % {what} at {pw['at'][11:16]}" for what, pw in st])
                    + f", {tail})")
        else:
            parts = [f"{src.get(pw['source'], pw['source'])}, {pw['battery_percent']} %, {pw['battery_state']}, "
                     f"{what} at {pw['at'][11:16]}" for what, pw in st]
            body = parts[0] + (f"; {_join(parts[1:])}" if len(parts) > 1 else "") + f" ({tail})"
        modes = {(pw.get("mode"), pw.get("mode_setting")) for _, pw in st}
        if len(modes) == 1 and st[0][1].get("mode"):
            mode, setting = modes.pop()
            m = (f"with pmset's power-mode setting at {mode}{'' if same else ' at every stamp'} (`{setting}`: the "
                 f"configured setting, which does not show that the machine ran in {mode} Mode)")
        else:
            m = "with pmset's power-mode setting " + _join([f"{pw.get('mode') or 'not recorded'} {what}"
                                                             for what, pw in st])
        return f"{body}, {m}; the encode and doc stages stamp nothing"


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
        it = ad.get("policy_iteration")
        fmt = ad.get("policy_format")
        tag = f" (iteration {it})" if it is not None else (f" ({fmt})" if fmt else "")
        L.append(f"- Policy: `{ad.get('policy')}`{tag}; physics: {ad.get('physics')}.")
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
        L.append(_power_line([("record", hw)] + [(f"render {label}", r.get("hardware")) for label, r, *_ in shot_rows]))
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
    notes = (task.notes(Take(task, rec_dir, rec, [(label, r) for label, r, *_ in shot_rows])) if callable(task.notes)
             else task.notes)
    if notes:
        L += ["", "## How it was made", "", notes.strip()]
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
