"""Print which Newton and Warp are installed (path, VCS commit or editable source, and for a git checkout its HEAD and dirty
state), so validation logs can be pinned to an install. usage: python scripts/diagnostics/newton_stamp.py [--json]"""
import json, os, subprocess, sys, glob, importlib.util


def _pkg(name):
    spec = importlib.util.find_spec(name); path = os.path.dirname(spec.origin) if spec and spec.origin else None
    out = {"path": path}
    for d in glob.glob(os.path.join(os.path.dirname(path or ""), f"{name}-*.dist-info")) + glob.glob(os.path.join(os.path.dirname(path or ""), f"{name}*.dist-info")):
        f = os.path.join(d, "direct_url.json")
        if os.path.exists(f):
            u = json.load(open(f)); out["url"] = u.get("url"); out["commit"] = (u.get("vcs_info") or {}).get("commit_id")
            out["editable"] = bool((u.get("dir_info") or {}).get("editable")); break
    if path and "site-packages" in path:           # a wheel/VCS install: no source checkout to ask
        return out
    try:
        root = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5).stdout.strip()
        if root:
            out["git_head"] = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
            out["git_dirty"] = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    return out


if __name__ == "__main__":
    s = {"newton": _pkg("newton"), "warp": _pkg("warp")}
    print(json.dumps(s) if "--json" in sys.argv else f"newton {s['newton']} | warp {s['warp']}")
