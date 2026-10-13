"""pytest plugin: at each test function's return/unwind, record numeric locals (scalars exactly; arrays as
shape/dtype/sha1 + max/mean/min) to $CAPTURE_OUT (jsonl). Uses sys.monitoring on the test's code object only."""
import hashlib, json, math, os, sys
import pytest

OUT = os.environ.get("CAPTURE_OUT")
TOOL = 4
_target = {}
_records = {}


def _arr(a):
    import numpy as np
    try:
        mod = type(a).__module__
        if mod.startswith("warp"):
            a = a.numpy()
        elif mod.startswith("torch"):
            a = a.detach().cpu().numpy()
        a = np.asarray(a)
        if a.dtype == object or a.size > 50_000_000:
            return None
        d = {"shape": list(a.shape), "dtype": str(a.dtype), "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]}
        if a.size and a.dtype.kind in "fiub":
            f = a.astype(np.float64)
            fin = np.isfinite(f)
            if fin.any():
                d.update(max=float(f[fin].max()), min=float(f[fin].min()), mean=float(f[fin].mean()))
            d["nonfinite"] = int((~fin).sum())
        return d
    except Exception as e:  # noqa
        return {"err": repr(e)[:80]}


def _val(v, depth=0):
    import numpy as np
    if isinstance(v, bool) or v is None or isinstance(v, str) and len(v) < 200:
        return v
    if isinstance(v, (int, float)):
        return v if not (isinstance(v, float) and not math.isfinite(v)) else repr(v)
    if isinstance(v, np.generic):
        return _val(v.item())
    t = type(v).__module__
    if isinstance(v, np.ndarray) or (t.startswith("warp") and hasattr(v, "numpy") and hasattr(v, "shape")) or (t.startswith("torch") and hasattr(v, "detach")):
        return _arr(v)
    if depth < 2 and isinstance(v, (list, tuple)) and len(v) <= 64:
        out = [_val(x, depth + 1) for x in v]
        return out if any(o is not None for o in out) else None
    if depth < 2 and isinstance(v, dict) and len(v) <= 64:
        out = {str(k): _val(x, depth + 1) for k, x in v.items()}
        out = {k: x for k, x in out.items() if x is not None}
        return out or None
    return None


def _snap(code):
    f = sys._getframe(2)
    while f is not None and f.f_code is not code:
        f = f.f_back
    if f is None:
        return
    nid = _target.get(code)
    rec = {}
    for k, v in list(f.f_locals.items()):
        try:
            x = _val(v)
        except Exception as e:  # noqa
            x = {"err": repr(e)[:80]}
        if x is not None:
            rec[k] = x
    _records[nid] = rec


def _on_return(code, off, retval):
    if code in _target:
        _snap(code)


def _on_unwind(code, off, exc):
    if code in _target:
        _snap(code)


def pytest_configure(config):
    m = sys.monitoring
    m.use_tool_id(TOOL, "capture_locals")
    m.register_callback(TOOL, m.events.PY_RETURN, _on_return)
    m.register_callback(TOOL, m.events.PY_UNWIND, _on_unwind)
    m.set_events(TOOL, m.events.PY_UNWIND)  # unwind is global-only; filtered by code in the callback


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    code = getattr(getattr(item, "obj", None), "__code__", None)
    if code is not None:
        _target.clear(); _target[code] = item.nodeid
        sys.monitoring.set_local_events(TOOL, code, sys.monitoring.events.PY_RETURN)
    yield
    if code is not None:
        sys.monitoring.set_local_events(TOOL, code, 0)
        _target.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    out = yield
    rep = out.get_result()
    if OUT and (rep.when == "call" or rep.outcome != "passed"):
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"nodeid": item.nodeid, "when": rep.when, "outcome": rep.outcome,
                                 "xfail": hasattr(rep, "wasxfail"), "duration": rep.duration,
                                 "longrepr": (str(rep.longrepr)[-1500:] if rep.failed else (str(rep.longrepr)[:300] if rep.skipped else None)),
                                 "locals": _records.pop(item.nodeid, None) if rep.when == "call" else None}, default=str) + "\n")
