"""Check bam_m6.budget (and so the patch's copy) against BAM's own Model.compute_frictions (clone b9e1111), m1..m6,
on random motor / external torques and velocities.  <main venv>/python -I check_budget.py"""
import os, sys
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/so101-servo-src/bam")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import types  # noqa: E402
for mod in ("colorama",):                      # bam.message's colours only; stubbed so no package is installed
    if mod not in sys.modules:
        st = types.ModuleType(mod)
        st.Fore = st.Back = st.Style = types.SimpleNamespace(__getattr__=lambda *_: "")
        class _C:
            def __getattr__(self, _):
                return ""
        st.Fore, st.Back, st.Style = _C(), _C(), _C()
        sys.modules[mod] = st
from bam.model import load_model  # noqa: E402
import bam_m6 as B  # noqa: E402
rng = np.random.default_rng(0)
tm, te, w = rng.uniform(-3, 3, 20000), rng.uniform(-3, 3, 20000), rng.uniform(-1.5, 1.5, 20000)
for k in ("m1", "m2", "m3", "m4", "m5", "m6"):
    ref = load_model(os.path.join(B.PDIR, f"{k}.json"))
    fl_ref, damp = ref.compute_frictions(tm, te, w)
    fl = B.budget(B.load(k), tm, te, w)
    print(k, "max |ours - BAM|", float(np.max(np.abs(fl - fl_ref))), "damping", damp)
