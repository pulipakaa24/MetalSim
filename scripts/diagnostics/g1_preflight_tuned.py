"""metalsim.learn.g1_preflight with a contact/limit tuning applied to the G1 model (no edit of the task):

    python scripts/diagnostics/g1_preflight_tuned.py --contact_tuning tau5_imp99_hardlimits --envs 1024 --physics_dt 0.0025 --probe_iters 20
"""
import sys

from metalsim.physics import contact_tuning

i = sys.argv.index("--contact_tuning")
name = sys.argv[i + 1]
del sys.argv[i:i + 2]
print(f"contact tuning: {name}: {contact_tuning.PRESETS[name]}", flush=True)
with contact_tuning.g1_model_tuning(name):
    from metalsim.learn import g1_preflight
    g1_preflight.main()
