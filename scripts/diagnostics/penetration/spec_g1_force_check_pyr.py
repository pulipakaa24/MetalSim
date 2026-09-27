import sys, runpy
from metalsim.physics import contact_tuning as ct
import dataclasses
ct.PRESETS["_pyr_tau5_specgap10mm"] = dataclasses.replace(ct.PRESETS["ellip10_tau5_specgap10mm"], cone="pyramidal", impratio=1.0)
ct.PRESETS["_ellip_imp1_tau5_specgap10mm"] = dataclasses.replace(ct.PRESETS["ellip10_tau5_specgap10mm"], impratio=1.0)
sys.argv = ["x", sys.argv[1], sys.argv[2], sys.argv[3]]
import os; runpy.run_path(os.path.join(os.path.dirname(__file__), "spec_g1_force_check.py"), run_name="__main__")
