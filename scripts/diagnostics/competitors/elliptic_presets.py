"""Registers the elliptic-cone variants of the final G1 contact preset in metalsim.physics.contact_tuning.PRESETS and
then runs another script (the fidelity drivers take preset names): elliptic cones on top of tau10_impact_hardlimits
(the adopted preset, DECISIONS 2026-09-25) with impratio 1 / 10 / 100 (100 = Menagerie Go2's own setting).

    python scripts/diagnostics/competitors/elliptic_presets.py scripts/diagnostics/bench_contact_tuning.py default tau10_impact_elliptic_hardlimits
    python scripts/diagnostics/competitors/elliptic_presets.py -m metalsim.parity.record_g1 --contact_tuning tau10_impact_elliptic_hardlimits ..."""
import runpy, sys
from metalsim.physics import contact_tuning as ct

_T10 = ct.PRESETS["tau10_impact_hardlimits"]
EXTRA = {
    "tau10_impact_elliptic_hardlimits": ct.Tuning(contact_solref=_T10.contact_solref, contact_solimp=_T10.contact_solimp,
                                                  limit_solref=_T10.limit_solref, limit_solimp=_T10.limit_solimp, cone="elliptic", impratio=1.0,
                                                  note="tau10_impact_hardlimits + elliptic cone, impratio 1"),
    "tau10_impact_elliptic_imp10_hardlimits": ct.Tuning(contact_solref=_T10.contact_solref, contact_solimp=_T10.contact_solimp,
                                                        limit_solref=_T10.limit_solref, limit_solimp=_T10.limit_solimp, cone="elliptic", impratio=10.0,
                                                        note="tau10_impact_hardlimits + elliptic cone, impratio 10"),
    "tau10_impact_elliptic_imp100_hardlimits": ct.Tuning(contact_solref=_T10.contact_solref, contact_solimp=_T10.contact_solimp,
                                                         limit_solref=_T10.limit_solref, limit_solimp=_T10.limit_solimp, cone="elliptic", impratio=100.0,
                                                         note="tau10_impact_hardlimits + elliptic cone, impratio 100"),
    "elliptic_imp100_hardlimits": ct.Tuning(cone="elliptic", impratio=100.0, limit_solref=_T10.limit_solref, limit_solimp=_T10.limit_solimp,
                                            note="elliptic cone, impratio 100, hard limits"),
}
ct.PRESETS.update(EXTRA)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "-m":
        sys.argv = [args[1]] + args[2:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")
