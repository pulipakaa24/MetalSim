# Superseded: mixed-preset il3 runs (2026-09-25)

These runs are kept for the record but do not measure Isaac's settings:

- `train_flat_il3_isaaclab3_every_substep_cap20_s0.*` (flat +24.9 at 1499, 44 blow-ups)
- `train_rough_il3_isaaclab3_every_substep_cap20_s0.*` (rough +6.1 / level 5.70, 1,615 blow-ups)
- `step2b.log` / `transfer_*.json`: the transfer columns "default" and "hardlimits" (both ran on the task's
  `contact_cfg="recommended"`), and the isaaclab3 columns (recommended limit impedance underneath)

Cause: since e3ba79f (14:52) `G1VelocityTask` applies `contact_cfg="recommended"` (tau10_impact_hardlimits) before
`solver_cfg`, and `solver_presets.apply` did not set `jnt_solimp` or `geom_margin`. The runs therefore combined Isaac's
soft joint-limit solref (time constants up to 0.46 s, damping ratio 0.03) with the hard-limit impedance 0.99–0.999
instead of Isaac's 0.9–0.95. The configuration is reproducible as `solver_cfg="isaaclab3_mixed_recommended_limits"`.
Fixed at 19:50: the solver preset owns every contact and limit field
(`tests/test_solver_presets.py::test_effective_model_fields_per_preset_combination`). The fidelity table
(`fidelity/`, `record_g1` does not use the task) is unaffected.
