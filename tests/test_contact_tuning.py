"""metalsim.physics.contact_tuning: presets set the intended MjModel / MjSpec fields, margin only on the
named geoms, and the G1 builder context manager tunes and then restores."""
import mujoco
import numpy as np

from metalsim.physics import contact_tuning as ct

XML = """<mujoco><worldbody><geom name="ground" type="plane" size="1 1 .1"/>
  <body><freejoint/><geom name="a" size=".1"/></body>
  <body pos="0 0 1"><joint name="h" range="-1 1"/><geom name="b" size=".1"/></body></worldbody></mujoco>"""


def test_presets_on_model_and_spec():
    m = mujoco.MjModel.from_xml_string(XML)
    ct.apply(m, "tau5_imp99_hardlimits")
    assert np.allclose(m.geom_solref, [0.005, 1.0]) and np.allclose(m.geom_solimp[:, :3], [0.99, 0.999, 0.001])
    assert np.allclose(m.jnt_solref, [0.005, 1.0]) and np.allclose(m.geom_margin, 0)
    ct.apply(m, "tau5_imp99_margin_gap1cm")
    assert m.geom_margin.tolist() == [0.01, 0, 0] and m.geom_gap.tolist() == [0.01, 0, 0]   # ground only
    s = mujoco.MjSpec.from_string(XML); ct.apply(s, "tau5_imp99_hardlimits"); m2 = s.compile()
    assert np.allclose(m2.geom_solref, m.geom_solref) and np.allclose(m2.jnt_solref, [0.005, 1.0])
    s = mujoco.MjSpec.from_string(XML); ct.apply(s, "recommended"); m3 = s.compile()     # tau10_impact_hardlimits
    assert np.allclose(m3.geom_solref, [0.01, 1.0]) and np.allclose(m3.geom_solimp[:, :3], [0.9, 0.999, 0.005])
    m2.opt.timestep = 0.005; assert len(ct.check_timestep(m2, "tau5_imp99_hardlimits")) == 2   # 5 ms < 2 dt: MuJoCo would clamp to 10 ms


def test_g1_builder_context_restores():
    import metalsim.learn.g1_velocity as g1
    orig = g1.build_g1_model
    with ct.g1_model_tuning("recommended"):
        assert g1.build_g1_model is not orig
    assert g1.build_g1_model is orig


def test_elliptic_presets_set_cone_and_impratio():
    """tau10_impact_hardlimits_ellip{1,10,100}: the adopted contact/limit stiffness plus the elliptic cone and impratio."""
    base = mujoco.MjModel.from_xml_string(XML); ct.apply(base, "tau10_impact_hardlimits")
    assert base.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL and base.opt.impratio == 1.0     # the base preset leaves the cone alone
    for imp in (1, 10, 100):
        m = mujoco.MjModel.from_xml_string(XML); ct.apply(m, f"tau10_impact_hardlimits_ellip{imp}")
        assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == imp
        assert np.allclose(m.geom_solref, base.geom_solref) and np.allclose(m.geom_solimp, base.geom_solimp)
        assert np.allclose(m.jnt_solref, base.jnt_solref) and np.allclose(m.jnt_solimp, base.jnt_solimp)
        assert np.allclose(m.geom_margin, base.geom_margin) and np.allclose(m.geom_friction, base.geom_friction)
        s = mujoco.MjSpec.from_string(XML); ct.apply(s, f"tau10_impact_hardlimits_ellip{imp}"); m2 = s.compile()
        assert m2.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m2.opt.impratio == imp
    # the G1 task's builder applies the preset through contact_cfg; check the model it would build (CPU, no GPU)
    from metalsim.learn.g1_velocity import build_g1_model
    g1m, _ = build_g1_model("flat", physics_dt=0.0025); ct.apply(g1m, "tau10_impact_hardlimits_ellip10")
    assert g1m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and g1m.opt.impratio == 10.0
    assert np.allclose(g1m.geom_solref, [0.01, 1.0]) and np.allclose(g1m.geom_solimp[:, :3], [0.9, 0.999, 0.005])
    assert np.allclose(g1m.jnt_solref[g1m.jnt_limited.astype(bool)], [0.005, 1.0])
