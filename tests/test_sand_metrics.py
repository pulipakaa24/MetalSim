"""CPU tests of the sand-fidelity measurements (metalsim/physics/mpm/probes/sand_metrics.py) on synthetic particle
sets with known answers, and of the batch-01 evaluator on synthetic results. No GPU."""
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metalsim/physics/mpm/probes"))
import sand_metrics as SM  # noqa: E402


def _cone(angle_deg, r_toe=0.08, sp=0.005 / 3, convex=False):
    """Particles filling a cone (or, convex=True, a cone with a rounded crest) on the lattice the probes use."""
    n = int(r_toe / sp) + 1
    g = (np.arange(-n, n) + 0.5) * sp
    x, y = np.meshgrid(g, g, indexing="ij")
    r = np.hypot(x, y)
    t = math.tan(math.radians(angle_deg))
    h = np.maximum(t * (r_toe - r), 0.0)
    if convex:
        cap = 0.6 * t * r_toe
        h = np.where(h > cap, cap + 0.3 * (h - cap), h)
    pts = []
    for xi, yi, hi in zip(x.ravel(), y.ravel(), h.ravel()):
        k = int(hi / sp)
        if k:
            z = (np.arange(k) + 0.5) * sp
            pts.append(np.stack([np.full(k, xi), np.full(k, yi), z], 1))
    return np.concatenate(pts).astype(np.float64), sp


def test_cone_band_angles_uniform():
    q, sp = _cone(30.0)
    prof = SM.heap_profiles(q, sp ** 3)
    res = SM.flank_angles_by_height(prof, 0.005)
    angles = [b["angle_deg"] for b in res["bands"][:3]]
    assert all(abs(a - 30.0) < 1.5 for a in angles), angles


def test_convex_heap_steeper_low_band():
    q, sp = _cone(30.0, convex=True)
    res = SM.flank_angles_by_height(SM.heap_profiles(q, sp ** 3), 0.005)
    low, high = res["bands"][0]["angle_deg"], res["bands"][3]["angle_deg"]
    assert low > high + 5, (low, high)


def test_heap_profiles_match_repose_lift_surface():
    from repose_lift import heap_metrics

    q, sp = _cone(30.0)
    m = heap_metrics(q, 0.005, sp ** 3)
    prof = SM.heap_profiles(q, sp ** 3)
    assert abs(max(prof["all"]["h"]) - m["h_max"]) < 1e-12


def test_stress_invariants_simple_shear_and_triaxial():
    p0, tau = 100.0, 30.0
    shear = np.array([[-p0, 0, tau], [0, -p0, 0], [tau, 0, -p0]])          # tension-positive Cauchy stress
    tri = np.diag([-p0 - 2 * tau, -p0 + tau, -p0 + tau])                    # axial compression, two equal lateral
    p, rj2, lode, sign = SM.stress_invariants(np.stack([shear, shear, tri]))
    assert sign == 1.0 and np.allclose(p, p0)
    assert abs(rj2[0] - tau) < 1e-9 and abs(lode[0]) < 1e-9                 # pure shear: Lode parameter 0
    assert abs(abs(lode[2]) - 1.0) < 1e-9                                   # triaxial meridian
    p2, _, lode2, sign2 = SM.stress_invariants(-np.stack([shear, tri]))     # compression-positive input
    assert sign2 == -1.0 and np.allclose(p2, p0) and np.allclose(lode2, lode[[0, 2]])


def test_flank_stress_ratio_on_cone():
    q, sp = _cone(30.0)
    mu, ratio = 0.68, 0.8
    S = np.zeros((q.shape[0], 3, 3))
    S[:, 0, 0] = S[:, 1, 1] = S[:, 2, 2] = -100.0
    S[:, 0, 2] = S[:, 2, 0] = ratio * mu * 100.0
    out = SM.flank_stress(q, S, mu, 0.005, sp ** 3, r_apex=0.005, r_toe=0.08, h_max=0.046)
    assert out["flank_particles"] > 0
    for s in out["shells"]:
        if s["n"]:
            assert abs(s["ratio_median"] - ratio) < 1e-9 and abs(s["lode_median"]) < 1e-9


def test_tilted_surface_angle():
    sp, theta, a_box = 0.005 / 3, 35.0, 4.0
    xs = np.arange(-0.15, 0.15, sp) + 0.5 * sp
    pts = []
    for x in xs:
        top = 0.05 + math.tan(math.radians(a_box)) * x
        z = np.arange(0.5 * sp, top, sp)
        pts.append(np.stack([np.full(z.size, x), np.zeros(z.size), z], 1))
    q = np.concatenate(pts)
    ang, ab, n = SM.surface_angle_world(q, theta, -0.135, 0.135, 2 * sp, 0.01)
    assert abs(ab - a_box) < 0.3 and abs(ang - (theta - a_box)) < 0.3 and n > 10


def test_deposit_angle_after_full_avalanche():
    """A layer that avalanched into a pile against the downslope wall: thin bare part upslope, a wedge whose face rises
    towards +x at 25 deg in the box frame. The deposit fit takes the face, not a line through the step (02b t1b)."""
    sp, theta, face = 0.005 / 3, 60.0, 25.0
    xs = np.arange(-0.15, 0.15, sp) + 0.5 * sp
    pts = []
    for x in xs:
        top = max(0.012, 0.012 + math.tan(math.radians(face)) * (x - 0.0))
        z = np.arange(0.5 * sp, top, sp)
        pts.append(np.stack([np.full(z.size, x), np.zeros(z.size), z], 1))
    q = np.concatenate(pts)
    ang, ab, n = SM.surface_angle_world(q, theta, -0.135, 0.135, 2 * sp, 0.01)
    assert abs(ab - face) < 1.0 and abs(ang - (theta - face)) < 1.0, (ang, ab, n)


def test_first_sustained():
    t = np.arange(0, 2, 0.025)
    x = np.where((t > 0.5) & (t < 0.6), 1.0, 0.0) + np.where(t >= 1.0, 1.0, 0.0)
    k = SM.first_sustained(t, x, 0.5, 0.25)
    assert abs(t[k] - 1.0) < 1e-9
    assert SM.first_sustained(t, np.zeros_like(t), 0.5, 0.25) is None


def test_uehara_plan_values():
    # plan section 1, G4: D 0.05 m, density ratio 1, mu 0.7: h 0.1 / 0.2 / 0.4 / 0.8 m -> ~13 / 16 / 20 / 26 mm
    for h, d in ((0.1, 0.013), (0.2, 0.016), (0.4, 0.020), (0.8, 0.026)):
        assert abs(SM.uehara_solve(0.7, 0.05, h) - d) < 0.0011


def test_box_overlap():
    q = np.array([[0.0, 0.0, -0.045], [0.0, 0.0, -0.06], [0.2, 0.0, 0.0]])
    o = SM.box_overlap(q, np.zeros(3), np.array([1.0, 0, 0, 0]), 0.05)
    assert o["inside"] == 1 and abs(o["max_depth_m"] - 0.005) < 1e-12


def test_impact_phase_ignores_late_events():
    fps, drop, depth = 480, 0.15, 0.08
    t = (np.arange(720) + 1) / fps
    tc = math.sqrt(2 * drop / 9.81)
    pen = np.where(t < tc, -0.01, 0.018)                    # 18 mm at impact ...
    pen[t > 1.2] = 0.025                                    # ... and a late sink to 25 mm
    r = {"drop": drop, "depth": depth, "fps": fps, "bottom": (depth - pen).tolist(), "peak_penetration_m": 0.025}
    ip = SM.impact_phase(r)
    assert abs(ip["peak_m"] - 0.018) < 1e-12 and abs(ip["v_impact"] - math.sqrt(2 * 9.81 * drop)) < 1e-12
    assert SM.impact_phase({"peak_penetration_m": 0.02})["peak_m"] == 0.02     # records without the time series


def test_tilt_box_scene_and_nonfinite_report():
    mujoco = pytest.importorskip("mujoco")
    import tilt_box as TB

    for ox in (0.0, 0.0025):
        m = mujoco.MjModel.from_xml_string(TB.box_xml(1 / 240, ox, ox))
        assert m.ngeom == 5
        x_face = m.geom_pos[4][0] - m.geom_size[4][0]                       # inner face of the downslope end wall
        assert abs(x_face - (ox + 0.5 * TB.L)) < 1e-9
    q = np.array([[0.149, 0.0, 0.03], [0.0, 0.0, 0.03], [0.149, 0.0195, 0.001]], np.float32)
    v = np.zeros_like(q)
    bad = np.array([True, False, True])
    w = TB.nonfinite_report(q, v, bad, 0.0, 0.0, 0.005)
    assert w["n_bad"] == 2 and w["frac_near_end_wall"] == 1.0 and w["frac_near_floor_edge"] == 0.5
    assert w["frac_near_wall_edge"] == 0.5


def test_sphere_scenes_and_collider_layout():
    mujoco = pytest.importorskip("mujoco")
    import sphere_drop as SD

    from metalsim.physics.mpm import colliders as C

    cases = {"plain": ({}, 1, 1), "walls": ({"walls": (0.2, 0.2, 0.12)}, 1, 5),
             "control": ({"ball": False, "walls": (0.2, 0.2, 0.12)}, 0, 5),
             "kinematic": ({"kinematic": True, "walls": (0.2, 0.2, 0.12)}, 1, 5)}
    for name, (kw, moving, static) in cases.items():
        m = mujoco.MjModel.from_xml_string(SD.scene_xml(1 / 1440, 0.025, 0.16, 0.146, ox=0.0025, oy=0.0025, **kw))
        lay = C.layout(m, list(range(m.ngeom)))
        assert (len(lay.body_map), len(lay.static_geoms)) == (moving, static), name
        if kw.get("walls"):                                 # wall faces half a voxel off the grid nodes
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall_xhi")
            face = m.geom_pos[g][0] - m.geom_size[g][0]
            assert abs(face - (0.0025 + 0.1)) < 1e-9 and abs((face / 0.005) % 1 - 0.5) < 1e-6


def test_solve_log_summary():
    import solve_log

    logs = [{"iterations": np.array([250, 250, 100]), "res_l2": np.array([1e-3, 2e-3, 1e-5]),
             "res_linf": np.array([1e-2, 1e-2, 1e-4])}, {"iterations": np.array([250]), "res_l2": np.array([1e-3]),
                                                         "res_linf": np.array([1e-2])}]
    s = solve_log.summary(logs, 250)
    assert s["solves"] == 4 and abs(s["frac_at_cap"] - 0.75) < 1e-12
    assert solve_log.summary([], 250) == {"solves": 0}


def test_tilt_box_ejection_mask():
    """The tilt probe's ejection test: particles > 0.5 m outside the box (02c t1b: the layer ended at 1-788 m)."""
    pytest.importorskip("mujoco")
    import tilt_box as TB

    q = np.array([[0.0, 0.0, 0.03], [0.149, 0.0, 0.0], [0.0, 0.0, 788.0], [0.0, 0.0, -46.0], [0.9, 0.0, 0.02]])
    out = TB.outside_box(q, 0.0, 0.0)
    assert out.tolist() == [False, False, True, True, True]


def test_evaluator_02d_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/sand_fidelity/evaluate_batch_02d.py"), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_evaluator_02c_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/sand_fidelity/evaluate_batch_02c.py"), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_evaluator_02b_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/sand_fidelity/evaluate_batch_02b.py"), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_evaluator_02a_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/sand_fidelity/evaluate_batch_02a.py"), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_evaluator_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/sand_fidelity/evaluate_batch_01.py"), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
