"""MetalSim's Lode-dependent yield surface (solver variants ``pr4660mc`` / ``pr4660mn``;
research_notes/sand_fidelity/PLAN_2026-10-14.md section 17). CPU only (Warp's CPU device).

What is pinned here, before any GPU run:

* the kernel's generalized-classical Lode function equals the exact Mohr-Coulomb (up to the documented corner
  rounding) and Matsuoka-Nakai radii, with the solver's compression-positive sign (triaxial compression = +30 deg);
* the deviatoric section is convex;
* the Lode angle is taken in the world frame, so it does not depend on the Delassus 5x5 rotation of the local solve;
* the local flow rule (a bracketed secant solve of the Lode factor, at most 8 anisotropic solves) returns stresses on
  the surface to < 1e-3 for Delassus anisotropy up to 100, and leaves stresses inside it alone;
* with the Lode factor fixed to 1 the flow rule is bitwise the ``pr4660`` one;
* the friction -> 0 limits (Mohr-Coulomb -> rounded Tresca, Matsuoka-Nakai -> a circle);
* the configuration keys (``MPMConfig(fidelity=..., yield_surface=...)``) and their validation;
* the material conventions the presets need, in the full solver on CPU: ``yield_stress`` is tau_y in simple shear
  and ``viscosity`` acts as tau = viscosity x gamma_dot / 2 (``test_viscosity_convention``).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

wp = pytest.importorskip("warp")
pytest.importorskip("newton")

import metalsim.physics.mpm.implicit_mpm_pr4660.rheology_solver_kernels as rk_p  # noqa: E402
import metalsim.physics.mpm.implicit_mpm_pr4660mc.rheology_solver_kernels as rk_mc  # noqa: E402
import metalsim.physics.mpm.implicit_mpm_pr4660mn.rheology_solver_kernels as rk_mn  # noqa: E402

vec6 = rk_mc.vec6
mat55 = rk_mc.mat55
YP = rk_mc.YieldParamVec
# kernel tests: Warp CPU by default; METALSIM_LODE_DEVICE=metal:0 runs them on Metal (batch 03a). The full-solver
# material measurements below always run on the CPU.
DEVICE = os.environ.get("METALSIM_LODE_DEVICE", "cpu")
RNG_SEED = 20261015

_flow_mc = rk_mc.make_solve_flow_rule(False, False)
_flow_mn = rk_mn.make_solve_flow_rule(False, False)
_flow_p = rk_p.make_solve_flow_rule(False, False)
_given_mc = rk_mc.make_solve_flow_rule_given_scale(False)


# ------------------------------------------------------------------------------------------------ kernels
@wp.kernel
def _k_lode(r: wp.array(dtype=vec6), mu: wp.array(dtype=float), s3: wp.array(dtype=float),
            k_mc: wp.array(dtype=float), k_mn: wp.array(dtype=float)):
    i = wp.tid()
    s = rk_mc.lode_sin3theta(r[i])
    s3[i] = s
    k_mc[i] = rk_mc.lode_scale(s, mu[i])
    k_mn[i] = rk_mn.lode_scale(s, mu[i])


@wp.kernel
def _k_local(r_world: wp.array(dtype=vec6), r_local: wp.array(dtype=vec6), rot: wp.array(dtype=mat55),
             yp: wp.array(dtype=YP), k_world: wp.array(dtype=float), k_local: wp.array(dtype=float)):
    i = wp.tid()
    k_world[i] = rk_mc.lode_scale_world(r_world[i], yp[i])
    k_local[i] = rk_mc.lode_scale_local(r_local[i], rot[i], yp[i])


@wp.kernel
def _k_flow(D: wp.array(dtype=vec6), b: wp.array(dtype=vec6), yp: wp.array(dtype=YP), rot: wp.array(dtype=mat55),
            u_mc: wp.array(dtype=vec6), u_mn: wp.array(dtype=vec6), u_p: wp.array(dtype=vec6),
            u_mc_k1: wp.array(dtype=vec6)):
    i = wp.tid()
    u_mc[i] = _flow_mc(D[i], b[i], vec6(0.0), yp[i], 1.0, rot[i])
    u_mn[i] = _flow_mn(D[i], b[i], vec6(0.0), yp[i], 1.0, rot[i])
    u_p[i] = _flow_p(D[i], b[i], vec6(0.0), yp[i], 1.0)
    u_mc_k1[i] = _given_mc(D[i], b[i], yp[i], 1.0)


@wp.kernel
def _k_bounds(mu: wp.array(dtype=float), lo_mc: wp.array(dtype=float), hi_mc: wp.array(dtype=float),
              lo_mn: wp.array(dtype=float), hi_mn: wp.array(dtype=float)):
    i = wp.tid()
    a, b = rk_mc.lode_scale_bounds(mu[i])
    lo_mc[i] = a
    hi_mc[i] = b
    c, d = rk_mn.lode_scale_bounds(mu[i])
    lo_mn[i] = c
    hi_mn[i] = d


@wp.kernel
def _k_project(r: wp.array(dtype=vec6), yp: wp.array(dtype=YP), out: wp.array(dtype=vec6)):
    i = wp.tid()
    out[i] = rk_mc.project_stress(r[i], yp[i])


def _run(kernel, n, inputs, outputs):
    wp.launch(kernel, dim=n, inputs=inputs + outputs, device=DEVICE)
    return [o.numpy() for o in outputs]


def _arr(a, dtype):
    return wp.array(np.ascontiguousarray(a), dtype=dtype, device=DEVICE)


def _zeros(n, dtype):
    return wp.zeros(n, dtype=dtype, device=DEVICE)


# ------------------------------------------------------------------------------------------------ numpy references
S3 = np.sqrt(3.0)


def tensor_to_dof(m):
    """Inverse of warp.fem's DB16 ``dof_to_value_3d`` (orthonormal for (tau : tau) / 2); (..., 3, 3) -> (..., 6)."""
    a = np.trace(m, axis1=-2, axis2=-1) / 3.0
    c = (m[..., 2, 2] - a) / 2.0
    b = m[..., 0, 0] - a + c
    return np.stack([a * np.sqrt(1.5), b, c * S3, m[..., 1, 2], m[..., 0, 2], m[..., 0, 1]], axis=-1)


def dof_to_tensor(d):
    a = d[..., 0] * np.sqrt(2.0 / 3.0)
    b, c, dd, e, f = d[..., 1], d[..., 2] / S3, d[..., 3], d[..., 4], d[..., 5]
    m = np.empty(d.shape[:-1] + (3, 3))
    m[..., 0, 0], m[..., 1, 1], m[..., 2, 2] = a + b - c, a - b - c, a + 2.0 * c
    m[..., 0, 1] = m[..., 1, 0] = f
    m[..., 0, 2] = m[..., 2, 0] = e
    m[..., 1, 2] = m[..., 2, 1] = dd
    return m


def principal(theta, p, sj2):
    """Compression-positive principal stresses with mean p, sqrt(J2) sj2 and Lode angle theta (+pi/6: triaxial
    compression, sigma_1 > sigma_2 = sigma_3)."""
    a = 2.0 / S3 * sj2
    return np.stack([p + a * np.cos(theta - np.pi / 6 + k * 2 * np.pi / 3) for k in (0, -1, 1)], axis=-1)


def random_rotation(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    return q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]


def stress_dofs(rng, theta, p, sj2):
    sig = principal(theta, p, sj2)
    R = random_rotation(rng, len(sig))
    return tensor_to_dof(np.einsum("nij,nj,nkj->nik", R, sig, R))


def gc_scale(sin3t, mu, surface, beta_mc=0.9999):
    """float64 replica of the kernel's k(theta) = M(theta) / mu (generalized-classical; Lagioia & Panteghini 2016)."""
    phi = np.maximum(np.arctan(mu), 0.01)
    s = np.sin(phi)
    mc = 6 * s / (3 - s)
    if surface == "mn":
        a1, a2 = (6 + 2 * s * s) / (8 * s * s), (9 - s * s) / (8 * s * s)
        alpha, beta, gamma = 2 / 3 * np.sqrt(a1) * mc, a2 / a1 ** 1.5, 0.0
    else:
        gbar = 6 / np.pi * np.arctan(s / S3)
        alpha, beta, gamma = 1 / np.cos((gbar + 1) * np.pi / 6), beta_mc, 1 - gbar
    shape = alpha * np.cos(np.arccos(np.clip(-beta * sin3t, -1, 1)) / 3 - gamma * np.pi / 6)
    return mc / (S3 * shape * np.tan(phi))


def exact_mc_ratio(theta, phi):
    """sqrt(J2) / p of the sharp Mohr-Coulomb surface (c = 0)."""
    return np.sin(phi) / (np.cos(theta) - np.sin(theta) * np.sin(phi) / S3)


def exact_mn_ratio(theta, phi):
    """sqrt(J2) / p of Matsuoka-Nakai (I1 I2 / I3 = (9 - sin^2) / (1 - sin^2), c = 0), by bisection."""
    s = np.sin(phi)
    K = (9 - s * s) / (1 - s * s)
    lo = np.zeros_like(theta)
    cneg = -np.min(np.stack([np.cos(theta - np.pi / 6 + k * 2 * np.pi / 3) for k in (0, -1, 1)]), axis=0)
    hi = 0.999999 * S3 / 2 / cneg  # the smallest principal stress stays positive
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s1, s2, s3 = np.moveaxis(principal(theta, 1.0, mid), -1, 0)
        f = (s1 + s2 + s3) * (s1 * s2 + s2 * s3 + s3 * s1) / (s1 * s2 * s3) - K
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)


def friction_angle(sig):
    s1, s3 = sig.max(axis=-1), sig.min(axis=-1)
    return np.degrees(np.arcsin((s1 - s3) / (s1 + s3)))


def yield_params(n, mu, s=0.0, p_max=1.0e6):
    sc = np.sqrt(1.5)
    yp = np.zeros((n, 6))
    yp[:, 0], yp[:, 1], yp[:, 2], yp[:, 3] = p_max * sc, 0.0, s, mu * p_max
    return yp


# ------------------------------------------------------------------------------------------------ tests
PHIS = np.radians(np.linspace(5.0, 50.0, 10))
THETAS = np.radians(np.linspace(-30.0, 30.0, 61))


def _lode_eval(r, mu):
    n = len(r)
    return _run(_k_lode, n, [_arr(r, vec6), _arr(mu, float)], [_zeros(n, float), _zeros(n, float), _zeros(n, float)])


def test_lode_angle_sign_and_kernel_matches_reference():
    """sin(3 theta) of rotated principal states is the constructed one (+1 = triaxial compression), and the
    kernel's k(theta) equals the float64 replica."""
    rng = np.random.default_rng(RNG_SEED)
    phi, th = np.meshgrid(PHIS, THETAS, indexing="ij")
    phi, th = phi.ravel(), th.ravel()
    r = stress_dofs(rng, th, p=rng.uniform(0.5, 2.0, th.size), sj2=rng.uniform(0.1, 1.0, th.size))
    s3, k_mc, k_mn = _lode_eval(r, np.tan(phi))
    np.testing.assert_allclose(s3, np.sin(3 * th), atol=2e-5)
    np.testing.assert_allclose(k_mc, gc_scale(np.sin(3 * th), np.tan(phi), "mc"), rtol=2e-5)
    np.testing.assert_allclose(k_mn, gc_scale(np.sin(3 * th), np.tan(phi), "mn"), rtol=2e-5)
    # uniaxial compression (compression-positive tensor diag(1, 0, 0)) is triaxial compression
    s3u, k_u, _ = _lode_eval(tensor_to_dof(np.diag([1.0, 0.0, 0.0])[None]), np.array([0.68]))
    assert s3u[0] > 0.9999 and k_u[0] > 1.0


def test_generalized_classical_equals_exact_radii():
    phi, th = np.meshgrid(PHIS, THETAS, indexing="ij")
    mu = np.tan(phi)
    # Mohr-Coulomb: sharp GC (beta = 1) is exact; the kernel's inner rounding (beta = 0.9999) costs <= 0.33 deg up to
    # phi = 50 deg, the most at the triaxial corners
    np.testing.assert_allclose(gc_scale(np.sin(3 * th), mu, "mc", beta_mc=1.0) * mu, exact_mc_ratio(th, phi),
                               rtol=1e-6)
    m_round = gc_scale(np.sin(3 * th), mu, "mc") * mu
    err = friction_angle(principal(th, 1.0, m_round)) - np.degrees(phi)
    assert err.max() < 0.01 and err.min() > -0.33, (err.min(), err.max())
    # Matsuoka-Nakai: exact
    np.testing.assert_allclose(gc_scale(np.sin(3 * th), mu, "mn") * mu, exact_mn_ratio(th, phi), rtol=1e-6)


def test_deviatoric_section_is_convex():
    """Midpoint test in the deviatoric plane: for boundary points P, Q the midpoint lies inside."""
    rng = np.random.default_rng(RNG_SEED + 1)
    w = np.linspace(0.0, 2 * np.pi, 721)[:-1]          # polar angle in the deviatoric plane
    for surface in ("mc", "mn"):
        for phi in PHIS:
            mu = np.tan(phi)
            radius = gc_scale(np.sin(3 * w), mu, surface)  # sin(3 theta) is 2pi/3-periodic and even about the axes
            pts = np.stack([radius * np.cos(w), radius * np.sin(w)], axis=-1)
            i, j = rng.integers(0, len(w), size=(2, 4000))
            mid = 0.5 * (pts[i] + pts[j])
            r_mid = np.hypot(mid[:, 0], mid[:, 1])
            w_mid = np.arctan2(mid[:, 1], mid[:, 0])
            assert np.all(r_mid <= gc_scale(np.sin(3 * w_mid), mu, surface) * (1 + 1e-9)), (surface, phi)


def test_lode_factor_bounds_bracket_every_lode_angle():
    """The secant iterate is clamped to [min, max] of k over the Lode angles; Mohr-Coulomb's minimum is mid-face."""
    mu = np.tan(np.concatenate([[0.0], PHIS]))
    n = len(mu)
    lo_mc, hi_mc, lo_mn, hi_mn = _run(_k_bounds, n, [_arr(mu, float)], [_zeros(n, float) for _ in range(4)])
    s3 = np.sin(3 * np.radians(np.linspace(-30.0, 30.0, 6001)))
    for lo, hi, surface in ((lo_mc, hi_mc, "mc"), (lo_mn, hi_mn, "mn")):
        k = gc_scale(s3[None, :], mu[:, None], surface)
        np.testing.assert_allclose(lo, k.min(axis=1), rtol=2e-5)
        np.testing.assert_allclose(hi, k.max(axis=1), rtol=2e-5)


def test_lode_angle_is_rotation_invariant():
    rng = np.random.default_rng(RNG_SEED + 2)
    n = 1000
    r_world = stress_dofs(rng, rng.uniform(-np.pi / 6, np.pi / 6, n), rng.uniform(0.5, 2, n), rng.uniform(0.1, 1, n))
    q, rr = np.linalg.qr(rng.normal(size=(n, 5, 5)))
    rot = q * np.sign(np.diagonal(rr, axis1=1, axis2=2))[:, None, :]
    r_local = r_world.copy()
    r_local[:, 1:] = np.einsum("ni,nij->nj", r_world[:, 1:], rot)        # _world_to_local: world[1:6] @ rot
    yp = yield_params(n, 0.68)
    k_world, k_local = _run(_k_local, n, [_arr(r_world, vec6), _arr(r_local, vec6), _arr(rot, mat55), _arr(yp, YP)],
                            [_zeros(n, float), _zeros(n, float)])
    np.testing.assert_allclose(k_local, k_world, rtol=2e-5)
    # and a non-trivial check that the rotation matters if ignored: local dofs read as world give other angles
    s3_naive, _, _ = _lode_eval(r_local, np.full(n, 0.68))
    s3_true, _, _ = _lode_eval(r_world, np.full(n, 0.68))
    assert np.abs(s3_naive - s3_true).max() > 0.1


def _flow_case(rng, n, mu, s=0.0, scale=3.0, d_lo=0.2):
    D = rng.uniform(d_lo, 2.0, size=(n, 6))
    q, rr = np.linalg.qr(rng.normal(size=(n, 5, 5)))
    rot = q * np.sign(np.diagonal(rr, axis1=1, axis2=2))[:, None, :]
    # trial local stress r0 = -b / D: compressive normal part, deviator `scale` x the DP radius (plastic) on average
    r0 = np.zeros((n, 6))
    r0[:, 0] = rng.uniform(0.5, 2.0, n)
    dirs = rng.normal(size=(n, 5))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mag = (s + mu / np.sqrt(1.5) * r0[:, 0]) * rng.uniform(0.2, scale, n)
    r0[:, 1:] = dirs * mag[:, None]
    b = -r0 * D
    return D, b, rot, yield_params(n, mu, s)


def _local_to_world(r, rot):
    w = r.copy()
    w[:, 1:] = np.einsum("nij,nj->ni", rot, r[:, 1:])
    return w


def _flow(D, b, yp, rot):
    n = len(D)
    return _run(_k_flow, n, [_arr(D, vec6), _arr(b, vec6), _arr(yp, YP), _arr(rot, mat55)],
                [_zeros(n, vec6) for _ in range(4)])


@pytest.mark.parametrize("d_lo", [0.2, 0.02])
def test_flow_rule_returns_to_the_surface(d_lo):
    """The bounded secant solve of the Lode factor puts plastic stresses on the Lode-scaled surface to < 1e-3 (gate G3,
    CPU part; residual reported) for Delassus anisotropy up to 100, and leaves elastic stresses alone;
    Matsuoka-Nakai stresses satisfy I1 I2 / I3 = K."""
    rng = np.random.default_rng(RNG_SEED + 3)
    n, mu = 4000, 0.68
    D, b, rot, yp = _flow_case(rng, n, mu, d_lo=d_lo)
    u_mc, u_mn, _u_p, _ = _flow(D, b, yp, rot)
    r0 = -b / D
    worst = {}
    for name, u in (("mc", u_mc), ("mn", u_mn)):
        r = (u - b) / D
        r_world = _local_to_world(r, rot)
        _, k_mc, k_mn = _lode_eval(r_world, np.full(n, mu))
        k = k_mc if name == "mc" else k_mn
        ys = mu / np.sqrt(1.5) * r[:, 0] * k
        rt = np.linalg.norm(r[:, 1:], axis=1)
        plastic = np.linalg.norm(u[:, 1:], axis=1) > 0
        assert plastic.sum() > n // 4 and (~plastic).sum() > n // 10
        res = np.abs(rt[plastic] / ys[plastic] - 1.0)
        worst[name] = (float(res.max()), float(np.quantile(res, 0.99)))
        assert res.max() < 1e-3, (name, res.max())
        # elastic: the trial stress is inside the surface and is kept
        np.testing.assert_allclose(r[~plastic], r0[~plastic], rtol=1e-5, atol=1e-6)
        _, k0_mc, k0_mn = _lode_eval(_local_to_world(r0[~plastic], rot[~plastic]), np.full((~plastic).sum(), mu))
        k0 = k0_mc if name == "mc" else k0_mn
        assert np.all(np.linalg.norm(r0[~plastic, 1:], axis=1) <= mu / np.sqrt(1.5) * r0[~plastic, 0] * k0 * (1 + 1e-5))
        if name == "mn":
            sig = np.linalg.eigvalsh(dof_to_tensor(r_world[plastic]))
            s1, s2, s3 = sig[:, 2], sig[:, 1], sig[:, 0]
            sphi = np.sin(np.arctan(mu))
            K = (9 - sphi ** 2) / (1 - sphi ** 2)
            ratio = (s1 + s2 + s3) * (s1 * s2 + s2 * s3 + s3 * s1) / (s1 * s2 * s3)
            np.testing.assert_allclose(ratio, K, rtol=5e-3)
    print(f"Lode solve residual, D in [{d_lo}, 2], (max, p99) over plastic samples:", worst)


def test_unit_lode_factor_is_bitwise_pr4660():
    rng = np.random.default_rng(RNG_SEED + 4)
    D, b, rot, yp = _flow_case(rng, 2000, 0.68)
    _, _, u_p, u_k1 = _flow(D, b, yp, rot)
    assert np.array_equal(u_p, u_k1)


def test_project_stress_lands_on_the_surface():
    rng = np.random.default_rng(RNG_SEED + 5)
    n, mu = 2000, 0.68
    r = stress_dofs(rng, rng.uniform(-np.pi / 6, np.pi / 6, n), rng.uniform(0.5, 2, n), rng.uniform(0.05, 3, n))
    yp = yield_params(n, mu)
    (out,) = _run(_k_project, n, [_arr(r, vec6), _arr(yp, YP)], [_zeros(n, vec6)])
    s3, k_mc, _ = _lode_eval(out, np.full(n, mu))
    ys = mu / np.sqrt(1.5) * out[:, 0] * k_mc
    rt = np.linalg.norm(out[:, 1:], axis=1)
    outside = np.linalg.norm(r[:, 1:], axis=1) > ys
    assert outside.sum() > n // 4
    np.testing.assert_allclose(rt[outside], ys[outside], rtol=1e-5)
    np.testing.assert_allclose(out[~outside], r[~outside], rtol=1e-6)


def test_zero_friction_limits():
    """friction -> 0: Mohr-Coulomb -> (rounded) Tresca with sqrt(J2) = c in pure shear (so yield_stress = tau_y);
    Matsuoka-Nakai -> a circle (von Mises), within 1 % at the kernel's smallest friction angle (0.01 rad), at the
    Tresca corner radius (1.155 c: Matsuoka-Nakai is matched to Mohr-Coulomb in triaxial compression)."""
    s3 = np.sin(3 * THETAS)
    k_mc = gc_scale(s3, 0.0, "mc")
    k_mn = gc_scale(s3, 0.0, "mn")
    _, kk_mc, _ = _lode_eval(np.zeros((61, 6)) + np.array([1, 0, 0, 0, 0, 0.0]), np.zeros(61))
    np.testing.assert_allclose(kk_mc, k_mc[30], rtol=1e-5)    # zero deviator -> theta = 0
    assert abs(k_mc[30] - 1.0) < 1e-4                          # pure shear: sqrt(J2) = c
    assert abs(k_mc[-1] - 1 / np.cos(np.pi / 6)) < 0.01        # Tresca corner, slightly rounded
    assert k_mn.max() / k_mn.min() < 1.01
    assert abs(k_mn.max() - 1 / np.cos(np.pi / 6)) < 0.01
    # the kernel agrees with the reference at friction 0 over the Lode range too
    r = stress_dofs(np.random.default_rng(RNG_SEED + 6), THETAS, np.ones(61), np.full(61, 0.5))
    _, kk_mc, kk_mn = _lode_eval(r, np.zeros(61))
    np.testing.assert_allclose(kk_mc, k_mc, rtol=2e-4)
    np.testing.assert_allclose(kk_mn, k_mn, rtol=2e-4)


def test_lode_variants_change_only_the_rheology_kernels():
    """pr4660mc / pr4660mn are pr4660 plus one file's change (apart from the notices), and differ from each other
    only in the surface constant; both are selectable."""
    def body(p):
        lines = p.read_text().splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("# Modified by MetalSim"))
        j = i
        while j < len(lines) and lines[j].startswith("#"):
            j += 1
        return lines[:i] + lines[j:]

    base = ROOT / "metalsim/physics/mpm/implicit_mpm_pr4660"
    for name in ("pr4660mc", "pr4660mn"):
        var = ROOT / f"metalsim/physics/mpm/implicit_mpm_{name}"
        for p in sorted(base.glob("*.py")):
            assert (body(p) == body(var / p.name)) == (p.name != "rheology_solver_kernels.py"), (name, p.name)
    mc = body(ROOT / "metalsim/physics/mpm/implicit_mpm_pr4660mc/rheology_solver_kernels.py")
    mn = body(ROOT / "metalsim/physics/mpm/implicit_mpm_pr4660mn/rheology_solver_kernels.py")
    diff = [(a, b) for a, b in zip(mc, mn) if a != b]
    assert len(mc) == len(mn) and diff == [("_LODE_SURFACE = wp.constant(1)", "_LODE_SURFACE = wp.constant(2)")]
    from metalsim.physics.mpm import solver  # noqa: PLC0415

    if solver.solver_kind() != "newton":
        assert solver.solver_class("pr4660mc").__module__.startswith("metalsim.physics.mpm.implicit_mpm_pr4660mc.")
        assert solver.solver_class("pr4660mn").__module__.startswith("metalsim.physics.mpm.implicit_mpm_pr4660mn.")


def test_mpm_config_fidelity_and_yield_surface():
    pytest.importorskip("mujoco")
    from metalsim.physics.mpm.coupled import MPMConfig

    c = MPMConfig()
    assert (c.fidelity, c.yield_surface, c.solver_variant, c.max_iterations, c.strain_basis) == \
        ("parity", "drucker_prager", "parity", 250, "P0")
    c = MPMConfig(fidelity="physical")
    assert (c.yield_surface, c.solver_variant, c.max_iterations, c.strain_basis) == \
        ("mohr_coulomb", "pr4660mc", 1000, "P1d")
    assert MPMConfig(fidelity="physical", yield_surface="matsuoka_nakai").solver_variant == "pr4660mn"
    assert MPMConfig(fidelity="physical", yield_surface="drucker_prager").solver_variant == "pr4660"
    assert MPMConfig(fidelity="physical", max_iterations=250).max_iterations == 250
    # research path (probes): an explicit variant decides the surface
    c = MPMConfig(solver_variant="pr4660mc")
    assert (c.yield_surface, c.max_iterations, c.strain_basis) == ("mohr_coulomb", 250, "P0")
    assert MPMConfig(solver_variant="pr4660").yield_surface == "drucker_prager"
    for kwargs, words in ((dict(fidelity="isaac"), "parity"), (dict(yield_surface="tresca"), "mohr_coulomb"),
                          (dict(yield_surface="mohr_coulomb"), "physical"),
                          (dict(solver_variant="pr4660mn", yield_surface="mohr_coulomb"), "matsuoka_nakai"),
                          (dict(solver_variant="nope"), "pr4660mc")):
        with pytest.raises(ValueError, match=words):
            MPMConfig(**kwargs)



# ------------------------------------------------------------------------------------------------ material conventions
# Each measurement runs the full solver in its own process: two copies of the solver package (parity and a variant)
# in one process collide in warp.fem's integrand handling (update_particle_strains, warmstart "particles"; seen with
# parity + pr4660 too, before this work; research_notes/sand_fidelity/PLAN_2026-10-14.md section 18).
def _block(variant, vel, *, eta, tau_y, dx=0.05, n_cells=10, ppc_axis=2, rho=1000.0, warmstart="auto", iters=200):
    """A free cube of material (no gravity, no colliders), velocity field ``vel(x)``, on Warp's CPU device."""
    import newton  # noqa: PLC0415

    from metalsim.physics.mpm.solver import solver_class  # noqa: PLC0415

    S = solver_class(variant)
    L = dx * n_cells
    g = (np.arange(n_cells * ppc_axis) + 0.5) * dx / ppc_axis
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    S.register_custom_attributes(builder)
    m = rho * (dx / ppc_axis) ** 3
    for x in pos:
        builder.add_particle(pos=tuple(x), vel=vel(x, L), mass=m, radius=0.5 * dx / ppc_axis,
                             custom_attributes={"mpm:young_modulus": 1.0e15, "mpm:friction": 0.0,
                                                "mpm:yield_pressure": 1.0e12, "mpm:yield_stress": tau_y,
                                                "mpm:viscosity": eta})
    model = builder.finalize(device="cpu")
    solver = S(model, S.Config(voxel_size=dx, grid_type="fixed", max_iterations=iters, tolerance=1.0e-7,
                               warmstart_mode=warmstart))
    return solver, model.state(), L


def measure_simple_shear_stress(variant, tau_y, eta, gdot):
    """Median interior sigma_xz after one step of uniform simple shear v_x = gdot (z - L/2)."""
    solver, st, L = _block(variant, lambda x, L: (gdot * (x[2] - 0.5 * L), 0.0, 0.0), eta=eta, tau_y=tau_y,
                           warmstart="particles")   # "particles": the solver writes mpm:particle_stress
    solver.step(st, st, None, None, 1.0e-3)
    sig = st.mpm.particle_stress.numpy().reshape(-1, 3, 3)
    q = st.particle_q.numpy()
    inner = np.all((q > 0.15) & (q < L - 0.15), axis=1)
    return float(np.median(sig[inner, 0, 2]))


def measure_viscous_decay(variant, eta=50.0, A=0.02, dt=1.0e-3, steps=100, rho=1000.0):
    """Decay rate of the free-block shear mode v_x = A cos(pi z / L), over eta k^2 / rho."""
    solver, st, L = _block(variant, lambda x, L: (A * np.cos(np.pi * x[2] / L), 0.0, 0.0), eta=eta, tau_y=0.0,
                           rho=rho)
    k = np.pi / L
    amps = []
    for i in range(steps + 1):
        if i:
            solver.step(st, st, None, None, dt)
        q, v = st.particle_q.numpy(), st.particle_qd.numpy()
        inner = np.all((q[:, :2] > 0.15) & (q[:, :2] < L - 0.15), axis=1)
        c = np.cos(k * q[inner, 2])
        amps.append(np.dot(c, v[inner, 0]) / np.dot(c, c))
    rate = -np.polyfit(np.arange(steps + 1) * dt, np.log(np.abs(np.array(amps))), 1)[0]
    return float(rate / (eta * k * k / rho))


def _measure(*args):
    """Run ``python tests/test_mpm_lode.py <measurement> <args...>`` in a fresh process; returns its float."""
    import subprocess  # noqa: PLC0415

    r = subprocess.run([sys.executable, __file__, *map(str, args)], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return float(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("variant", ["parity", "pr4660mc"])
def test_yield_stress_is_the_simple_shear_yield_stress(variant):
    """Uniform simple shear of a cohesive frictionless block: the stored particle stress in the interior is
    sigma_xz = -yield_stress (the solver's stress is compression-positive, i.e. minus Cauchy), whatever the viscosity
    and rate: ``yield_stress`` is tau_y. ``mpm:particle_stress`` is the stress projected onto the yield surface, so it
    never shows the viscous part (see test_viscosity_convention)."""
    for eta, gdot in ((0.0, 2.0), (10.0, 4.0)):
        sxz = _measure("shear", variant, 10.0, eta, gdot)
        np.testing.assert_allclose(sxz, -10.0, rtol=1e-3)


@pytest.mark.parametrize("variant", ["parity", "pr4660mc"])
def test_viscosity_convention(variant):
    """Decay of the free-block shear mode v_x = A cos(pi z / L) (traction-free top and bottom), friction 0, yield
    stress 0, viscosity eta = 50, rho 1000, 100 steps of 1 ms: a Newtonian fluid with shear stress tau = eta_N gamma_dot
    decays at eta_N k^2 / rho. Measured: 0.47 x eta k^2 / rho (0.45 at half the cell size, 0.48 at twice dt; the
    inviscid background is 0.6 % of it), so ``mpm:viscosity`` acts as tau = eta eps_dot = eta gamma_dot / 2: a
    Bingham plastic viscosity eta_pl is ``viscosity = 2 eta_pl``. The bounds discriminate 1/2 from 1."""
    ratio = _measure("decay", variant)
    print(f"{variant}: viscous decay rate / (eta k^2 / rho) = {ratio:.4f}")
    assert 0.40 < ratio < 0.55, ratio


if __name__ == "__main__":
    wp.config.quiet = True
    kind, variant, *rest = sys.argv[1:]
    if kind == "shear":
        print(measure_simple_shear_stress(variant, *map(float, rest)))
    elif kind == "decay":
        print(measure_viscous_decay(variant))
    else:
        raise SystemExit(f"unknown measurement {kind!r}")
