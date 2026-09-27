"""MuJoCo C (float64) reference implementation of the speculative-contact rule (2026-09-27), for oracle checks of the MuJoCo
Warp prototype (no MuJoCo C equivalent exists, so the rule is built on top of MuJoCo C's own pipeline):

  mj_fwdPosition on a copy of the model whose gap geoms carry margin = gap (so C detects the gap-zone contacts and makes
  rows for them), mj_fwdVelocity, then every contact row is re-parameterised from the true distance (margin 0):
    pos = dist; imp = solimp(|pos|), B = 2 / (dmax tau), K = 1 / (dmax^2 tau^2 zeta^2), tau = max(solref[0], 2 h) (refsafe);
    R_normal = (1 - imp) / imp * diagApprox, elliptic friction rows R_j = R_normal / impratio * (mu_0 / mu_j)^2, D = 1 / R;
    normal row: pos <= 0: aref = -B v - K imp pos (MuJoCo);  pos > 0: imp = solimp[0], aref = -(v + pos / h) / h (speculative);
    friction rows: aref = -B v, R from imp(|pos|) (fricimp "pos", prototype) or from the normal row's d0 (fricimp "d0");
    friction gate (mode "live"): friction rows of a speculative contact with pos + h v_n >= 0 get J = 0, v = 0, aref = 0
    (mode "none": all speculative friction rows; mode "cone": none);
  then mj_fwdActuation, mj_fwdAcceleration, mj_fwdConstraint (Newton, ls_iterations as set) and the model's integrator.
With gap = 0 everywhere the re-parameterisation reproduces MuJoCo's own rows (checked by selftest()).
"""
import numpy as np, mujoco

class SpecC:
    def __init__(self, m, mode="cone", speculative=True, fricimp="pos"):
        self.m0 = m                                   # the model as MuJoCo Warp sees it (margin, gap)
        import copy
        self.mc = copy.copy(m)
        self.true_margin = m.geom_margin.copy()
        if speculative:
            self.mc.geom_margin[:] = m.geom_margin + m.geom_gap
        self.mc.geom_gap[:] = 0.0
        self.mc.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        # MuJoCo C's island solver gathers efc_D / efc_R into per-island arrays inside mj_fwdPosition, before the rows can be
        # re-parameterised; the monolithic solver reads efc_* directly (MuJoCo Warp has no islands either)
        self.mc.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
        self.mode = mode; self.spec = speculative; self.fricimp = fricimp

    def _imp(self, solimp, pos):
        d0, dw, width, mid, power = solimp
        d0 = min(max(d0, 1e-4), 0.9999); dw = min(max(dw, 1e-4), 0.9999)
        x = abs(pos) / width if width > 1e-15 else 1.0
        if d0 == dw or width <= 1e-15: return 0.5 * (d0 + dw)
        if x >= 1: return dw
        if x <= 0: return d0
        if power == 1: y = x
        elif x <= mid: y = (1.0 / mid ** (power - 1)) * x ** power
        else: y = 1 - (1.0 / (1 - mid) ** (power - 1)) * (1 - x) ** power
        return d0 + y * (dw - d0)

    def step(self, d):
        m = self.mc; h = m.opt.timestep
        mujoco.mj_fwdPosition(m, d); mujoco.mj_fwdVelocity(m, d)
        if self.spec:
            nv = m.nv; J = d.efc_J.reshape(-1, nv) if d.nefc else None
            refsafe = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
            for i in range(d.ncon):
                c = d.contact[i]; a = c.efc_address
                if a < 0: continue
                g1, g2 = c.geom
                margin = max(self.true_margin[g1], self.true_margin[g2])      # MuJoCo: pair margin = max of the geoms'
                pos = c.dist - margin
                tau, zeta = c.solref; dmax = min(max(c.solimp[1], 1e-4), 0.9999)
                if refsafe: tau = max(tau, 2 * h)
                B = 2.0 / (dmax * tau); K = 1.0 / (dmax * dmax * tau * tau * zeta * zeta)
                imp = self._imp(c.solimp, pos)
                impC = self._imp(c.solimp, c.dist - c.includemargin)            # what C used for this row
                diag = d.efc_R[a] * impC / (1 - impC)                           # C's diagApprox of the normal row
                dim = c.dim if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else 1
                Rn = (1 - imp) / imp * diag
                vn = d.efc_vel[a]
                impn = min(max(c.solimp[0], 1e-4), 0.9999); Rn_s = (1 - impn) / impn * diag
                # (MuJoCo C's solver re-derives R from efc_KBIP's impedance, so K, B, imp are patched there too)
                if pos > 0:
                    d.efc_R[a] = Rn_s; d.efc_D[a] = 1 / Rn_s; d.efc_aref[a] = -(vn + pos / h) / h; d.efc_KBIP[a] = (0.0, 0.0, impn, 0.0)
                else:
                    d.efc_R[a] = Rn; d.efc_D[a] = 1 / Rn; d.efc_aref[a] = -B * vn - K * imp * pos; d.efc_KBIP[a] = (K, B, imp, 0.0)
                d.efc_pos[a] = pos
                Rf = Rn_s if (pos > 0 and self.fricimp == "d0") else Rn
                for j in range(1, dim):
                    Rj = Rf / m.opt.impratio * (c.friction[0] / c.friction[j - 1]) ** 2
                    d.efc_R[a + j] = Rj; d.efc_D[a + j] = 1 / Rj; d.efc_aref[a + j] = -B * d.efc_vel[a + j]
                    d.efc_KBIP[a + j] = (0.0, B, impn if (pos > 0 and self.fricimp == "d0") else imp, 0.0)
                    inert = pos > 0 and (self.mode == "none" or (self.mode == "live" and pos + h * vn >= 0))
                    if inert:
                        J[a + j] = 0.0; d.efc_vel[a + j] = 0.0; d.efc_aref[a + j] = 0.0
        mujoco.mj_fwdActuation(m, d); mujoco.mj_fwdAcceleration(m, d); mujoco.mj_fwdConstraint(m, d)
        if m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER: mujoco.mj_Euler(m, d)
        elif m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST or m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT: mujoco.mj_implicit(m, d)
        else: raise ValueError("integrator")


def selftest(m, d0, S=200):
    """re-parameterised pipeline with gap removed == mj_step (max |dq|)"""
    import copy
    m2 = copy.copy(m); m2.geom_gap[:] = 0; m2.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE; m2.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    a = copy.copy(d0); b = copy.copy(d0); o = SpecC(m2, "cone", True); worst = 0.0
    for s in range(S):
        mujoco.mj_step(m2, a); o.step(b); worst = max(worst, np.abs(a.qpos - b.qpos).max())
    return worst
