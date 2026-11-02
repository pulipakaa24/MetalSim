"""Joint-motion smoothness from the state at every physics step (not from commands).

``joint_motion(q, qd, dt)``: q, qd (episodes, steps, joints) sampled every ``dt`` (5 ms), NaN after an episode's end.
Per joint, pooled over all episodes' steps:

* ``rms_acc``  rad/s^2: RMS of the finite-difference acceleration (qd[k+1] - qd[k]) / dt;
* ``rms_jerk`` rad/s^3: RMS of the finite-difference jerk of that acceleration;
* ``reversals_per_s``: sign changes of the joint velocity per second, with hysteresis: a reversal counts only between
  stretches moving faster than ``V_HYST`` = 0.05 rad/s in opposite directions (a target overshoot or a ring shows
  as one or more reversals; slow drift through zero does not);
* ``osc_rms`` rad/s and ``osc_frac``: the joint velocity above ``F_BAND`` = 5 Hz (one-sided FFT per episode, the
  episode's mean removed, Hann window not used so that the energy is Parseval-exact): its RMS and its share of the
  velocity variance. The task's intended motion (reach, descend, close, lift over 1-6 s, 20 ms commands) lies below a
  few Hz; a servo that rings at 10-25 Hz or a policy that flips its command every step puts energy above 5 Hz.
  (inference: the band edge is a judgement, not a published value; the 2-25 Hz split is reported too).
"""
import numpy as np

V_HYST = 0.05
F_BAND = 5.0


def _reversals(v):
    s = np.where(v > V_HYST, 1, np.where(v < -V_HYST, -1, 0))
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


def joint_motion(q, qd, dt, f_band=F_BAND):
    q, qd = np.asarray(q, float), np.asarray(qd, float)
    E, T, J = qd.shape
    acc_sq, jerk_sq, n_acc, n_jerk = np.zeros(J), np.zeros(J), np.zeros(J), np.zeros(J)
    rev, secs = np.zeros(J), 0.0
    hi_e, tot_e, mid_e = np.zeros(J), np.zeros(J), np.zeros(J)
    peak_v, peak_a = np.zeros(J), np.zeros(J)
    for e in range(E):
        ok = ~np.isnan(qd[e, :, 0])
        n = int(ok.sum())
        if n < 8:
            continue
        v = qd[e, :n]
        a = np.diff(v, axis=0) / dt
        jk = np.diff(a, axis=0) / dt
        acc_sq += (a ** 2).sum(0); n_acc += len(a)
        jerk_sq += (jk ** 2).sum(0); n_jerk += len(jk)
        peak_v = np.maximum(peak_v, np.abs(v).max(0)); peak_a = np.maximum(peak_a, np.abs(a).max(0))
        secs += n * dt
        for j in range(J):
            rev[j] += _reversals(v[:, j])
        x = v - v.mean(0)
        P = np.abs(np.fft.rfft(x, axis=0)) ** 2
        P[1:] *= 2
        if n % 2 == 0:
            P[-1] /= 2
        f = np.fft.rfftfreq(n, dt)
        tot_e += P.sum(0) / n ** 2
        hi_e += P[f > f_band].sum(0) / n ** 2
        mid_e += P[f > 2.0].sum(0) / n ** 2
    r = lambda x: [round(float(v), 4) for v in x]
    return dict(rms_acc=r(np.sqrt(acc_sq / np.maximum(n_acc, 1))), rms_jerk=r(np.sqrt(jerk_sq / np.maximum(n_jerk, 1))),
                reversals_per_s=r(rev / max(secs, 1e-9)), osc_rms=r(np.sqrt(hi_e / max(E, 1))),
                osc_frac=r(hi_e / np.maximum(tot_e, 1e-12)), above2hz_frac=r(mid_e / np.maximum(tot_e, 1e-12)),
                peak_speed=r(peak_v), peak_acc=r(peak_a), seconds=round(secs, 2), f_band_hz=f_band, v_hyst=V_HYST)
