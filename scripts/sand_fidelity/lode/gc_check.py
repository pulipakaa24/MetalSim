"""Check the generalized-classical (Lagioia & Panteghini 2016) Lode function against exact Mohr-Coulomb and
Matsuoka-Nakai radii, and the friction-angle error of the inner-rounded Mohr-Coulomb (beta < 1).

Plan section 17.2. GC theta_s = +30 deg is triaxial compression here (compression-positive).
"""
import numpy as np
from scipy.optimize import brentq

mu = 0.68
ph = np.arctan(mu)
s = np.sin(ph)
Mc = 6 * s / (3 - s)


def G(th, a, b, g):
    return a * np.cos(np.arccos(-b * np.sin(3 * th)) / 3 - g * np.pi / 6)


gb = 6 / np.pi * np.arctan(s / np.sqrt(3))
aMC = 1 / np.cos((gb + 1) * np.pi / 6)
K = (9 - s * s) / (1 - s * s)
A1 = (K - 3) / (K - 9)
A2 = K / (K - 9)
cands = {"sqrtA1": 2 / 3 * np.sqrt(A1) * Mc, "A1": 2 / 3 * A1 * Mc}


def M_mc(t):
    return s / (np.cos(t) - np.sin(t) * s / np.sqrt(3))


def princ(th, m):
    a = 2 / np.sqrt(3) * m
    return [1 + a * np.cos(th - np.pi / 6 + k * 2 * np.pi / 3) for k in (0, -1, 1)]


def mn(sig):
    s1, s2, s3 = sig
    return (s1 + s2 + s3) * (s1 * s2 + s2 * s3 + s3 * s1) / (s1 * s2 * s3)


def phi_of(t, m):
    sig = princ(t, m)
    return np.degrees(np.arcsin((max(sig) - min(sig)) / (max(sig) + min(sig))))


for d in [-30, -15, 0, 15, 30]:
    th = np.radians(d)
    ours = th  # +30 deg = triaxial compression in both conventions here
    line = f"th_s {d:4d} MC-GC {Mc / (np.sqrt(3) * G(th, aMC, 1, 1 - gb)):.4f} exactMC {M_mc(ours):.4f}"
    for k, a in cands.items():
        line += f"  MN[{k}] {Mc / (np.sqrt(3) * G(th, a, A2 / A1 ** 1.5, 0)):.4f}"
    hi = 0.999 / (2 / np.sqrt(3) * max(abs(np.cos(ours - np.pi / 6 + k * 2 * np.pi / 3)) for k in (0, -1, 1)))
    line += f"  exactMN {brentq(lambda m: mn(princ(ours, m)) - K, 1e-4, hi):.4f}"
    print(line)

for b in [0.99, 0.995, 0.999, 0.9999]:
    errs = []
    for th_s in np.radians(np.linspace(-30, 30, 121)):
        m = Mc / (np.sqrt(3) * G(th_s, aMC, b, 1 - gb))
        errs.append(phi_of(th_s, m) - np.degrees(ph))
    print("beta", b, "phi err range", round(min(errs), 3), round(max(errs), 3))
