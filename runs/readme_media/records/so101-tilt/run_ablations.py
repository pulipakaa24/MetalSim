"""so101-tilt: one-at-a-time ablations (100 seeds each, 7v4_bam unless set, CPU MuJoCo 3.14), 8 in parallel.
    .venv/bin/python runs/readme_media/scratch/so101-tilt/run_ablations.py [name ...]     (from the MetalSim root)"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys

P = "runs/readme_media/scratch/so101-tilt"
ABL = {
    "base": {},
    "servo_7v4_real": {"servo": "7v4_real"},
    "condim3": {"grip": {"condim": 3}},
    "condim4": {"grip": {"condim": 4}},
    "tors0p03": {"grip": {"friction": [1, 0.03, 0.0005]}},
    "tors1e-5": {"grip": {"friction": [1, 1e-5, 0.0005]}},
    "mu0p5": {"grip": {"friction": [0.5, 0.005, 0.0005]}},
    "impratio1": {"opt": {"impratio": 1}},
    "impratio100": {"opt": {"impratio": 100}},
    "pyramidal": {"opt": {"cone": 0}},
    "noslip10": {"opt": {"noslip_iterations": 10}},
    "noslip2": {"opt": {"noslip_iterations": 2}},
    "solref0p02": {"grip": {"solref": [0.02, 1]}},
    "dz0": {"grasp_dz": 0.0},
    "dz-0p008": {"grasp_dz": -0.008},
    "dz0p016": {"grasp_dz": 0.016},
    "pitch_hold": {"pitch_hold": True},
    "pitch_hold_real": {"pitch_hold": True, "servo": "7v4_real"},
    "box_mu0p5": {"box_mu": 0.5},
    "noslip10_pitch_hold": {"opt": {"noslip_iterations": 10}, "pitch_hold": True},
}
names = sys.argv[1:] or [n for n in ABL if not os.path.exists(f"{P}/out/{n}.json")]


def run(n):
    with open(f"{P}/logs/{n}.log", "w") as log:
        rc = subprocess.call([".venv/bin/python", f"{P}/tilt_probe.py", "100", json.dumps(ABL[n]), f"out/{n}.json"],
                             stdout=log, stderr=subprocess.STDOUT)
    return n, rc


with cf.ThreadPoolExecutor(8) as ex:
    for n, rc in ex.map(run, names):
        print(n, "rc", rc, flush=True)
