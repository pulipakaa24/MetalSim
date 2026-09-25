"""Isaac Lab 3.0 PhysX-backend deformable recording vs MetalSim's backends (flex, XPBD) on the same scenes, with the
3.0 defaults mapped where a physical mapping exists (PHYS) and the rest fitted. Metrics: il3_analysis (identical code for
the recording and for ours). Writes runs/parity3/isaac/deformable/il3_physx_ours.json."""
import warp as wp
wp.set_device("cpu")   # CPU device only: no GPU work, no queue needed
import sys, os, json, copy, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physx_protocol as pp, il3_analysis as ia

REC = "runs/parity3/isaac/deformable/deformable_il3/isaacsim_physx"
m3 = json.load(open(os.path.join(REC, "meta.json")))
ref = ia.analyse(REC)
cm = m3["scenes"]["cloth"]["material"]; vm = m3["scenes"]["rod_n1"]["material"]
E, NU, RHO, BETA = vm["youngs_modulus"], vm["poissons_ratio"], vm["density"], vm["elasticity_damping"]
REST = 0.02      # PhysX 3.0 deformable rest offset as recorded (cloth and cube rest at z = 0.020 on the ground)

# a meta in the 5.1 protocol format with 3.0 values (PHYS)
meta = copy.deepcopy(json.load(open("runs/deformable/isaac51/meta.json")))
c = meta["scenes"]["cloth"]
c.update(verts_per_side=33, size_m=1.0, z0=0.5, particle_rest_offset=min(REST, 0.45 / 32),   # flexcomp: radius < spacing/2
         mass=cm["density"] * cm["surface_thickness"] * 1.0,
         particle_mass=cm["density"] * cm["surface_thickness"] / 1089, pbd_friction=0.5 * (cm["dynamic_friction"] + 0.5),
         spring_stretch_stiffness=cm["youngs_modulus"] * cm["surface_thickness"], spring_shear_stiffness=cm["youngs_modulus"] * cm["surface_thickness"],
         spring_bend_stiffness=0.0, spring_damping=cm["elasticity_damping"] * cm["youngs_modulus"] * cm["surface_thickness"])
meta["ground"]["dynamic_friction"] = 0.5
r = meta["scenes"]["rope"]; r["material"].update(youngs_modulus=E, poissons_ratio=NU, density=RHO, elasticity_damping=BETA, dynamic_friction=0.5)
r["mass"] = RHO * 0.5 * 0.02 * 0.02
q = meta["scenes"]["cube"]; q["material"].update(youngs_modulus=E, poissons_ratio=NU, density=RHO, elasticity_damping=BETA, dynamic_friction=0.5)
q["mass"] = RHO * 0.2 ** 3

out = {"physx": {"cloth": ref["cloth"], "rope": ref["rope"], "cube": ref["cube"]}, "ours": {}}


def rec(label, obj, pos, vel, pin=None):
    if obj == "cloth":
        mm = ia.cloth(pos, vel)
    elif obj == "rope":
        mm = ia.rope(pos, pin)
    else:
        mm = ia.cube(pos, vel)
    out["ours"][label] = mm
    print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items()}, flush=True)


def flex_rope_pin(pos):
    return np.abs(pos[0, :, 0] - pos[0, :, 0].min()) < 1e-6


ONLY = os.environ.get('ONLY', 'rope,cube,cloth').split(',')
# rope: XPBD physical preset (thin rod), flex FEM rod (the 11x2x2 mesh = PhysX n1), implicit damping at the PhysX value
for nseg in ((10, 20) if 'rope' in ONLY else ()):
    pos, vel, info = pp.run_xpbd("rope", meta, {"rope_segments": nseg}, device="cpu")
    pin = np.zeros(pos.shape[1], bool); pin[1] = True
    rec(f"rope xpbd physical {nseg} seg", "rope", pos, vel, pin)
for dt in ((2.5e-4,) if 'rope' in ONLY else ()):
    try:
        pos, vel, info = pp.run_flex("rope", meta, {"elastic_damping": BETA, "implicit_damping": True, "dt": dt}, device="cpu")
        rec(f"rope flex 11x2x2 implicit beta {BETA} dt {dt}", "rope", pos, vel, flex_rope_pin(pos))
    except Exception as e:  # noqa: BLE001
        print("rope flex failed", repr(e)[:200])

# cube: flex, radius = PhysX rest offset (PHYS), implicit damping PhysX value and fitted, stiff direct contact
for P in (({"elastic_damping": BETA}, {"elastic_damping": 0.02}, {"elastic_damping": 0.05}) if "cube" in ONLY else ()):
    Q = {"implicit_damping": True, "contact_solref": "-1e6 -300", "cube_radius": REST, "dt": 2.5e-4, **P}
    pos, vel, info = pp.run_flex("cube", meta, Q, device="cpu")
    rec(f"cube flex implicit {P['elastic_damping']}", "cube", pos, vel)

# cloth: XPBD with PhysX FEM-membrane-equivalent springs (k = E t, no bending), flex StVK membrane (elastic2d stretch)
for P in ({}, {"damping": 0.5}):
    pos, vel, info = pp.run_xpbd("cloth", meta, {"stretch_compliance": 1 / c["spring_stretch_stiffness"], "bend_compliance": 1e9,
                                                "stretch_damping": c["spring_damping"], "bend_damping": 0.0, "substeps": 16, **P}, device="cpu")
    rec(f"cloth xpbd k=Et {P}", "cloth", pos, vel)
try:
    pos, vel, info = pp.run_flex("cloth", meta, {"young": cm["youngs_modulus"], "poisson": cm["poissons_ratio"], "thickness": cm["surface_thickness"],
                                                "elastic_damping": cm["elasticity_damping"], "implicit_damping": True, "contact_solref": "0.01 1", "dt": 5e-4},
                                 device="cpu")
    rec("cloth flex StVK membrane implicit", "cloth", pos, vel)
except Exception as e:  # noqa: BLE001
    print("cloth flex failed", repr(e)[:200])

fn = "runs/parity3/isaac/deformable/il3_physx_ours.json"
if os.path.exists(fn):
    old = json.load(open(fn)); old["ours"].update(out["ours"]); out["ours"] = old["ours"]
json.dump(out, open(fn, "w"), indent=1, default=float)
