import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "newton/_src/solvers/mujoco"
k = (root / "kernels.py").read_text()
a = """        mj_contact_worldid: wp.array[wp.int32],
        mj_efc_force: wp.array2d[float],"""
assert k.count(a) == 1
k = k.replace(a, """        mj_contact_worldid: wp.array[wp.int32],
        mj_contact_adhesion: wp.array[float],
        mj_efc_force: wp.array2d[float],""")
b = """                mj_contact_efc_address,
                mj_efc_force,
                njmax,"""
assert k.count(b) == 1
k = k.replace(b, """                mj_contact_efc_address,
                mj_contact_adhesion,
                mj_efc_force,
                njmax,""")
(root / "kernels.py").write_text(k)

s = (root / "solver_mujoco.py").read_text()
c = """                mj_contact.worldid,
                mj_data.efc.force,"""
assert s.count(c) == 1
s = s.replace(c, """                mj_contact.worldid,
                mj_contact.adhesion,
                mj_data.efc.force,""")
(root / "solver_mujoco.py").write_text(s)
print("ok")
