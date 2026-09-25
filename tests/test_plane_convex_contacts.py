"""Plane-mesh contacts in MuJoCo Warp (fork, branch metalsim) equal MuJoCo C's mjc_PlaneConvex:
support vertex + the adjacent face most anti-aligned with the plane, pruned to its max-area quad,
vertices within the margin and below the mesh centre. Before the fix MuJoCo Warp kept only vertices
within 1 mm of the deepest one, so a tilted foot got 2 contacts where C finds 4 (61 of 691 G1
landing steps had a different contact set, tests/test_sensors_contact.py), and it ignored the margin."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.physics.batch import BatchSim, BatchSimOptions

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

CUBE = " ".join(f"{x} {y} {z}" for x in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (-0.02, 0.02))
NGON = " ".join(f"{0.08 * np.cos(a):.6f} {0.08 * np.sin(a):.6f} {z}" for a in np.linspace(0, 2 * np.pi, 12, endpoint=False) for z in (-0.03, 0.03))


def _xml(mesh, margin=0.0):
    return f"""<mujoco><option gravity="0 0 0"/>
  <asset><mesh name="m" vertex="{mesh}"/></asset>
  <worldbody><geom name="floor" type="plane" size="2 2 0.1" margin="{margin}"/>
    <body name="b" pos="0 0 0.1"><freejoint/><geom name="g" type="mesh" mesh="m" margin="{margin}"/></body>
  </worldbody></mujoco>"""


def _contacts(d):
    c = [(d.contact[i].dist, *d.contact[i].pos) for i in range(d.ncon)]
    return np.array(sorted(c, key=lambda r: (round(r[1], 4), round(r[2], 4)))).reshape(-1, 4)


def _compare(xml, poses):
    m = mujoco.MjModel.from_xml_string(xml)
    sim = BatchSim(m, len(poses), options=BatchSimOptions(njmax=64))
    q = np.tile(m.qpos0, (len(poses), 1)); q[:] = poses
    sim.set_state(q.astype(np.float32)); sim.synchronize()
    out = []
    for w, qp in enumerate(poses):
        dc = mujoco.MjData(m); dc.qpos[:] = qp; mujoco.mj_forward(m, dc)
        dw = sim.get_world(w)
        a, b = _contacts(dc), _contacts(dw)
        assert a.shape == b.shape, (w, a, b)
        assert np.allclose(a, b, atol=2e-5), (w, a, b)
        out.append(len(a))
    return out


def _pose(z, angle_deg, axis):
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), np.deg2rad(angle_deg))
    return np.concatenate([[0, 0, z], q])


def test_tilted_cube_gets_c_contact_set():
    """Cube mesh tilted 3 deg (edges 5 mm apart in depth, beyond the old 1 mm band): 4 contacts as C."""
    poses = [_pose(0.012, 3.0, [0, 1, 0]), _pose(0.012, -3.0, [0, 1, 0]), _pose(0.013, 2.0, [1, 1, 0]),
             _pose(0.0195, 0.0, [0, 0, 1]), _pose(0.05, 10.0, [1, 0, 0])]
    n = _compare(_xml(CUBE), poses)
    assert n[:3] == [4, 4, 4] and n[4] == 0, n


def test_speculative_margin_contacts():
    """Mesh 5 mm above the plane with 1 cm margin (0.5 cm per geom, summed): C reports the contacts at
    positive distance; the old MuJoCo Warp returned none."""
    poses = [_pose(0.025, 0.0, [0, 0, 1]), _pose(0.026, 1.0, [0, 1, 0])]
    n = _compare(_xml(CUBE, margin=0.005), poses)
    assert n == [4, 4], n


def test_polygon_face_pruned_to_max_area_quad():
    """12-gon prism resting on its cap: C prunes the face to its maximum-area quadrilateral (hull4f)."""
    poses = [_pose(0.029, 0.0, [0, 0, 1]), _pose(0.03, 1.5, [1, 0.3, 0])]
    n = _compare(_xml(NGON), poses)
    assert n[0] == 4, n
