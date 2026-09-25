"""Flex edge-equality row order vs MuJoCo C (FlexConstraintTest.test_constraint_parity 3x3 cloth): efc_pos compared in order and sorted. usage: row_order_repro.py [metal:0|cpu]"""
import sys, numpy as np, mujoco, warp as wp
import mujoco_warp as mjw
from mujoco_warp import test_data
dev = sys.argv[1] if len(sys.argv) > 1 else "metal:0"
wp.set_device(dev)
XMLS = {
 "rope": '<mujoco><worldbody><flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" dim="1" mass="1"><edge equality="true"/></flexcomp></worldbody></mujoco>',
 "cloth": '<mujoco><worldbody><flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1"><edge equality="true"/></flexcomp></worldbody></mujoco>',
}
for name, xml in XMLS.items():
  for nworld in (1, 2):
    mjm, mjd, m, d = test_data.fixture(xml=xml, qpos_noise=0.05, nworld=nworld)
    d.nefc.fill_(-1); d.efc.pos.fill_(wp.inf); d.efc.J.fill_(wp.inf)
    mjw.fwd_position(m, d); mjw.make_constraint(m, d)
    mujoco.mj_forward(mjm, mjd)
    nefc = mjd.nefc
    print(f"== {name} nworld={nworld} is_sparse={m.is_sparse} mj_isSparse={mujoco.mj_isSparse(mjm)} nefc C={nefc} warp={d.nefc.numpy()}")
    ref = mjd.efc_pos[:nefc]
    for w in range(nworld):
      p = d.efc.pos.numpy()[w, :nefc]
      err_unsorted = np.abs(p - ref).max()
      err_sorted = np.abs(np.sort(p) - np.sort(ref)).max()
      ids = d.efc.id.numpy()[w, :nefc]
      print(f"  w{w}: max|pos-C| in order {err_unsorted:.2e}, sorted {err_sorted:.2e}; efc_id order {ids.tolist()}; C id {mjd.efc_id[:nefc].tolist()}")
      # flexedge lengths vs C
      fl = d.flexedge_length.numpy()[w]
      print(f"       flexedge_length max err {np.abs(fl - mjd.flexedge_length).max():.2e}")
      if m.is_sparse:
        J = np.zeros((nefc, mjm.nv))
        mujoco.mju_sparse2dense(J, d.efc.J.numpy()[w, 0], d.efc.J_rownnz.numpy()[w, :nefc], d.efc.J_rowadr.numpy()[w, :nefc], d.efc.J_colind.numpy()[w, 0])
      else:
        J = d.efc.J.numpy()[w, :nefc, :mjm.nv]
      if mujoco.mj_isSparse(mjm):
        JC = np.zeros((nefc, mjm.nv)); mujoco.mju_sparse2dense(JC, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
      else:
        JC = mjd.efc_J.reshape(nefc, mjm.nv)
      order = np.argsort(p); orderC = np.argsort(ref)
      print(f"       J max err in order {np.abs(J - JC).max():.2e}; after matching rows by pos {np.abs(J[order]-JC[orderC]).max():.2e}")
