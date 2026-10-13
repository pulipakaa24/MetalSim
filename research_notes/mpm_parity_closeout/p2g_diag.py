"""Where Metal's first-step node mass / velocity differ from the L4's (dump_p2g.py outputs)."""
import numpy as np

W = "/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c/runs/mpm/closeout"
a = np.load(f"/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c/runs/parity/isaaclab3_mpm_closeout/p2g_mm_base/p2g.npz")
b = np.load(f"{W}/p2g_mm_metal/p2g.npz")
n = a["inv_mass"].shape[0]
pi_a, pi_b = a["vtest.space_restriction._dof_partition_indices"][:n], b["vtest.space_restriction._dof_partition_indices"][:n]
print("first n partition indices equal:", np.array_equal(pi_a, pi_b))
ia, ib = a["inv_mass"], b["inv_mass"]
d = ib.view(np.int32).astype(np.int64) - ia.view(np.int32).astype(np.int64)
u, c = np.unique(np.clip(d, -10, 10), return_counts=True)
print("inv_mass ulp diff histogram:", dict(zip(u.tolist(), c.tolist())))
# node mass = 1/inv_mass (no drag); which nodes: by number of contributing particles
off = a["vtest.space_restriction._dof_partition_element_offsets"]
nel = np.diff(off)[:n]
for k in np.unique(nel):
    s = nel == k
    print(f"  nodes touching {k} elements: {s.sum()}, differ {np.mean(d[s] != 0) * 100:.1f} %")
ijk = a["pic._domain.geometry_partition.geometry._node_ijk"]
z = ijk[:, 2]
for zz in np.unique(z)[:: max(1, len(np.unique(z)) // 12)]:
    s = z == zz
    print(f"  node k={zz}: {s.sum()} nodes, differ {np.mean(d[s] != 0) * 100:.1f} %")
