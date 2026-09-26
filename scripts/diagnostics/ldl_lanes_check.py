"""Check of the MuJoCo Warp fork's lane-parallel and chain-parallel sparse L'DL kernels (Metal, one world per SIMD group)
against the one-world-per-thread serial kernels, on the G1 task model (43 dofs, tree-sparse M) and the
menagerie Go2 / Panda if present: factor of M and of M - dt*D, solve with random right-hand sides, over random
states. Run with PYTHONPATH pointing at the fork worktree (both kernels are in it; MJW_METAL_LDL_LANES selects
the default path, the check launches both explicitly).

usage: python scripts/diagnostics/ldl_lanes_check.py [N=64]"""
import sys
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import smooth as SM, types as T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
dev = wp.get_device("metal:0")


def models():
    from metalsim.learn.g1_velocity import build_g1_model
    m, _ = build_g1_model(physics_dt=0.0025)
    yield "G1 task (nv 43)", m, 32
    import os
    for name, path in (("Go2 menagerie", "upstream/mujoco_menagerie/unitree_go2/scene.xml"),
                       ("humanoid (dm)", "upstream/mujoco_warp/benchmarks/humanoid/humanoid.xml")):
        if os.path.exists(path):
            spec = mujoco.MjSpec.from_file(path)
            for g in spec.geoms:
                g.margin = 0.0          # the fork rejects margins with MULTICCD / NATIVECCD (collision is not exercised here)
            yield name + " (m_dense_max 0: all sparse)", spec.compile(), 0


ok = True
for name, m, dense_max in models():
    T.M_BLOCK_DENSE_MAX = dense_max
    mjd = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, mjd, 0)
    mujoco.mj_forward(m, mjd)
    with wp.ScopedDevice(dev):
        mw = mjw.put_model(m)
        d = mjw.put_data(m, mjd, nworld=N, nconmax=64, njmax=256)
        rng = np.random.default_rng(0)
        q = np.tile(mjd.qpos, (N, 1)); q[:, 7:] += rng.uniform(-0.5, 0.5, (N, m.nq - 7)) if m.nq > 7 else 0
        d.qpos.assign(q.astype(np.float32))
        mjw.kinematics(mw, d); mjw.com_pos(mw, d); mjw.crb(mw, d)
        wp.synchronize_device(dev)
        M = d.M
        nlev = len(mw.qLD_updates)
        print(f"{name}: nM {mw.nM}, levels {nlev}, updates {len(mw.qLD_all_updates)}, lane pairs {len(mw.qLD_lane_pairs)}, "
              f"sparse dofs {int((mw.qLD_block_adr.numpy() == T.Q_LD_BLOCK_SPARSE).sum())} of {m.nv}")
        Ls, Ds = wp.zeros_like(M), wp.zeros((N, m.nv), dtype=float)
        Ll, Dl = wp.zeros_like(M), wp.zeros((N, m.nv), dtype=float)
        wp.launch(SM._factor_i_sparse_serial(nlev), dim=N, inputs=[mw.M_rownnz, mw.M_rowadr, mw.qLD_all_updates, mw.qLD_level_offsets, M],
                  outputs=[Ls, Ds], block_dim=32)
        wp.launch_tiled(SM._factor_i_sparse_lanes(nlev, mw.nM), dim=N,
                        inputs=[mw.M_rownnz, mw.M_rowadr, mw.qLD_updates_byrow, mw.qLD_lane_pairs, mw.qLD_lane_pair_offsets, M],
                        outputs=[Ll, Dl], block_dim=32)
        wp.synchronize_device(dev)
        ls, ll, ds, dl = Ls.numpy(), Ll.numpy(), Ds.numpy(), Dl.numpy()
        sparse = mw.qLD_block_adr.numpy() == T.Q_LD_BLOCK_SPARSE
        # only sparse-block rows are defined by these kernels; compare the entries of those rows
        rows = [(int(mw.M_rowadr.numpy()[i]), int(mw.M_rownnz.numpy()[i])) for i in range(m.nv) if sparse[i]]
        sel = np.concatenate([np.arange(a, a + n) for a, n in rows]) if rows else np.arange(0)
        fbit = np.array_equal(ls[:, sel], ll[:, sel]) and np.array_equal(ds[:, sparse], dl[:, sparse])
        fmax = max(np.abs(ls[:, sel] - ll[:, sel]).max() if sel.size else 0.0, np.abs(ds[:, sparse] - dl[:, sparse]).max() if sparse.any() else 0.0)
        # reference: MuJoCo C factor of world 0
        y = wp.array(rng.standard_normal((N, m.nv)).astype(np.float32), dtype=float)
        xs, xl = wp.zeros((N, m.nv), dtype=float), wp.zeros((N, m.nv), dtype=float)
        wp.launch(SM._solve_LD_sparse_serial(m.nv, nlev), dim=N, inputs=[mw.qLD_block_adr, Ls, Ds, mw.qLD_all_updates, mw.qLD_level_offsets, y],
                  outputs=[xs], block_dim=32)
        wp.launch_tiled(SM._solve_LD_sparse_lanes(m.nv, nlev), dim=N,
                        inputs=[mw.qLD_block_adr, Ls, Ds, mw.qLD_updates_byrow, mw.qLD_level_offsets, mw.qLD_lane_rows, mw.qLD_lane_row_offsets, y],
                        outputs=[xl], block_dim=32)
        wp.synchronize_device(dev)
        xs_, xl_ = xs.numpy()[:, sparse], xl.numpy()[:, sparse]
        sbit = np.array_equal(xs_, xl_); smax = np.abs(xs_ - xl_).max() if sparse.any() else 0.0
        # residual against a dense solve of world 0..3 (sparse-only models)
        res = float("nan")
        if sparse.all():
            Mf = np.zeros((4, m.nv, m.nv))
            for w in range(4):
                mujoco.mju_sym2dense(Mf[w], M.numpy()[w].astype(np.float64), m.M_rownnz, m.M_rowadr, m.M_colind)
            res = np.abs(np.einsum("wij,wj->wi", Mf, xl.numpy()[:4].astype(np.float64)) - y.numpy()[:4]).max() / np.abs(y.numpy()[:4]).max()
        print(f"  factor: {'bitwise' if fbit else f'DIFF max {fmax:.2e}'} | solve: {'bitwise' if sbit else f'DIFF max {smax:.2e}'} | solve residual vs dense M {res:.1e}")
        ok &= fbit and sbit and (not sparse.all() or res < 1e-4)
        # chain-parallel kernels: the solve (on the serial factor) must be bitwise; the factor differs by the order in
        # which chains of one level update a shared ancestor row (float noise)
        nlc = len(mw.qLD_chain_level_offsets) - 1
        Lc, Dc = wp.zeros_like(M), wp.zeros((N, m.nv), dtype=float)
        wp.launch_tiled(SM._factor_i_sparse_chains(nlc, mw.nM), dim=N,
                        inputs=[mw.M_rownnz, mw.M_rowadr, mw.qLD_chain_rows, mw.qLD_chain_adr, mw.qLD_chain_level_offsets, mw.qLD_updates_bysrc, mw.qLD_src_adr, M],
                        outputs=[Lc, Dc], block_dim=32)
        xc = wp.zeros((N, m.nv), dtype=float)
        wp.launch_tiled(SM._solve_LD_sparse_chains(m.nv, nlc), dim=N,
                        inputs=[mw.qLD_block_adr, Ls, Ds, mw.qLD_chain_rows, mw.qLD_chain_adr, mw.qLD_chain_level_offsets, mw.qLD_updates_byrow,
                                mw.qLD_lane_rows, mw.qLD_row_adr, mw.qLD_updates_bysrc, mw.qLD_src_adr, y],
                        outputs=[xc], block_dim=32)
        wp.synchronize_device(dev)
        lc, dc, xc_ = Lc.numpy(), Dc.numpy(), xc.numpy()[:, sparse]
        frel = np.abs(lc[:, sel] - ls[:, sel]).max() / np.abs(ls[:, sel]).max() if sel.size else 0.0
        drel = np.abs(dc[:, sparse] - ds[:, sparse]).max() / np.abs(ds[:, sparse]).max() if sparse.any() else 0.0
        cbit = np.array_equal(xc_, xs_)
        resc = float("nan")
        if sparse.all():
            xcf = wp.zeros((N, m.nv), dtype=float)
            wp.launch_tiled(SM._solve_LD_sparse_chains(m.nv, nlc), dim=N,
                            inputs=[mw.qLD_block_adr, Lc, Dc, mw.qLD_chain_rows, mw.qLD_chain_adr, mw.qLD_chain_level_offsets, mw.qLD_updates_byrow,
                                    mw.qLD_lane_rows, mw.qLD_row_adr, mw.qLD_updates_bysrc, mw.qLD_src_adr, y],
                            outputs=[xcf], block_dim=32)
            wp.synchronize_device(dev)
            resc = np.abs(np.einsum("wij,wj->wi", Mf, xcf.numpy()[:4].astype(np.float64)) - y.numpy()[:4]).max() / np.abs(y.numpy()[:4]).max()
        # unrolled register form (per-model native snippet): factor and solve must be bitwise the serial kernels
        key = SM._ldl_schedule(mw)
        Lu, Du = wp.zeros_like(M), wp.zeros((N, m.nv), dtype=float); xu = wp.zeros((N, m.nv), dtype=float)
        wp.launch_tiled(SM._factor_i_sparse_unrolled(key), dim=N, inputs=[M], outputs=[Lu, Du], block_dim=32)
        wp.launch_tiled(SM._solve_LD_sparse_unrolled(key), dim=N, inputs=[Ls, Ds, y], outputs=[xu], block_dim=32)
        wp.synchronize_device(dev)
        ubit = np.array_equal(Lu.numpy()[:, sel], ls[:, sel]) and np.array_equal(Du.numpy()[:, sparse], ds[:, sparse])
        usbit = np.array_equal(xu.numpy()[:, sparse], xs_)
        print(f"  unrolled: factor {'bitwise' if ubit else f'DIFF max {np.abs(Lu.numpy()[:, sel] - ls[:, sel]).max():.2e}'} | solve on the serial factor: "
              f"{'bitwise' if usbit else f'DIFF max {np.abs(xu.numpy()[:, sparse] - xs_).max():.2e}'}")
        ok &= ubit and usbit
        print(f"  chains ({nlc} levels, {len(mw.qLD_chain_adr)} chains): factor rel diff vs serial {frel:.1e} (D {drel:.1e}) | solve on the serial factor: "
              f"{'bitwise' if cbit else f'DIFF max {np.abs(xc_ - xs_).max():.2e}'} | chain factor + chain solve residual vs dense M {resc:.1e}")
        ok &= cbit and frel < 1e-5 and drel < 1e-5 and (not sparse.all() or resc < 1e-4)
print("ALL OK" if ok else "FAILURES")
