"""Cost of the MuJoCo Warp fork's sparse L'DL kernels on Metal at 4096 worlds of the G1 task model: factorization
and solve, one-world-per-thread serial vs one-world-per-SIMD-group lanes (bitwise the same results,
scripts/diagnostics/ldl_lanes_check.py). Graph-replayed, 50 launches per timing, median of 3.

usage: PYTHONPATH=<fork worktree> python scripts/diagnostics/ldl_lanes_bench.py [N=4096]"""
import sys, time
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import smooth as SM, types as T
from metalsim.learn.g1_velocity import build_g1_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
K = 50
dev = wp.get_device("metal:0")
m, _ = build_g1_model(physics_dt=0.0025)
T.M_BLOCK_DENSE_MAX = 32
mjd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, mjd, 0); mujoco.mj_forward(m, mjd)
with wp.ScopedDevice(dev):
    mw = mjw.put_model(m)
    d = mjw.put_data(m, mjd, nworld=N, nconmax=64, njmax=256)
    rng = np.random.default_rng(0)
    q = np.tile(mjd.qpos, (N, 1)); q[:, 7:] += rng.uniform(-0.5, 0.5, (N, m.nq - 7))
    d.qpos.assign(q.astype(np.float32))
    mjw.kinematics(mw, d); mjw.com_pos(mw, d); mjw.crb(mw, d); wp.synchronize_device(dev)
    M = d.M; nlev = len(mw.qLD_updates)
    L, D = wp.zeros_like(M), wp.zeros((N, m.nv), dtype=float)
    y = wp.array(rng.standard_normal((N, m.nv)).astype(np.float32), dtype=float); x = wp.zeros((N, m.nv), dtype=float)
    launches = {
        "factor serial": lambda: wp.launch(SM._factor_i_sparse_serial(nlev), dim=N, inputs=[mw.M_rownnz, mw.M_rowadr, mw.qLD_all_updates, mw.qLD_level_offsets, M], outputs=[L, D], block_dim=32),
        "factor lanes": lambda: wp.launch_tiled(SM._factor_i_sparse_lanes(nlev, mw.nM), dim=N, inputs=[mw.M_rownnz, mw.M_rowadr, mw.qLD_updates_byrow, mw.qLD_lane_pairs, mw.qLD_lane_pair_offsets, M], outputs=[L, D], block_dim=32),
        "solve serial": lambda: wp.launch(SM._solve_LD_sparse_serial(m.nv, nlev), dim=N, inputs=[mw.qLD_block_adr, L, D, mw.qLD_all_updates, mw.qLD_level_offsets, y], outputs=[x], block_dim=32),
        "solve lanes": lambda: wp.launch_tiled(SM._solve_LD_sparse_lanes(m.nv, nlev), dim=N, inputs=[mw.qLD_block_adr, L, D, mw.qLD_updates_byrow, mw.qLD_level_offsets, mw.qLD_lane_rows, mw.qLD_lane_row_offsets, y], outputs=[x], block_dim=32),
    }
    nlc = len(mw.qLD_chain_level_offsets) - 1
    launches["factor chains"] = lambda: wp.launch_tiled(SM._factor_i_sparse_chains(nlc, mw.nM), dim=N, inputs=[mw.M_rownnz, mw.M_rowadr, mw.qLD_chain_rows, mw.qLD_chain_adr, mw.qLD_chain_level_offsets, mw.qLD_updates_bysrc, mw.qLD_src_adr, M], outputs=[L, D], block_dim=32)
    launches["solve chains"] = lambda: wp.launch_tiled(SM._solve_LD_sparse_chains(m.nv, nlc), dim=N, inputs=[mw.qLD_block_adr, L, D, mw.qLD_chain_rows, mw.qLD_chain_adr, mw.qLD_chain_level_offsets, mw.qLD_updates_byrow, mw.qLD_lane_rows, mw.qLD_row_adr, mw.qLD_updates_bysrc, mw.qLD_src_adr, y], outputs=[x], block_dim=32)
    key = SM._ldl_schedule(mw)
    launches["factor unrolled"] = lambda: wp.launch_tiled(SM._factor_i_sparse_unrolled(key), dim=N, inputs=[M], outputs=[L, D], block_dim=32)
    launches["solve unrolled"] = lambda: wp.launch_tiled(SM._solve_LD_sparse_unrolled(key), dim=N, inputs=[L, D, y], outputs=[x], block_dim=32)
    for name, fn in launches.items():
        fn(); wp.synchronize_device(dev)
        with wp.ScopedCapture(device=dev) as cap:
            for _ in range(K):
                fn()
        wp.capture_launch(cap.graph); wp.synchronize_device(dev)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize_device(dev); ts.append((time.perf_counter() - t0) / K)
        print(f"{name:14s}: {sorted(ts)[1]*1e3:.3f} ms per launch of {N} worlds (G1, nv {m.nv}, nM {mw.nM}, {nlev} levels)", flush=True)
