"""Record an MPM reference trajectory with Newton 1.5.2's SolverImplicitMPM, on CUDA (the L4) or Metal (the Mac).

    python record_mpm.py --scene column --set aspect=2 --device cuda:0 --out ~/mpm_ref/R1_a2
    python record_mpm.py --scene example:mpm_granular --device cuda:0 --out ~/mpm_ref/R3_granular
    python record_mpm.py --scene column --device metal:0 --out runs/mpm/metal/R1_a2       # Mac, .venv-newton152

Both sides run the same two files next to this one (on the L4 machine) or from metalsim.physics.mpm (on the Mac):
``scenes.py`` (scene construction) and ``newton_compat.py`` (float32 accumulation; the CUDA-graph stopping rule run
from the host, with per-solve iteration/residual logging). Defaults per device:
    cuda:  --accumulate native  --loop cuda   (Isaac's arithmetic; host-run graph semantics so iterations are logged)
    metal: --accumulate float32 --loop cuda
``--loop native`` on CUDA runs Newton's own conditional graph; the stock-vs-host-loop recording pair checks that the
host loop changes nothing (docs/research/mpm_2026-10-09.md, R0).

Writes to --out: init.npz (initial particle state), traj.npz (float32 pos/vel every --stride frames, body_q if
any), solves.npz (iterations, res_l2, res_linf per solve), meta.json (versions, device, settings, scene spec,
per-frame wall times, input hash).
"""
import argparse, hashlib, json, os, platform, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--accumulate", default=None, choices=["float32", "native"])
ap.add_argument("--loop", default="cuda", choices=["cuda", "native"])
ap.add_argument("--stride", type=int, default=1, help="record every N frames")
ap.add_argument("--seconds", type=float, default=None)
ap.add_argument("--set", action="append", default=[], help="scene parameter key=json_value (repeatable)")
ap.add_argument("--chunk-graph", action="store_true", help="Metal: replay the 5-iteration chunk as a graph")
ap.add_argument("--gpu-exit", action="store_true", help="Metal (with --chunk-graph): decide the solver stop on the GPU")
ap.add_argument("--fused", action="store_true", help="Metal: one-dispatch Gauss-Seidel solve (metalsim.physics.mpm.fused_gs)")
ap.add_argument("--no-log", action="store_true", help="do not log per-solve residuals (timing runs)")
ap.add_argument("--cuda-nofma", action="store_true", help="CUDA: compile every kernel with fuse_fp off (tie floor)")
ap.add_argument("--cuda-cubin", action="store_true", help="CUDA: assemble with the toolkit's ptxas (cubin), not the driver's")
ap.add_argument("--cuda-sah", action="store_true", help="CUDA: build mesh BVHs with SAH instead of LBVH (tie floor)")
ap.add_argument("--precise-cuda", action="store_true",
                help="CUDA: compile every kernel without fast_math (exact / and sqrt), into a separate kernel cache; "
                     "measures how far CUDA's own approximate math moves a trajectory")
ap.add_argument("--jitter-ulp", type=int, default=0,
                help="perturb every initial particle coordinate by k float32 ulps, k uniform in [-N, N] (floor ensemble; "
                     "init.npz / init_md5 keep the unperturbed state so runs stay comparable); 0 = off")
ap.add_argument("--seed", type=int, default=0, help="RNG seed for --jitter-ulp")
ap.add_argument("--jitter-body-ulp", type=int, default=0,
                help="perturb every initial rigid joint coordinate (state joint_q) by k float32 ulps, k uniform in "
                     "[-N, N] (rigid-side floor ensemble; init.npz / init_md5 keep the particle state); 0 = off")
ap.add_argument("--metal-fast-math", choices=["off", "cuda", "relaxed"],
                help="Metal form of fast_math modules (newton_compat.install(metal_fast_math=...)); default: install's")
ap.add_argument("--no-fast-math", action="store_true",
                help="compile every module with fast_math off (any device; Metal: the integrated fork honours fast_math)")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
_variant = "_".join(n for n, on in (("precise", args.precise_cuda), ("nofma", args.cuda_nofma), ("cubin", args.cuda_cubin)) if on)
if _variant:
    wp.config.kernel_cache_dir = os.path.expanduser(f"~/.cache/warp_{_variant}")
if args.cuda_cubin:
    wp.config.cuda_output = "cubin"
wp.init()
if args.no_fast_math:
    from warp._src.context import Module as _Module

    _resolve = _Module.resolve_options

    def _resolve_no_fast_math(self, *a, **kw):
        o = _resolve(self, *a, **kw)
        o["fast_math"] = False
        return o

    _Module.resolve_options = _resolve_no_fast_math
if args.cuda_sah:
    _mesh_init = wp.Mesh.__init__

    def _mesh_init_sah(self, *a, **kw):
        kw.setdefault("bvh_constructor", "sah")
        _mesh_init(self, *a, **kw)

    wp.Mesh.__init__ = _mesh_init_sah
if args.precise_cuda or args.cuda_nofma:
    from warp._src.context import runtime

    _compile = runtime.core.wp_cuda_compile_program

    def _compile_precise(*a):
        a = list(a)
        # Warp 1.16 argument order: ..., verbose, verify_fp, fast_math, fuse_fp, ...
        if args.precise_cuda:
            a[11] = False
        if args.cuda_nofma:
            a[12] = False
        return _compile(*a)

    runtime.core.wp_cuda_compile_program = _compile_precise
device = args.device or ("cuda:0" if wp.is_cuda_available() else "metal:0")
dev = wp.get_device(device)
accumulate = args.accumulate or ("native" if dev.is_cuda else "float32")

here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(here, "newton_compat.py")):
    sys.path.insert(0, here)
    import newton_compat, scenes  # noqa: E401
else:
    from metalsim.physics.mpm import newton_compat, scenes

import newton

_install_kw = {"gpu_exit": True} if args.gpu_exit else {}  # older newton_compat copies (on the L4 machine) lack the argument
if args.metal_fast_math:
    _install_kw["metal_fast_math"] = args.metal_fast_math
newton_compat.install(accumulate=accumulate, loop=args.loop, log=not args.no_log, chunk_graph=args.chunk_graph,
                      fused=args.fused, **_install_kw)

kw = {}
for s in args.set:
    k, v = s.split("=", 1)
    kw[k] = json.loads(v)
if args.seconds is not None:
    kw["seconds"] = args.seconds

os.makedirs(args.out, exist_ok=True)
with wp.ScopedDevice(device):
    sc = scenes.make_scene(args.scene, **kw)
    st = sc.state
    q0 = st.particle_q.numpy().copy()
    qd0 = st.particle_qd.numpy().copy()
    m = sc.state  # noqa: F841
    model = sc.model
    pm = model.particle_mass.numpy() if model.particle_mass is not None and model.particle_count else np.zeros(0)
    pr = model.particle_radius.numpy() if model.particle_radius is not None and model.particle_count else np.zeros(0)
    np.savez_compressed(os.path.join(args.out, "init.npz"), q=q0, qd=qd0, mass=pm, radius=pr)
    init_hash = hashlib.md5(q0.tobytes() + qd0.tobytes()).hexdigest()
    jitter = None
    if args.jitter_ulp:
        rng = np.random.default_rng(args.seed)
        q32 = q0.astype(np.float32)
        k = rng.integers(-args.jitter_ulp, args.jitter_ulp + 1, size=q32.shape).astype(np.float32)
        qj = (q32 + k * np.spacing(np.abs(q32))).astype(np.float32)
        st.particle_q.assign(qj.astype(q0.dtype))
        jitter = {"ulp": args.jitter_ulp, "seed": args.seed, "coords_changed": int((qj != q32).sum()),
                  "max_abs_m": float(np.abs(qj.astype(np.float64) - q32).max()),
                  "perturbed_md5": hashlib.md5(qj.astype(q0.dtype).tobytes() + qd0.tobytes()).hexdigest()}

    if args.jitter_body_ulp:
        bs0 = sc.body_state
        rng_b = np.random.default_rng(1000 + args.seed)
        jq = bs0.joint_q.numpy().astype(np.float32)
        kb = rng_b.integers(-args.jitter_body_ulp, args.jitter_body_ulp + 1, size=jq.shape).astype(np.float32)
        jqj = (jq + kb * np.spacing(np.abs(jq))).astype(np.float32)
        bs0.joint_q.assign(jqj)
        # body poses follow the perturbed joint coordinates (Newton's forward kinematics)
        newton.eval_fk(sc.model, bs0.joint_q, bs0.joint_qd, bs0)
        jitter = dict(jitter or {}, body_ulp=args.jitter_body_ulp, body_coords_changed=int((jqj != jq).sum()))
    frames = sc.frames
    pos, vel, body, fidx, wall = [], [], [], [], []
    extras = {}
    solves_per_frame = []
    for f in range(frames):
        n_before = len(newton_compat.SOLVE_LOG)
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        sc.step()
        wp.synchronize_device(device)
        wall.append(time.perf_counter() - t0)
        solves_per_frame.append(len(newton_compat.SOLVE_LOG) - n_before)
        if hasattr(sc, "extra"):
            for k, v in sc.extra().items():
                extras.setdefault(k, []).append(v)
        if (f + 1) % args.stride == 0 or f == frames - 1:
            s = sc.state
            pos.append(s.particle_q.numpy().astype(np.float32))
            vel.append(s.particle_qd.numpy().astype(np.float32))
            bs = sc.body_state
            if bs is not None and bs.body_q is not None:
                body.append(bs.body_q.numpy().astype(np.float32))
            fidx.append(f + 1)
        if not np.isfinite(pos[-1]).all() if pos else False:
            print(f"non-finite positions at frame {f + 1}; stopping")
            break
        if (f + 1) % max(1, frames // 20) == 0:
            print(f"frame {f + 1}/{frames} wall {wall[-1] * 1e3:.1f} ms", flush=True)

    traj = {"pos": np.stack(pos), "vel": np.stack(vel), "frame": np.array(fidx, dtype=np.int32)}
    if body:
        traj["body_q"] = np.stack(body)
    for k, v in extras.items():
        traj["x_" + k] = np.stack(v)  # every frame, not strided
    np.savez_compressed(os.path.join(args.out, "traj.npz"), **traj)
    log = newton_compat.drain_log()
    log["solves_per_frame"] = np.array(solves_per_frame, dtype=np.int32)
    np.savez_compressed(os.path.join(args.out, "solves.npz"), **log)

    meta = {
        "scene": args.scene, "scene_kwargs": kw, "info": sc.info, "device": device, "device_name": dev.name,
        "compat": newton_compat.settings(), "stride": args.stride, "frames": frames, "frame_dt": sc.frame_dt,
        "init_md5": init_hash, "jitter": jitter, "no_fast_math": args.no_fast_math, "particle_count": int(q0.shape[0]),
        "wall_s": wall, "wall_total_s": float(np.sum(wall)),
        "iterations_mean": float(log["iterations"].mean()) if log["iterations"].size else None,
        "iterations_max": int(log["iterations"].max()) if log["iterations"].size else None,
        "res_l2_max": float(log["res_l2"].max()) if log["res_l2"].size else None,
        "versions": {"newton": newton.__version__, "warp": wp.__version__, "numpy": np.__version__,
                     "python": platform.python_version(), "platform": platform.platform()},
        "argv": sys.argv,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
print(f"done: {frames} frames, {q0.shape[0]} particles, {meta['wall_total_s']:.2f} s, "
      f"iters mean {meta['iterations_mean']} max {meta['iterations_max']}, init {init_hash}")
