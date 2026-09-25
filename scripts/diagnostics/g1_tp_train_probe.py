"""20-iteration G1 flat training probe (MuJoCo Warp, 2.5 ms, 4096 envs, seed 0, Isaac's flat PPO config) under a
throughput configuration variant (MJW_TP_VARIANT, keys as in g1_tp_variants.py); prints the per-iteration
episode-length / return trend and the training rate, for comparison with the committed configuration.

usage: MJW_TP_VARIANT='{"m_dense_max":0}' python scripts/diagnostics/g1_tp_train_probe.py [N] [ITERS] [LOG]"""
import json, os, sys
import warp as wp
wp.config.quiet = True
import metalsim.learn.g1_velocity as g1v
from metalsim.physics.batch import BatchSimOptions

V = json.loads(os.environ.get("MJW_TP_VARIANT", "{}") or "{}")


def _opts(**kw):
    for k in ("njmax", "nconmax", "jacobian", "block_dim", "m_dense_max", "metal_register_cholesky_max"):
        if k in V:
            kw[k] = V[k]
    return BatchSimOptions(**kw)


g1v.BatchSimOptions = _opts
args = [a for a in sys.argv[1:] if not a.startswith("--")]
n = int(args[0]) if args else 4096
iters = int(args[1]) if len(args) > 1 else 20
log = args[2] if len(args) > 2 else None
print(f"variant {V or 'committed'}", flush=True)
g1v.train_g1(n, "flat", iters, seed=0, log_path=log, physics_dt=0.0025)
