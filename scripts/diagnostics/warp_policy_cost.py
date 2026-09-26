"""Cost of the Warp rollout policy (metalsim.learn.warp_policy) at 4096 envs: the G1 flat (obs 123, 256-128-128,
act 37) and rough (obs 310, 512-256-128) actor-critic MLPs, graph-replayed, and bit-identical variants of the
layer kernel's thread mapping (every output keeps the same sequential dot product over its inputs, so the results
are bitwise the same; only which threads share which loads changes):
  A: one thread per (env, output), outputs fastest (the committed kernel: a SIMD group reads 32 weight rows, one x row)
  B: one thread per (output, env), envs fastest (a SIMD group reads one weight row, 32 x rows)
  C: one thread per (env, group of 4 outputs): x[e, i] read once per 4 weight rows
  D: one thread per (group of 4 outputs, env), envs fastest

usage: python scripts/diagnostics/warp_policy_cost.py [N=4096]"""
import sys, time
import numpy as np, torch, warp as wp
wp.config.quiet = True
from metalsim.learn import warp_policy as WPL
from metalsim.learn.ppo_warp import ActorCriticMLP

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
dev = "metal:0"


@wp.kernel
def mlp_layer_B(x: wp.array2d(dtype=float), W: wp.array2d(dtype=float), b: wp.array(dtype=float), act: int, y: wp.array2d(dtype=float)):
    j, e = wp.tid()
    n_in = W.shape[1]
    s = b[j]
    for i in range(n_in):
        s += x[e, i] * W[j, i]
    if act == 1:
        s = WPL.elu(s)
    y[e, j] = s


@wp.kernel
def mlp_layer_C(x: wp.array2d(dtype=float), W: wp.array2d(dtype=float), b: wp.array(dtype=float), act: int, y: wp.array2d(dtype=float)):
    e, g = wp.tid()
    n_in = W.shape[1]; n_out = W.shape[0]
    j0 = g * 4
    s0 = b[j0]; s1 = float(0.0); s2 = float(0.0); s3 = float(0.0)
    if j0 + 1 < n_out: s1 = b[j0 + 1]
    if j0 + 2 < n_out: s2 = b[j0 + 2]
    if j0 + 3 < n_out: s3 = b[j0 + 3]
    for i in range(n_in):
        xi = x[e, i]
        s0 += xi * W[j0, i]
        if j0 + 1 < n_out: s1 += xi * W[j0 + 1, i]
        if j0 + 2 < n_out: s2 += xi * W[j0 + 2, i]
        if j0 + 3 < n_out: s3 += xi * W[j0 + 3, i]
    if act == 1:
        s0 = WPL.elu(s0); s1 = WPL.elu(s1); s2 = WPL.elu(s2); s3 = WPL.elu(s3)
    y[e, j0] = s0
    if j0 + 1 < n_out: y[e, j0 + 1] = s1
    if j0 + 2 < n_out: y[e, j0 + 2] = s2
    if j0 + 3 < n_out: y[e, j0 + 3] = s3


@wp.kernel
def mlp_layer_D(x: wp.array2d(dtype=float), W: wp.array2d(dtype=float), b: wp.array(dtype=float), act: int, y: wp.array2d(dtype=float)):
    g, e = wp.tid()
    n_in = W.shape[1]; n_out = W.shape[0]
    j0 = g * 4
    s0 = b[j0]; s1 = float(0.0); s2 = float(0.0); s3 = float(0.0)
    if j0 + 1 < n_out: s1 = b[j0 + 1]
    if j0 + 2 < n_out: s2 = b[j0 + 2]
    if j0 + 3 < n_out: s3 = b[j0 + 3]
    for i in range(n_in):
        xi = x[e, i]
        s0 += xi * W[j0, i]
        if j0 + 1 < n_out: s1 += xi * W[j0 + 1, i]
        if j0 + 2 < n_out: s2 += xi * W[j0 + 2, i]
        if j0 + 3 < n_out: s3 += xi * W[j0 + 3, i]
    if act == 1:
        s0 = WPL.elu(s0); s1 = WPL.elu(s1); s2 = WPL.elu(s2); s3 = WPL.elu(s3)
    y[e, j0] = s0
    if j0 + 1 < n_out: y[e, j0 + 1] = s1
    if j0 + 2 < n_out: y[e, j0 + 2] = s2
    if j0 + 3 < n_out: y[e, j0 + 3] = s3


def launch(variant, x, W, b, act, y):
    n_out = W.shape[0]
    if variant == "A":
        wp.launch(WPL.mlp_layer, dim=(N, n_out), inputs=[x, W, b, act, y], device=dev)
    elif variant == "B":
        wp.launch(mlp_layer_B, dim=(n_out, N), inputs=[x, W, b, act, y], device=dev)
    elif variant == "C":
        wp.launch(mlp_layer_C, dim=(N, (n_out + 3) // 4), inputs=[x, W, b, act, y], device=dev)
    else:
        wp.launch(mlp_layer_D, dim=((n_out + 3) // 4, N), inputs=[x, W, b, act, y], device=dev)


for name, obs_dim, hidden, act_dim in (("flat", 123, (256, 128, 128), 37), ("rough", 310, (512, 256, 128), 37)):
    torch.manual_seed(0)
    net = ActorCriticMLP(obs_dim, act_dim, hidden=hidden).to("mps")
    pol = WPL.WarpMLPPolicy(net, N, obs_dim, act_dim, [-1.0] * act_dim, [1.0] * act_dim, device=dev)
    obs = wp.array(np.random.default_rng(0).standard_normal((N, obs_dim)).astype(np.float32), dtype=float, device=dev)
    ctrl = wp.zeros((N, act_dim), dtype=float, device=dev)
    # whole act() (actor + critic + sampling), as captured in the rollout graph
    with wp.ScopedCapture(device=dev) as cap:
        for _ in range(20):
            pol.act(obs, ctrl)
    wp.capture_launch(cap.graph); wp.synchronize_device(dev)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize_device(dev); ts.append((time.perf_counter() - t0) / 20)
    print(f"{name}: act() (actor {hidden} + critic + sample) {sorted(ts)[1]*1e3:.3f} ms per step at N={N}", flush=True)
    # per-variant: the actor's layers only, outputs compared bitwise with variant A
    layers = pol.actor_layers
    outs = {}
    for v in "ABCD":
        acts = [wp.zeros((N, W.shape[0]), dtype=float, device=dev) for W, _, _ in layers]
        def run():
            x = obs
            for (W, b, act), y in zip(layers, acts):
                launch(v, x, W, b, act, y); x = y
        run(); wp.synchronize_device(dev)
        with wp.ScopedCapture(device=dev) as cap:
            for _ in range(20):
                run()
        wp.capture_launch(cap.graph); wp.synchronize_device(dev)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize_device(dev); ts.append((time.perf_counter() - t0) / 20)
        outs[v] = acts[-1].numpy().copy()
        same = np.array_equal(outs[v], outs["A"])
        print(f"   actor layers, mapping {v}: {sorted(ts)[1]*1e3:.3f} ms per forward  ({'bitwise = A' if same else 'DIFFERS from A: max ' + str(np.abs(outs[v]-outs['A']).max())})", flush=True)
