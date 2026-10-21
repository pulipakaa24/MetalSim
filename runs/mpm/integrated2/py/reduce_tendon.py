"""Well-defined reduction of the _tendon_dot compiler-service failure (device-space kernel arguments).

Keeps Warp's native headers and the original _tendon_dot kernel unchanged; removes the other kernels, the adjoint
functions and every user function the failure does not need (whole functions only, never statements), then probes
which features the failure needs with well-defined edits. usage: reduce_tendon.py MODULE.metal OUTDIR"""
import os, re, subprocess, sys

src_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
MSLC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mslc.py")
PY = ".venv/bin/python"
FN = "_tendon_dot_62db264d_metal_forward"
lines = open(src_path).read().split("\n")
user0 = next(i for i, l in enumerate(lines) if re.match(r"^// /Users/.*(mujoco_warp|warp)/.*\.py:\d+$", l) and i > 1000)
starts = [i for i, l in enumerate(lines) if l.startswith("struct wp_args_")]
k0 = starts[0]
tgt = next(i for i in starts if "_tendon_dot_62db264d" in lines[i])
nxt = min([i for i in starts if i > tgt] + [len(lines)])
head = lines[:user0]
funcs = [l for l in lines[user0:k0]]
kernel = lines[tgt:nxt]
TOP = re.compile(r"^(static |WP_FORCE_INLINE|template|inline )")

def blocks(ls):
    us, i = [], 0
    while i < len(ls):
        if TOP.match(ls[i]):
            j, depth, opened = i, 0, False
            while j < len(ls):
                depth += ls[j].count("{") - ls[j].count("}")
                opened = opened or "{" in ls[j]
                j += 1
                if opened and depth == 0:
                    break
            us.append((i, j)); i = j
        else:
            i += 1
    return us

# drop adjoint functions up front (forward-only kernel)
keep = []
for a, b in blocks(funcs):
    if not re.search(r"\badj_\w+\(", funcs[a]):
        keep += funcs[a:b]
funcs = keep

def test(fs, ks=kernel, define=()):
    """define non-empty: build with constant-space arguments (the module's first line selects device ones)"""
    p = os.path.join(out, "t.metal")
    h = list(head)
    if define:
        h[0] = h[0].replace("#define WP_METAL_DEVICE_ARGS 1", "#define WP_METAL_DEVICE_ARGS 0")
    open(p, "w").write("\n".join(h + fs + ks) + "\n")
    return subprocess.run([PY, MSLC, p, FN], capture_output=True, text=True).returncode

assert test(funcs) == 2, "does not reproduce"
changed = True
while changed:
    changed = False
    for a, b in sorted(blocks(funcs), key=lambda u: u[0] - u[1]):
        cand = funcs[:a] + funcs[b:]
        if test(cand) == 2:
            funcs, changed = cand, True
            break
names = [re.search(r"(\w+)\(", funcs[a]).group(1) for a, _ in blocks(funcs)]
print("user functions kept:", names)
repro = os.path.join(out, "tendon_dot_device_args_crash.metal")
open(repro, "w").write("\n".join(head + funcs + kernel) + "\n")
print("repro:", repro, len(head), "header lines +", len(funcs) + len(kernel), "kernel/function lines")
print("device args:", test(funcs), " constant args:", test(funcs, define=["WP_METAL_CONSTANT_ARGS=1"]), "(2 = pipeline failure)")
body = "\n".join(funcs)
variants = {
    "float atomic_add -> plain add": body.replace("wp::atomic_add(var_75, var_19, var_72);", "*wp::address(var_75, var_19) += var_72; var_76 = 0;"),
    "motion_cross branch removed": body.replace("var_58 = motion_cross_0(WP_FUNC_ARG var_59, var_32);", "var_58 = var_59;"),
}
for name, b in variants.items():
    print(f"{name:34s} device: {test(b.split(chr(10)))}  constant: {test(b.split(chr(10)), define=['WP_METAL_CONSTANT_ARGS=1'])}")
kl = list(kernel)
calls = [i for i, l in enumerate(kl) if "_accumulate_jac_dot_chain_0(WP_FUNC_ARG" in l]
for c in calls:
    k2 = [l for i, l in enumerate(kl) if i != c]
    print(f"{'drop chain call at line ' + str(c):34s} device: {test(funcs, k2)}  constant: {test(funcs, k2, ['WP_METAL_CONSTANT_ARGS=1'])}")
