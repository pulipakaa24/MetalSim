"""Join the L4 nsys per-kernel sums with Metal kernel_profile.py JSON, per scene.

python kjoin.py L4_DIR METAL_DIR SCENE L4_FRAMES OUT_MD [--l4-name NAME]
"""
import csv, json, os, re, sys
from collections import defaultdict

l4dir, mdir, scene, l4frames, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
l4name = sys.argv[6] if len(sys.argv) > 6 else scene

CAT = [
    (r"radix|onesweep|RadixSort|sort_|segmented_sort|DeviceSegmentedSort|Histogram|scan_|DeviceScan|Scan|reduce|DeviceReduce|DeviceSelect|select_|run_length|RunLength|bitonic", "sort/scan/reduce (native)"),
    (r"memset|memcpy|copy_|array_copy|zero|fill", "copy/fill"),
]


def norm(n):
    n = n.split("(")[0].strip()
    n = re.sub(r"^void\s+", "", n)
    n = re.sub(r"_(cuda_kernel|metal)_(forward|backward)$", "", n)
    n = re.sub(r"(_[0-9a-f]{8})+$", "", n)
    n = re.sub(r"(_[0-9a-f]{8})(?=_)", "", n)
    return n


def cat(n):
    for pat, c in CAT:
        if re.search(pat, n):
            return c
    return n


def read_l4(prefix):
    f = os.path.join(l4dir, f"{prefix}_cuda_gpu_kern_sum.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    agg = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        n = cat(norm(r["Name"]))
        agg[n][0] += float(r["Total Time (ns)"]) * 1e-6 / l4frames
        agg[n][1] += float(r["Instances"]) / l4frames
    return agg


def read_metal(name):
    f = os.path.join(mdir, f"profile_{name}.json")
    if not os.path.exists(f):
        return None, None
    d = json.load(open(f))
    agg = defaultdict(lambda: [0.0, 0.0])
    for k in d["kernels"]:
        n = cat(norm(k["kernel"]))
        agg[n][0] += k["ms_per_frame"]
        agg[n][1] += k["launches_per_frame"]
    return agg, d


L = read_l4(l4name)
M, md = read_metal(scene)
keys = sorted(set(L) | set(M), key=lambda k: -(L.get(k, [0])[0] + M.get(k, [0])[0]))
lt, mt = sum(v[0] for v in L.values()), sum(v[0] for v in M.values())
ln, mn = sum(v[1] for v in L.values()), sum(v[1] for v in M.values())
calib = md.get("empty_cb_us_dim1", 0.0) if md else 0.0
lines = [f"### {scene}", "",
         f"L4 kernel sum {lt:.2f} ms/frame over {ln:.0f} launches; Metal (isolated-dispatch profile) {mt:.2f} ms/frame over "
         f"{mn:.0f} launches; Metal empty command buffer {calib:.2f} us/launch -> calibrated Metal sum "
         f"{mt - mn * calib * 1e-3:.2f} ms/frame.", "",
         "| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |",
         "|---|---|---|---|---|---|---|"]
for k in keys[:30]:
    l = L.get(k, [0.0, 0.0]); m = M.get(k, [0.0, 0.0])
    mc = max(m[0] - m[1] * calib * 1e-3, 0.0)
    r = f"{mc / l[0]:.2f}" if l[0] > 0 and m[0] > 0 else ""
    lines.append(f"| {k[:70]} | {l[0]:.3f} | {l[1]:.0f} | {m[0]:.3f} | {m[1]:.0f} | {mc:.3f} | {r} |")
rest = keys[30:]
if rest:
    lines.append(f"| ({len(rest)} more) | {sum(L.get(k, [0])[0] for k in rest):.3f} | | {sum(M.get(k, [0])[0] for k in rest):.3f} | | | |")
open(out, "a").write("\n".join(lines) + "\n\n")
print("\n".join(lines[:16]))
