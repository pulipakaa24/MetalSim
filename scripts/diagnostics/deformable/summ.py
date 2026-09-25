import sys, json, re
t = open(sys.argv[1]).read()
mode_filter = sys.argv[2] if len(sys.argv) > 2 else None
# find "name {json}" blocks by brace matching
i = 0
while True:
    m = re.search(r'(\w+) \{', t[i:])
    if not m: break
    start = i + m.start(2) if False else i + m.end() - 1
    depth = 0
    for j in range(start, len(t)):
        if t[j] == '{': depth += 1
        elif t[j] == '}':
            depth -= 1
            if depth == 0: break
    try:
        d = json.loads(t[start:j + 1])
    except Exception:
        i = start + 1; continue
    for tag, v in d.items():
        if mode_filter and tag != mode_filter: continue
        e = v['traj_err_m@50/100/200/400']; c = v['C_vs_Cprime_m']
        print(f"{m.group(1):15s} vs {tag:4s}: sets {v['contact_sets_equal']:6s} dqvel med {v['onestep_dqvel_median']:.1e} max {v['onestep_dqvel_max']:.1e} | traj @100 {e[1]:.1e} @200 {e[2]:.1e} @400 {e[3]:.1e} | C-C' @200 {c[2]:.1e} @400 {c[3]:.1e} | dz@400 {v['mean_z_diff_m@400']:.1e}")
    i = j + 1
