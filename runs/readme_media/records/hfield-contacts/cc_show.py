import json, sys
r = json.load(open(sys.argv[1]))
print('==', r['shape'], {k: v for k, v in r.items() if k.startswith('distinct')})
for e, v in r['engines'].items():
  for ref, m in v.items():
    print(f'{e:12s} {ref:8s} eq {m["count_equal_min_ref_cap"]:.4f} few {m["fewer"]:3d} more {m["more"]:3d} none {m["no_contact"]:3d} '
          f'n>10 {m["share_normal_over_10deg"]} nerr {m["normal_err_deg_p50_p95_p99"]} derr {m["deepest_depth_err_mm_p50_p95_max"]} '
          f'wN95 {m["weighted_normal_err_deg_p95"]} cov {m["coverage_of_ref_mean"]}')
