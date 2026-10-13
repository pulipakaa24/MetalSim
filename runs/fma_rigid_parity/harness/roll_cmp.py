import numpy as np, sys
O = sys.argv[1]
def L(n): return np.load(f"{O}/{n}.npz")
for terr in ("flat", "rough"):
    print(f"\n== {terr}: pair | first differing step | worlds differing at step 1 / 10 / 200 | median over worlds of max|dq_joint| at step 1 / 5 / 20 / 50 / 200 | |mean rew diff| | dones")
    for a, b in [("B1","B2"),("C1","C2"),("F1","F2"),("B1","C1"),("B2","C2"),("B1","F1"),("F1","C1")]:
        A, B = L(f"{a}_{terr}"), L(f"{b}_{terr}")
        dq = np.abs(A["qpos"][..., 7:].astype(np.float64) - B["qpos"][..., 7:])   # joints only (37), [step, world, j]
        wmax = dq.max(-1)                                                         # [step, world]
        anyd = (A["qpos"] != B["qpos"]).any(-1) | (A["qvel"] != B["qvel"]).any(-1)
        first = int(np.argmax(anyd.any(1))) if anyd.any() else -1
        med = [np.median(wmax[s]) for s in (0, 4, 19, 49, 199)]
        print(f"  {a}-{b} | {first} | {anyd[0].sum()}/{anyd[9].sum()}/{anyd[199].sum()} of 512 | " + " / ".join(f"{x:.2e}" for x in med)
              + f" | {abs(A['rew'].mean()-B['rew'].mean()):.2e} | {int(A['done'].sum())} vs {int(B['done'].sum())}")
