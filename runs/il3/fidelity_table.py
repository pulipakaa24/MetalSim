import json, numpy as np, sys
P=["default","hardlimits","tau10_impact_hardlimits","isaaclab3","isaaclab3_collide_every_substep","isaaclab3_hardlimits"]
R={p:json.load(open(f"runs/il3/fidelity/report_{p}/report.json"))["physics"] for p in P}
rows=[]
def g(p,t,*k):
    v=R[p][t]
    for kk in k:
        if v is None: return None
        v=v.get(kk) if isinstance(v,dict) else None
    return v
f=lambda v,fmt="{:.4f}": "-" if v is None else fmt.format(v)
lines=[("joint RMSE hold @1s / end [rad]",lambda p:f"{f(g(p,'A_hold','joint_rmse_rad','t1s'))} / {f(g(p,'A_hold','joint_rmse_rad','end'))}"),
 ("joint RMSE drop @1s / end [rad]",lambda p:f"{f(g(p,'C_drop','joint_rmse_rad','t1s'))} / {f(g(p,'C_drop','joint_rmse_rad','end'))}"),
 ("joint RMSE random @0.5s / @1s [rad]",lambda p:f"{f(g(p,'B_random','joint_rmse_rad','t0.5s'))} / {f(g(p,'B_random','joint_rmse_rad','t1s'))}"),
 ("random: divergence >0.1 rad [s]",lambda p:f(g(p,'B_random','divergence_time_s'),"{:.2f}")),
 ("root height RMSE hold / drop [m]",lambda p:f"{f(g(p,'A_hold','root_height','rmse'))} / {f(g(p,'C_drop','root_height','rmse'))}"),
 ("drop landing impulse [N s] (Isaac {:.1f})".format(g('default','C_drop','contact_momentum','landing','impulse_Ns','isaac')),lambda p:f(g(p,'C_drop','contact_momentum','landing','impulse_Ns','metalsim'),"{:.1f}")),
 ("drop landing max 20 ms mean force [N] (Isaac {:.0f})".format(g('default','C_drop','contact_momentum','landing','max_20ms_mean_force_N','isaac')),lambda p:f(g(p,'C_drop','contact_momentum','landing','max_20ms_mean_force_N','metalsim'),"{:.0f}")),
 ("drop torso impact impulse [N s] (Isaac {:.1f})".format(g('default','C_drop','contact_momentum','torso','impulse_Ns','isaac')),lambda p:f(g(p,'C_drop','contact_momentum','torso','impulse_Ns','metalsim'),"{:.1f}")),
 ("hold torso impact impulse [N s] (Isaac {:.1f})".format(g('default','A_hold','contact_momentum','torso','impulse_Ns','isaac') or 0),lambda p:f(g(p,'A_hold','contact_momentum','torso','impulse_Ns','metalsim'),"{:.1f}")),
 ("limit excursion max, hold / drop / random [rad] (Isaac {:.4f} / {:.4f} / {:.4f})".format(*[g('default',t,'limit_excursion_rad','isaac_max_ctrl') for t in ('A_hold','C_drop','B_random')]),
   lambda p:" / ".join(f(g(p,t,'limit_excursion_rad','metalsim_max_ctrl')) for t in ('A_hold','C_drop','B_random'))),
 ("steps with excursion > 0.01 rad, hold / drop / random (Isaac {} / {} / {})".format(*[g('default',t,'limit_excursion_rad','isaac_steps_over_0.01') for t in ('A_hold','C_drop','B_random')]),
   lambda p:" / ".join(str(g(p,t,'limit_excursion_rad','metalsim_steps_over_0.01')) for t in ('A_hold','C_drop','B_random'))),
 ("peak joint speed drop [rad/s] (Isaac {:.1f})".format(g('default','C_drop','peak_joint_speed_rad_s','isaac')),lambda p:f(g(p,'C_drop','peak_joint_speed_rad_s','metalsim'),"{:.1f}")),
 ("max penetration drop [cm]",lambda p:f(100*g(p,'C_drop','penetration_m','metalsim_max'),"{:.2f}")),
]
print("| measure | "+" | ".join(P)+" |"); print("|---|"+"---|"*len(P))
for name,fn in lines: print(f"| {name} | "+" | ".join(fn(p) for p in P)+" |")
# solver iterations per substep (last substep of each control step)
isa={t:np.load(f"runs/parity3/isaac/fidelity/newton_mjwarp/{t}.npz")["solver_niter"] for t in ("A_hold","B_random","C_drop")}
print("\nsolver iterations (last substep per control step) mean / max: Isaac "+", ".join(f"{t} {isa[t].mean():.2f}/{isa[t].max()}" for t in isa))
for p in P:
    s={t:np.load(f"runs/il3/fidelity/{p}/{t}.npz")["solver_niter"][:,0] for t in isa}
    print(f"  {p}: "+", ".join(f"{t} {s[t].mean():.2f}/{s[t].max()}" for t in s))
