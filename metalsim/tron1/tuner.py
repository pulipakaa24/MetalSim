"""Live gain-tuning panel for the classical TRON1 controller, served by the viewer process.

The viewer (`metalsim.tron1.view --tune`) starts this on http://127.0.0.1:8777 and opens it in the
browser. Every `ClassicalGains` field is a slider; changes are applied between frames of the
running sim. Balance weights (Q, R) and ride height recompute the stance, the LQR gains and the
feedforward immediately (`ClassicalController.set_height`). Overrides survive resets, falls and
code reloads; "Save" writes them to JSON, which `--gains PATH` loads at start.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# name -> (group, min, max, step, description). Q is split into its four weights.
META = {
    "Q_x": ("Balance (LQR)", 0.0, 200.0, 0.5, "state weight on position error"),
    "Q_phi": ("Balance (LQR)", 1.0, 400.0, 1.0, "state weight on body lean angle"),
    "Q_v": ("Balance (LQR)", 0.0, 50.0, 0.5, "state weight on speed error"),
    "Q_phidot": ("Balance (LQR)", 0.0, 50.0, 0.5, "state weight on lean rate"),
    "R": ("Balance (LQR)", 0.01, 5.0, 0.01, "weight on wheel torque (higher = gentler)"),
    "manual_K": ("Balance: manual gains", 0.0, 1.0, 1.0, "1 = use the four gains below instead of the LQR"),
    "K_pos": ("Balance: manual gains", 0.0, 60.0, 0.1, "position P [N m per m]"),
    "K_vel": ("Balance: manual gains", 0.0, 60.0, 0.1, "position D = speed [N m per m/s]"),
    "K_lean": ("Balance: manual gains", 0.0, 300.0, 0.5, "lean P [N m per rad]"),
    "K_lean_rate": ("Balance: manual gains", 0.0, 60.0, 0.1, "lean D [N m per rad/s]"),
    "latency_comp": ("Balance (LQR)", 0.0, 0.03, 0.001, "predict the state this far ahead [s]"),
    "trim_rate": ("Balance (LQR)", 0.0, 0.1, 0.005, "balance-point integrator [rad/(m s)] (0 = off)"),
    "trim_limit": ("Balance (LQR)", 0.0, 0.3, 0.01, "balance-point integrator clamp [rad]"),
    "hold_kp": ("Position hold (parked)", 0.0, 5.0, 0.05, "speed correction per metre of error [1/s]"),
    "hold_speed_limit": ("Position hold (parked)", 0.0, 0.5, 0.01, "cap on that correction [m/s]"),
    "hold_deadband": ("Position hold (parked)", 0.0, 0.2, 0.005, "no position correction inside this [m]"),
    "x_err_limit": ("Position hold (parked)", 0.05, 2.0, 0.05, "beyond this the spot is re-anchored [m]"),
    "park_speed": ("Position hold (parked)", 0.0, 0.3, 0.005, "counts as stopped below this speed [m/s]"),
    "park_yaw_rate": ("Position hold (parked)", 0.0, 0.5, 0.01, "counts as stopped below this yaw rate [rad/s]"),
    "park_time": ("Position hold (parked)", 0.0, 3.0, 0.05, "stopped this long before parking [s]"),
    "park_timeout": ("Position hold (parked)", 0.0, 5.0, 0.1, "park anyway this long after the command ends [s]"),
    "heading_kp": ("Heading", 0.0, 30.0, 0.25, "wheel torque difference per rad of heading error"),
    "heading_kd": ("Heading", 0.0, 20.0, 0.25, "per rad/s of yaw-rate error (gyro)"),
    "heading_err_limit": ("Heading", 0.0, 1.5, 0.05, "heading error clamp [rad]"),
    "accel_limit": ("Driving", 0.1, 5.0, 0.1, "ramp on the speed command [m/s^2]"),
    "wheel_torque_limit": ("Driving", 5.0, 40.0, 0.5, "per wheel [N m]"),
    "wheel_friction_comp": ("Driving", 0.0, 0.5, 0.01, "Coulomb friction fed forward [N m]"),
    "wheel_friction_eps": ("Driving", 0.05, 3.0, 0.05, "smoothing of that near zero speed [rad/s]"),
    "pack_mass": ("Sensor pack: actual (sim)", 0.0, 6.0, 0.05, "mass of the pack on the simulated robot [kg]"),
    "pack_com_x": ("Sensor pack: actual (sim)", -0.12, 0.12, 0.005, "its COM, forward of the base origin [m]"),
    "pack_com_y": ("Sensor pack: actual (sim)", -0.08, 0.08, 0.005, "its COM, to the left [m]"),
    "pack_com_z": ("Sensor pack: actual (sim)", -0.05, 0.30, 0.005, "its COM, above the base origin [m] (base top 0.023)"),
    "belief_mass": ("Sensor pack: controller's belief", 0.0, 6.0, 0.05, "pack mass the controller assumes [kg]"),
    "belief_com_x": ("Sensor pack: controller's belief", -0.12, 0.12, 0.005, "assumed COM forward [m]"),
    "belief_com_y": ("Sensor pack: controller's belief", -0.08, 0.08, 0.005, "assumed COM left [m]"),
    "belief_com_z": ("Sensor pack: controller's belief", -0.05, 0.30, 0.005, "assumed COM up [m]"),
    "height": ("Legs", 0.55, 0.76, 0.005, "axle below the base origin [m]"),
    "leg_kp": ("Legs", 0.0, 200.0, 1.0, "hip/knee stiffness [N m/rad]"),
    "leg_kd": ("Legs", 0.0, 10.0, 0.1, "hip/knee damping [N m s/rad]"),
    "leg_ki": ("Legs", 0.0, 100.0, 1.0, "leg joint integral [N m/(rad s)]"),
    "leg_ki_limit": ("Legs", 0.0, 40.0, 0.5, "leg integral clamp [N m]"),
    "abad_kp": ("Legs", 0.0, 300.0, 1.0, "hip-roll stiffness [N m/rad]"),
    "abad_kd": ("Legs", 0.0, 15.0, 0.1, "hip-roll damping [N m s/rad]"),
    "roll_kp": ("Roll and sideways", 0.0, 5.0, 0.05, "leg-length difference per unit roll"),
    "roll_kd": ("Roll and sideways", 0.0, 1.0, 0.01, "same, per roll rate"),
    "lateral_kp": ("Roll and sideways", 0.0, 3.0, 0.05, "sideways wheel step per unit roll (sensitive)"),
    "lateral_kd": ("Roll and sideways", 0.0, 0.5, 0.01, "same, per roll rate (sensitive: 0.1 fell)"),
    "lateral_limit": ("Roll and sideways", 0.0, 0.2, 0.005, "max sideways step [m]"),
}
Q_NAMES = ("Q_x", "Q_phi", "Q_v", "Q_phidot")
PACK = ("pack_mass", "pack_com_x", "pack_com_y", "pack_com_z")
BELIEF = ("belief_mass", "belief_com_x", "belief_com_y", "belief_com_z")


def pack_defaults(sim_params, model):
    """Slider defaults: the sim's pack and the controller model's assumed pack."""
    c = sim_params.payload_pos
    b = model.payload_pos
    return {"pack_mass": float(sim_params.payload_mass), "pack_com_x": float(c[0]), "pack_com_y": float(c[1]),
            "pack_com_z": float(c[2]), "belief_mass": float(model.payload_mass), "belief_com_x": float(b[0]),
            "belief_com_y": float(b[1]), "belief_com_z": float(b[2])}


def apply_pack(ctrl, sim, flat, current):
    """Apply pack (sim) and belief (controller) sliders present in `flat`; `current` has all values."""
    if any(k in flat for k in PACK):
        sim.set_pack(current["pack_mass"], (current["pack_com_x"], current["pack_com_y"], current["pack_com_z"]))
    if any(k in flat for k in BELIEF):
        ctrl.set_pack_belief(current["belief_mass"], (current["belief_com_x"], current["belief_com_y"], current["belief_com_z"]))
RECOMPUTE = set(Q_NAMES) | {"R", "height", "manual_K", "K_pos", "K_vel", "K_lean", "K_lean_rate"}


def gains_to_flat(g):
    out = {f.name: getattr(g, f.name) for f in fields(g) if f.name != "Q"}
    out.update(dict(zip(Q_NAMES, g.Q)))
    return {k: float(v) for k, v in out.items()}


def apply_flat(ctrl, flat):
    """Write a {name: value} dict into a ClassicalController; recompute the LQR when needed."""
    g = ctrl.g
    q = list(g.Q)
    recompute = False
    for k, v in flat.items():
        if k in PACK or k in BELIEF:
            continue
        if k in Q_NAMES:
            q[Q_NAMES.index(k)] = float(v)
        elif hasattr(g, k):
            setattr(g, k, float(v))
        recompute |= k in RECOMPUTE
    g.Q = tuple(q)
    if recompute:
        ctrl.set_height(g.height)


class Tuner:
    def __init__(self, defaults, port=8777, gains_file=None, save_path="runs/tron1_gains.json"):
        self.defaults = defaults                # flat dict of ClassicalGains defaults
        self.overrides = {}
        if gains_file:
            self.overrides.update(json.loads(Path(gains_file).read_text()))
        self.pending = dict(self.overrides)
        self.lock = threading.Lock()
        self.telemetry = {}
        self.history = []                      # (t, x_err, v, pitch_deg)
        self.actions = []                      # "push_fwd", "push_side", "reset", "stop"
        self.save_path = Path(save_path)
        self.port = port
        tuner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype="application/json"):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/":
                    self._send(PAGE, "text/html; charset=utf-8")
                elif self.path == "/state":
                    with tuner.lock:
                        cur = {**tuner.defaults, **tuner.overrides}
                        body = json.dumps(dict(gains=cur, defaults=tuner.defaults, meta=META,
                                               telemetry=tuner.telemetry, history=tuner.history[-600:]))
                    self._send(body)
                else:
                    self.send_error(404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                msg = json.loads(self.rfile.read(n) or b"{}")
                with tuner.lock:
                    if self.path == "/set":
                        tuner.overrides[msg["name"]] = float(msg["value"])
                        tuner.pending[msg["name"]] = float(msg["value"])
                    elif self.path == "/reset_gains":
                        tuner.pending.update(tuner.defaults)
                        tuner.overrides.clear()
                    elif self.path == "/save":
                        tuner.save_path.parent.mkdir(parents=True, exist_ok=True)
                        tuner.save_path.write_text(json.dumps(tuner.overrides, indent=1))
                    elif self.path == "/action":
                        tuner.actions.append(msg["action"])
                self._send(json.dumps(dict(ok=True, saved=str(tuner.save_path.resolve()))))

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def open(self):
        webbrowser.open(f"http://127.0.0.1:{self.port}/")

    def attach(self, ctrl, sim):
        """A new robot (start, reset, fall): apply every override to it."""
        with self.lock:
            apply_flat(ctrl, self.overrides)
            apply_pack(ctrl, sim, self.overrides, {**self.defaults, **self.overrides})
            self.pending.clear()
            self.history.clear()

    def update(self, ctrl, sim, vel):
        """Once per frame: apply pending changes, publish telemetry, return queued actions."""
        with self.lock:
            if self.pending:
                apply_flat(ctrl, self.pending)
                apply_pack(ctrl, sim, self.pending, {**self.defaults, **self.overrides})
                self.pending.clear()
            acts, self.actions = self.actions, []
        tr = sim.truth()
        g = ctrl.g
        poles = np.linalg.eigvals(ctrl.A - ctrl.B @ ctrl.K)
        x_err = float(ctrl.x - ctrl.x_ref)
        tel = dict(t=tr["t"], v=float(tr["v_head"][0]), yaw_rate=tr["yaw_rate"], pitch=float(np.degrees(tr["pitch"])),
                   roll=float(np.degrees(tr["roll"])), cmd_vx=vel[0], cmd_wz=vel[2], x_err=x_err,
                   parked=bool(getattr(ctrl, "parked", False)), tau_bal=float(ctrl.tau_bal),
                   phi_trim_deg=float(np.degrees(ctrl.phi_trim)),
                   K=[round(float(-k), 2) for k in ctrl.K[0]],
                   poles=[f"{p.real:.2f}{p.imag:+.2f}j" if abs(p.imag) > 1e-6 else f"{p.real:.2f}" for p in poles],
                   height=g.height, total_mass=float(sim.total_mass))
        with self.lock:
            self.telemetry = tel
            self.history.append((round(tr["t"], 3), round(x_err, 4), round(tel["v"], 4), round(tel["pitch"], 3)))
            if len(self.history) > 1200:
                del self.history[:200]
        return acts


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRON1 Gain Tuner</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#16181d;--mute:#6b7280;--line:#e3e6eb;--acc:#2563eb;--warn:#b45309;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--panel:#171a21;--ink:#e6e8ec;--mute:#9aa3af;--line:#2a2f3a;--acc:#60a5fa;--warn:#f59e0b;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 -apple-system,system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center}
h1{font-size:15px;margin:0 8px 0 0}button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:4px 10px;cursor:pointer}
button:hover{border-color:var(--acc)}.tel{display:flex;flex-wrap:wrap;gap:4px 14px;font-variant-numeric:tabular-nums;color:var(--mute)}.tel b{color:var(--ink);font-weight:600}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;padding:12px 16px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}h2{font-size:13px;margin:0 0 8px}
.row{display:grid;grid-template-columns:1fr 84px 22px;gap:2px 8px;align-items:center;margin:6px 0}
.row label{grid-column:1/4;display:flex;justify-content:space-between;gap:8px}.row label span{color:var(--mute);font-size:11.5px;text-align:right}
input[type=range]{width:100%;accent-color:var(--acc)}input[type=number]{width:84px;font:inherit;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:2px 4px}
.changed input[type=number]{border-color:var(--warn);color:var(--warn)}.undo{padding:0 4px;font-size:11px;visibility:hidden}.changed .undo{visibility:visible}
canvas{width:100%;height:90px;display:block}.plots{grid-column:1/-1}.legend{display:flex;gap:14px;color:var(--mute);font-size:11.5px}
.legend i{display:inline-block;width:10px;height:3px;margin-right:4px;vertical-align:middle}
</style></head><body>
<header><h1>TRON1 gain tuner</h1>
<button onclick="act('push_fwd')">Shove forward</button><button onclick="act('push_side')">Shove sideways</button>
<button onclick="act('stop')">Stop</button><button onclick="act('reset')">Reset robot</button>
<button onclick="post('/reset_gains',{}).then(load)">All gains to default</button><button id="save" onclick="save()">Save</button>
<div class="tel" id="tel"></div></header>
<main id="main"><section class="plots"><h2>Last 12 s</h2>
<div class="legend"><span><i style="background:var(--acc)"></i>position error (m, parked hold)</span><span><i style="background:var(--warn)"></i>speed (m/s)</span><span><i style="background:var(--ok)"></i>pitch (deg / 10)</span></div>
<canvas id="plot"></canvas></section></main>
<script>
let S=null;
const post=(u,b)=>fetch(u,{method:'POST',body:JSON.stringify(b)}).then(r=>r.json());
const act=a=>post('/action',{action:a});
function save(){post('/save',{}).then(r=>{const b=document.getElementById('save');b.textContent='Saved to '+r.saved;setTimeout(()=>b.textContent='Save',3000)})}
function fmt(v,st){const d=Math.max(0,-Math.floor(Math.log10(st)));return Number(v).toFixed(d)}
function build(){
  const m=document.getElementById('main');[...m.querySelectorAll('section:not(.plots)')].forEach(e=>e.remove());
  const groups={};
  for(const [k,[g,lo,hi,st,doc]] of Object.entries(S.meta)){(groups[g]=groups[g]||[]).push([k,lo,hi,st,doc])}
  for(const [g,items] of Object.entries(groups)){
    const sec=document.createElement('section');sec.innerHTML='<h2>'+g+'</h2>';
    for(const [k,lo,hi,st,doc] of items){
      const v=S.gains[k],d=S.defaults[k];const row=document.createElement('div');row.className='row'+(v!==d?' changed':'');
      row.innerHTML=`<label><b>${k}</b><span>${doc} · default ${fmt(d,st)}</span></label>
        <input type=range min=${lo} max=${Math.max(hi,v)} step=${st} value=${v}><input type=number step=${st} value=${fmt(v,st)}><button class=undo title="back to default">↺</button>`;
      const [r,n,u]=row.querySelectorAll('input,button');
      const setv=x=>{r.value=x;n.value=fmt(x,st);row.classList.toggle('changed',Math.abs(x-d)>1e-12);post('/set',{name:k,value:x})};
      r.oninput=()=>setv(+r.value);n.onchange=()=>setv(+n.value);u.onclick=()=>setv(d);
      sec.appendChild(row)}
    m.appendChild(sec)}
}
function load(){fetch('/state').then(r=>r.json()).then(s=>{S=s;build()})}
function tel(){fetch('/state').then(r=>r.json()).then(s=>{const t=s.telemetry;if(!t.t)return;
  document.getElementById('tel').innerHTML=
   `<span>t <b>${t.t.toFixed(1)} s</b></span><span>robot <b>${t.total_mass.toFixed(2)} kg</b></span><span>cmd <b>${t.cmd_vx.toFixed(2)} m/s, ${t.cmd_wz.toFixed(2)} rad/s</b></span>`+
   `<span>speed <b>${t.v.toFixed(3)}</b></span><span>pitch <b>${t.pitch.toFixed(2)}°</b></span><span>roll <b>${t.roll.toFixed(2)}°</b></span>`+
   `<span>pos err <b>${(t.x_err*100).toFixed(1)} cm</b></span><span><b style="color:${t.parked?'var(--ok)':'var(--mute)'}">${t.parked?'PARKED':'driving'}</b></span>`+
   `<span>wheel τ <b>${t.tau_bal.toFixed(2)} N·m</b></span><span>gains in use: pos P <b>${t.K[0]}</b> pos D <b>${t.K[2]}</b> lean P <b>${t.K[1]}</b> lean D <b>${t.K[3]}</b></span><span>poles <b>${t.poles.join(', ')}</b></span>`;
  draw(s.history)})}
function draw(h){const c=document.getElementById('plot'),x=c.getContext('2d'),W=c.width=c.clientWidth*devicePixelRatio,H=c.height=90*devicePixelRatio;
  if(h.length<2)return;const t1=h[h.length-1][0],t0=t1-12;const pts=h.filter(p=>p[0]>=t0);
  const cs=getComputedStyle(document.documentElement);x.strokeStyle=cs.getPropertyValue('--line');x.beginPath();x.moveTo(0,H/2);x.lineTo(W,H/2);x.stroke();
  const ser=[[1,'--acc',1],[2,'--warn',1],[3,'--ok',0.1]];let lim=0.05;for(const p of pts)for(const [i,,s] of ser)lim=Math.max(lim,Math.abs(p[i]*s));
  for(const [i,col,s] of ser){x.strokeStyle=cs.getPropertyValue(col);x.lineWidth=1.5*devicePixelRatio;x.beginPath();
    pts.forEach((p,j)=>{const X=(p[0]-t0)/12*W,Y=H/2-p[i]*s/lim*(H/2-4);j?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke()}
  x.fillStyle=cs.getPropertyValue('--mute');x.font=`${11*devicePixelRatio}px system-ui`;x.fillText('±'+lim.toFixed(3),4,12*devicePixelRatio)}
load();setInterval(tel,200);
</script></body></html>
"""
