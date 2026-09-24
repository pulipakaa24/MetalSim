#!/usr/bin/env bash
# Copy the TRON1 classical controller + LimX SDK to the Jetson and build its Python env there.
#
#   scripts/tron1_jetson_deploy.sh [ssh-host]      (default: guest@172.16.15.31 with ~/.ssh/tron1)
#
# Installs into ~/tron1_ctrl on the Jetson with its own Python 3.12 (uv), leaving the ROS Noetic
# Python 3.8 setup untouched. MuJoCo is pinned to the version this code was tested with (3.14.0);
# the controller needs MjSpec (MuJoCo >= 3.2, Python >= 3.9). limxsdk's wheel is cp38-abi3, so
# it installs on 3.12.
set -euo pipefail
HOST="${1:-guest@172.16.15.31}"
SSH=(ssh -i "$HOME/.ssh/tron1" -o ConnectTimeout=8 "$HOST")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK_WHL="${SDK_WHL:-/private/tmp/claude-501/-Users-aditya/bb1d4029-a069-4fee-a180-4e72d456a105/scratchpad/limx/limxsdk-lowlevel/python3/aarch64/limxsdk-4.1.1-cp38-abi3-manylinux_2_29_aarch64.whl}"

"${SSH[@]}" 'mkdir -p ~/tron1_ctrl/metalsim/tron1 ~/tron1_ctrl/assets/tron1 ~/tron1_ctrl/runs'
rsync -az -e "ssh -i $HOME/.ssh/tron1" --exclude __pycache__ \
  "$ROOT/metalsim/__init__.py" "$HOST:tron1_ctrl/metalsim/"
rsync -az -e "ssh -i $HOME/.ssh/tron1" --exclude __pycache__ \
  "$ROOT/metalsim/tron1/" "$HOST:tron1_ctrl/metalsim/tron1/"
rsync -az -e "ssh -i $HOME/.ssh/tron1" "$ROOT/assets/tron1/WF_TRON1A" "$HOST:tron1_ctrl/assets/tron1/"
rsync -az -e "ssh -i $HOME/.ssh/tron1" "$ROOT/assets/tron1/limx_policy" "$HOST:tron1_ctrl/assets/tron1/"
[ -f "$ROOT/runs/tron1_gains.json" ] && rsync -az -e "ssh -i $HOME/.ssh/tron1" "$ROOT/runs/tron1_gains.json" "$HOST:tron1_ctrl/runs/"
rsync -az -e "ssh -i $HOME/.ssh/tron1" "$SDK_WHL" "$HOST:tron1_ctrl/"

"${SSH[@]}" bash -s <<'EOF'
set -euo pipefail
cd ~/tron1_ctrl
command -v uv >/dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -d .venv ] || $UV venv --python 3.12 .venv
$UV pip install --python .venv/bin/python -q numpy scipy websocket-client onnxruntime "mujoco==3.14.0" ./limxsdk-*.whl
.venv/bin/python - <<'PY'
import time, numpy as np
import limxsdk, mujoco
from metalsim.tron1.limx_bridge import make_controller, BridgeConfig, simulate_procedure, summarize
c = make_controller("runs/tron1_gains.json" if __import__("os").path.exists("runs/tron1_gains.json") else None)
from metalsim.tron1.sim import Tron1Sim
from metalsim.tron1.realism import SimParams
s = Tron1Sim(SimParams.nominal()).reset(c.q_stance)
st, imu = s.latest_state, s.latest_imu
for _ in range(100): c.step(st, imu, (0, 0, 0), 0.1)
t = time.perf_counter(); n = 1000
dts = []
for i in range(n):
    t1 = time.perf_counter(); c.step(st, imu, (0, 0, 0), 0.1 + i * 0.002); dts.append(time.perf_counter() - t1)
d = np.array(dts) * 1e3
print(f"controller step on this Jetson: mean {d.mean():.2f} ms, p99 {np.percentile(d, 99):.2f} ms, max {d.max():.2f} ms (budget 2 ms)")
r = simulate_procedure(c, BridgeConfig())
print("sim procedure through the bridge:", summarize(r), "(falling after the final DAMP is expected)")
print("mujoco", mujoco.__version__, "| limxsdk import ok")
PY
EOF
echo "deployed to $HOST:~/tron1_ctrl"
