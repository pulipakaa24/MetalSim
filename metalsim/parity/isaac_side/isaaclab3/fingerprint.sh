#!/bin/bash
# Software fingerprint of an Isaac Lab 3.0-EA L4 machine (read-only), so a machine can be checked against the setup that ran the
# Franka-Pour seed-0 reference (l4a, 2026-10-12) before its runs are pooled with it.
#   ssh <l4-host> 'bash -s' < fingerprint.sh > fp_<l4-host>.txt
# FP_* lines are the checked keys (compared against the seed-0 reference); the rest is recorded for the PR.
export PATH=$HOME/.local/bin:$PATH OMNI_KIT_ACCEPT_EULA=YES
. $HOME/env_isaaclab3/bin/activate
cd $HOME/IsaacLab3 || { echo "FP_ERROR: no ~/IsaacLab3"; exit 1; }
echo "FP_DRIVER: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "FP_ISAACLAB_HEAD: $(git rev-parse HEAD)"
echo "FP_ISAACLAB_DIRTY: $(git status --porcelain | wc -l)"
git status --porcelain | sed 's/^/  dirty: /'
echo "FP_RSL_RL: $(python -c "import importlib.metadata as m; print(m.version('rsl-rl-lib'))" 2>&1 | tail -1)"
# Kit's console prints 'Initializing Version 2.72.3-release.7151+gl.5390bed9' (the omni client library); count
# installed client libraries carrying that exact version string
libs=$(find "$VIRTUAL_ENV" -name "libomniclient*.so*" 2>/dev/null)
n=0; for l in $libs; do grep -q "2.72.3-release.7151" "$l" && n=$((n+1)); done
echo "FP_OMNICLIENT_7151: $n"
for l in $libs; do echo "  omniclient lib: ${l#$VIRTUAL_ENV/}"; done
f=source/isaaclab_newton/isaaclab_newton/physics/newton_collision_cfg.py
echo "FP_MTP_CFG: $(grep -o 'max_triangle_pairs: int = [0-9_]*' $f)"
echo "FP_MTP_POUR_OVERRIDES: $(grep -rl max_triangle_pairs source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour | wc -l)"
echo "== recorded, not checked"
nvidia-smi --query-gpu=name --format=csv,noheader; nvidia-smi | grep -o "CUDA Version: [0-9.]*"
uname -r; grep PRETTY_NAME /etc/os-release; python --version
echo "franka_pour task files sha256: $(find source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-16)"
python - <<'EOF'
import importlib.metadata as md
for d in ("isaacsim", "isaaclab", "isaaclab_newton", "isaaclab_tasks", "newton", "warp-lang", "mujoco-warp", "mujoco",
          "torch", "rsl-rl-lib", "numpy", "usd-core"):
    try:
        print(f"{d}=={md.version(d)}")
    except Exception:
        print(f"{d}: not installed")
EOF
echo "parity3_scripts: $(ls $HOME/parity3_scripts 2>/dev/null | tr '\n' ' ')"
