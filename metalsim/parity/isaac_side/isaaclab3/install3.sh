#!/bin/bash
# Isaac Sim 6.1 + Isaac Lab v3.0.0-EA in a second environment (~/env_isaaclab3); leaves ~/env_isaaclab untouched.
# Follows the official "Python environment with Isaac Sim" path (docs/source/setup/installation/index.rst at v3.0.0-EA).
exec > $HOME/parity3/install3.log 2>&1
set -x
export PATH=$HOME/.local/bin:$PATH TERM=xterm OMNI_KIT_ACCEPT_EULA=YES
cd $HOME
[ -d IsaacLab3 ] || git clone --branch v3.0.0-EA https://github.com/isaac-sim/IsaacLab.git IsaacLab3
(cd IsaacLab3 && git log -1 --format='%H %cd')
[ -d env_isaaclab3 ] || uv venv --python 3.12 --seed env_isaaclab3
source env_isaaclab3/bin/activate
python --version
uv pip install --upgrade pip
uv pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
echo "isaacsim exit $?"
uv pip install -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
echo "torch exit $?"
cd IsaacLab3
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[kit]'
echo "isaaclab -i exit $?"
uv pip install imageio
python -c "import torch, warp, newton, mujoco, mujoco_warp; print('torch', torch.__version__, torch.cuda.is_available()); print('warp', warp.__version__); print('newton', newton.__version__); print('mujoco', mujoco.__version__, 'mjwarp', getattr(mujoco_warp,'__version__','?'))"
uv pip list 2>/dev/null | grep -i -E "^(isaacsim |isaaclab|torch |warp-lang|newton|mujoco|rsl-rl)" 
touch $HOME/parity3/INSTALL_DONE
