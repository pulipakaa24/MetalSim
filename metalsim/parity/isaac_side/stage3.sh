#!/bin/bash
# Fidelity protocol + Isaac's own G1 training on the VM. Waits for the benchmark stage to finish.
exec > $HOME/stage3.log 2>&1
set -x
until [ -f $HOME/ISAAC_DONE ]; do sleep 60; done
export PATH=$HOME/.local/bin:$PATH TERM=xterm OMNI_KIT_ACCEPT_EULA=YES
source $HOME/env_isaaclab/bin/activate
cd $HOME/IsaacLab
python -m pip install -q imageio 2>/dev/null || uv pip install -q imageio
python $HOME/record_g1.py --headless --out $HOME/parity_out/rt --render rt --frame_every 5
python $HOME/record_g1.py --headless --out $HOME/parity_out/pt --render pt --frame_every 25
# Isaac's own training with Isaac's config (checkpoints every 50 iterations under logs/rsl_rl/g1_flat)
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 4096 --headless --max_iterations 1500 --seed 0 > $HOME/parity_out/train_g1_flat.log 2>&1
cp -r logs/rsl_rl/g1_flat $HOME/parity_out/rsl_rl_g1_flat
tar czf $HOME/parity_out.tgz -C $HOME parity_out
touch $HOME/STAGE3_DONE
