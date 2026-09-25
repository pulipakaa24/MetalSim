#!/bin/bash
# Isaac's own rough-terrain G1 training (rsl_rl, Isaac's G1RoughPPORunnerCfg, seed 0, 1500 iterations) for the
# like-for-like rough comparison against MetalSim. debug_vis off: with it on, scene setup takes ~65 min.
# Checkpoints every 50 iterations under logs/rsl_rl/g1_rough; the terrain seed follows the env seed (0).
exec > $HOME/stage10.log 2>&1
set -x
export PATH=$HOME/.local/bin:$PATH TERM=xterm OMNI_KIT_ACCEPT_EULA=YES
source $HOME/env_isaaclab/bin/activate
cd $HOME/IsaacLab
mkdir -p $HOME/parity_out
rm -f $HOME/ROUGH_DONE
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --num_envs 4096 --headless \
  --max_iterations 1500 --seed 0 env.commands.base_velocity.debug_vis=false env.scene.height_scanner.debug_vis=false \
  > $HOME/parity_out/train_g1_rough.log 2>&1
echo "train exit $?"
rm -rf $HOME/parity_out/rsl_rl_g1_rough
cp -r logs/rsl_rl/g1_rough $HOME/parity_out/rsl_rl_g1_rough
touch $HOME/ROUGH_DONE
