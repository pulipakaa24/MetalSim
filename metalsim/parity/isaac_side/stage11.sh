#!/bin/bash
# Stage 11: rough-terrain transfer. Plays checkpoints in Isaac-Velocity-Rough-G1-v0 on the training run's terrain
# (env seed 0), curriculum off, every env at terrain row $LEVELS in its own column (4 envs -> columns 0, 4, 9, 14 (float32 floor of i / 0.2)),
# command (0.5, 0, 0), no observation noise, no termination, 400 steps, mean action (play_policy.py).
#   WHO=isaac   : Isaac's own rsl_rl checkpoints 500 / 1000 / 1499 of logs/rsl_rl/g1_rough (latest run)
#   WHO=metalsim: MetalSim exports in ~/metalsim_ckpts_rough/*.pt (metalsim/parity/export_policy.py)
exec >> $HOME/stage11.log 2>&1
set -x
WHO=${WHO:-isaac}; LEVELS=${LEVELS:-3}
export PATH=$HOME/.local/bin:$PATH TERM=xterm OMNI_KIT_ACCEPT_EULA=YES
source $HOME/env_isaaclab/bin/activate
cd $HOME/IsaacLab
rm -f $HOME/STAGE11_DONE
run=$(ls -d logs/rsl_rl/g1_rough/* | tail -1)
for lv in $LEVELS; do
  if [ "$WHO" = isaac ]; then
    for it in 500 1000 1499; do
      f=$run/model_$it.pt; [ -f $f ] || f=$(ls $run/model_*.pt | sort -t_ -k2 -n | tail -1)
      python $HOME/play_policy.py --headless --task Isaac-Velocity-Rough-G1-v0 --level $lv --seed 0 --no_camera --ckpt $f --out $HOME/parity_out/play_rough/L$lv/isaac_it$it
    done
  else
    for f in $HOME/metalsim_ckpts_rough/*.pt; do
      b=$(basename $f .pt)
      python $HOME/play_policy.py --headless --task Isaac-Velocity-Rough-G1-v0 --level $lv --seed 0 --no_camera --ckpt $f --out $HOME/parity_out/play_rough/L$lv/metalsim_$b
    done
  fi
done
tar czf $HOME/parity_play_rough_$WHO.tgz -C $HOME/parity_out play_rough
touch $HOME/STAGE11_DONE
