#!/bin/bash
# Stage 8: play Isaac's final checkpoint (model_1499; stage 5 picked model_950 for the "it1500" slot
# because rsl_rl saved 1499 as the last one). Waits for stage 7 (Kit single-instance lock).
set -x
until [ -f $HOME/STAGE7_DONE ]; do sleep 60; done
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
source $HOME/env_isaaclab/bin/activate
cd $HOME/IsaacLab
d=$(ls -d $HOME/parity_out/rsl_rl_g1_flat/*/ | tail -1)
python $HOME/play_policy.py --headless --ckpt $d/model_1499.pt --out $HOME/parity_out/play/isaac_it1499 --render rt > $HOME/stage8.log 2>&1
tar czf $HOME/parity_play_1499.tgz -C $HOME/parity_out/play isaac_it1499
touch $HOME/STAGE8_DONE
