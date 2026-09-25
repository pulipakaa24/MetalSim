#!/bin/bash
# Stage 6: re-record the fidelity protocols with the grey ground actually bound (the first recording showed
# Isaac's default grid plane and the other envs on the horizon), one env, RT then PT. Waits for stage 5.
set -x
until [ -f $HOME/STAGE5_DONE ]; do sleep 60; done
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
source $HOME/env_isaaclab/bin/activate
cd $HOME/IsaacLab
python $HOME/record_g1.py --headless --out $HOME/parity_out2/rt --render rt --num_envs 1 > $HOME/parity_out2_rt.log 2>&1
python $HOME/record_g1.py --headless --out $HOME/parity_out2/pt --render pt --num_envs 1 --frame_every 25 > $HOME/parity_out2_pt.log 2>&1
touch $HOME/STAGE6_DONE
