#!/bin/bash
# Play policies in Isaac for the side-by-side videos: Isaac's own checkpoints (rsl_rl) and MetalSim's
# exported ones (uploaded to ~/metalsim_ckpts/*.pt). Waits for stage 4 and for the upload marker.
exec > $HOME/stage5.log 2>&1
set -x
until [ -f $HOME/STAGE4_DONE ] && [ -f $HOME/metalsim_ckpts/UPLOADED ]; do sleep 60; done
export PATH=$HOME/.local/bin:$PATH TERM=xterm OMNI_KIT_ACCEPT_EULA=YES
source $HOME/env_isaaclab/bin/activate
cd $HOME/IsaacLab
run=$(ls -d logs/rsl_rl/g1_flat/* | tail -1)
for it in 100 500 1000 1500; do
  f=$run/model_$it.pt; [ -f $f ] || f=$(ls $run/model_*.pt | sort -t_ -k2 -n | tail -1)
  python $HOME/play_policy.py --headless --ckpt $f --out $HOME/parity_out/play/isaac_it$it --render rt
done
for f in $HOME/metalsim_ckpts/*.pt; do
  b=$(basename $f .pt)
  python $HOME/play_policy.py --headless --ckpt $f --out $HOME/parity_out/play/metalsim_$b --render rt
done
tar czf $HOME/parity_play.tgz -C $HOME parity_out/play
touch $HOME/STAGE5_DONE
