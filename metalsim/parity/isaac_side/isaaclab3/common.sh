# sourced by the parity3 stage scripts: Isaac Sim 6.1 + Isaac Lab v3.0.0-EA environment
export PATH=$HOME/.local/bin:$PATH TERM=xterm OMNI_KIT_ACCEPT_EULA=YES
source $HOME/env_isaaclab3/bin/activate
cd $HOME/IsaacLab3
OUT=$HOME/parity3; mkdir -p $OUT
# one Kit process at a time on this VM (other stages / other users of the 5.1 env)
wait_for_gpu() { while pgrep -f "[k]it/python|[i]saaclab train|[i]saaclab benchmark|[r]ecord_g1|[r]ecord_deformables|[t]rain.py|[p]lay_policy" > /dev/null; do sleep 30; done; }
NOVIS="env.commands.base_velocity.debug_vis=false"
