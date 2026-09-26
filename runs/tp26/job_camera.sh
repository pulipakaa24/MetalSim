#!/bin/bash
# tp26 round 2: camera-RL path (Isaac-Cartpole-RGB, 1024 envs, tier 0, Isaac's skrl config), 400 K env-steps of training,
# landed state (installed forks, MetalSim HEAD) vs the day's starting forks (worktrees 07a51a6 / f194006a via PYTHONPATH).
cd /Users/aditya/robosim && source .venv/bin/activate
OLD=/Users/aditya/robosim/upstream/warp-innate-merge:/Users/aditya/robosim/upstream/mujoco_warp-07a51a6
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) installed warp $(git -C upstream/warp-innate rev-parse --short HEAD) mjw $(git -C upstream/mujoco_warp rev-parse --short HEAD) power: $(pmset -g batt | head -1)"
for r in 1 2; do
  echo "--- after $r (installed forks) $(date +%H:%M:%S)"; python -m metalsim.learn.train_cartpole_rgb --envs 1024 --tier 0 --steps 400000 --log runs/tp26/camera_after_$r.log 2>&1 | grep "done:\|Error\|Traceback" | cut -c1-220
  echo "--- before $r (forks 07a51a6 / f194006a) $(date +%H:%M:%S)"; PYTHONPATH=$OLD python -c "import warp, mujoco_warp; print('imports', warp.__file__, mujoco_warp.__file__)"; PYTHONPATH=$OLD python -m metalsim.learn.train_cartpole_rgb --envs 1024 --tier 0 --steps 400000 --log runs/tp26/camera_before_$r.log 2>&1 | grep "done:\|Error\|Traceback" | cut -c1-220
done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
