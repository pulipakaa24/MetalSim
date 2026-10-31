#!/bin/zsh
# Fetches Unitree's unitree_rl_gym (BSD-3-Clause, https://github.com/unitreerobotics/unitree_rl_gym) at a pinned
# commit into assets/unitree_rl_gym/ for the film tool's g1_unitree task (metalsim/render/film/tasks.py G1_UNITREE):
# Unitree's pretrained G1 walking policy (deploy/pre_train/g1/motion.pt, TorchScript), its MuJoCo sim2sim playback
# (deploy/deploy_mujoco/), the G1 model it plays on (resources/robots/g1_description/) and the training config the
# policy came from (legged_gym/envs/g1/). Shallow, sparse checkout of those paths only; git-ignored, not committed.
set -e
cd "$(dirname "$0")/.."
REPO=${UNITREE_RL_GYM_REPO:-https://github.com/unitreerobotics/unitree_rl_gym.git}
SHA=276801e46c5d433564f24658bac64f254b7d2d4b     # main as of 2026-10-31
DEST=assets/unitree_rl_gym
# SHA-256 of the files the film reads, as fetched on 2026-10-31
POLICY_SHA256=cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d
if [ ! -d $DEST/.git ]; then
  mkdir -p $DEST
  git -C $DEST init -q
  git -C $DEST remote add origin "$REPO"
  git -C $DEST sparse-checkout set --no-cone /LICENSE /README.md /deploy/pre_train/g1/ /deploy/deploy_mujoco/ \
      /resources/robots/g1_description/ /legged_gym/envs/g1/ /legged_gym/envs/base/legged_robot_config.py
fi
if [ "$(git -C $DEST rev-parse -q --verify HEAD 2>/dev/null)" != "$SHA" ]; then
  git -C $DEST fetch -q --depth 1 origin $SHA
  git -C $DEST checkout -q --detach FETCH_HEAD
fi
[ "$(git -C $DEST rev-parse HEAD)" = "$SHA" ] || { echo "unitree_rl_gym: HEAD is not $SHA"; exit 1; }
h=$(shasum -a 256 $DEST/deploy/pre_train/g1/motion.pt | cut -d' ' -f1)
[ "$POLICY_SHA256" = PENDING ] || [ "$h" = "$POLICY_SHA256" ] || { echo "SHA-256 mismatch for motion.pt: $h"; exit 1; }
echo "unitree_rl_gym $SHA: $DEST (motion.pt sha256 $h)"
