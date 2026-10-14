#!/bin/zsh
# Fetches the part of Isaac Lab 3.0-EA's source that tests/test_franka_pour_terms.py uses as its oracle (Isaac's own
# Franka-Pour reward / observation / termination / reset-sampler code and the isaaclab math and MDP modules it calls,
# loaded by tests/franka_pour_isaac_ref.py without importing isaaclab): a sparse, shallow checkout of the pinned
# commit, a few hundred KB instead of the whole repository. Isaac Lab is BSD-3-Clause (its LICENSE comes with the
# checkout); nothing of it is copied into this repository.
# Usage: scripts/fetch_isaaclab_source.sh   (run from anywhere)
# Env: ISAACLAB3_DIR (default <repo>/upstream/IsaacLab3; tests/franka_pour_isaac_ref.py reads ISAACLAB3_SOURCE =
#      that directory's source/), ISAACLAB3_REPO (default github.com/isaac-sim/IsaacLab).
# An existing checkout (e.g. a full clone) is left as it is; a note is printed when it is not at the pin.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DIR=${ISAACLAB3_DIR:-$REPO/upstream/IsaacLab3}
URL=${ISAACLAB3_REPO:-https://github.com/isaac-sim/IsaacLab}
PIN=ae37b028ea415c91ea2bc32609efcd759ed2b974     # tag v3.0.0-EA, the Isaac Lab 3.0-EA the parity work was recorded with
PATHS=(source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour
       source/isaaclab/isaaclab/utils
       source/isaaclab/isaaclab/envs/mdp)
if [ -e "$DIR/.git" ]; then
  HEAD=$(git -C "$DIR" rev-parse HEAD)
  [ "$HEAD" = "$PIN" ] || echo "note: $DIR is at ${HEAD:0:10}, not the pinned ${PIN:0:10} (left as is)"
else
  mkdir -p "$DIR"
  git -C "$DIR" init -q
  git -C "$DIR" remote add origin "$URL"
  git -C "$DIR" sparse-checkout set --cone $PATHS
  git -C "$DIR" fetch -q --depth 1 --filter=blob:none origin "$PIN"
  git -C "$DIR" checkout -q FETCH_HEAD
fi
for f in source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py source/isaaclab/isaaclab/utils/math.py \
         source/isaaclab/isaaclab/envs/mdp/rewards.py LICENSE; do
  [ -f "$DIR/$f" ] || { echo "missing $DIR/$f after the fetch"; exit 1; }
done
echo "Isaac Lab 3.0-EA source: $DIR at $(git -C "$DIR" rev-parse --short HEAD) ($(du -sh "$DIR" | cut -f1))"
