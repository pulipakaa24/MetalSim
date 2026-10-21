#!/bin/bash
# The render / sensor test files of this branch, one pytest process per file (GPU check job).
WT=.
cd $WT
for f in tests/test_render_tendons.py tests/test_render_env_textures.py tests/test_render_extras.py tests/test_render_appearance.py \
         tests/test_render_world.py tests/test_render_tier0.py tests/test_render_tier2.py tests/test_render_terrain_slots.py \
         tests/test_camera_projection.py tests/test_sensors_rt.py tests/test_render_particles.py tests/test_render_newton_bridge.py \
         tests/test_render_newton_perworld.py tests/test_replicator.py tests/test_replicator_annotators.py ${EXTRA:-}; do
  echo "== $f"
  .venv/bin/python -m pytest -q -p no:cacheprovider "$f" 2>&1 | tail -15
done
