#!/bin/bash
# suite17b.sh ID TREE   suite17.sh without test_contact_reduction_global.py (that one runs alone, exclusive)
T=upstream/$2/newton/tests
exec runs/rigid17/tools/ntest.sh "$1" "$2" \
  .venv-newtonfork/bin/python -- -rfE \
  --junitxml=runs/rigid17/out/$1.xml \
  $T/test_contact_reduction.py $T/test_hashtable.py $T/test_contact_matching.py \
  $T/test_collision_pipeline.py $T/test_narrow_phase.py $T/test_sdf_contact.py $T/test_heightfield.py \
  $T/test_mesh_backface.py $T/test_rigid_contact.py $T/test_solver_xpbd.py $T/kamino/test_kamino_solvers_metrics.py
