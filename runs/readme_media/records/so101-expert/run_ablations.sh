#!/bin/bash
# CPU ablations of the inferred / uncertain parts of 7v4_real for the deployment expert (so101Sim venv).
S=runs/readme_media/scratch/so101-expert; PY=~/so101Sim/urlab_bridge/.venv/bin/python; SO=~/so101Sim
$PY $S/eval_cpu.py $SO 7v4_real $S/cpu/abl_no_accel_limit 100 --over '{"accel_limit": Infinity}'
$PY $S/eval_cpu.py $SO 7v4_real $S/cpu/abl_host_10ms 100 --over '{"host_delay": 0.010}'
$PY $S/eval_cpu.py $SO 7v4_real $S/cpu/abl_bam_stall 100 --over '{"stall_Nm": 3.41}'
$PY $S/eval_cpu.py $SO 7v4_real $S/cpu/abl_hz50 100 --hz 50
$PY $S/eval_cpu.py $SO 12v_real $S/cpu/deploy_12v_real 100
$PY $S/eval_cpu.py $SO 7v4 $S/cpu/deploy_7v4_old 100
