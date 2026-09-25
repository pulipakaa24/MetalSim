#!/bin/bash
# round 1: profile + variant scan + correctness checks (one queue slot)
cd /Users/aditya/robosim
bash runs/mjw_tp/job_profile.sh   > runs/mjw_tp/profile.log 2>&1
bash runs/mjw_tp/job_variants1.sh > runs/mjw_tp/variants1.log 2>&1
bash runs/mjw_tp/job_checks1.sh   > runs/mjw_tp/checks1.log 2>&1
