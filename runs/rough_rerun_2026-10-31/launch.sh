#!/bin/bash
# launch.sh  [FROM=<job>] [SKIP="<job> ..."]: start run_all.sh detached so it survives the agent and Claude Code exiting:
# a new session (perl POSIX::setsid: no controlling terminal, own process group), nohup (SIGHUP ignored), stdin from
# /dev/null, and caffeinate -i -s (no idle / system sleep while the driver runs on AC power; closing the lid still sleeps).
R=~/MetalSim; RR=runs/rough_rerun_2026-10-31; cd "$R" || exit 1
[ -e $RR/STOP ] && { echo "$RR/STOP exists: remove it to launch"; exit 1; }
if [ -e $RR/driver.pid ] && kill -0 "$(cat $RR/driver.pid)" 2>/dev/null; then echo "driver $(cat $RR/driver.pid) already running"; exit 1; fi
rm -f $RR/driver.pid
env -u PYTHONPATH FROM="${FROM:-}" SKIP="${SKIP:-}" nohup /usr/bin/perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' \
  /usr/bin/caffeinate -i -s /bin/bash $RR/run_all.sh >> $RR/driver.out 2>&1 < /dev/null &
LP=$!; disown $LP 2>/dev/null
echo "$LP" > $RR/launcher.pid
for i in 1 2 3 4 5 6 7 8 9 10; do [ -s $RR/driver.pid ] && break; sleep 1; done
echo "caffeinate (launcher) pid $LP; driver (run_all.sh) pid $(cat $RR/driver.pid 2>/dev/null)"
ps -o pid,ppid,pgid,sess,command -p $LP,$(cat $RR/driver.pid 2>/dev/null) 2>/dev/null | cut -c1-150
