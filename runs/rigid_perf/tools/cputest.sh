#!/bin/bash
# CPU-only Newton unit tests on the final fork (Newton's test device list has no Metal device)
N=_forks/newton
cd $N
export PYTHONPATH=upstream/warp-innate-integrated:$N
export WARP_CACHE_PATH=runs/rigid_perf/kcache/kc_cputest
exec .venv-newton152/bin/python "$@"
