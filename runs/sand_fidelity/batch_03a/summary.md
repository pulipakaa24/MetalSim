# sand_fidelity batch 03a: results

Gates (plan section 18.5): CPU tests, Lode kernels on Metal, pr4660 Metal tests, parity (main and 1.5.2), all
with zero skips; Lode smokes finite. Heap angles at R/dx 5 are reported, not gated (03b is the acceptance).

* TESTS_03A: PASS 11:06:23 50 passed in 37.77s
* lode_metal: DONE 11:06:41 18 s
* pr4660_metal: DONE 11:08:20 99 s
* parity_main: DONE 11:13:24 304 s
* parity_152: DONE 11:21:43 499 s
* mc_smoke: DONE 11:24:57 194 s
* mn_smoke: DONE 11:28:14 164 s
* mc_smoke heap: 26.94 deg (atan mu 34.22 deg, R/dx 5, P1d, 250 it)
* mn_smoke heap: 29.19 deg (atan mu 34.22 deg, R/dx 5, P1d, 250 it)
* cpu_tests pytest: 50 passed in 37.77s
* lode_metal pytest: 10 passed, 6 deselected in 10.34s
* pr4660_metal pytest: 7 passed in 92.38s (0:01:32)
* parity_main pytest: 17 passed in 295.57s (0:04:55)
* parity_152 pytest: 16 passed, 1 xfailed in 108.72s (0:01:48)
