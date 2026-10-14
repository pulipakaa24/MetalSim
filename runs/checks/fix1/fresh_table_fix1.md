| step | 16-inch (macOS 26) | 14-inch (macOS 27, another user; correctness only) |
|---|---|---|
| 01_clone | ok | ok |
| 02_venv | ok | ok |
| 03_pip_metalsim | ok | ok |
| 04_setup_warp | ok | ok |
| 05_pip_mujoco_warp | ok | ok |
| 06_fetch_isaac | ok | ok |
| 07_menagerie | ok | ok |
| 08_setup_newton152 | ok | ok |
| 09_fetch_polyhaven_demo | ok | ok |
| 09b_fetch_isaaclab_source | ok | ok |
| 10_metal_check | ok | ok |
| 11_rigid_suite | **FAIL** (rc 1): 1 failed, 233 passed, 20 warnings in 512.54s (0:08:32) | **FAIL** (rc 1): 1 failed, 233 passed, 20 warnings |
| 12_render_suite | ok: 235 passed, 5 skipped, 2 warnings in 211.26s (0:03:31) | ok: 235 passed, 5 skipped, 2 warnings |
| 13_test_mpm | ok: 16 passed, 1 xfailed in 122.92s (0:02:02) | ok: 16 passed, 1 xfailed |
| 14_test_mpm_main | ok: 17 passed in 359.23s (0:05:59) | ok: 17 passed |
| 15_pins | ok | ok |
| 17_test_franka_pour | ok: 8 passed, 48 warnings in 122.43s (0:02:02) | ok: 8 passed, 48 warnings |
| 20_demo_g1 | ok | ok |
| 21_demo_franka_pour | ok | ok |
| 22_demo_box_on_sand | ok | ok |
| 23_demo_anymal_sim | ok | ok |
| 24_demo_anymal_render | ok | ok |
| 25_demo_transparency_lidar | ok | ok |
| 30_skips | ok: 5 skipped | ok: 5 skipped |
