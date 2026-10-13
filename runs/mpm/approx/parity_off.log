## T1 short horizon vs the L4 float32 floor
| run | l4_ref | early_frame | early_rms | early_floor | early_ratio | end_frame | end_rms | end_floor | end_ratio | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_a2_host_1 | R0_a2_f32 | 10 | 2.795e-08 | 3.363e-08 | 0.8308 | 180 | 1.649e-07 | 1.886e-07 | 0.8744 | PASS |
| R0_granular_host_1 | R0_granular_f32 | 6 | 0.0002444 | 5.284e-08 | 4625 | 60 | 0.0174 | 0.000629 | 27.66 | FAIL |

## T2 column collapse (normalized runout / height; seed means)
| a | lattice_runout_l4 | lattice_runout_metal | runout_l4 | runout_l4_sd | runout_metal | runout_metal_sd | runout_rel | height_l4 | height_metal | height_rel | iters_l4 | iters_metal | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

fitted exponents: {}

## T3 examples, coupling, robot, plate
| run | particles | frames | com_travel_m | com_err_rel | ke_curve_rel_rms | body_travel_m | body_err_rel | iters_l4 | iters_metal | ms_l4 | ms_metal | pass | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3_granular | 226981 | 180 | 2.238 | 0.000509 | 0.0005209 |  |  | 249.2 | 249.1 | 54.99 | 180 | PASS |  |
| R3_multi_material | 178669 | 180 | 0.6332 | 0.002737 | 0.001453 |  |  | 241.9 | 241.8 | 55.41 | 177.7 | PASS |  |
| R3_snow_ball | 641657 | 120 | 0.6019 | 0.001438 | 0.001084 | 0 | 0 | 149.7 | 149.5 | 116 | 232.4 | PASS |  |
| R3_viscous | 100190 | 480 | 0.3145 | 0.001524 | 0.008162 |  |  | 250 | 250 | 54.01 | 81.5 | PASS |  |
| R3_beam_twist | 10309 | 480 | 0.08416 | 0.0004947 | 0.0001786 |  |  | None | None | 26.43 | 33.77 | PASS |  |
| R4_twoway | 453871 | 300 | 0.0835 | 0.0004323 | 4.94e-05 | 4.283 | 0.01282 | 50 | 50 | 90.5 | 161 | PASS |  |
| R6_anymal | 486416 | 150 | 0.03614 | 0.001114 | 0.001955 | 2.854 | 1.089e-06 | 50 | 50 | 62.89 | 47.28 | PASS |  |

| run | particles | force_peak_l4_N | force_peak_metal_N | force_final_l4_N | force_final_metal_N | force_curve_rel_rms | iters_l4 | iters_metal | ms_l4 | ms_metal | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R7_plate_intrude | 280478 | 821 | 796.2 | 809.2 | 786.5 | 0.01924 | 250 | 250 | 57.33 | 68.54 | PASS |
| R7_plate_drag | 278846 | 283.5 | 266.8 | 262.8 | 239.6 | 0.03624 | 250 | 250 | 56.76 | 43.97 | PASS |
