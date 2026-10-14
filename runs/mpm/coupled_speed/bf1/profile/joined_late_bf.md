### twoway_coupling

L4 kernel sum 28.90 ms/frame over 1298 launches; Metal (isolated-dispatch profile) 38.45 ms/frame over 1890 launches; Metal empty command buffer 4.97 us/launch -> calibrated Metal sum 29.06 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| sort/scan/reduce (native) | 15.481 | 190 | 8.130 | 471 | 5.790 | 0.37 |
| gs_solve_kernel_impl | 3.207 | 400 | 9.026 | 400 | 7.039 | 2.19 |
| copy/fill | 0.890 | 38 | 1.463 | 145 | 0.743 | 0.83 |
| solve_subgrid_friction | 0.440 | 50 | 1.722 | 50 | 1.473 | 3.35 |
| integrate_velocity_apic | 1.037 | 1 | 0.830 | 1 | 0.825 | 0.80 |
| make_bsr_mv_tiled_kernel__locals__bsr_mv_tiled_kernel | 0.117 | 51 | 1.579 | 51 | 1.326 | 11.29 |
| apply_subgrid_impulse | 0.660 | 51 | 0.930 | 51 | 0.676 | 1.02 |
| make_bsr_mv_kernel__locals__bsr_mv_kernel | 1.340 | 51 | 0.127 | 1 | 0.122 | 0.09 |
| wp_memtile | 0.000 | 0 | 1.081 | 21 | 0.976 |  |
| NodePartition___scatter_partition_indices | 0.268 | 3 | 0.782 | 3 | 0.767 | 2.87 |
| collision_weight_field | 0.350 | 1 | 0.686 | 1 | 0.681 | 1.95 |
| dispatch_linear_kernel_fn | 0.588 | 4 | 0.307 | 4 | 0.287 | 0.49 |
| _identity_field | 0.433 | 3 | 0.288 | 3 | 0.273 | 0.63 |
| _update_gradient_cholesky_blocked__locals__kernel | 0.061 | 4 | 0.629 | 4 | 0.609 | 10.01 |
| create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mp | 0.159 | 4 | 0.513 | 4 | 0.493 | 3.11 |
| strain_delta_form | 0.347 | 1 | 0.227 | 1 | 0.222 | 0.64 |
| _triplet_segments | 0.000 | 0 | 0.535 | 6 | 0.505 |  |
| _segment_heads | 0.000 | 0 | 0.531 | 6 | 0.501 |  |
| _bsr_accumulate_triplet_values | 0.429 | 5 | 0.066 | 5 | 0.041 | 0.10 |
| advect_particles | 0.284 | 1 | 0.211 | 1 | 0.207 | 0.73 |
| rle_lengths | 0.000 | 0 | 0.437 | 9 | 0.393 |  |
| integrate_yield_parameters | 0.186 | 1 | 0.180 | 1 | 0.175 | 0.94 |
| wp_bounded_prep | 0.000 | 0 | 0.357 | 89 | 0.000 |  |
| _triplet_keys_u64 | 0.000 | 0 | 0.335 | 2 | 0.325 |  |
| convert_newton_contacts_to_mjwarp_kernel | 0.047 | 4 | 0.253 | 8 | 0.213 | 4.49 |
| dispatch_bilinear_kernel_fn | 0.172 | 2 | 0.111 | 2 | 0.101 | 0.59 |
| det_counter_resolve_small | 0.000 | 0 | 0.270 | 20 | 0.170 |  |
| _transpose_columns_counted | 0.000 | 0 | 0.254 | 1 | 0.249 |  |
| _transpose_keys_u64 | 0.000 | 0 | 0.254 | 1 | 0.249 |  |
| _bsr_mv_listed_rows | 0.000 | 0 | 0.250 | 50 | 0.001 |  |
| (126 more) | 2.402 | | 6.082 | | | |

