### twoway_coupling

L4 kernel sum 28.90 ms/frame over 1298 launches; Metal (isolated-dispatch profile) 41.29 ms/frame over 1870 launches; Metal empty command buffer 2.61 us/launch -> calibrated Metal sum 36.41 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| sort/scan/reduce (native) | 15.481 | 190 | 8.191 | 459 | 6.993 | 0.45 |
| gs_solve_kernel_impl | 3.207 | 400 | 8.990 | 400 | 7.945 | 2.48 |
| compute_body_forces_lanes | 0.000 | 0 | 2.881 | 4 | 2.870 |  |
| copy/fill | 0.890 | 38 | 1.484 | 145 | 1.105 | 1.24 |
| solve_subgrid_friction | 0.440 | 50 | 1.744 | 50 | 1.614 | 3.67 |
| integrate_velocity_apic | 1.037 | 1 | 0.834 | 1 | 0.832 | 0.80 |
| make_bsr_mv_tiled_kernel__locals__bsr_mv_tiled_kernel | 0.117 | 51 | 1.551 | 51 | 1.418 | 12.07 |
| apply_subgrid_impulse | 0.660 | 51 | 0.925 | 51 | 0.792 | 1.20 |
| make_bsr_mv_kernel__locals__bsr_mv_kernel | 1.340 | 51 | 0.127 | 1 | 0.124 | 0.09 |
| wp_memtile | 0.000 | 0 | 1.093 | 21 | 1.038 |  |
| NodePartition___scatter_partition_indices | 0.268 | 3 | 0.791 | 3 | 0.783 | 2.92 |
| collision_weight_field | 0.350 | 1 | 0.687 | 1 | 0.685 | 1.96 |
| dispatch_linear_kernel_fn | 0.588 | 4 | 0.307 | 4 | 0.296 | 0.50 |
| _identity_field | 0.433 | 3 | 0.293 | 3 | 0.285 | 0.66 |
| create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mp | 0.159 | 4 | 0.528 | 4 | 0.517 | 3.26 |
| _update_gradient_cholesky_blocked__locals__kernel | 0.061 | 4 | 0.623 | 4 | 0.612 | 10.07 |
| strain_delta_form | 0.347 | 1 | 0.230 | 1 | 0.227 | 0.65 |
| _triplet_segments | 0.000 | 0 | 0.543 | 6 | 0.528 |  |
| _segment_heads | 0.000 | 0 | 0.533 | 6 | 0.517 |  |
| advect_particles | 0.284 | 1 | 0.215 | 1 | 0.212 | 0.75 |
| _bsr_accumulate_triplet_values | 0.429 | 5 | 0.066 | 5 | 0.052 | 0.12 |
| rle_lengths | 0.000 | 0 | 0.439 | 9 | 0.415 |  |
| integrate_yield_parameters | 0.186 | 1 | 0.177 | 1 | 0.175 | 0.94 |
| wp_bounded_prep | 0.000 | 0 | 0.362 | 89 | 0.129 |  |
| _triplet_keys_u64 | 0.000 | 0 | 0.336 | 2 | 0.331 |  |
| convert_newton_contacts_to_mjwarp_kernel | 0.047 | 4 | 0.262 | 8 | 0.241 | 5.08 |
| dispatch_bilinear_kernel_fn | 0.172 | 2 | 0.111 | 2 | 0.106 | 0.62 |
| det_counter_resolve_small | 0.000 | 0 | 0.276 | 20 | 0.223 |  |
| _transpose_keys_u64 | 0.000 | 0 | 0.254 | 1 | 0.251 |  |
| _transpose_columns_counted | 0.000 | 0 | 0.253 | 1 | 0.251 |  |
| (124 more) | 2.402 | | 6.189 | | | |

