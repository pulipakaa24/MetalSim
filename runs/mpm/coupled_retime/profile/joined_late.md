### twoway_coupling

L4 kernel sum 28.90 ms/frame over 1298 launches; Metal (isolated-dispatch profile) 60.17 ms/frame over 1925 launches; Metal empty command buffer 2.51 us/launch -> calibrated Metal sum 55.34 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| sort/scan/reduce (native) | 15.481 | 190 | 8.291 | 462 | 7.133 | 0.46 |
| _bsr_mv_listed_rows | 0.000 | 0 | 19.543 | 100 | 19.292 |  |
| gs_solve_kernel_impl | 3.207 | 400 | 9.294 | 400 | 8.291 | 2.59 |
| compute_body_forces_lanes | 0.000 | 0 | 2.880 | 4 | 2.870 |  |
| copy/fill | 0.890 | 38 | 1.481 | 145 | 1.118 | 1.26 |
| solve_subgrid_friction | 0.440 | 50 | 1.777 | 50 | 1.651 | 3.75 |
| make_bsr_mv_kernel__locals__bsr_mv_kernel | 1.340 | 51 | 0.571 | 2 | 0.565 | 0.42 |
| integrate_velocity_apic | 1.037 | 1 | 0.840 | 1 | 0.837 | 0.81 |
| apply_subgrid_impulse | 0.660 | 51 | 0.939 | 51 | 0.811 | 1.23 |
| wp_memtile | 0.000 | 0 | 1.089 | 21 | 1.037 |  |
| NodePartition___scatter_partition_indices | 0.268 | 3 | 0.789 | 3 | 0.782 | 2.92 |
| collision_weight_field | 0.350 | 1 | 0.687 | 1 | 0.685 | 1.96 |
| dispatch_linear_kernel_fn | 0.588 | 4 | 0.309 | 4 | 0.299 | 0.51 |
| _identity_field | 0.433 | 3 | 0.290 | 3 | 0.283 | 0.65 |
| create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mp | 0.159 | 4 | 0.533 | 4 | 0.523 | 3.30 |
| _update_gradient_cholesky_blocked__locals__kernel | 0.061 | 4 | 0.630 | 4 | 0.620 | 10.19 |
| strain_delta_form | 0.347 | 1 | 0.227 | 1 | 0.224 | 0.65 |
| wp_bounded_prep | 0.000 | 0 | 0.572 | 139 | 0.224 |  |
| _triplet_segments | 0.000 | 0 | 0.547 | 6 | 0.532 |  |
| _segment_heads | 0.000 | 0 | 0.536 | 6 | 0.521 |  |
| _bsr_accumulate_triplet_values | 0.429 | 5 | 0.065 | 5 | 0.053 | 0.12 |
| advect_particles | 0.284 | 1 | 0.210 | 1 | 0.208 | 0.73 |
| rle_lengths | 0.000 | 0 | 0.440 | 9 | 0.417 |  |
| integrate_yield_parameters | 0.186 | 1 | 0.176 | 1 | 0.174 | 0.93 |
| _triplet_keys_u64 | 0.000 | 0 | 0.336 | 2 | 0.331 |  |
| convert_newton_contacts_to_mjwarp_kernel | 0.047 | 4 | 0.250 | 8 | 0.230 | 4.85 |
| dispatch_bilinear_kernel_fn | 0.172 | 2 | 0.114 | 2 | 0.109 | 0.63 |
| det_counter_resolve_small | 0.000 | 0 | 0.277 | 20 | 0.227 |  |
| _scatter_node_counts | 0.050 | 8 | 0.209 | 8 | 0.189 | 3.77 |
| _transpose_columns_counted | 0.000 | 0 | 0.255 | 1 | 0.252 |  |
| (124 more) | 2.469 | | 6.011 | | | |

