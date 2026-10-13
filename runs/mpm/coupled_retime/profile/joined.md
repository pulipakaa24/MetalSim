### twoway_coupling

L4 kernel sum 28.90 ms/frame over 1298 launches; Metal (isolated-dispatch profile) 41.47 ms/frame over 1925 launches; Metal empty command buffer 4.97 us/launch -> calibrated Metal sum 31.90 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| sort/scan/reduce (native) | 15.481 | 190 | 8.321 | 462 | 6.023 | 0.39 |
| gs_solve_kernel_impl | 3.207 | 400 | 9.847 | 400 | 7.858 | 2.45 |
| compute_body_forces_lanes | 0.000 | 0 | 3.045 | 4 | 3.025 |  |
| copy/fill | 0.890 | 38 | 1.493 | 145 | 0.772 | 0.87 |
| solve_subgrid_friction | 0.440 | 50 | 1.758 | 50 | 1.510 | 3.43 |
| integrate_velocity_apic | 1.037 | 1 | 0.782 | 1 | 0.777 | 0.75 |
| apply_subgrid_impulse | 0.660 | 51 | 0.924 | 51 | 0.670 | 1.01 |
| make_bsr_mv_kernel__locals__bsr_mv_kernel | 1.340 | 51 | 0.131 | 2 | 0.121 | 0.09 |
| wp_memtile | 0.000 | 0 | 1.113 | 21 | 1.009 |  |
| NodePartition___scatter_partition_indices | 0.268 | 3 | 0.793 | 3 | 0.778 | 2.90 |
| collision_weight_field | 0.350 | 1 | 0.664 | 1 | 0.659 | 1.88 |
| dispatch_linear_kernel_fn | 0.588 | 4 | 0.319 | 4 | 0.299 | 0.51 |
| _identity_field | 0.433 | 3 | 0.284 | 3 | 0.269 | 0.62 |
| _update_gradient_cholesky_blocked__locals__kernel | 0.061 | 4 | 0.636 | 4 | 0.616 | 10.14 |
| create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mp | 0.159 | 4 | 0.498 | 4 | 0.478 | 3.02 |
| _update_gradient_cholesky_blocked_skip_unchanged__locals__kernel | 0.043 | 4 | 0.560 | 4 | 0.540 | 12.68 |
| _bsr_mv_listed_rows | 0.000 | 0 | 0.575 | 100 | 0.078 |  |
| strain_delta_form | 0.347 | 1 | 0.224 | 1 | 0.219 | 0.63 |
| wp_bounded_prep | 0.000 | 0 | 0.570 | 139 | 0.000 |  |
| _triplet_segments | 0.000 | 0 | 0.568 | 6 | 0.538 |  |
| _segment_heads | 0.000 | 0 | 0.538 | 6 | 0.508 |  |
| advect_particles | 0.284 | 1 | 0.211 | 1 | 0.206 | 0.73 |
| _bsr_accumulate_triplet_values | 0.429 | 5 | 0.045 | 5 | 0.020 | 0.05 |
| rle_lengths | 0.000 | 0 | 0.453 | 9 | 0.408 |  |
| integrate_yield_parameters | 0.186 | 1 | 0.166 | 1 | 0.161 | 0.86 |
| _triplet_keys_u64 | 0.000 | 0 | 0.340 | 2 | 0.330 |  |
| convert_newton_contacts_to_mjwarp_kernel | 0.047 | 4 | 0.256 | 8 | 0.216 | 4.55 |
| dispatch_bilinear_kernel_fn | 0.172 | 2 | 0.108 | 2 | 0.099 | 0.57 |
| _transpose_columns_counted | 0.000 | 0 | 0.275 | 1 | 0.270 |  |
| _scatter_node_counts | 0.050 | 8 | 0.211 | 8 | 0.172 | 3.43 |
| (124 more) | 2.427 | | 5.765 | | | |

### anymal

L4 kernel sum 10.54 ms/frame over 1358 launches; Metal (isolated-dispatch profile) 26.22 ms/frame over 1829 launches; Metal empty command buffer 2.55 us/launch -> calibrated Metal sum 21.56 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| gs_solve_kernel_impl | 3.307 | 400 | 9.501 | 400 | 8.482 | 2.57 |
| sort/scan/reduce (native) | 1.557 | 217 | 3.491 | 468 | 2.299 | 1.48 |
| collision_weight_field | 0.695 | 1 | 1.554 | 1 | 1.552 | 2.23 |
| advect_particles | 0.393 | 1 | 0.607 | 1 | 0.604 | 1.54 |
| solve_subgrid_friction | 0.363 | 50 | 0.540 | 50 | 0.412 | 1.14 |
| copy/fill | 0.140 | 40 | 0.742 | 150 | 0.359 | 2.57 |
| dispatch_linear_kernel_fn | 0.389 | 4 | 0.315 | 4 | 0.304 | 0.78 |
| rasterize_collider_kernel | 0.347 | 1 | 0.330 | 1 | 0.327 | 0.94 |
| _identity_field | 0.211 | 3 | 0.420 | 3 | 0.412 | 1.95 |
| strain_delta_form | 0.327 | 1 | 0.256 | 1 | 0.253 | 0.77 |
| det_counter_resolve_small | 0.000 | 0 | 0.536 | 37 | 0.442 |  |
| apply_subgrid_impulse | 0.195 | 51 | 0.334 | 51 | 0.204 | 1.05 |
| _primitive_narrowphase__locals__primitive_narrowphase | 0.039 | 4 | 0.448 | 8 | 0.428 | 10.92 |
| _linesearch_iterative_kernel__locals__kernel | 0.113 | 10 | 0.363 | 9 | 0.340 | 3.00 |
| _update_gradient_cholesky__locals__kernel | 0.077 | 14 | 0.380 | 13 | 0.347 | 4.51 |
| _crb_accumulate | 0.032 | 20 | 0.356 | 20 | 0.305 | 9.57 |
| _cfrc_backward | 0.029 | 20 | 0.310 | 20 | 0.259 | 8.78 |
| integrate_yield_parameters | 0.154 | 1 | 0.179 | 1 | 0.176 | 1.15 |
| _tile_cholesky_factorize_solve_block__locals__kernel | 0.053 | 8 | 0.279 | 8 | 0.258 | 4.91 |
| dispatch_bilinear_kernel_fn | 0.175 | 2 | 0.117 | 2 | 0.112 | 0.64 |
| _subtree_com_acc | 0.030 | 20 | 0.216 | 20 | 0.166 | 5.45 |
| _tiled_sum_kernel | 0.052 | 20 | 0.182 | 20 | 0.131 | 2.52 |
| _update_constraint_efc__locals__kernel | 0.047 | 14 | 0.185 | 22 | 0.128 | 2.71 |
| wp_memtile | 0.000 | 0 | 0.224 | 16 | 0.183 |  |
| compute_delassus_diagonal | 0.046 | 1 | 0.139 | 1 | 0.137 | 2.97 |
| integrate_velocity | 0.081 | 1 | 0.101 | 1 | 0.099 | 1.21 |
| eval_articulation_fk | 0.068 | 4 | 0.099 | 4 | 0.089 | 1.32 |
| _efc_contact_jac_dense__locals__kernel | 0.038 | 4 | 0.118 | 4 | 0.108 | 2.82 |
| mul_m_kernel__locals___mul_m | 0.056 | 14 | 0.095 | 13 | 0.062 | 1.11 |
| _update_gradient_grad__locals__kernel | 0.029 | 14 | 0.107 | 13 | 0.074 | 2.57 |
| (143 more) | 1.501 | | 3.693 | | | |

