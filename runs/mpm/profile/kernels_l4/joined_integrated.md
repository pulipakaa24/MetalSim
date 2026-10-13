### granular

L4 kernel sum 18.67 ms/frame over 2823 launches; Metal (isolated-dispatch profile) 37.27 ms/frame over 3036 launches; Metal empty command buffer 2.54 us/launch -> calibrated Metal sum 29.55 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| gs_solve_kernel_impl | 13.293 | 1790 | 27.411 | 1787 | 22.867 | 1.72 |
| sort/scan/reduce (native) | 1.135 | 217 | 2.464 | 420 | 1.396 | 1.23 |
| solve_subgrid_friction | 0.842 | 224 | 1.261 | 223 | 0.693 | 0.82 |
| apply_subgrid_impulse | 0.687 | 225 | 1.135 | 224 | 0.564 | 0.82 |
| integrate_velocity_apic | 0.443 | 1 | 0.615 | 1 | 0.613 | 1.38 |
| _tiled_sum_kernel | 0.231 | 90 | 0.771 | 89 | 0.544 | 2.36 |
| collision_weight_field | 0.282 | 1 | 0.593 | 1 | 0.591 | 2.10 |
| advect_particles | 0.181 | 1 | 0.284 | 1 | 0.281 | 1.55 |
| dispatch_linear_kernel_fn | 0.205 | 4 | 0.247 | 4 | 0.237 | 1.16 |
| copy/fill | 0.068 | 11 | 0.264 | 42 | 0.157 | 2.31 |
| evaluate_strain_residual | 0.117 | 45 | 0.203 | 45 | 0.089 | 0.77 |
| _identity_field | 0.090 | 3 | 0.196 | 3 | 0.189 | 2.09 |
| strain_delta_form | 0.158 | 1 | 0.120 | 1 | 0.118 | 0.74 |
| update_condition | 0.074 | 45 | 0.140 | 45 | 0.026 | 0.36 |
| project_outside_collider | 0.092 | 1 | 0.118 | 1 | 0.116 | 1.26 |
| integrate_yield_parameters | 0.078 | 1 | 0.086 | 1 | 0.083 | 1.07 |
| dispatch_bilinear_kernel_fn | 0.091 | 2 | 0.068 | 2 | 0.063 | 0.70 |
| compute_delassus_diagonal | 0.028 | 1 | 0.088 | 1 | 0.086 | 3.11 |
| vb_world_bbox | 0.000 | 0 | 0.100 | 3 | 0.092 |  |
| apply_stress_gs | 0.039 | 8 | 0.057 | 8 | 0.037 | 0.94 |
| integrate_velocity | 0.028 | 1 | 0.046 | 1 | 0.043 | 1.53 |
| set_conditional_if_handle_kernel | 0.066 | 46 | 0.000 | 0 | 0.000 |  |
| wp_memtile | 0.000 | 0 | 0.066 | 4 | 0.055 |  |
| rasterize_collider_kernel | 0.021 | 1 | 0.033 | 1 | 0.031 | 1.43 |
| vb_unique_scatter | 0.000 | 0 | 0.049 | 6 | 0.034 |  |
| vb_unique_count | 0.000 | 0 | 0.044 | 6 | 0.029 |  |
| vb_init_grid | 0.000 | 0 | 0.041 | 3 | 0.034 |  |
| postprocess_stress_and_strain | 0.014 | 1 | 0.026 | 1 | 0.024 | 1.71 |
| quadrature_point_element_indices | 0.032 | 1 | 0.008 | 1 | 0.005 | 0.17 |
| _triplet_keys_u32 | 0.000 | 0 | 0.038 | 4 | 0.028 |  |
| (73 more) | 0.374 | | 0.699 | | | |

### multi_material

L4 kernel sum 10.83 ms/frame over 1546 launches; Metal (isolated-dispatch profile) 28.13 ms/frame over 2007 launches; Metal empty command buffer 2.56 us/launch -> calibrated Metal sum 23.00 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| gs_solve_kernel_impl | 6.708 | 860 | 19.133 | 1020 | 16.525 | 2.46 |
| sort/scan/reduce (native) | 1.126 | 217 | 2.658 | 420 | 1.584 | 1.41 |
| solve_subgrid_friction | 0.417 | 108 | 0.790 | 128 | 0.464 | 1.11 |
| apply_subgrid_impulse | 0.322 | 108 | 0.693 | 128 | 0.364 | 1.13 |
| collision_weight_field | 0.266 | 1 | 0.692 | 1 | 0.689 | 2.59 |
| integrate_velocity_apic | 0.311 | 1 | 0.618 | 1 | 0.615 | 1.98 |
| _tiled_sum_kernel | 0.111 | 43 | 0.473 | 51 | 0.342 | 3.08 |
| copy/fill | 0.067 | 11 | 0.343 | 46 | 0.226 | 3.39 |
| dispatch_linear_kernel_fn | 0.171 | 4 | 0.222 | 4 | 0.212 | 1.24 |
| advect_particles | 0.144 | 1 | 0.248 | 1 | 0.245 | 1.70 |
| _identity_field | 0.096 | 3 | 0.241 | 3 | 0.233 | 2.42 |
| update_particle_strains | 0.147 | 1 | 0.165 | 1 | 0.162 | 1.10 |
| strain_delta_form | 0.125 | 1 | 0.105 | 1 | 0.103 | 0.82 |
| evaluate_strain_residual | 0.055 | 22 | 0.115 | 26 | 0.050 | 0.90 |
| dispatch_bilinear_kernel_fn | 0.077 | 2 | 0.068 | 2 | 0.063 | 0.82 |
| integrate_yield_parameters | 0.059 | 1 | 0.075 | 1 | 0.072 | 1.23 |
| update_condition | 0.036 | 22 | 0.083 | 26 | 0.017 | 0.48 |
| compute_delassus_diagonal | 0.022 | 1 | 0.096 | 1 | 0.093 | 4.21 |
| vb_world_bbox | 0.000 | 0 | 0.106 | 3 | 0.098 |  |
| apply_stress_gs | 0.038 | 8 | 0.063 | 8 | 0.043 | 1.11 |
| project_outside_collider | 0.054 | 1 | 0.037 | 1 | 0.035 | 0.65 |
| wp_memtile | 0.000 | 0 | 0.071 | 4 | 0.061 |  |
| integrate_velocity | 0.021 | 1 | 0.041 | 1 | 0.038 | 1.83 |
| vb_unique_scatter | 0.000 | 0 | 0.056 | 6 | 0.040 |  |
| postprocess_stress_and_strain | 0.023 | 1 | 0.033 | 1 | 0.030 | 1.32 |
| vb_init_grid | 0.000 | 0 | 0.049 | 3 | 0.041 |  |
| vb_unique_count | 0.000 | 0 | 0.048 | 6 | 0.032 |  |
| rasterize_collider_kernel | 0.012 | 1 | 0.032 | 1 | 0.029 | 2.43 |
| _triplet_keys_u32 | 0.000 | 0 | 0.040 | 4 | 0.029 |  |
| _bsr_accumulate_triplet_values | 0.013 | 3 | 0.025 | 3 | 0.018 | 1.42 |
| (75 more) | 0.407 | | 0.713 | | | |

### twoway_coupling

L4 kernel sum 28.90 ms/frame over 1287 launches; Metal (isolated-dispatch profile) 42.74 ms/frame over 1983 launches; Metal empty command buffer 2.49 us/launch -> calibrated Metal sum 37.81 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| sort/scan/reduce (native) | 15.398 | 190 | 8.725 | 533 | 7.400 | 0.48 |
| gs_solve_kernel_impl | 3.221 | 400 | 7.106 | 400 | 6.112 | 1.90 |
| make_bsr_mv_kernel__locals__bsr_mv_kernel | 1.336 | 51 | 5.406 | 102 | 5.152 | 3.86 |
| compute_body_forces_lanes | 0.000 | 0 | 2.741 | 4 | 2.731 |  |
| copy/fill | 0.909 | 38 | 1.443 | 142 | 1.090 | 1.20 |
| solve_subgrid_friction | 0.440 | 50 | 1.711 | 50 | 1.587 | 3.61 |
| integrate_velocity_apic | 1.025 | 1 | 0.769 | 1 | 0.767 | 0.75 |
| apply_subgrid_impulse | 0.660 | 51 | 0.916 | 51 | 0.789 | 1.19 |
| wp_memtile | 0.000 | 0 | 1.089 | 21 | 1.036 |  |
| NodePartition___scatter_partition_indices | 0.282 | 3 | 0.794 | 3 | 0.786 | 2.79 |
| collision_weight_field | 0.344 | 1 | 0.649 | 1 | 0.647 | 1.88 |
| dispatch_linear_kernel_fn | 0.588 | 4 | 0.311 | 4 | 0.301 | 0.51 |
| _identity_field | 0.431 | 3 | 0.283 | 3 | 0.276 | 0.64 |
| _update_gradient_cholesky_blocked__locals__kernel | 0.065 | 4 | 0.624 | 4 | 0.614 | 9.47 |
| create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mp | 0.160 | 4 | 0.471 | 4 | 0.461 | 2.89 |
| _update_gradient_cholesky_blocked_skip_unchanged__locals__kernel | 0.043 | 4 | 0.546 | 4 | 0.536 | 12.61 |
| strain_delta_form | 0.347 | 1 | 0.221 | 1 | 0.218 | 0.63 |
| _triplet_segments | 0.000 | 0 | 0.551 | 6 | 0.536 |  |
| _segment_heads | 0.000 | 0 | 0.535 | 6 | 0.520 |  |
| _bsr_accumulate_triplet_values | 0.434 | 5 | 0.046 | 5 | 0.034 | 0.08 |
| advect_particles | 0.278 | 1 | 0.202 | 1 | 0.200 | 0.72 |
| rle_lengths | 0.000 | 0 | 0.438 | 9 | 0.416 |  |
| integrate_yield_parameters | 0.188 | 1 | 0.168 | 1 | 0.166 | 0.88 |
| det_scatter_finalize | 0.000 | 0 | 0.346 | 76 | 0.157 |  |
| _triplet_keys_u64 | 0.000 | 0 | 0.337 | 2 | 0.333 |  |
| _transpose_columns | 0.000 | 0 | 0.309 | 1 | 0.307 |  |
| _bsr_transpose_values | 0.059 | 1 | 0.244 | 1 | 0.242 | 4.13 |
| convert_newton_contacts_to_mjwarp_kernel | 0.041 | 4 | 0.245 | 8 | 0.225 | 5.42 |
| dispatch_bilinear_kernel_fn | 0.177 | 2 | 0.108 | 2 | 0.103 | 0.58 |
| SpaceRestriction___split_vertex_element_index | 0.056 | 4 | 0.199 | 4 | 0.189 | 3.37 |
| (120 more) | 2.416 | | 5.202 | | | |

### anymal

L4 kernel sum 10.56 ms/frame over 1344 launches; Metal (isolated-dispatch profile) 25.73 ms/frame over 2189 launches; Metal empty command buffer 4.56 us/launch -> calibrated Metal sum 15.74 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| gs_solve_kernel_impl | 3.315 | 400 | 7.176 | 400 | 5.351 | 1.61 |
| sort/scan/reduce (native) | 1.559 | 217 | 4.557 | 617 | 1.742 | 1.12 |
| collision_weight_field | 0.693 | 1 | 1.613 | 1 | 1.609 | 2.32 |
| advect_particles | 0.390 | 1 | 0.625 | 1 | 0.620 | 1.59 |
| det_scatter_finalize | 0.000 | 0 | 0.953 | 138 | 0.323 |  |
| solve_subgrid_friction | 0.364 | 50 | 0.550 | 50 | 0.322 | 0.89 |
| copy/fill | 0.139 | 40 | 0.759 | 150 | 0.073 | 0.53 |
| dispatch_linear_kernel_fn | 0.390 | 4 | 0.339 | 4 | 0.320 | 0.82 |
| rasterize_collider_kernel | 0.353 | 1 | 0.334 | 1 | 0.330 | 0.93 |
| _identity_field | 0.213 | 3 | 0.435 | 3 | 0.422 | 1.98 |
| strain_delta_form | 0.327 | 1 | 0.258 | 1 | 0.253 | 0.77 |
| apply_subgrid_impulse | 0.197 | 51 | 0.331 | 51 | 0.098 | 0.50 |
| _primitive_narrowphase__locals__primitive_narrowphase | 0.044 | 4 | 0.478 | 8 | 0.442 | 10.07 |
| _update_gradient_cholesky__locals__kernel | 0.079 | 14 | 0.394 | 13 | 0.334 | 4.22 |
| _linesearch_iterative_kernel__locals__kernel | 0.113 | 10 | 0.303 | 9 | 0.261 | 2.31 |
| det_seg_down | 0.000 | 0 | 0.339 | 37 | 0.170 |  |
| integrate_yield_parameters | 0.156 | 1 | 0.183 | 1 | 0.178 | 1.14 |
| _tile_cholesky_factorize_solve_block__locals__kernel | 0.056 | 8 | 0.277 | 8 | 0.241 | 4.27 |
| dispatch_bilinear_kernel_fn | 0.174 | 2 | 0.119 | 2 | 0.110 | 0.63 |
| _crb_accumulate | 0.036 | 20 | 0.246 | 20 | 0.155 | 4.30 |
| _update_constraint_efc__locals__kernel | 0.048 | 14 | 0.196 | 22 | 0.094 | 1.94 |
| _cfrc_backward | 0.034 | 20 | 0.201 | 20 | 0.110 | 3.28 |
| _tiled_sum_kernel | 0.052 | 20 | 0.179 | 20 | 0.088 | 1.70 |
| wp_memtile | 0.000 | 0 | 0.231 | 16 | 0.158 |  |
| _subtree_com_acc | 0.035 | 20 | 0.160 | 20 | 0.069 | 2.00 |
| compute_delassus_diagonal | 0.046 | 1 | 0.142 | 1 | 0.138 | 2.99 |
| integrate_velocity | 0.081 | 1 | 0.100 | 1 | 0.096 | 1.18 |
| eval_articulation_fk | 0.071 | 4 | 0.102 | 4 | 0.084 | 1.18 |
| det_seg_spine | 0.000 | 0 | 0.158 | 37 | 0.000 |  |
| mul_m_kernel__locals___mul_m | 0.054 | 14 | 0.096 | 13 | 0.036 | 0.67 |
| (146 more) | 1.545 | | 3.896 | | | |

