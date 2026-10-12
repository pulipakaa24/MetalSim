### granular

L4 kernel sum 18.67 ms/frame over 2823 launches; Metal (isolated-dispatch profile) 47.45 ms/frame over 3077 launches; Metal empty command buffer 5.07 us/launch -> calibrated Metal sum 31.84 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| gs_solve_kernel_impl | 13.293 | 1790 | 37.706 | 1820 | 28.475 | 2.14 |
| sort/scan/reduce (native) | 1.135 | 217 | 2.455 | 420 | 0.325 | 0.29 |
| solve_subgrid_friction | 0.842 | 224 | 1.548 | 228 | 0.394 | 0.47 |
| apply_subgrid_impulse | 0.687 | 225 | 1.159 | 228 | 0.000 | 0.00 |
| _tiled_sum_kernel | 0.231 | 90 | 0.747 | 91 | 0.286 | 1.24 |
| collision_weight_field | 0.282 | 1 | 0.568 | 1 | 0.562 | 2.00 |
| integrate_velocity_apic | 0.443 | 1 | 0.313 | 1 | 0.308 | 0.70 |
| advect_particles | 0.181 | 1 | 0.277 | 1 | 0.272 | 1.50 |
| dispatch_linear_kernel_fn | 0.205 | 4 | 0.247 | 4 | 0.226 | 1.11 |
| copy/fill | 0.068 | 11 | 0.254 | 41 | 0.046 | 0.68 |
| evaluate_strain_residual | 0.117 | 45 | 0.198 | 46 | 0.000 | 0.00 |
| strain_delta_form | 0.158 | 1 | 0.110 | 1 | 0.105 | 0.66 |
| _identity_field | 0.090 | 3 | 0.176 | 3 | 0.161 | 1.78 |
| project_outside_collider | 0.092 | 1 | 0.122 | 1 | 0.117 | 1.27 |
| update_condition | 0.074 | 45 | 0.127 | 46 | 0.000 | 0.00 |
| integrate_yield_parameters | 0.078 | 1 | 0.086 | 1 | 0.081 | 1.05 |
| dispatch_bilinear_kernel_fn | 0.091 | 2 | 0.069 | 2 | 0.059 | 0.65 |
| compute_delassus_diagonal | 0.028 | 1 | 0.085 | 1 | 0.080 | 2.90 |
| vb_world_bbox | 0.000 | 0 | 0.100 | 3 | 0.085 |  |
| apply_stress_gs | 0.039 | 8 | 0.058 | 8 | 0.017 | 0.43 |
| integrate_velocity | 0.028 | 1 | 0.048 | 1 | 0.042 | 1.49 |
| wp_memtile | 0.000 | 0 | 0.066 | 4 | 0.046 |  |
| set_conditional_if_handle_kernel | 0.066 | 46 | 0.000 | 0 | 0.000 |  |
| rasterize_collider_kernel | 0.021 | 1 | 0.038 | 1 | 0.033 | 1.55 |
| _bsr_accumulate_triplet_values | 0.013 | 3 | 0.037 | 3 | 0.022 | 1.65 |
| vb_unique_scatter | 0.000 | 0 | 0.049 | 6 | 0.019 |  |
| vb_unique_count | 0.000 | 0 | 0.043 | 6 | 0.013 |  |
| vb_init_grid | 0.000 | 0 | 0.041 | 3 | 0.026 |  |
| postprocess_stress_and_strain | 0.014 | 1 | 0.026 | 1 | 0.021 | 1.55 |
| _triplet_keys_u32 | 0.000 | 0 | 0.040 | 4 | 0.020 |  |
| (72 more) | 0.392 | | 0.654 | | | |

### multi_material

L4 kernel sum 10.83 ms/frame over 1546 launches; Metal (isolated-dispatch profile) 31.53 ms/frame over 1774 launches; Metal empty command buffer 2.40 us/launch -> calibrated Metal sum 27.27 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| gs_solve_kernel_impl | 6.708 | 860 | 23.678 | 850 | 21.635 | 3.23 |
| sort/scan/reduce (native) | 1.126 | 217 | 2.558 | 420 | 1.548 | 1.38 |
| solve_subgrid_friction | 0.417 | 108 | 0.735 | 106 | 0.480 | 1.15 |
| apply_subgrid_impulse | 0.322 | 108 | 0.550 | 107 | 0.292 | 0.91 |
| collision_weight_field | 0.266 | 1 | 0.596 | 1 | 0.593 | 2.23 |
| integrate_velocity_apic | 0.311 | 1 | 0.251 | 1 | 0.248 | 0.80 |
| _tiled_sum_kernel | 0.111 | 43 | 0.357 | 42 | 0.255 | 2.29 |
| copy/fill | 0.067 | 11 | 0.317 | 45 | 0.209 | 3.14 |
| dispatch_linear_kernel_fn | 0.171 | 4 | 0.202 | 4 | 0.192 | 1.12 |
| advect_particles | 0.144 | 1 | 0.223 | 1 | 0.221 | 1.53 |
| _identity_field | 0.096 | 3 | 0.214 | 3 | 0.207 | 2.15 |
| update_particle_strains | 0.147 | 1 | 0.152 | 1 | 0.149 | 1.01 |
| strain_delta_form | 0.125 | 1 | 0.090 | 1 | 0.088 | 0.70 |
| evaluate_strain_residual | 0.055 | 22 | 0.093 | 21 | 0.042 | 0.76 |
| dispatch_bilinear_kernel_fn | 0.077 | 2 | 0.064 | 2 | 0.059 | 0.77 |
| integrate_yield_parameters | 0.059 | 1 | 0.071 | 1 | 0.068 | 1.16 |
| compute_delassus_diagonal | 0.022 | 1 | 0.083 | 1 | 0.081 | 3.65 |
| vb_world_bbox | 0.000 | 0 | 0.103 | 3 | 0.096 |  |
| apply_stress_gs | 0.038 | 8 | 0.059 | 8 | 0.040 | 1.05 |
| update_condition | 0.036 | 22 | 0.059 | 21 | 0.008 | 0.21 |
| project_outside_collider | 0.054 | 1 | 0.038 | 1 | 0.036 | 0.66 |
| wp_memtile | 0.000 | 0 | 0.065 | 4 | 0.056 |  |
| integrate_velocity | 0.021 | 1 | 0.039 | 1 | 0.036 | 1.75 |
| postprocess_stress_and_strain | 0.023 | 1 | 0.033 | 1 | 0.030 | 1.33 |
| vb_unique_scatter | 0.000 | 0 | 0.054 | 6 | 0.040 |  |
| _bsr_accumulate_triplet_values | 0.013 | 3 | 0.035 | 3 | 0.028 | 2.24 |
| rasterize_collider_kernel | 0.012 | 1 | 0.035 | 1 | 0.032 | 2.65 |
| vb_unique_count | 0.000 | 0 | 0.046 | 6 | 0.032 |  |
| vb_init_grid | 0.000 | 0 | 0.046 | 3 | 0.039 |  |
| _triplet_keys_u32 | 0.000 | 0 | 0.039 | 4 | 0.029 |  |
| (74 more) | 0.407 | | 0.651 | | | |

### twoway_coupling

L4 kernel sum 28.90 ms/frame over 1287 launches; Metal (isolated-dispatch profile) 45.85 ms/frame over 1566 launches; Metal empty command buffer 2.42 us/launch -> calibrated Metal sum 42.07 ms/frame.

| kernel (normalised) | L4 ms/frame | L4 launches | Metal ms/frame | Metal launches | Metal calibrated ms | Metal / L4 |
|---|---|---|---|---|---|---|
| sort/scan/reduce (native) | 15.398 | 190 | 9.603 | 337 | 8.788 | 0.57 |
| gs_solve_kernel_impl | 3.221 | 400 | 9.128 | 400 | 8.161 | 2.53 |
| make_bsr_mv_kernel__locals__bsr_mv_kernel | 1.336 | 51 | 5.454 | 102 | 5.207 | 3.90 |
| _bsr_accumulate_triplet_values | 0.434 | 5 | 1.921 | 5 | 1.908 | 4.40 |
| copy/fill | 0.909 | 38 | 1.425 | 138 | 1.092 | 1.20 |
| sum_body_force_chunks | 0.000 | 0 | 2.228 | 4 | 2.219 |  |
| solve_subgrid_friction | 0.440 | 50 | 1.788 | 50 | 1.667 | 3.79 |
| apply_subgrid_impulse | 0.660 | 51 | 0.962 | 51 | 0.839 | 1.27 |
| integrate_velocity_apic | 1.025 | 1 | 0.544 | 1 | 0.542 | 0.53 |
| wp_memtile | 0.000 | 0 | 1.088 | 21 | 1.038 |  |
| NodePartition___scatter_partition_indices | 0.282 | 3 | 0.776 | 3 | 0.769 | 2.73 |
| collision_weight_field | 0.344 | 1 | 0.613 | 1 | 0.610 | 1.77 |
| dispatch_linear_kernel_fn | 0.588 | 4 | 0.323 | 4 | 0.313 | 0.53 |
| _identity_field | 0.431 | 3 | 0.282 | 3 | 0.275 | 0.64 |
| _update_gradient_cholesky_blocked__locals__kernel | 0.065 | 4 | 0.619 | 4 | 0.610 | 9.41 |
| _update_gradient_cholesky_blocked_skip_unchanged__locals__kernel | 0.043 | 4 | 0.544 | 4 | 0.534 | 12.57 |
| create_narrow_phase_kernel_gjk_mpr__locals__narrow_phase_kernel_gjk_mp | 0.160 | 4 | 0.417 | 4 | 0.407 | 2.55 |
| strain_delta_form | 0.347 | 1 | 0.216 | 1 | 0.214 | 0.62 |
| _triplet_segments | 0.000 | 0 | 0.551 | 6 | 0.536 |  |
| _segment_heads | 0.000 | 0 | 0.529 | 6 | 0.515 |  |
| advect_particles | 0.278 | 1 | 0.192 | 1 | 0.190 | 0.68 |
| rle_lengths | 0.000 | 0 | 0.436 | 9 | 0.414 |  |
| integrate_yield_parameters | 0.188 | 1 | 0.166 | 1 | 0.163 | 0.87 |
| _triplet_keys_u64 | 0.000 | 0 | 0.336 | 2 | 0.331 |  |
| _transpose_columns | 0.000 | 0 | 0.317 | 1 | 0.315 |  |
| _bsr_transpose_values | 0.059 | 1 | 0.247 | 1 | 0.245 | 4.18 |
| dispatch_bilinear_kernel_fn | 0.177 | 2 | 0.110 | 2 | 0.105 | 0.60 |
| SpaceRestriction___split_vertex_element_index | 0.056 | 4 | 0.200 | 4 | 0.190 | 3.39 |
| _transpose_keys_u64 | 0.000 | 0 | 0.252 | 1 | 0.249 |  |
| _triplet_keys_u32 | 0.000 | 0 | 0.246 | 4 | 0.236 |  |
| (125 more) | 2.457 | | 4.342 | | | |

