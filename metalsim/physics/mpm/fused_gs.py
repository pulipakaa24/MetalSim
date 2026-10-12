"""Whole rheology solve in one Metal dispatch: Newton 1.5.2's colored Gauss-Seidel loop in a persistent threadgroup.

Why: on the M4 Max a Gauss-Seidel iteration of the granular example is 8 dependent color dispatches plus a contact
dispatch of ~1,100 strain nodes each, and each dispatch costs ~40 us of latency regardless of its size, so the strain
solve (90 % of a frame) is latency-bound (runs/mpm/profile/loop_breakdown_granular.json). Here one threadgroup runs
every iteration, color and contact pass with threadgroup barriers in between, evaluates the residual with tile
reductions, and applies Newton's conditional-graph stopping rule (check every 5 iterations, counting from 1, stop when
converged or past ``max_iterations``) on the device: one dispatch and one host read per solve.

It calls Newton's own per-node functions (``make_compute_local_strain``, ``make_solve_local_stress``,
``make_apply_stress_delta``, ``solve_coulomb_isotropic``) with the same fast-math option as Newton's kernel, and a
color's blocks touch disjoint nodes, so the iterates match Newton's dispatch-per-color loop. The residual's sum
order differs from Newton's tiled reduction, which can only move a stop decision that sits within float rounding of
the tolerance.

Scope: the plain Gauss-Seidel solver with nodal contact, no rigid coupling operator, one environment. Everything else
falls back to the dispatch-per-color loop (``eligible``).
"""
from __future__ import annotations

import numpy as np
import warp as wp
import warp.fem as fem

from newton._src.solvers.implicit_mpm import rheology_solver_kernels as rk
from newton._src.solvers.implicit_mpm.contact_solver_kernels import solve_coulomb_isotropic

BLOCK = 1024
GRANULARITY = 5


@wp.func_native("metal::threadgroup_barrier(metal::mem_flags::mem_device | metal::mem_flags::mem_threadgroup);")
def _tg_barrier(): ...


def eligible(rheology_solver, contact_solver, device, max_color_blocks: int) -> bool:
    if not getattr(device, "is_metal", False):
        return False
    if type(rheology_solver).__name__ != "_GaussSeidelSolver" or type(contact_solver).__name__ != "_NodalContactSolver":
        return False
    if contact_solver.collision.rigidity_operator is not None:
        return False
    if rheology_solver.rheology.strain_environment_offsets is not None:
        return False
    sizes = np.diff(rheology_solver.rheology.color_offsets.numpy())
    return int(sizes.max(initial=0)) <= max_color_blocks


def make_fused_kernel(has_viscosity: bool, has_dilatancy: bool, has_compliance_mat: bool, strain_velocity_node_count: int):
    key = ("fused", BLOCK, has_viscosity, has_dilatancy, has_compliance_mat, strain_velocity_node_count)
    compute_local_strain = rk.make_compute_local_strain(has_compliance_mat, strain_velocity_node_count)
    solve_local_stress = rk.make_solve_local_stress(has_viscosity, has_dilatancy)
    apply_stress_delta = rk.make_apply_stress_delta(strain_velocity_node_count)

    @fem.cache.dynamic_kernel(suffix=key, kernel_options={"fast_math": True})
    def fused_gs_solve(
        max_chunks: int,
        max_iterations: int,
        tol_sq: float,
        l2_scale_sq: float,
        color_count: int,
        color_offsets: wp.array[int],
        color_blocks: wp.array2d[int],
        yield_params: wp.array[rk.YieldParamVec],
        strain_node_volume: wp.array[float],
        compliance_mat_offsets: wp.array[int],
        compliance_mat_columns: wp.array[int],
        compliance_mat_values: wp.array[rk.mat66],
        strain_mat_offsets: wp.array[int],
        strain_mat_columns: wp.array[int],
        strain_mat_values: wp.array[rk.mat13],
        delassus_diagonal: wp.array[rk.vec6],
        delassus_rotation: wp.array[rk.mat55],
        inv_mass_matrix: wp.array[float],
        local_strain_rhs: wp.array[rk.vec6],
        velocities: wp.array[wp.vec3],
        local_stress: wp.array[rk.vec6],
        delta_correction: wp.array[rk.vec6],
        collider_friction: wp.array[float],
        collider_adhesion: wp.array[float],
        collider_normals: wp.array[wp.vec3],
        collider_inv_mass: wp.array[float],
        collider_velocities: wp.array[wp.vec3],
        impulse: wp.array[wp.vec3],
        delta_impulse: wp.array[wp.vec3],
        iteration_and_condition: wp.array[int],
        residual_out: wp.array[float],
    ):
        tid = wp.tid()
        n_strain = local_stress.shape[0]
        n_contact = impulse.shape[0]

        for _chunk in range(max_chunks):
            if iteration_and_condition[1] == 0:
                return

            for _k in range(wp.static(GRANULARITY)):
                # nodal contact (solve_nodal_friction)
                for i in range(tid, n_contact, wp.static(BLOCK)):
                    friction_coeff = collider_friction[i]
                    if friction_coeff >= 0.0:
                        n = collider_normals[i]
                        u0 = velocities[i] - collider_velocities[i]
                        w = inv_mass_matrix[i] + collider_inv_mass[i]
                        u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)
                        delta_lambda = (u - u0) / w
                        delta_impulse[i] = delta_lambda
                        impulse[i] += delta_lambda
                        velocities[i] += inv_mass_matrix[i] * delta_lambda
                _tg_barrier()

                # colored Gauss-Seidel sweep (gs_solve_kernel_impl, one color at a time)
                for color in range(color_count):
                    color_end = color_offsets[color + 1]
                    for color_offset in range(color_offsets[color] + tid, color_end, wp.static(BLOCK)):
                        beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
                        for tau_i in range(beg, end):
                            if strain_mat_offsets[tau_i] == strain_mat_offsets[tau_i + 1]:
                                delta_correction[tau_i] = rk.vec6(0.0)
                                continue
                            local_strain = wp.static(compute_local_strain)(
                                tau_i, compliance_mat_offsets, compliance_mat_columns, compliance_mat_values,
                                strain_mat_offsets, strain_mat_columns, strain_mat_values, local_strain_rhs,
                                velocities, local_stress,
                            )
                            delta_stress = wp.static(solve_local_stress)(
                                tau_i, local_strain, yield_params, strain_node_volume, delassus_diagonal,
                                delassus_rotation, local_stress,
                            )
                            local_stress[tau_i] += delta_stress
                            delta_correction[tau_i] = delta_stress
                            wp.static(apply_stress_delta)(
                                tau_i, delta_stress, strain_mat_offsets, strain_mat_columns, strain_mat_values,
                                inv_mass_matrix, velocities,
                            )
                    _tg_barrier()

            # residual (evaluate_strain_residual + squared norm / max), then update_condition
            s = float(0.0)
            m = float(0.0)
            for tau in range(tid, n_strain, wp.static(BLOCK)):
                d = wp.cw_mul(rk._world_to_local(delta_correction[tau], delassus_rotation[tau]), delassus_diagonal[tau])
                r = wp.length_sq(d)
                s += r
                m = wp.max(m, r)
            total = wp.tile_extract(wp.tile_sum(wp.tile(s)), 0)
            peak = wp.tile_extract(wp.tile_max(wp.tile(m)), 0)
            if tid == 0:
                cur_it = iteration_and_condition[0] + wp.static(GRANULARITY)
                converged = total < tol_sq * l2_scale_sq and peak < tol_sq
                stop = converged or cur_it > max_iterations
                iteration_and_condition[0] = cur_it
                iteration_and_condition[1] = wp.where(stop, 0, 1)
                residual_out[0] = total
                residual_out[1] = peak
            _tg_barrier()

    return fused_gs_solve


def run(rheology_solver, contact_solver, max_iterations: int, tolerance: float, l2_tolerance_scale: float,
        iteration_and_condition: wp.array, residual_out: wp.array):
    """Run the whole solve; ``iteration_and_condition`` must hold ``[1, 1]`` and ``residual_out`` 2 floats."""
    rh = rheology_solver.rheology
    mo = rheology_solver.momentum
    de = rheology_solver.delassus_operator
    co = contact_solver.collision
    kernel = make_fused_kernel(rh.has_viscosity, rh.has_dilatancy, rh.compliance_mat.nnz > 0,
                               rh.strain_velocity_node_count)
    max_chunks = max_iterations // GRANULARITY + 2
    wp.launch(
        kernel,
        dim=BLOCK,
        block_dim=BLOCK,
        inputs=[
            max_chunks, max_iterations, tolerance * tolerance, float(l2_tolerance_scale) ** 2,
            rh.color_offsets.shape[0] - 1, rh.color_offsets, rh.color_blocks, rh.yield_params, rh.strain_node_volume,
            rh.compliance_mat.offsets, rh.compliance_mat.columns, rh.compliance_mat.values,
            rh.strain_mat.offsets, rh.strain_mat.columns, rh.strain_mat.values.view(dtype=rk.mat13),
            de.delassus_diagonal, de.delassus_rotation, mo.inv_volume, rh.elastic_strain_delta,
            mo.velocity, rh.stress, rheology_solver.delta_stress,
            co.collider_friction, co.collider_adhesion, co.collider_normals, contact_solver.collider_inv_mass,
            co.collider_velocities, co.collider_impulse, contact_solver.delta_impulse,
            iteration_and_condition, residual_out,
        ],
        device=mo.velocity.device,
    )
