"""SIMD-cooperative colored Gauss-Seidel sweep for Newton 1.5.2's implicit MPM on Metal.

Why: one color of the granular example holds ~1,100 strain nodes, one thread each in Newton's kernel, i.e. under one
SIMD group per GPU core; each thread then walks its strain row (8 velocity nodes for a Q1 velocity / P0 strain cell),
its compliance row and the local flow-rule solve serially, so a color pass is memory-latency bound at very low
occupancy (~40-55 us per color, runs/mpm/profile/loop_breakdown_granular.json).

Here 8 lanes of one SIMD group serve one strain node. Lane j loads strain-row entry j and its velocity and computes
its term; every lane then accumulates the terms in Newton's serial order through ``simd_shuffle``, so the local strain
is bit-identical to the serial loop. The compliance row is handled the same way in chunks of 8. The local stress solve
runs redundantly on all 8 lanes (identical inputs, identical results), and lane j applies the stress update to its own
velocity node (distinct nodes; the same arithmetic as Newton's loop). Colors keep nodes disjoint, so the sweep matches
Newton's kernel bit for bit.

Scope: Newton's ``_GaussSeidelSolver`` with a fixed strain row length of 8 (Q1 velocity basis, shared-grid
``strain_velocity_node_count == 8``); anything else keeps Newton's kernel (``eligible``).
"""
from __future__ import annotations

import numpy as np
import warp as wp
import warp.fem as fem

from newton._src.solvers.implicit_mpm import rheology_solver_kernels as rk

LANES = 8
BLOCK_DIM = 256


@wp.func_native("return metal::simd_shuffle(x, (ushort)src);")
def _shfl(x: float, src: int) -> float: ...


@wp.func
def _shfl6(v: rk.vec6, src: int):
    return rk.vec6(_shfl(v[0], src), _shfl(v[1], src), _shfl(v[2], src), _shfl(v[3], src), _shfl(v[4], src),
                   _shfl(v[5], src))


def eligible(rheology_solver) -> bool:
    if not getattr(rheology_solver.device, "is_metal", False):
        return False
    if type(rheology_solver).__name__ != "_GaussSeidelSolver":
        return False
    return rheology_solver.rheology.strain_velocity_node_count == LANES


def make_simd_gs_kernel(has_viscosity: bool, has_dilatancy: bool, has_compliance_mat: bool):
    key = ("simd_gs", LANES, has_viscosity, has_dilatancy, has_compliance_mat)
    solve_local_stress = rk.make_solve_local_stress(has_viscosity, has_dilatancy)

    @fem.cache.dynamic_kernel(suffix=key, kernel_options={"fast_math": True})
    def simd_gs_kernel(
        color: int,
        slot_count: int,
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
    ):
        tid = wp.tid()
        slot = tid // wp.static(LANES)
        lane = tid - slot * wp.static(LANES)
        simd_lane = tid & 31
        group_base = simd_lane - lane

        color_end = color_offsets[color + 1]
        for color_offset in range(color_offsets[color] + slot, color_end, slot_count):
            beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
            for tau_i in range(beg, end):
                row_beg = strain_mat_offsets[tau_i]
                if row_beg == strain_mat_offsets[tau_i + 1]:
                    if lane == 0:
                        delta_correction[tau_i] = rk.vec6(0.0)
                    continue

                # tau = rhs + sum_j B_j v_j, summed in Newton's order
                b = row_beg + lane
                u_i = strain_mat_columns[b]
                b_val = strain_mat_values[b]
                term = rk._symmetric_part_op(b_val, velocities[u_i])
                tau = local_strain_rhs[tau_i]
                for k in range(wp.static(LANES)):
                    tau += _shfl6(term, group_base + k)

                # tau += C sigma, in chunks of LANES, in Newton's order
                if wp.static(has_compliance_mat):
                    c_beg = compliance_mat_offsets[tau_i]
                    c_end = compliance_mat_offsets[tau_i + 1]
                    for c0 in range(c_beg, c_end, wp.static(LANES)):
                        cb = c0 + lane
                        c_term = rk.vec6(0.0)
                        if cb < c_end:
                            c_term = compliance_mat_values[cb] @ local_stress[compliance_mat_columns[cb]]
                        n_chunk = wp.min(c_end - c0, wp.static(LANES))
                        for k in range(wp.static(LANES)):
                            t = _shfl6(c_term, group_base + k)
                            if k < n_chunk:
                                tau += t

                delta_stress = wp.static(solve_local_stress)(
                    tau_i, tau, yield_params, strain_node_volume, delassus_diagonal, delassus_rotation, local_stress
                )
                if lane == 0:
                    local_stress[tau_i] += delta_stress
                    delta_correction[tau_i] = delta_stress
                velocities[u_i] += inv_mass_matrix[u_i] * rk._symmetric_part_transposed_op(b_val, delta_stress)

    return simd_gs_kernel


class SimdSweep:
    """Drop-in replacement for ``_GaussSeidelSolver.solve`` (one sweep over all colors)."""

    def __init__(self, rheology_solver):
        rs = rheology_solver
        rh, mo, de = rs.rheology, rs.momentum, rs.delassus_operator
        self.color_count = rh.color_offsets.shape[0] - 1
        kernel = make_simd_gs_kernel(rh.has_viscosity, rh.has_dilatancy, rh.compliance_mat.nnz > 0)
        dev = mo.velocity.device
        slot_count = dev.sm_count * 2 * (BLOCK_DIM // LANES)
        self.launch = wp.launch(
            kernel,
            dim=slot_count * LANES,
            block_dim=BLOCK_DIM,
            inputs=[
                0, slot_count, rh.color_offsets, rh.color_blocks, rh.yield_params, rh.strain_node_volume,
                rh.compliance_mat.offsets, rh.compliance_mat.columns, rh.compliance_mat.values,
                rh.strain_mat.offsets, rh.strain_mat.columns, rh.strain_mat.values.view(dtype=rk.mat13),
                de.delassus_diagonal, de.delassus_rotation, mo.inv_volume, rh.elastic_strain_delta,
                mo.velocity, rh.stress, rs.delta_stress,
            ],
            device=dev,
            record_cmd=True,
        )

    def __call__(self):
        for color in range(self.color_count):
            self.launch.set_param_at_index(0, color)
            self.launch.launch()
