"""PhysX-style PBD particle cloth (metalsim.physics.physx_cloth): PhysX's own spring rules and partitioning,
PhysX's rest/contact-offset semantics, hanging and dropped cloth against MetalSim's XPBDSim (PhysX ships no
numeric reference: its snippets print nothing and the repository has no tests), Metal = CPU, graph = eager.
GPU tests: run through scripts/gpu_run.sh; METALSIM_DEFORMABLE_DEVICE=cpu for a dry run of the logic."""
import os

import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.physics import physx_cloth as pc

DEV = os.environ.get("METALSIM_DEFORMABLE_DEVICE", "metal:0")
needs_dev = pytest.mark.skipif(DEV.startswith("metal") and not wp.is_metal_available(), reason="needs Metal")

HANGING_XML = """<mujoco><option timestep="0.005"/><worldbody><geom type="plane" size="2 2 .1"/>
<flexcomp name="cloth" type="grid" count="21 21 1" spacing="0.05 0.05 0.05" pos="0 0 1.0" euler="90 0 0" dim="2"
          radius="0.005" mass="8.82"><edge equality="true"/><pin gridrange="0 20  20 20"/></flexcomp>
</worldbody></mujoco>"""

DROP_XML = """<mujoco><option timestep="0.005"/><worldbody><geom type="plane" size="2 2 .1"/>
<geom type="box" size="0.2 0.2 0.2" pos="0 0 0.2"/>
<flexcomp name="cloth" type="grid" count="21 21 1" spacing="0.05 0.05 0.05" pos="0 0 0.5" dim="2"
          radius="0.005" mass="8.82"><edge equality="true"/></flexcomp>
</worldbody></mujoco>"""


def _snippet_grid(nx: int, nz: int, sp: float = 0.2):
    """SnippetPBDCloth.cpp initCloth: vertices, triangles and the explicit spring list (stretch + both diagonals)."""
    idx = lambda i, j: i * nz + j  # noqa: E731
    x = np.array([(i * sp, 0.0, j * sp) for i in range(nx) for j in range(nz)], np.float32)
    springs, tris = set(), []
    for i in range(nx):
        for j in range(nz):
            if i > 0:
                springs.add(tuple(sorted((idx(i - 1, j), idx(i, j)))))
            if j > 0:
                springs.add(tuple(sorted((idx(i, j - 1), idx(i, j)))))
            if i > 0 and j > 0:
                springs.add(tuple(sorted((idx(i - 1, j - 1), idx(i, j)))))
                springs.add(tuple(sorted((idx(i - 1, j), idx(i, j - 1)))))
                tris += [(idx(i - 1, j - 1), idx(i - 1, j), idx(i, j - 1)), (idx(i - 1, j), idx(i, j - 1), idx(i, j))]
    return pc.ClothMesh(x, np.array(tris, np.int32), np.ones(len(x), np.float32)), springs


def test_cooker_reproduces_snippet_springs():
    """ExtParticleClothCooker rules on SnippetPBDCloth's mesh give exactly the snippet's hand-written stretch and
    shear springs (the quad diagonal is the longest edge of both triangles → both diagonals), plus bending."""
    mesh, ref = _snippet_grid(7, 5)
    si, sj, rest, typ = pc.cook_springs(mesh, vertical_dir=(0.0, 1.0, 0.0))
    got = {(int(a), int(b)) for a, b, t in zip(si, sj, typ) if t != pc.BENDING}
    assert got == ref
    diag = typ == pc.DIAGONAL
    np.testing.assert_allclose(rest[diag], 0.2 * np.sqrt(2.0), rtol=1e-6)
    np.testing.assert_allclose(rest[~diag & (typ != pc.BENDING)], 0.2, rtol=1e-6)
    # bending: straight-line neighbours two grid steps apart
    bend = typ == pc.BENDING
    assert bend.sum() == 7 * 3 + 5 * 5 and np.allclose(rest[bend], 0.4, rtol=1e-6)


def test_partition_invariants():
    """NpParticleClothPreProcessor semantics: 8 combined partitions, no particle twice in an original partition,
    every copy chains to a later partition or to exactly one accumulation slot, blend = 1 / (max copies + 1)."""
    mesh = pc.grid_cloth(21, 1.0)
    si, sj, _, _ = pc.cook_springs(mesh)
    order, part_end, remap, copy_end, blend = pc.partition_springs(si, sj, len(mesh.x))
    ns = len(si)
    assert len(part_end) == pc.PARTITIONS_FINAL and part_end[-1] == ns and sorted(order) == list(range(ns))
    targets = remap.tolist()
    assert len(set(targets)) == 2 * ns                           # a bijection onto chain inputs + accumulation slots
    acc = [t for t in targets if t >= 2 * ns]
    assert len(acc) == copy_end[-1]
    a, b = si[order], sj[order]
    bounds = [0] + part_end.tolist()
    part = np.searchsorted(np.array(bounds[1:]), np.arange(ns), side="right")
    for s in range(2 * ns):
        t = remap[s]
        p_here = part[s % ns]
        v_here = (a if s < ns else b)[s % ns]
        if t < 2 * ns:
            assert part[t % ns] > p_here                          # chains move forward
            assert (a if t < ns else b)[t % ns] == v_here         # same particle
    ncop = np.diff(np.concatenate([[0], copy_end]))
    assert blend == pytest.approx(1.0 / (ncop.max() + 1))


@needs_dev
def test_dropped_cloth_rests_at_physx_offsets():
    """PhysX semantics of rest and contact offsets: after 3 s the particles touching the floor sit at z = particle rest
    offset + shape rest offset (none below), those on the box top at box height + the same; the cloth is at rest."""
    cfg = pc.PhysXClothCfg()                                    # recorder values: rest 0.025, contact 0.0375
    sim = pc.PhysXClothSim(pc.grid_cloth(21, 1.0, z0=0.5), 4, obstacles=[pc.plane(), pc.box((0, 0, 0.2), (0.2, 0.2, 0.2))],
                           cfg=cfg, device=DEV)
    sim.randomize(seed=0, xy=0.03, z=0.02)
    for _ in range(600):
        sim.step()
    sim.synchronize()
    x = sim.x.numpy()
    assert np.isfinite(x).all()
    z = x[:, :, 2]
    assert z.min() > cfg.rest_offset - 2e-4                         # nothing inside the floor's rest band
    assert (np.abs(z - cfg.rest_offset) < 2e-4).sum(1).min() > 0    # and the hanging corners rest on it in every world
    top = (np.abs(x[:, :, 0]) < 0.15) & (np.abs(x[:, :, 1]) < 0.15)
    ztop = x[:, :, 2][top]
    assert ztop.min() > 0.4 + cfg.rest_offset - 2e-4 and abs(np.median(ztop) - (0.4 + cfg.rest_offset)) < 1e-3   # wrinkles lift some
    assert sim.energy_arr.numpy()[:, 1].max() < 0.05


@needs_dev
def test_contacts_follow_contact_offset():
    """A contact exists exactly when the step-start distance to the shape is within the summed contact offsets."""
    cfg = pc.PhysXClothCfg(contact_offset=0.03)
    mesh = pc.grid_cloth(11, 0.5, z0=0.0)
    rng = np.random.default_rng(1)
    n = 8
    x = mesh.x[None].repeat(n, 0)
    x[:, :, 2] = rng.uniform(0.0, 0.12, (n, len(mesh.x)))
    sim = pc.PhysXClothSim(mesh, n, obstacles=[pc.plane(contact_offset=0.02)], cfg=cfg, device=DEV, capture=False)
    sim.set_state(x)
    sim.step()
    sim.synchronize()
    active = sim.c_active.numpy()[:, :, 0].astype(bool)
    expect = x[:, :, 2] <= 0.03 + 0.02                             # detection uses the step-start positions
    assert np.array_equal(active, expect)


def _run_pair(xml: str, steps: int, cfg: "pc.PhysXClothCfg", obstacles):
    from metalsim.physics import deformable as dfm
    m = mujoco.MjModel.from_xml_string(xml)
    a = pc.PhysXClothSim(pc.mesh_from_flex(m), 1, obstacles=obstacles, cfg=cfg, device=DEV)
    b = dfm.XPBDSim(m, 1, device=DEV, cfg=dfm.XPBDCfg(substeps=cfg.iterations, stretch_compliance=1.0 / cfg.stretch_stiffness,
                                                      bend_compliance=1.0 / cfg.bend_stiffness, damping=0.0))
    for _ in range(steps):
        a.step(); b.step()
    a.synchronize(); b.synchronize()
    return a, b


@needs_dev
def test_hanging_cloth_against_xpbd():
    """1 m cloth, 441 particles of 0.02 kg, top row pinned, 3 s: settles (kinetic energy < 0.01 J), stretch within
    5 %, and hangs within 2 cm of XPBDSim's cloth of the same topology and compliance (different solver and
    bending rule: PhysX's softer effective stretch sags ~1 cm more, measured)."""
    cfg = pc.PhysXClothCfg(rest_offset=0.005, contact_offset=0.0075)
    a, b = _run_pair(HANGING_XML, 600, cfg, [pc.plane()])
    xa, xb = a.x.numpy()[0], b.x.numpy()[0]
    assert a.energy_arr.numpy()[0, 1] < 0.01
    assert np.abs(a.spring_strain()).max() < 0.05
    assert abs(xa[:, 2].min() - xb[:, 2].min()) < 0.02
    assert abs(xa[:, 2].mean() - xb[:, 2].mean()) < 0.02
    np.testing.assert_allclose(xa[:, 0], xb[:, 0], atol=0.02)     # no sideways drift


@needs_dev
def test_dropped_cloth_against_xpbd():
    """Cloth dropped over a 0.4 m box: the centre rests on the box top in both, mean height within 4 cm (measured
    3.1 cm: 0.206 vs 0.237 m; different solvers and bending rules, and XPBDSim's per-substep SDF contacts vs PhysX's
    once-per-step contact list)."""
    cfg = pc.PhysXClothCfg(rest_offset=0.005, contact_offset=0.0075)
    a, b = _run_pair(DROP_XML, 600, cfg, [pc.plane(), pc.box((0, 0, 0.2), (0.2, 0.2, 0.2))])
    xa, xb = a.x.numpy()[0], b.x.numpy()[0]
    centre = np.argmin(np.linalg.norm(xa[:, :2], axis=1))
    assert abs(xa[centre, 2] - 0.405) < 0.005 and abs(xb[centre, 2] - 0.405) < 0.005
    assert abs(xa[:, 2].mean() - xb[:, 2].mean()) < 0.04, (xa[:, 2].mean(), xb[:, 2].mean())


@pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
def test_metal_equals_cpu_and_graph_equals_eager():
    mesh = pc.grid_cloth(21, 1.0, z0=0.5)
    obs = [pc.plane(), pc.box((0, 0, 0.2), (0.2, 0.2, 0.2))]
    g = pc.PhysXClothSim(mesh, 4, obstacles=obs, device="metal:0", capture=True)
    e = pc.PhysXClothSim(mesh, 4, obstacles=obs, device="metal:0", capture=False)
    c = pc.PhysXClothSim(mesh, 4, obstacles=obs, device="cpu", capture=False)
    x = g.randomize(seed=3)
    e.set_state(x); c.set_state(x)
    for _ in range(40):                                            # free fall + first contacts
        g.step(); e.step(eager=True); c.step()
    g.synchronize(); e.synchronize()
    assert np.array_equal(g.x.numpy(), e.x.numpy())
    assert np.abs(g.x.numpy() - c.x.numpy()).max() < 1e-3        # measured 1.9e-4 m at contact onset (float32 op order)
