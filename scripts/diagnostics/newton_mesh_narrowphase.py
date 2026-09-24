"""Newton's mesh / convex narrow phase (GJK/MPR, mesh-plane, triangle-contact reduction) on Metal vs CPU.

Needs the MetalSim Warp fork with Metal fixed-size arrays (wp.zeros inside kernels; Newton's
multicontact manifold builder uses them) and `thread`-qualified reference casts in @wp.func_native
snippets (Newton's float_flip). Before that fork change these modules failed to compile on Metal.

A 0.2 m box (5 kg) represented as a box primitive, a convex hull of its 8 corners, and a triangle
mesh, on a ground plane and stacked on a static box / hull. For each case and device:
  - static query: the box placed 1 mm into its support, one collide(): contact count, signed
    distances (world frame, from the body-frame contact points) and normals;
  - dynamics: dropped from 5 mm above rest, 1 s of SolverXPBD (4 iterations, 2.5 ms): resting height
    of the box centre and the final contact set.
usage: python scripts/diagnostics/newton_mesh_narrowphase.py"""
import numpy as np, warp as wp, newton
wp.config.quiet = True

H = 0.1                      # half extent [m]
CORNERS = np.array([[x, y, z] for x in (-H, H) for y in (-H, H) for z in (-H, H)], np.float32)
FACES = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3], np.int32)


def add_box(b, body, kind, **kw):
    if kind == "box":
        return b.add_shape_box(body, hx=H, hy=H, hz=H, **kw)
    mesh = newton.Mesh(CORNERS, FACES)
    return (b.add_shape_convex_hull if kind == "hull" else b.add_shape_mesh)(body, mesh=mesh, **kw)


def scene(kind, support, z):
    b = newton.ModelBuilder(); b.gravity = -9.81
    if support == "plane":
        b.add_ground_plane()
    else:                                         # static support centred at z = H
        add_box(b, -1, support, xform=wp.transform((0, 0, H), wp.quat_identity()))
    body = b.add_body(xform=wp.transform((0, 0, z), wp.quat_identity()))
    add_box(b, body, kind, cfg=newton.ModelBuilder.ShapeConfig(density=625.0))
    return b.finalize()


def contact_report(m, s, c):
    n = int(c.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0, np.nan, np.nan, np.zeros(3)
    sb = m.shape_body.numpy(); bq = s.body_q.numpy()
    s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
    p0, p1 = c.rigid_contact_point0.numpy()[:n], c.rigid_contact_point1.numpy()[:n]
    nrm = c.rigid_contact_normal.numpy()[:n]
    def world(shape, p):
        bi = sb[shape]
        if bi < 0:
            return p
        t = bq[bi]; return np.array(wp.transform_point(wp.transform(wp.vec3(*t[:3]), wp.quat(*t[3:])), wp.vec3(*p)))
    w0 = np.array([world(a, p) for a, p in zip(s0, p0)]); w1 = np.array([world(a, p) for a, p in zip(s1, p1)])
    off = c.rigid_contact_margin0.numpy()[:n] + c.rigid_contact_margin1.numpy()[:n] if c.rigid_contact_margin0 is not None else 0.0
    d = np.einsum("ij,ij->i", nrm, w1 - w0) - off
    return n, d.min(), d.max(), nrm.mean(0)


def run(kind, support, dev):
    rest = H if support == "plane" else 3 * H
    with wp.ScopedDevice(dev):
        m = scene(kind, support, rest - 0.001); s = m.state(); c = m.collide(s)
        static = contact_report(m, s, c)
        m = scene(kind, support, rest + 0.005); solver = newton.solvers.SolverXPBD(m, iterations=4)
        s0, s1 = m.state(), m.state(); ctrl = m.control(); c = m.collide(s0)
        for _ in range(400):
            s0.clear_forces(); c = m.collide(s0, c); solver.step(s0, s1, ctrl, c, 0.0025); s0, s1 = s1, s0
        z = float(s0.body_q.numpy()[0][2]); dyn = contact_report(m, s0, c)
    return static, z - rest, dyn


if __name__ == "__main__":
    print(f"{'case':22s} {'device':8s} | static (1 mm in): n, min/max signed dist [mm], mean normal | after 1 s: z - rest [mm], n, min dist [mm]")
    for kind, support in (("box", "plane"), ("hull", "plane"), ("mesh", "plane"), ("box", "box"), ("hull", "box"), ("hull", "hull")):
        for dev in ("metal:0", "cpu"):
            (n, dmin, dmax, nm), dz, (n2, d2, _, _) = run(kind, support, dev)
            print(f"{kind + ' on ' + support:22s} {dev:8s} | {n:2d}, {dmin*1e3:+7.3f} / {dmax*1e3:+7.3f}, "
                  f"({nm[0]:+.2f} {nm[1]:+.2f} {nm[2]:+.2f}) | {dz*1e3:+7.3f}, {n2:2d}, {d2*1e3:+7.3f}", flush=True)
