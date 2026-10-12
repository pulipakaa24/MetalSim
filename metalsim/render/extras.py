"""Non-geom visuals for the scene tables: what MuJoCo's own renderer draws besides geoms.

``mujoco.Renderer`` (mjv_updateScene + mjr_render) also draws, under its default ``mjvOption``:

* sites in the visible site groups (0-2), as their primitive (sphere, capsule, ellipsoid, cylinder, box, mesh);
* flexes (``mjVIS_FLEXSKIN`` and ``mjVIS_FLEXEDGE`` are on by default): a 2D flex as its smoothed surface offset by
  +-radius along the vertex normals with the boundary closed by side faces, a 3D flex as its outer shell offset by
  +radius, a 1D flex (cable) as one open cylinder of the flex radius per edge (mjv_updateActiveFlex, render_gl3.c);
* skins: linear blend skinning of the skin vertices by their bones, area-weighted vertex normals, ``inflate``
  (mjv_updateActiveSkin);
* spatial tendons: one capsule of the tendon width per segment between consecutive wrap points, half width inside a
  wrapping geom, none across a pulley (addSpatialTendonGeoms).

Sites are rigid and become ordinary instanced slots whose pose rows follow the geoms' (``SceneTables.n_pose`` rows per
env: the ``ngeom`` geom poses, then the drawn sites' ``site_xpos`` / ``site_xmat``). Flexes, skins, tendons and
user-supplied deforming meshes (``DeformableMesh``: XPBD / PhysX-port cloth, any (N, V, 3) vertex array) are dynamic
slots: their vertices are generated on the GPU each frame (``metalsim.render.dynamic_mesh``).

Every extra object gets one slot (material, colour randomization, segmentation). Segmentation in geom mode writes
``ngeom + 1 + j`` for the j-th extra object (``SceneTables.extras[j]["seg_id"]``), so geom ids are unchanged; body mode
writes the object's body + 1 (site: its body; flex: the body of its first vertex or node; skin: its first bone's body;
tendon: the body of its first site).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from metalsim.render import scene_tables as st

G_ = mujoco.mjtGeom
V_ = mujoco.mjtVisFlag


@dataclass
class DeformableMesh:
    """A deforming triangle mesh drawn from a per-env vertex array each frame (``render(..., sources={name: array})``).

    ``tris`` (F, 3) vertex indices, ``nvert`` vertices; the array is (N, nvert, 3) float32 (a Warp array on the GPU
    path, numpy on the host path). ``radius`` > 0 draws it as MuJoCo draws a 2D flex skin of that radius (two layers
    offset along the smoothed vertex normals, boundary closed); 0 draws the surface itself (two-sided)."""
    name: str
    tris: np.ndarray
    nvert: int
    rgba: tuple = (0.5, 0.5, 0.5, 1.0)
    radius: float = 0.0
    uv: np.ndarray | None = None
    body: int = 0
    material: dict | None = None      # metallic, roughness, specular, emission, reflectance (defaults as an untextured geom)

    def __post_init__(self):
        self.tris = np.asarray(self.tris, np.int64).reshape(-1, 3)
        if self.tris.size and (self.tris.min() < 0 or self.tris.max() >= self.nvert):
            raise ValueError(f"mesh {self.name!r}: triangle index outside [0, {self.nvert})")


def visual_option(visual):
    """None: MuJoCo's default mjvOption (what ``mujoco.Renderer`` draws); False: no extras (geoms only, the behaviour
    before extras existed); an ``mujoco.MjvOption``: as given."""
    if visual is False:
        return None
    if visual is None or visual is True:
        return mujoco.MjvOption()
    return visual


def _group_on(groups, g):
    return bool(groups[max(0, min(len(groups) - 1, int(g)))])


_DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0], np.float32)


def _rgba(m, matid, rgba):
    """Colour MuJoCo draws the object with (``setMaterial``): the material's rgba unless the object's own rgba differs
    from the default (0.5, 0.5, 0.5, 1) or there is no material (as ``scene_tables.geom_rgba``)."""
    rgba = np.asarray(rgba, np.float32)
    if matid >= 0 and np.array_equal(rgba, _DEFAULT_RGBA):
        return np.asarray(m.mat_rgba[matid], np.float32).copy()
    return rgba.copy()


def material_row(m, matid, rgba, atlas, *, textured=True, gtype=None, size=None):
    """(16,) material block as ``build_scene_tables`` writes for a geom (same rules), for a non-geom object."""
    row = np.zeros(16, np.float32)
    row[0:4] = _rgba(m, matid, rgba)
    if matid < 0:
        row[4:8] = (0.0, 0.6, 0.5, 0.0)
        return row
    metallic = float(m.mat_metallic[matid]) if hasattr(m, "mat_metallic") else -1.0
    roughness = float(m.mat_roughness[matid]) if hasattr(m, "mat_roughness") else -1.0
    if metallic < 0:
        metallic = 0.0
    if roughness < 0:
        roughness = 1.0 - float(m.mat_shininess[matid])
    row[4:8] = (metallic, float(np.clip(roughness, 0.02, 1.0)), float(m.mat_specular[matid]), float(m.mat_emission[matid]))
    row[15] = float(m.mat_reflectance[matid])
    texid = -1
    if m.ntex and textured:
        roles = np.asarray(m.mat_texid[matid]).reshape(-1)
        texid = int(roles[1]) if len(roles) > 1 else int(roles[0])
    if texid >= 0 and texid in atlas.rects:
        row[8:12] = atlas.rects[texid]
        row[12:14] = m.mat_texrepeat[matid]
        if m.mat_texuniform[matid] and size is not None:
            row[12:14] = m.mat_texrepeat[matid] * (np.sort(np.asarray(size, np.float64)[:3])[-2:] * 2)
        row[14] = 1.0
    return row


def _prim_tris(m, gtype, size, dataid):
    """(P, U, N) per-corner triangles of a primitive or mesh as MuJoCo's renderer draws it (``builtin_geoms``, the
    geoms' tessellation)."""
    from metalsim.render import builtin_geoms as bg
    if gtype == G_.mjGEOM_MESH:
        return bg.mesh_triangles(m, dataid) if dataid >= 0 else (None, None, None)
    r = bg.primitive(m, gtype, size)
    return r if r is not None else (None, None, None)


def _quat_mat(q):
    return st.quats_to_mats(np.asarray(q, np.float64).reshape(1, 4))[0]


def collect(m, n_envs, atlas, visual=None, meshes=(), smooth=True):
    """Extra objects of ``m`` under ``visual`` (see ``visual_option``) plus the user meshes. Returns a dict with:
    sites (site ids, in slot order), site_meshes (indexed (verts, normals, uvs, idx) per site), dyn (dynamic object
    descriptors), rows (material (16,), semantic (4,) per extra slot, sites first), extras (descriptors), vis flags."""
    vopt = visual_option(visual)
    out = {"sites": [], "site_meshes": [], "dyn": [], "mats": [], "sem": [], "extras": [], "vis": {}}
    meshes = list(meshes or ())
    if vopt is None and not meshes:
        return out
    ngeom = m.ngeom
    flags = np.asarray(vopt.flags) if vopt is not None else np.zeros(int(V_.mjNVISFLAG), np.uint8)
    out["vis"] = {"flexvert": bool(flags[V_.mjVIS_FLEXVERT]), "flexedge": bool(flags[V_.mjVIS_FLEXEDGE]),
                  "flexface": bool(flags[V_.mjVIS_FLEXFACE]), "flexskin": bool(flags[V_.mjVIS_FLEXSKIN]),
                  "flex_layer": int(vopt.flex_layer) if vopt is not None else 0}

    def add(kind, objtype, objid, name, mat, body, group, extra=None):
        j = len(out["extras"])
        body = int(body)
        root = int(m.body_rootid[body]) if 0 <= body < m.nbody else 0
        out["mats"].append(mat)
        out["sem"].append((ngeom + j, body, root, int(group)))
        d = {"kind": kind, "objtype": int(objtype), "objid": int(objid), "name": name, "seg_id": ngeom + j + 1}
        if extra:
            d.update(extra)
        out["extras"].append(d)
        return j

    if vopt is not None:
        # sites (addSiteGeoms): visible group, alpha > 0 after the material
        for i in range(m.nsite):
            if not _group_on(vopt.sitegroup, m.site_group[i]):
                continue
            matid = int(m.site_matid[i])
            if _rgba(m, matid, m.site_rgba[i])[3] == 0:
                continue
            gtype = int(m.site_type[i])
            size = np.asarray(m.site_size[i], np.float64)
            tri, uv, nrm = _prim_tris(m, gtype, size, int(m.site_dataid[i]))
            if tri is None:
                continue
            if gtype == G_.mjGEOM_MESH:      # MuJoCo places a mesh site by the mesh's own frame (mesh_pos / mesh_quat)
                did = int(m.site_dataid[i])
                R = _quat_mat(m.mesh_quat[did])
                tri = tri @ R.T + m.mesh_pos[did]
                nrm = nrm @ R.T
            out["site_meshes"].append(st._tris_to_indexed_normals(tri, uv, nrm))
            out["sites"].append(i)
            textured = gtype != G_.mjGEOM_MESH or (int(m.site_dataid[i]) >= 0 and int(m.mesh_texcoordadr[int(m.site_dataid[i])]) >= 0)
            add("site", mujoco.mjtObj.mjOBJ_SITE, i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i),
                material_row(m, matid, m.site_rgba[i], atlas, textured=textured, gtype=gtype, size=size),
                m.site_bodyid[i], m.site_group[i])
        # flexes (addFlexGeoms)
        vis = out["vis"]
        if vis["flexvert"] or vis["flexedge"] or vis["flexface"] or vis["flexskin"]:
            for f in range(m.nflex):
                if not _group_on(vopt.flexgroup, m.flex_group[f]):
                    continue
                matid = int(m.flex_matid[f])
                if _rgba(m, matid, m.flex_rgba[f])[3] == 0:
                    continue
                dim = int(m.flex_dim[f])
                faces = dim > 1 and (vis["flexskin"] or vis["flexface"])
                lines = float(m.flex_radius[f]) > 0 and not (vis["flexskin"] and faces) and (vis["flexedge"] or vis["flexvert"])
                if not faces and not lines:
                    continue
                if m.flex_interp[f]:
                    body = m.flex_nodebodyid[m.flex_nodeadr[f]]
                else:
                    body = m.flex_vertbodyid[m.flex_vertadr[f]]
                textured = int(m.flex_texcoordadr[f]) >= 0
                j = add("flex", mujoco.mjtObj.mjOBJ_FLEX, f, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, f),
                        material_row(m, matid, m.flex_rgba[f], atlas, textured=textured), body, m.flex_group[f])
                out["dyn"].append({"kind": "flex", "id": f, "extra": j, "faces": faces, "lines": lines})
        # skins (addSkinGeoms)
        if flags[V_.mjVIS_SKIN]:
            for i in range(m.nskin):
                if not _group_on(vopt.skingroup, m.skin_group[i]):
                    continue
                matid = int(m.skin_matid[i])
                if _rgba(m, matid, m.skin_rgba[i])[3] == 0 or m.skin_vertnum[i] == 0 or m.skin_facenum[i] == 0:
                    continue
                j = add("skin", mujoco.mjtObj.mjOBJ_SKIN, i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SKIN, i),
                        material_row(m, matid, m.skin_rgba[i], atlas, textured=int(m.skin_texcoordadr[i]) >= 0),
                        m.skin_bonebodyid[m.skin_boneadr[i]], m.skin_group[i])
                out["dyn"].append({"kind": "skin", "id": i, "extra": j})
        # spatial tendons (addSpatialTendonGeoms): straight capsules between wrap points
        if flags[V_.mjVIS_TENDON]:
            for i in range(m.ntendon):
                if not _group_on(vopt.tendongroup, m.tendon_group[i]):
                    continue
                adr, num = int(m.tendon_adr[i]), int(m.tendon_num[i])
                types = [int(m.wrap_type[adr + k]) for k in range(num)]
                if mujoco.mjtWrap.mjWRAP_SITE not in types or float(m.tendon_width[i]) <= 0:
                    continue                              # fixed tendon (joints): no path, nothing drawn
                matid = int(m.tendon_matid[i])
                if _rgba(m, matid, m.tendon_rgba[i])[3] == 0:
                    continue
                first_site = next(int(m.wrap_objid[adr + k]) for k in range(num) if types[k] == mujoco.mjtWrap.mjWRAP_SITE)
                j = add("tendon", mujoco.mjtObj.mjOBJ_TENDON, i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, i),
                        material_row(m, matid, m.tendon_rgba[i], atlas, textured=False),
                        m.site_bodyid[first_site], m.tendon_group[i])
                # MuJoCo Data holds up to 2 points per path element (a wrapping geom adds two tangent points)
                out["dyn"].append({"kind": "tendon", "id": i, "extra": j, "max_points": 2 * num})
    # user deformable meshes
    names = set()
    for mesh in meshes:
        if mesh.name in names:
            raise ValueError(f"duplicate deformable mesh name {mesh.name!r}")
        names.add(mesh.name)
        row = np.zeros(16, np.float32)
        row[0:4] = mesh.rgba
        mt = dict(metallic=0.0, roughness=0.6, specular=0.5, emission=0.0, reflectance=0.0)
        mt.update(mesh.material or {})
        row[4:8] = (mt["metallic"], mt["roughness"], mt["specular"], mt["emission"])
        row[15] = mt["reflectance"]
        j = add("mesh", -1, len(out["dyn"]), mesh.name, row, mesh.body, 0, {"mesh": mesh})
        out["dyn"].append({"kind": "mesh", "mesh": mesh, "extra": j})
    return out


def extend_tables(t, m, n_envs, visual=None, meshes=(), smooth=True):
    """Append the extra objects (``collect``) to scene tables built for the geoms: site meshes and their instanced
    (env, slot) rows after the geoms' (the geoms' draws and instance rows are unchanged), and one material / semantic
    row per extra slot. Returns ``t`` (modified in place)."""
    ex = collect(m, n_envs, t.atlas, visual, meshes, smooth)
    t.extras = ex["extras"]
    t.vis = ex["vis"]
    G0 = len(t.geoms)
    t.n_geom_slots = G0
    if not t.extras:
        return t
    ns = len(ex["sites"])
    t.site_ids = list(ex["sites"])
    t.n_pose = m.ngeom + ns
    # site meshes (one per site; never shared with a geom so the geoms' instance rows keep their order)
    verts, inds = [t.vertices], [t.indices]
    v_off, i_off = len(t.vertices), len(t.indices)
    site_mesh = []
    for (v, n, uv, idx) in ex["site_meshes"]:
        rec = np.zeros((len(v), 8), np.float32)
        rec[:, 0:3] = v; rec[:, 3:6] = n; rec[:, 6:8] = uv
        verts.append(rec)
        inds.append(idx.reshape(-1).astype(np.uint32) + v_off)
        site_mesh.append(len(t.meshes))
        t.meshes.append({"i_off": i_off, "i_count": int(idx.size), "radius": float(np.linalg.norm(v, axis=1).max())})
        v_off += len(v); i_off += idx.size
    t.vertices = np.concatenate(verts).astype(np.float32)
    t.indices = np.concatenate(inds).astype(np.uint32)
    first = int(t.draws[-1][2] + t.draws[-1][3]) if t.draws else 0
    rows = [t.inst_table]
    for k, mi in enumerate(site_mesh):
        slot = G0 + k
        rows.append(np.stack([np.arange(n_envs), np.full(n_envs, slot)], 1).astype(np.uint32))
        t.draws.append((t.meshes[mi]["i_off"], t.meshes[mi]["i_count"], first, n_envs))
        if t.draws_opaque is not None:
            t.draws_opaque.append((t.meshes[mi]["i_off"], t.meshes[mi]["i_count"], first, n_envs))
        first += n_envs
    t.inst_table = np.concatenate(rows).astype(np.uint32).reshape(-1, 2)
    nx = len(t.extras)
    t.materials = np.concatenate([t.materials, np.asarray(ex["mats"], np.float32).reshape(nx, 16)])
    t.semantic = np.concatenate([t.semantic, np.asarray(ex["sem"], np.int32).reshape(nx, 4)])
    t.static_geom = np.concatenate([t.static_geom, np.zeros(nx, bool)])
    if t.material_ext is not None:            # texture-mapping extension rows: the defaults (2D, no offset, unit scale)
        ext = np.zeros((nx, 8), np.float32); ext[:, 4:7] = 1.0; ext[:, 7] = 1.0
        t.material_ext = np.concatenate([t.material_ext, ext])
    if t.transparent is not None:             # extras are drawn opaque (MuJoCo blends alpha < 1; not reproduced here)
        t.transparent = np.concatenate([t.transparent, np.zeros(nx, bool)])
    if t.sized is not None:
        t.sized = np.concatenate([t.sized, np.zeros(nx, bool)])
    # instanced slots (geoms, then sites): pose row and mesh per slot
    t.slot_pose = np.concatenate([np.asarray(t.geoms, np.int64), m.ngeom + np.arange(ns)]).astype(np.int32)
    t.slot_mesh_inst = np.concatenate([np.asarray(t.geom_mesh, np.int64), np.asarray(site_mesh, np.int64)]).astype(np.int32)
    t.dyn = ex["dyn"]
    for d in t.dyn:
        d["slot"] = G0 + d["extra"]
    t.G = G0 + nx
    return t
