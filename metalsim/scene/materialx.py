"""MaterialX materials for the USD loader (UsdMtlx without UsdMtlx).

Isaac Sim reads MaterialX two ways: ``.mtlx`` documents referenced from USD (the UsdMtlx file format
plugin composes them as ``/MaterialX/Materials/<name>`` prims) and MaterialX networks written directly
as UsdShade prims (a material's ``outputs:mtlx:surface`` connected to a shader whose ``info:id`` is a
MaterialX node definition, ``ND_*``). RTX then translates the network to MDL. The PyPI ``usd-core``
wheel has no UsdMtlx, so a ``.mtlx`` arc does not compose here; this module

1. reads the document with the MaterialX package (XInclude, version upgrade, ``fileprefix``) and
   writes the UsdShade layer UsdMtlx would produce (in memory; texture paths made absolute),
2. swaps the ``.mtlx`` arcs for that layer in the stage's session layer (user files untouched),
3. flattens each MaterialX surface (``standard_surface``, ``open_pbr_surface``,
   ``UsdPreviewSurface``, ``gltf_pbr``) to the parameters our material table carries: base colour,
   one RGB texture with repeat, metallic, roughness, specular (F0 = 0.08 * specular), emission (a
   multiple of the base colour) and opacity. Everything else is listed per material in
   ``MaterialParams.unmapped`` / ``approximations`` (see docs/research/materialx_2026-09-25.md).

Entry points: ``import_materials`` (the loader hook), ``load_mtlx`` (standalone documents),
``material_params`` (one UsdShade material), ``translate_mtlx`` (document -> Sdf layer).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field

import numpy as np
from pxr import Sdf, Usd, UsdShade

SURFACE_MODELS = ("standard_surface", "open_pbr_surface", "UsdPreviewSurface", "gltf_pbr")
OPENPBR_NITS_PER_UNIT = 1000.0      # open_pbr emission_luminance (cd/m^2) that maps to emission 1
MAX_EMISSION = 10.0                 # same cap as the OmniPBR branch of the loader

_LIB = None
_LAYERS: dict = {}                  # translated layers kept alive (anonymous layers die with their last handle)


def _library():
    """MaterialX standard data libraries (node definitions and their defaults), loaded once."""
    global _LIB
    if _LIB is None:
        import MaterialX as mx
        lib = mx.createDocument()
        mx.loadLibraries(mx.getDefaultDataLibraryFolders(), mx.getDefaultDataSearchPath(), lib)
        _LIB = lib
    return _LIB


def _py(v):
    """MaterialX / Gf value -> float, np.ndarray, bool or str."""
    if hasattr(v, "asTuple"):
        return np.asarray(v.asTuple(), float)
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Sdf.AssetPath):
        return v.resolvedPath or v.path
    try:
        return np.asarray(list(v), float)
    except TypeError:
        return v


@dataclass
class MaterialParams:
    name: str
    model: str                                  # surface node category
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8, 1.0]))
    metallic: float = 0.0
    roughness: float = 0.5
    specular: float = 0.5                       # renderer: dielectric F0 = 0.08 * specular
    emission: float = 0.0                       # renderer: emitted = emission * base colour
    texture: str | None = None                  # base colour texture (absolute path)
    texrepeat: np.ndarray = field(default_factory=lambda: np.ones(2))
    unmapped: list = field(default_factory=list)        # authored features the table cannot carry
    approximations: list = field(default_factory=list)  # mapped, but not exactly


# ----------------------------------------------------------------------------------------------------
# .mtlx document -> UsdShade layer (the subset of UsdMtlx the loader needs)

_SDF_TYPES = {
    "float": "Float", "integer": "Int", "boolean": "Bool", "string": "String", "filename": "Asset",
    "color3": "Color3f", "color4": "Color4f", "vector2": "Float2", "vector3": "Float3", "vector4": "Float4",
    "matrix33": "Matrix3d", "matrix44": "Matrix4d",
}


def _sdf_type(mxtype):
    return getattr(Sdf.ValueTypeNames, _SDF_TYPES.get(mxtype, "Token"))


def _usd_value(inp, mxtype, base_dir):
    if mxtype == "filename":
        p = inp.getResolvedValueString()           # applies fileprefix
        return Sdf.AssetPath(p if not p or os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p)))
    v = inp.getValue()
    if v is None:
        return None
    if mxtype in ("color3", "vector3"):
        return tuple(v.asTuple())
    if mxtype in ("color4", "vector4", "vector2"):
        return tuple(v.asTuple())
    return v


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name) or "_"


def translate_mtlx(path: str) -> Sdf.Layer:
    """Translate a MaterialX document into an anonymous USD layer with UsdMtlx's material paths:
    ``/MaterialX/Materials/<material>`` (default prim ``/MaterialX``). Each material carries its own
    copy of the shader nodes and node graphs it uses, so a reference to a single material is
    self-contained. Cached per file and modification time."""
    import MaterialX as mx
    path = os.path.abspath(path)
    key = (path, os.path.getmtime(path))
    if key in _LAYERS:
        return _LAYERS[key]
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, path)
    doc.setDataLibrary(_library())
    base_dir = os.path.dirname(path)
    layer = Sdf.Layer.CreateAnonymous(os.path.basename(path) + ".usda")
    stage = Usd.Stage.Open(layer)
    root = stage.DefinePrim("/MaterialX"); stage.SetDefaultPrim(root)
    stage.DefinePrim("/MaterialX/Materials")

    def node_def_id(node):
        nd = node.getNodeDef()
        if nd is None:     # inexact typing (e.g. a mis-typed output): fall back to the category's definitions
            cands = _library().getMatchingNodeDefs(node.getCategory())
            nd = next((c for c in cands if c.getType() == node.getType()), cands[0] if cands else None)
        return nd.getName() if nd else f"ND_{node.getCategory()}_{node.getType()}"

    def define_node(parent_path, node, graph_prim):
        """Shader prim for a MaterialX node under ``parent_path``; inputs connected recursively."""
        sp = parent_path.AppendChild(_safe(node.getName()))
        if stage.GetPrimAtPath(sp):
            return UsdShade.Shader(stage.GetPrimAtPath(sp))
        sh = UsdShade.Shader.Define(stage, sp)
        sh.SetShaderId(node_def_id(node))
        outs = node.getOutputs() or []
        if outs:
            for o in outs:
                sh.CreateOutput(o.getName(), _sdf_type(o.getType()))
        else:
            sh.CreateOutput("out", _sdf_type(node.getType()))
        for inp in node.getInputs():
            t = inp.getType()
            ui = sh.CreateInput(inp.getName(), _sdf_type(t))
            connect_input(ui, inp, parent_path, graph_prim, node.getParent())
            if not ui.HasConnectedSource():
                v = _usd_value(inp, t, base_dir)
                if v is not None:
                    ui.Set(v)
            if inp.getColorSpace():
                ui.GetAttr().SetColorSpace(inp.getColorSpace())
        return sh

    def define_graph(mat_path, graph):
        gp = mat_path.AppendChild(_safe(graph.getName()))
        if stage.GetPrimAtPath(gp):
            return UsdShade.NodeGraph(stage.GetPrimAtPath(gp))
        ng = UsdShade.NodeGraph.Define(stage, gp)
        for inp in graph.getInputs():                       # interface inputs
            gi = ng.CreateInput(inp.getName(), _sdf_type(inp.getType()))
            v = _usd_value(inp, inp.getType(), base_dir)
            if v is not None:
                gi.Set(v)
        for out in graph.getOutputs():
            go = ng.CreateOutput(out.getName(), _sdf_type(out.getType()))
            src = graph.getNode(out.getNodeName()) if out.getNodeName() else None
            if src is not None:
                sh = define_node(gp, src, ng)
                go.ConnectToSource(sh.ConnectableAPI(), out.getOutputString() or "out")
        return ng

    def connect_input(ui, inp, parent_path, graph_prim, scope):
        if inp.getInterfaceName() and graph_prim is not None:
            ui.ConnectToSource(graph_prim.ConnectableAPI(), inp.getInterfaceName(), UsdShade.AttributeType.Input)
            return
        if inp.getNodeGraphString():
            graph = doc.getNodeGraph(inp.getNodeGraphString())
            if graph is not None:
                ng = define_graph(mat_path_cur[0], graph)
                outname = inp.getOutputString() or (graph.getOutputs()[0].getName() if graph.getOutputs() else "out")
                ui.ConnectToSource(ng.ConnectableAPI(), outname)
            return
        if inp.getNodeName():
            src = scope.getNode(inp.getNodeName()) if hasattr(scope, "getNode") else None
            if src is None:
                src = doc.getNode(inp.getNodeName())
            if src is not None:
                sh = define_node(parent_path, src, graph_prim)
                ui.ConnectToSource(sh.ConnectableAPI(), inp.getOutputString() or "out")

    mat_path_cur = [None]
    for mnode in doc.getMaterialNodes():
        mp = Sdf.Path("/MaterialX/Materials").AppendChild(_safe(mnode.getName()))
        mat_path_cur[0] = mp
        mat = UsdShade.Material.Define(stage, mp)
        mat.GetPrim().SetCustomDataByKey("mtlx:source", path)
        for inp in mnode.getInputs():
            if inp.getType() != "surfaceshader":
                continue
            if inp.getNodeName() and doc.getNode(inp.getNodeName()) is not None:
                sh = define_node(mp, doc.getNode(inp.getNodeName()), None)
                src = sh.ConnectableAPI(); outname = inp.getOutputString() or "out"
            elif inp.getNodeGraphString() and doc.getNodeGraph(inp.getNodeGraphString()) is not None:
                graph = doc.getNodeGraph(inp.getNodeGraphString())
                ng = define_graph(mp, graph)
                src = ng.ConnectableAPI(); outname = inp.getOutputString() or graph.getOutputs()[0].getName()
            else:
                continue
            mat.CreateSurfaceOutput("mtlx").ConnectToSource(src, outname)
    _LAYERS[key] = layer
    return layer


# ----------------------------------------------------------------------------------------------------
# stage preparation: compose .mtlx arcs through the translated layers

def _mtlx_arcs(prim):
    """(spec, kind, item, absolute .mtlx path) for every reference/payload to a .mtlx asset."""
    out = []
    for spec in prim.GetPrimStack():
        for kind in ("referenceList", "payloadList"):
            lst = getattr(spec, kind)
            for item in lst.GetAddedOrExplicitItems():
                ap = item.assetPath
                if ap and ap.lower().endswith(".mtlx"):
                    out.append((spec, kind, item, spec.layer.ComputeAbsolutePath(ap)))
    return out


def resolve_mtlx_references(stage: Usd.Stage) -> list:
    """Replace every ``.mtlx`` reference/payload on the stage by its translated layer (authored in the
    session layer; arcs from the root layer stack are also deleted there, arcs inside referenced
    assets stay as failed arcs next to the working one). Returns the prim paths changed."""
    todo = []
    for prim in stage.TraverseAll():
        if prim.IsInstanceProxy():
            continue
        if not (prim.HasAuthoredReferences() or prim.HasAuthoredPayloads()):
            continue
        arcs = _mtlx_arcs(prim)
        if arcs:
            todo.append((prim.GetPath(), arcs))
    if not todo:
        return []
    layer = stage.GetSessionLayer() or stage.GetRootLayer()
    root_stack = set(stage.GetLayerStack())
    for _, arcs in todo:                     # translate first: stage authoring cannot run inside a change block
        for *_, apath in arcs:
            if os.path.exists(apath):
                translate_mtlx(apath)
    with Sdf.ChangeBlock():
        for path, arcs in todo:
            ps = Sdf.CreatePrimInLayer(layer, path)
            for spec, kind, item, apath in arcs:
                if not os.path.exists(apath):
                    continue
                tl = translate_mtlx(apath)
                if spec.layer in root_stack:
                    lst = getattr(ps, kind)
                    if item not in lst.deletedItems:
                        lst.deletedItems.append(item)
                new = Sdf.Reference(tl.identifier, item.primPath, item.layerOffset)
                if new not in ps.referenceList.prependedItems:
                    ps.referenceList.prependedItems.append(new)
    return [p for p, _ in todo]


# ----------------------------------------------------------------------------------------------------
# graph flattening

@dataclass
class _Tex:
    file: str
    scale: np.ndarray                  # multiplier (broadcast to the image channels)
    bias: np.ndarray
    channel: int | None = None
    repeat: np.ndarray = field(default_factory=lambda: np.ones(2))


@dataclass
class _UV:
    repeat: np.ndarray = field(default_factory=lambda: np.ones(2))


_NO_OP_NODES = {"normalmap", "bump", "heighttonormal", "normal", "tangent", "bitangent", "position"}


def _nodedef(shader_id):
    return _library().getNodeDef(shader_id) if shader_id else None


def _category(shader_id):
    nd = _nodedef(shader_id)
    if nd is not None:
        return nd.getNodeString(), nd
    m = re.match(r"ND_(.+?)_(float|color3|color4|vector2|vector3|vector4|surfaceshader|integer|boolean)", shader_id or "")
    return (m.group(1) if m else shader_id or ""), None


class _Flattener:
    def __init__(self, notes):
        self.notes = notes

    def input(self, shader, name, default=None):
        """Evaluate a shader input: constant (np.ndarray/float/str/bool), _Tex, _UV, or ``default``."""
        inp = shader.GetInput(name)
        if inp:
            attrs = UsdShade.Utils.GetValueProducingAttributes(inp)
            if attrs:
                a = attrs[0]
                if UsdShade.Utils.GetType(a.GetName()) == UsdShade.AttributeType.Output:
                    v = self.node(a.GetPrim(), UsdShade.Utils.GetBaseNameAndType(a.GetName())[0])
                    return default if v is None else v
                v = a.Get()
                if v is not None:
                    return _py(v)
        return default

    def node(self, prim, output):
        sh = UsdShade.Shader(prim)
        if not sh:
            return None
        sid = sh.GetShaderId() or ""
        cat, nd = _category(sid)
        d = (lambda n: _py(nd.getActiveInput(n).getValue()) if nd is not None and nd.getActiveInput(n) is not None
             and nd.getActiveInput(n).getValue() is not None else None)
        out_type = nd.getType() if nd is not None else ""
        I = lambda n: self.input(sh, n, d(n))
        if cat in ("constant", "dot"):
            return I("value" if cat == "constant" else "in")
        if cat == "convert":
            return _coerce(I("in"), out_type)
        if cat in ("texcoord", "geompropvalue", "UsdPrimvarReader"):
            return _UV()
        if cat == "place2d":
            uv = I("texcoord"); sc = np.asarray(I("scale") if I("scale") is not None else (1, 1), float)
            if any(np.abs(np.atleast_1d(I(k) if I(k) is not None else 0)).sum() > 1e-6 for k in ("offset", "rotate")):
                self.notes.append("place2d offset/rotate")
            rep = uv.repeat if isinstance(uv, _UV) else np.ones(2)
            return _UV(rep / np.where(sc == 0, 1, sc))
        if cat == "UsdTransform2d":
            uv = I("in"); sc = np.asarray(I("scale") if I("scale") is not None else (1, 1), float)
            rep = uv.repeat if isinstance(uv, _UV) else np.ones(2)
            return _UV(rep * sc)
        if cat in ("image", "tiledimage", "UsdUVTexture", "gltf_image", "gltf_colorimage"):
            f = I("file")
            if not f:
                return I("default") if cat in ("image", "tiledimage") else None
            uv = I("st" if cat == "UsdUVTexture" else "texcoord")
            rep = uv.repeat.copy() if isinstance(uv, _UV) else np.ones(2)
            if cat == "tiledimage":
                rep = rep * np.asarray(I("uvtiling") if I("uvtiling") is not None else (1, 1), float)
            if cat.startswith("gltf_"):
                sc = np.asarray(I("scale") if I("scale") is not None else (1, 1), float)
                rep = rep * sc
            t = _Tex(str(f), np.ones(1), np.zeros(1), None, rep)
            if cat == "UsdUVTexture":
                t.scale = np.asarray(I("scale") if I("scale") is not None else (1, 1, 1, 1), float)
                t.bias = np.asarray(I("bias") if I("bias") is not None else (0, 0, 0, 0), float)
                ch = {"r": 0, "g": 1, "b": 2, "a": 3}.get(output)
                if ch is not None:
                    t.channel = ch; t.scale = t.scale[ch:ch + 1]; t.bias = t.bias[ch:ch + 1]
                else:
                    t.scale = t.scale[:3]; t.bias = t.bias[:3]
            elif cat == "gltf_colorimage":
                fac = I("color")
                if fac is not None:
                    t.scale = np.asarray(fac, float)[:3]
                if output == "outa":
                    t.channel = 3; t.scale = np.ones(1)
            return t
        if cat in ("multiply", "divide", "add", "subtract"):
            a, b = I("in1"), I("in2")
            if isinstance(a, (_Tex, _UV)) or isinstance(b, (_Tex, _UV)):
                x, c = (a, b) if isinstance(a, (_Tex, _UV)) else (b, a)
                if isinstance(c, (_Tex, _UV)) or c is None or isinstance(c, (str, bool)):
                    self.notes.append(f"{cat} of two textures")
                    return x
                c = np.atleast_1d(np.asarray(c, float))
                if cat == "divide" and x is a:
                    c = 1.0 / np.where(c == 0, 1, c)
                elif cat == "subtract" and x is a:
                    c = -c
                elif cat in ("divide", "subtract"):
                    self.notes.append(f"{cat} with texture as second operand")
                    return x
                if isinstance(x, _UV):
                    if cat in ("multiply", "divide"):
                        return _UV(x.repeat * (c[:2] if c.size >= 2 else c))
                    self.notes.append("texcoord offset")
                    return x
                if cat in ("multiply", "divide"):
                    return _Tex(x.file, x.scale * c, x.bias * c, x.channel, x.repeat)
                return _Tex(x.file, x.scale, x.bias + c, x.channel, x.repeat)
            if a is None or b is None:
                return None
            a, b = np.asarray(a, float), np.asarray(b, float)
            return {"multiply": lambda: a * b, "divide": lambda: a / np.where(b == 0, 1, b),
                    "add": lambda: a + b, "subtract": lambda: a - b}[cat]()
        if cat == "mix":
            fg, bg, mx_ = I("fg"), I("bg"), I("mix")
            if isinstance(mx_, (int, float)) or (isinstance(mx_, np.ndarray) and mx_.size == 1):
                w = float(np.asarray(mx_).reshape(-1)[0])
                if w >= 1 - 1e-6:
                    return fg
                if w <= 1e-6:
                    return bg
                if isinstance(fg, np.ndarray) or isinstance(fg, float):
                    if isinstance(bg, (np.ndarray, float)):
                        return w * np.asarray(fg, float) + (1 - w) * np.asarray(bg, float)
            self.notes.append("mix with a texture")
            return fg if isinstance(fg, _Tex) else bg
        if cat == "extract":
            v, k = I("in"), int(I("index") or 0)
            if isinstance(v, _Tex):
                return _Tex(v.file, v.scale[min(k, v.scale.size - 1):][:1], v.bias[min(k, v.bias.size - 1):][:1], k, v.repeat)
            return None if v is None else float(np.atleast_1d(v)[k])
        if cat in ("separate2", "separate3", "separate4"):
            v = I("in"); k = {"outr": 0, "outx": 0, "outg": 1, "outy": 1, "outb": 2, "outz": 2, "outa": 3, "outw": 3}.get(output, 0)
            if isinstance(v, _Tex):
                return _Tex(v.file, v.scale[min(k, v.scale.size - 1):][:1], v.bias[min(k, v.bias.size - 1):][:1], k, v.repeat)
            return None if v is None else float(np.atleast_1d(v)[k])
        if cat in ("combine2", "combine3", "combine4"):
            vs = [I(f"in{k}") for k in range(1, int(cat[-1]) + 1)]
            if all(isinstance(v, (float, np.ndarray)) for v in vs):
                return np.concatenate([np.atleast_1d(v) for v in vs]).astype(float)
        if cat in _NO_OP_NODES:
            return None
        self.notes.append(f"node '{cat or sid}'")
        return None


def _coerce(v, out_type):
    if not isinstance(v, (float, np.ndarray)):
        return v
    a = np.atleast_1d(np.asarray(v, float))
    n = {"float": 1, "vector2": 2, "color3": 3, "vector3": 3, "color4": 4, "vector4": 4}.get(out_type)
    if n is None or a.size == n:
        return float(a[0]) if n == 1 else a
    if a.size == 1:
        return float(a[0]) if n == 1 else np.full(n, a[0])
    if n == 1:
        return float(a[0])
    return np.concatenate([a, np.ones(n - a.size)])[:n] if a.size < n else a[:n]


def _load_image(path):
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGBA"), float) / 255.0
    except Exception:
        return None


def _scalar(v, name, default, notes_approx, image_cache):
    """Reduce an evaluated input to one float (textures -> image mean of the channel)."""
    if v is None or isinstance(v, (str, bool, _UV)):
        return default
    if isinstance(v, _Tex):
        img = image_cache.setdefault(v.file, _load_image(v.file))
        notes_approx.append(f"{name}: texture reduced to its mean")
        if img is None:
            return default
        ch = v.channel if v.channel is not None else 0
        return float(img[..., ch].mean() * v.scale.reshape(-1)[0] + v.bias.reshape(-1)[0])
    return float(np.mean(np.atleast_1d(v)))


def _color(v, default, name, notes_approx, image_cache):
    """Reduce to a constant rgb (textures -> mean colour)."""
    if isinstance(v, _Tex):
        img = image_cache.setdefault(v.file, _load_image(v.file))
        notes_approx.append(f"{name}: texture reduced to its mean colour")
        if img is None:
            return np.asarray(default, float)
        m = img[..., :3].reshape(-1, 3).mean(0) if v.channel is None else np.full(3, img[..., v.channel].mean())
        return m * _bc3(v.scale) + _bc3(v.bias)
    if v is None or isinstance(v, (str, bool, _UV)):
        return np.asarray(default, float)
    return _bc3(v)


def _bc3(v):
    a = np.atleast_1d(np.asarray(v, float))
    return np.full(3, a[0]) if a.size == 1 else a[:3]


def _luma(c):
    return float(np.dot(_bc3(c), (0.2126, 0.7152, 0.0722)))


def _f0(ior):
    return ((ior - 1.0) / (ior + 1.0)) ** 2


# per model: inputs the table cannot carry (reported when authored away from the node default)
_UNMAPPED = {
    "standard_surface": ("diffuse_roughness", "specular_anisotropy", "specular_rotation", "transmission",
                         "subsurface", "sheen", "coat", "thin_film_thickness", "thin_walled", "normal",
                         "coat_normal", "tangent"),
    "open_pbr_surface": ("base_diffuse_roughness", "specular_roughness_anisotropy", "transmission_weight",
                         "subsurface_weight", "fuzz_weight", "coat_weight", "thin_film_weight",
                         "geometry_thin_walled", "geometry_normal", "geometry_coat_normal", "geometry_tangent"),
    "UsdPreviewSurface": ("clearcoat", "normal", "displacement", "occlusion", "opacityThreshold"),
    "gltf_pbr": ("occlusion", "normal", "clearcoat", "transmission", "sheen_color", "iridescence",
                 "specular", "specular_color", "thickness"),
}


def is_materialx(material: UsdShade.Material) -> bool:
    """A material whose surface is a MaterialX network (``outputs:mtlx:surface``, or a universal
    surface connected to a MaterialX node definition)."""
    out = material.GetSurfaceOutput("mtlx")
    if out and out.HasConnectedSource():
        return True
    out = material.GetSurfaceOutput()
    if out and out.HasConnectedSource():
        attrs = UsdShade.Utils.GetValueProducingAttributes(out)
        if attrs:
            sid = UsdShade.Shader(attrs[0].GetPrim()).GetShaderId() or ""
            return sid.startswith("ND_")
    return False


def material_params(material: UsdShade.Material) -> MaterialParams | None:
    """Flatten a MaterialX material to our table's parameters; None if it is not MaterialX or its
    surface is not one of ``SURFACE_MODELS`` (e.g. a decomposed BSDF graph)."""
    if not is_materialx(material):
        return None
    out = material.GetSurfaceOutput("mtlx")
    if not (out and out.HasConnectedSource()):
        out = material.GetSurfaceOutput()
    attrs = UsdShade.Utils.GetValueProducingAttributes(out)
    if not attrs:
        return None
    sh = UsdShade.Shader(attrs[0].GetPrim())
    cat, nd = _category(sh.GetShaderId())
    name = material.GetPrim().GetName()
    if cat not in SURFACE_MODELS:      # decomposed BSDF graphs etc.: the regular importer's result stands
        return None
    graph_notes: list = []
    fl = _Flattener(graph_notes)
    p = MaterialParams(name, cat)
    cache: dict = {}

    def dflt(n):
        i = nd.getActiveInput(n) if nd is not None else None
        return _py(i.getValue()) if i is not None and i.getValue() is not None else None

    def I(n):
        return fl.input(sh, n, dflt(n))

    def S(n):
        return _scalar(I(n), n, dflt(n) if dflt(n) is not None else 0.0, p.approximations, cache)

    def C(n):
        return _color(I(n), dflt(n) if dflt(n) is not None else (1, 1, 1), n, p.approximations, cache)

    if cat == "standard_surface":
        base_in, base_w = "base_color", S("base")
        p.metallic = S("metalness"); p.roughness = S("specular_roughness")
        f0 = S("specular") * float(np.mean(C("specular_color"))) * _f0(S("specular_IOR"))
        emit = S("emission") * C("emission_color")
        p.rgba[3] = float(np.mean(C("opacity")))
    elif cat == "open_pbr_surface":
        base_in, base_w = "base_color", S("base_weight")
        p.metallic = S("base_metalness"); p.roughness = S("specular_roughness")
        f0 = S("specular_weight") * float(np.mean(C("specular_color"))) * _f0(S("specular_ior"))
        emit = S("emission_luminance") / OPENPBR_NITS_PER_UNIT * C("emission_color")
        p.rgba[3] = S("geometry_opacity")
    elif cat == "UsdPreviewSurface":
        base_in, base_w = "diffuseColor", 1.0
        p.metallic = S("metallic"); p.roughness = S("roughness")
        if int(S("useSpecularWorkflow") or 0):
            f0 = float(np.mean(C("specularColor"))); p.metallic = 0.0
            p.approximations.append("specular workflow: specularColor taken as dielectric F0")
        else:
            f0 = _f0(S("ior"))
        emit = C("emissiveColor")
        p.rgba[3] = S("opacity")
    else:  # gltf_pbr
        base_in, base_w = "base_color", 1.0
        p.metallic = S("metallic"); p.roughness = S("roughness")
        f0 = _f0(S("ior") if dflt("ior") is not None else 1.5)
        emit = C("emissive") * (S("emissive_strength") if dflt("emissive_strength") is not None else 1.0)
        p.rgba[3] = S("alpha")

    # base colour: constant, or one texture (constant factors fold into rgba, tiling into texrepeat)
    bv = I(base_in)
    if isinstance(bv, _Tex) and bv.channel is None and os.path.exists(bv.file):
        p.texture = bv.file
        p.rgba[:3] = _bc3(bv.scale) * base_w
        p.texrepeat = np.asarray(bv.repeat, float)[:2]
        if np.abs(bv.bias).max() > 1e-6:
            p.unmapped.append(f"{base_in}: texture bias")
        img = cache.setdefault(bv.file, _load_image(bv.file))
        base_lum = _luma(p.rgba[:3] * (img[..., :3].reshape(-1, 3).mean(0) if img is not None else 1.0))
    else:
        p.rgba[:3] = _color(bv, dflt(base_in) if dflt(base_in) is not None else (0.8, 0.8, 0.8), base_in,
                            p.approximations, cache) * base_w
        base_lum = _luma(p.rgba[:3])
    p.specular = float(np.clip(f0 / 0.08, 0.0, 1.0))
    if f0 / 0.08 > 1.0:
        p.approximations.append(f"specular F0 {f0:.3f} clipped to 0.08")
    e_lum = _luma(emit)
    if e_lum > 1e-6:
        p.emission = float(min(e_lum / max(base_lum, 1e-3), MAX_EMISSION))
        en = emit / max(np.linalg.norm(emit), 1e-9); bn = p.rgba[:3] / max(np.linalg.norm(p.rgba[:3]), 1e-9)
        if float(np.dot(en, bn)) < 0.99:
            p.approximations.append("emission hue replaced by the base colour (renderer emits emission * base)")
    p.roughness = float(np.clip(p.roughness, 0.0, 1.0)); p.metallic = float(np.clip(p.metallic, 0.0, 1.0))
    p.rgba = np.clip(p.rgba, 0.0, None)
    # authored features the table cannot carry
    for n in _UNMAPPED.get(cat, ()):
        inp = sh.GetInput(n)
        if not inp:
            continue
        if inp.HasConnectedSource():
            p.unmapped.append(n); continue
        v, dv = inp.Get(), dflt(n)
        if v is not None and dv is not None and not np.allclose(np.atleast_1d(_py(v)).astype(float),
                                                               np.atleast_1d(dv).astype(float)):
            p.unmapped.append(n)
    if cat in ("standard_surface", "open_pbr_surface") and p.metallic > 0:
        sc = sh.GetInput("specular_color")
        if sc and (sc.HasConnectedSource() or (sc.Get() is not None and not np.allclose(_py(sc.Get()), 1.0))):
            p.unmapped.append("metal edge tint (specular_color)")
    p.unmapped += [f"graph: {n}" for n in dict.fromkeys(graph_notes)]
    return p


def load_mtlx(path: str) -> dict:
    """Standalone ``.mtlx`` document -> {material name: MaterialParams} (unsupported surfaces omitted)."""
    stage = Usd.Stage.Open(translate_mtlx(path))
    out = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            p = material_params(UsdShade.Material(prim))
            if p is not None:
                out[prim.GetName()] = p
    return out


# ----------------------------------------------------------------------------------------------------
# loader hook

def _mujoco_texture_file(path):
    """MuJoCo decodes PNG; other formats are converted once into a cache directory."""
    if path.lower().endswith(".png"):
        return os.path.abspath(path)
    d = os.path.join(tempfile.gettempdir(), "metalsim_mtlx_tex"); os.makedirs(d, exist_ok=True)
    h = hashlib.sha1(f"{os.path.abspath(path)}:{os.path.getmtime(path)}".encode()).hexdigest()[:16]
    out = os.path.join(d, f"{h}.png")
    if not os.path.exists(out):
        from PIL import Image
        Image.open(path).convert("RGBA").save(out)
    return out


def apply_to_mjspec(spec, mm, p: MaterialParams) -> None:
    """Write flattened parameters into an MjSpec material (replacing what it had)."""
    import mujoco
    mm.rgba = [float(x) for x in p.rgba]
    mm.metallic = p.metallic
    mm.roughness = p.roughness
    mm.shininess = max(0.0, 1.0 - p.roughness)
    mm.specular = p.specular
    mm.emission = p.emission
    role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    if p.texture and os.path.exists(p.texture):
        t = spec.add_texture()
        t.name = mm.name + "_mtlx"; t.type = mujoco.mjtTexture.mjTEXTURE_2D
        t.file = _mujoco_texture_file(p.texture)
        mm.textures[role] = t.name
        mm.texrepeat = [float(x) for x in p.texrepeat]
    else:
        mm.textures[role] = ""


def stage_has_materialx(stage: Usd.Stage) -> bool:
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdShade.Material) and is_materialx(UsdShade.Material(prim)):
            return True
        if (prim.HasAuthoredReferences() or prim.HasAuthoredPayloads()) and not prim.IsInstanceProxy() and _mtlx_arcs(prim):
            return True
    return False


def import_materials(stage: Usd.Stage, spec, mat_by_path: dict, base_dir: str = ".") -> dict:
    """Loader hook: compose ``.mtlx`` arcs, then for every MaterialX material write its flattened
    parameters into the MjSpec (updating the material the regular importer made for that prim, or
    adding one and registering it in ``mat_by_path`` so geom bindings find it). Materials with an
    unsupported surface keep the regular importer's result. Returns {prim path: MaterialParams}."""
    resolve_mtlx_references(stage)
    used = {m.name for m in spec.materials}
    done = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdShade.Material):
            continue
        p = material_params(UsdShade.Material(prim))
        if p is None:
            continue
        path = prim.GetPath()
        mm = spec.material(mat_by_path[path]) if path in mat_by_path else None
        if mm is None:
            mm = spec.add_material()
            name = base = prim.GetName(); k = 2
            while name in used:
                name = f"{base}_{k}"; k += 1
            mm.name = name; used.add(name)
            mat_by_path[path] = name
        apply_to_mjspec(spec, mm, p)
        done[path] = p
    return done
