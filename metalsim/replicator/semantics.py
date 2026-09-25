"""Semantic labels on scene prims: the equivalent of Isaac Sim's Semantics schema.

Isaac labels USD prims (``UsdSemantics.LabelsAPI`` in Isaac Sim 5.x, the older
``Semantics.SemanticsAPI`` with ``semanticType``/``semanticData`` before that; ``rep.modify.semantics``
and ``isaacsim.core.utils.semantics.add_labels`` write them). A mesh without a label inherits the
label of its nearest labelled ancestor, and that ancestor is the *instance* the annotators report
(``instance_segmentation``, ``bounding_box_*``). ``instance_id_segmentation`` instead reports every
render prim (mesh) separately.

Here the prims are the MuJoCo bodies and geoms under the same paths ``metalsim.scene.mjcf_to_usd``
writes (``/World/<body>/<child body>/<geom>``). ``Semantics`` holds, per render slot of a renderer:

* ``slot_entity``: which labelled prim (instance) the slot belongs to, -1 if unlabelled;
* ``entities``: per instance its prim path, body id (the frame for 3-D boxes) and labels
  (``{"class": "cube", ...}``);
* the semantic id table with Isaac's conventions: id 0 ``BACKGROUND``, id 1 ``UNLABELLED``, labelled
  classes from 2 in sorted label order.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

BACKGROUND, UNLABELLED = 0, 1


def _clean(n: str) -> str:
    # same sanitising as metalsim.scene.mjcf_to_usd._name
    return "".join(c if c.isalnum() or c == "_" else "_" for c in n)


def _name(m, objtype, i, fallback):
    n = mujoco.mj_id2name(m, objtype, i)
    n = n if n else f"{fallback}_{i}"
    return _clean(n).lstrip("0123456789") or f"{fallback}_{i}"


def prim_paths(m: mujoco.MjModel) -> tuple[list[str], list[str]]:
    """USD prim paths of every body and geom, as ``mjcf_to_usd`` names them."""
    body = ["/World"] * m.nbody
    for b in range(1, m.nbody):   # MuJoCo orders parents before children
        body[b] = f"{body[int(m.body_parentid[b])]}/{_name(m, mujoco.mjtObj.mjOBJ_BODY, b, 'body')}"
    geom = [f"{body[int(m.geom_bodyid[g])]}/{_name(m, mujoco.mjtObj.mjOBJ_GEOM, g, 'geom')}" for g in range(m.ngeom)]
    return body, geom


@dataclass
class Entity:
    path: str               # prim path of the labelled prim (the instance)
    body: int               # MuJoCo body whose frame the instance is expressed in
    geoms: list             # model geom ids under it
    labels: dict            # semantic type -> label, e.g. {"class": "cube"}


@dataclass
class Semantics:
    model: mujoco.MjModel
    entities: list = field(default_factory=list)
    geom_entity: np.ndarray = None          # (ngeom,) instance index or -1
    body_paths: list = None
    geom_paths: list = None

    # -- construction ----------------------------------------------------------------------------------

    @classmethod
    def _empty(cls, m):
        bp, gp = prim_paths(m)
        return cls(m, [], -np.ones(m.ngeom, np.int64), bp, gp)

    def label_body(self, body: int, labels: dict, include_children: bool = True) -> "Semantics":
        """Label a body prim; its geoms (and, as in USD, those of unlabelled descendant bodies) become
        one instance. Geoms already owned by a more specific label keep it."""
        m = self.model
        bodies = {body}
        if include_children:
            for b in range(body + 1, m.nbody):
                if int(m.body_parentid[b]) in bodies and b != 0:
                    bodies.add(b)
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and self.geom_entity[g] < 0]
        self.entities.append(Entity(self.body_paths[body], body, geoms, dict(labels)))
        self.geom_entity[geoms] = len(self.entities) - 1
        return self

    def label_geom(self, geom: int, labels: dict) -> "Semantics":
        """Label a single geom prim (it becomes its own instance)."""
        old = self.geom_entity[geom]
        if old >= 0:
            self.entities[old].geoms.remove(geom)
        self.entities.append(Entity(self.geom_paths[geom], int(self.model.geom_bodyid[geom]), [geom], dict(labels)))
        self.geom_entity[geom] = len(self.entities) - 1
        return self

    @classmethod
    def from_rules(cls, m: mujoco.MjModel, bodies: dict | None = None, geoms: dict | None = None,
                   semantic_type: str = "class") -> "Semantics":
        """Label bodies / geoms whose name fully matches a regex: ``{"cube_.*": "cube"}``. Like
        ``rep.modify.semantics([("class", "cube")])`` applied to the matching prims. Bodies are labelled
        deepest-first so that a labelled child body stays its own instance."""
        s = cls._empty(m)
        depth = np.zeros(m.nbody, int)
        for b in range(1, m.nbody):
            depth[b] = depth[int(m.body_parentid[b])] + 1
        for g, lab in _matches(m, mujoco.mjtObj.mjOBJ_GEOM, m.ngeom, geoms or {}):
            s.label_geom(g, {semantic_type: lab})
        for b, lab in sorted(_matches(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody, bodies or {}), key=lambda t: -depth[t[0]]):
            s.label_body(b, {semantic_type: lab})
        return s

    @classmethod
    def from_body_names(cls, m: mujoco.MjModel, semantic_type: str = "class") -> "Semantics":
        """Every non-world body labelled with its own name (what ``mjcf_to_usd`` writes); world geoms
        (floor, walls) stay unlabelled."""
        s = cls._empty(m)
        for b in range(m.nbody - 1, 0, -1):
            s.label_body(b, {semantic_type: _name(m, mujoco.mjtObj.mjOBJ_BODY, b, "body")}, include_children=False)
        return s

    @classmethod
    def from_usd(cls, m: mujoco.MjModel, stage, semantic_types=("class",)) -> "Semantics":
        """Read ``UsdSemantics.LabelsAPI`` (and legacy ``Semantics.SemanticsAPI``) labels from a stage
        written by ``mjcf_to_usd`` (prim paths identify bodies and geoms)."""
        s = cls._empty(m)
        body_of = {p: b for b, p in enumerate(s.body_paths)}
        geom_of = {p: g for g, p in enumerate(s.geom_paths)}
        found_b, found_g = {}, {}
        for prim in stage.Traverse():
            labels = _usd_labels(prim, semantic_types)
            if not labels:
                continue
            p = str(prim.GetPath())
            if p in geom_of:
                found_g[geom_of[p]] = labels
            elif p in body_of and body_of[p] != 0:
                found_b[body_of[p]] = labels
        # a geom whose labels equal its body's is part of the body instance (mjcf_to_usd labels both)
        for g, lab in found_g.items():
            b = int(m.geom_bodyid[g])
            if found_b.get(b) != lab:
                s.label_geom(g, lab)
        for b in sorted(found_b, reverse=True):
            s.label_body(b, found_b[b])
        return s

    # -- tables -------------------------------------------------------------------------------------------

    def label_key(self, labels: dict, semantic_types=None) -> str:
        items = {k: v for k, v in labels.items() if semantic_types is None or k in semantic_types}
        return json.dumps(items, sort_keys=True)

    def semantic_table(self, semantic_types=("class",)):
        """(entity -> semantic id, idToLabels) with 0 BACKGROUND, 1 UNLABELLED, labelled from 2."""
        keys = sorted({self.label_key(e.labels, semantic_types) for e in self.entities
                       if self.label_key(e.labels, semantic_types) != "{}"})
        sid = {k: i + 2 for i, k in enumerate(keys)}
        ent_sem = np.array([sid.get(self.label_key(e.labels, semantic_types), UNLABELLED) for e in self.entities], np.int64)
        id_to_labels = {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}}
        for k, i in sid.items():
            id_to_labels[str(i)] = json.loads(k)
        return ent_sem, id_to_labels


def _matches(m, objtype, count, rules):
    out = []
    for i in range(count):
        n = mujoco.mj_id2name(m, objtype, i) or ""
        for pat, lab in rules.items():
            if re.fullmatch(pat, n):
                out.append((i, lab))
                break
    return out


def _usd_labels(prim, semantic_types):
    labels = {}
    try:
        from pxr import UsdSemantics
        for t in semantic_types:
            if prim.HasAPI(UsdSemantics.LabelsAPI, t):
                v = UsdSemantics.LabelsAPI(prim, t).GetLabelsAttr().Get()
                if v:
                    labels[t] = ",".join(str(x) for x in v)
    except ImportError:
        pass
    # legacy Semantics.SemanticsAPI: attributes semantic:<inst>:params:semanticType / semanticData
    for a in prim.GetAttributes():
        n = a.GetName()
        if n.startswith("semantic:") and n.endswith(":params:semanticType"):
            t = a.Get()
            d = prim.GetAttribute(n.replace("semanticType", "semanticData")).Get()
            if t in semantic_types and d and t not in labels:
                labels[t] = str(d)
    return labels
