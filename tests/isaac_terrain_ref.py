"""Runs Isaac Lab v2.3.2's own terrain generator code (reference copies in assets/isaac/terrains/)
on the CPU without Isaac Sim, as the ground truth for metalsim.learn.isaac_terrain.

Only the Isaac Lab utilities the generator touches are stubbed (configclass, md5 hash, yaml dump,
timer, warp-mesh conversion); the generator, the sub-terrain functions, the height-field-to-mesh
conversion and the rough-terrain config are Isaac's files executed unmodified. ``TerrainGenerator``
seeds itself exactly as in Isaac: ``ROUGH_TERRAINS_CFG.seed`` is None, so the generator takes
``np.random.get_state()[1][0]`` after the env's ``configure_seed`` (np.random.seed / torch.manual_seed).
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import os
import sys
import types
from dataclasses import MISSING

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "assets", "isaac", "terrains")


def _configclass(cls):
    """Minimal stand-in for isaaclab.utils.configclass: keyword construction over the annotated
    fields (defaults deep-copied from the MRO, MISSING allowed), ``copy`` and ``to_dict``."""
    names = []
    for c in reversed(cls.__mro__):
        for n in c.__dict__.get("__annotations__", {}):
            if n not in names:
                names.append(n)

    def default(n):
        for c in cls.__mro__:
            if n in c.__dict__:
                return c.__dict__[n]
        return MISSING

    def __init__(self, **kw):
        for n in names:
            v = kw.pop(n) if n in kw else default(n)
            object.__setattr__(self, n, v if callable(v) and not isinstance(v, type) else copy.deepcopy(v))
        for n, v in kw.items():
            object.__setattr__(self, n, v)
        if hasattr(self, "__post_init__"):
            self.__post_init__()

    def to_dict(self):
        out = {}
        for n, v in self.__dict__.items():
            out[n] = v.to_dict() if hasattr(v, "to_dict") else (getattr(v, "__name__", v) if callable(v) else v)
        return out

    cls.__init__ = __init__
    cls.copy = lambda self: copy.deepcopy(self)
    cls.to_dict = to_dict
    return cls


class _Timer(contextlib.ContextDecorator):
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


_loaded = None


def load():
    """Import Isaac's terrain package from the reference copies; returns the ``isaaclab.terrains`` module."""
    global _loaded
    if _loaded is not None:
        return _loaded
    for k in [k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")]:
        del sys.modules[k]
    _stub("isaaclab").__path__ = []
    _stub("isaaclab.utils", configclass=_configclass).__path__ = []
    _stub("isaaclab.utils.dict", dict_to_md5_hash=lambda d: "nohash")
    _stub("isaaclab.utils.io", dump_yaml=lambda *a, **k: None)
    _stub("isaaclab.utils.timer", Timer=_Timer)
    _stub("isaaclab.utils.warp", convert_to_warp_mesh=lambda *a, **k: None, raycast_mesh=lambda *a, **k: None)
    tg = _stub("isaaclab.terrains")
    tg.__path__ = [os.path.abspath(ROOT)]
    hf = importlib.import_module("isaaclab.terrains.height_field")
    tm = importlib.import_module("isaaclab.terrains.trimesh")
    gen = importlib.import_module("isaaclab.terrains.terrain_generator")
    gcfg = importlib.import_module("isaaclab.terrains.terrain_generator_cfg")
    for mod in (hf, tm):
        for n in dir(mod):
            if n.endswith("Cfg"):
                setattr(tg, n, getattr(mod, n))
    tg.TerrainGenerator = gen.TerrainGenerator
    tg.TerrainGeneratorCfg = gcfg.TerrainGeneratorCfg
    tg.hf_terrains = importlib.import_module("isaaclab.terrains.height_field.hf_terrains")
    tg.mesh_terrains = importlib.import_module("isaaclab.terrains.trimesh.mesh_terrains")
    tg.hf_utils = importlib.import_module("isaaclab.terrains.height_field.utils")
    _loaded = tg
    return tg


def rough_cfg():
    load()
    return copy.deepcopy(importlib.import_module("isaaclab.terrains.config.rough").ROUGH_TERRAINS_CFG)


def isaac_rough_generator(seed=0, num_rows=10, num_cols=20):
    """Isaac's TerrainGenerator on ROUGH_TERRAINS_CFG as Isaac-Velocity-Rough-G1-v0 builds it:
    curriculum on (the env has a terrain_levels curriculum term), seed from the env seed."""
    tg = load()
    cfg = rough_cfg()
    cfg.curriculum = True
    cfg.num_rows, cfg.num_cols = num_rows, num_cols
    np.random.seed(seed)        # isaaclab.utils.seed.configure_seed, called by ManagerBasedEnv before the scene
    torch.manual_seed(seed)
    return tg.TerrainGenerator(cfg, device="cpu")
