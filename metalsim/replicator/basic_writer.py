"""Replicator ``BasicWriter`` on-disk layout (omni.replicator.core 1.12.27, as in Isaac Sim 5.1).

``write(data)`` takes Isaac's per-frame dict, ``{"annotators": {name: {render_product: {"data": ...,
<info keys>}}}, "trigger_outputs": {...}}``, and writes the same files Isaac's BasicWriter writes:

* one render product -> flat in ``output_dir``; several -> ``<rp>/<annotator>/<file>`` (or
  ``<annotator>/<rp>_<file>`` with ``use_common_output_dir``); ``_fast`` suffixes dropped;
* ``rgb_####.png``; ``normals_####.png`` (``(n * 0.5 + 0.5) * 255`` of the 4 channels);
  ``distance_to_{camera,image_plane}_####.npy``; ``semantic_segmentation_####.png`` +
  ``semantic_segmentation_labels_####.json``; ``instance_id_segmentation_####.png`` +
  ``instance_id_segmentation_mapping_####.json``; ``instance_segmentation_####.png`` +
  ``instance_segmentation_mapping_####.json`` + ``instance_segmentation_semantics_mapping_####.json``;
  ``bounding_box_{2d_tight,2d_loose,3d}_####.npy`` (structured) + ``_labels_`` / ``_prim_paths_`` JSON;
  ``motion_vectors_####.npy``; ``camera_params_####.json``; ``pointcloud_####.npy`` + ``_rgb`` /
  ``_normals`` / ``_semantic`` / ``_instance``.

Segmentation colourization follows Isaac: RGBA PNG with the JSON keyed by ``"(r, g, b, a)"``,
``(0, 0, 0, 0)`` BACKGROUND and ``(0, 0, 0, 255)`` UNLABELLED, other ids a fixed pseudo-random
palette. Uncolourized ids are written as 16-bit PNGs (Isaac writes uint32 PNGs through its own
image backend; 16 bits hold every id MetalSim scenes produce, and larger ids raise).

``write_batch(ann, ...)`` builds that dict for a batch of envs (one render product per env) from an
``IsaacAnnotators``: every annotator runs on the GPU for the whole batch, then one synchronization
and the CPU file writes.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

_ANNOTATORS = ("rgb", "bounding_box_2d_tight", "bounding_box_2d_loose", "semantic_segmentation",
               "instance_id_segmentation", "instance_segmentation", "distance_to_camera",
               "distance_to_image_plane", "bounding_box_3d", "normals", "motion_vectors", "camera_params",
               "pointcloud")


def _png(path, arr):
    import imageio.v2 as iio
    iio.imwrite(path, arr)


def palette(ids) -> np.ndarray:
    """Fixed RGBA colour per id (uint8): 0 -> (0,0,0,0), 1 -> (0,0,0,255), others hashed, alpha 255."""
    ids = np.asarray(ids, np.uint64)
    h = (ids * np.uint64(2654435761) + np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15); h = (h * np.uint64(0x2C1B3C6D)) & np.uint64(0xFFFFFFFF); h ^= h >> np.uint64(12)
    c = np.stack([(h >> np.uint64(s)) & np.uint64(255) for s in (0, 8, 16)], -1).astype(np.uint8)
    c = np.maximum(c, 16)            # never pure black, so no clash with BACKGROUND / UNLABELLED
    out = np.concatenate([c, np.full(ids.shape + (1,), 255, np.uint8)], -1)
    out[ids == 0] = (0, 0, 0, 0)
    out[ids == 1] = (0, 0, 0, 255)
    return out


def colorize(ids: np.ndarray, id_to_labels: dict):
    """(H,W) ids -> (H,W,4) uint8 image and the JSON map keyed ``"(r, g, b, a)"``."""
    uniq, inv = np.unique(ids, return_inverse=True)
    cols = palette(uniq)
    if len({tuple(c) for c in cols}) != len(cols):
        raise ValueError("palette collision")
    img = cols[inv].reshape(ids.shape + (4,))
    mapping = {str(tuple(int(x) for x in cols[i])): id_to_labels.get(str(int(u)), id_to_labels.get(int(u)))
               for i, u in enumerate(uniq)}
    return img, mapping


class BasicWriter:
    def __init__(self, output_dir: str, rgb: bool = True, bounding_box_2d_tight: bool = False,
                 bounding_box_2d_loose: bool = False, semantic_segmentation: bool = False,
                 instance_id_segmentation: bool = False, instance_segmentation: bool = False,
                 distance_to_camera: bool = False, distance_to_image_plane: bool = False,
                 bounding_box_3d: bool = False, occlusion: bool = False, normals: bool = False,
                 motion_vectors: bool = False, camera_params: bool = False, pointcloud: bool = False,
                 pointcloud_include_unlabelled: bool = False, skeleton_data: bool = False,
                 image_output_format: str = "png", colorize_semantic_segmentation: bool = True,
                 colorize_instance_id_segmentation: bool = True, colorize_instance_segmentation: bool = True,
                 colorize_depth: bool = False, frame_padding: int = 4, semantic_types=None,
                 use_common_output_dir: bool = False):
        if occlusion or skeleton_data:
            raise NotImplementedError("occlusion and skeleton_data annotators are not available in MetalSim")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.enabled = {k: v for k, v in dict(
            rgb=rgb, bounding_box_2d_tight=bounding_box_2d_tight, bounding_box_2d_loose=bounding_box_2d_loose,
            semantic_segmentation=semantic_segmentation, instance_id_segmentation=instance_id_segmentation,
            instance_segmentation=instance_segmentation, distance_to_camera=distance_to_camera,
            distance_to_image_plane=distance_to_image_plane, bounding_box_3d=bounding_box_3d, normals=normals,
            motion_vectors=motion_vectors, camera_params=camera_params, pointcloud=pointcloud).items() if v}
        self.pointcloud_include_unlabelled = pointcloud_include_unlabelled
        self.fmt = image_output_format
        self.colorize = {"semantic_segmentation": colorize_semantic_segmentation,
                         "instance_id_segmentation": colorize_instance_id_segmentation,
                         "instance_segmentation": colorize_instance_segmentation}
        self.colorize_depth = colorize_depth
        self.pad = frame_padding
        self.semantic_types = semantic_types or ["class"]
        self.use_common_output_dir = use_common_output_dir
        self._frame_id = 0
        self._sequence_id = ""

    # -- Isaac's write(data) ----------------------------------------------------------------------------------

    def _f(self, stem, ext):
        return f"{stem}_{self._sequence_id}{self._frame_id:0{self.pad}}.{ext}"

    def write(self, data: dict):
        if "annotators" not in data:     # the pre-2026-09-25 NPZ interface
            from metalsim.replicator.writers import NpzWriter
            if not hasattr(self, "_legacy"):
                self._legacy = NpzWriter(self.output_dir)
            return self._legacy.write(data)
        seq = ""
        for trig, count in data.get("trigger_outputs", {}).items():
            if "on_time" in trig:
                seq = f"{count}_{seq}"
        if seq != self._sequence_id:
            self._frame_id, self._sequence_id = 0, seq
        for name, per_rp in data["annotators"].items():
            if name.endswith("_fast"):
                name = name[:-5]
            multi = len(per_rp) > 1
            for rp, d in per_rp.items():
                if multi:
                    rel = os.path.join(name, rp) + "_" if self.use_common_output_dir else os.path.join(rp, name) + os.sep
                else:
                    rel = ""
                prefix = os.path.join(self.output_dir, rel)
                os.makedirs(os.path.dirname(prefix) or self.output_dir, exist_ok=True)
                getattr(self, "_write_" + name)(d, prefix)
        self._frame_id += 1

    def _write_rgb(self, d, p):
        _png(p + self._f("rgb", self.fmt), d["data"])

    def _write_normals(self, d, p):
        _png(p + self._f("normals", "png"), ((d["data"] * 0.5 + 0.5) * 255).astype(np.uint8))

    def _write_depth(self, name, d, p):
        np.save(p + self._f(name, "npy"), d["data"])
        if self.colorize_depth:
            x = np.asarray(d["data"], np.float64)
            fin = np.isfinite(x) & (x > 0)
            img = np.zeros(x.shape + (4,), np.uint8)
            if fin.any():
                lx = np.log(x[fin]); lo, hi = lx.min(), lx.max()
                v = (255 * (1 - (lx - lo) / max(hi - lo, 1e-9))).astype(np.uint8)
                img[fin] = np.stack([v, v, v, np.full_like(v, 255)], -1)
            _png(p + self._f(name, "png"), img)

    def _write_distance_to_camera(self, d, p):
        self._write_depth("distance_to_camera", d, p)

    def _write_distance_to_image_plane(self, d, p):
        self._write_depth("distance_to_image_plane", d, p)

    def _write_seg(self, name, d, p, maps):
        ids = np.asarray(d["data"]).reshape(np.asarray(d["data"]).shape[:2])
        if self.colorize[name]:
            img, mapping = colorize(ids, maps[0][1])
            _png(p + self._f(name, "png"), img)
            maps = [(maps[0][0], mapping)] + [(stem, {str(tuple(int(c) for c in palette([int(k)])[0])): v for k, v in m.items()
                                                      if int(k) in set(np.unique(ids).tolist())}) for stem, m in maps[1:]]
        else:
            if ids.max(initial=0) >= 1 << 16:
                raise ValueError("ids exceed 16 bits; use colorized output")
            _png(p + self._f(name, "png"), ids.astype(np.uint16))
            maps = [(stem, {str(k): v for k, v in m.items()}) for stem, m in maps]
        for stem, m in maps:
            with open(p + self._f(stem, "json"), "w") as f:
                json.dump(m, f)

    def _write_semantic_segmentation(self, d, p):
        self._write_seg("semantic_segmentation", d, p, [("semantic_segmentation_labels", d["idToLabels"])])

    def _write_instance_id_segmentation(self, d, p):
        self._write_seg("instance_id_segmentation", d, p, [("instance_id_segmentation_mapping", d["idToLabels"])])

    def _write_instance_segmentation(self, d, p):
        self._write_seg("instance_segmentation", d, p, [("instance_segmentation_mapping", d["idToLabels"]),
                                                        ("instance_segmentation_semantics_mapping", d["idToSemantics"])])

    def _write_motion_vectors(self, d, p):
        np.save(p + self._f("motion_vectors", "npy"), d["data"])

    def _write_bbox(self, kind, d, p):
        np.save(p + self._f(f"bounding_box_{kind}", "npy"), d["data"])
        with open(p + self._f(f"bounding_box_{kind}_labels", "json"), "w") as f:
            json.dump({str(k): v for k, v in d["idToLabels"].items()}, f)
        with open(p + self._f(f"bounding_box_{kind}_prim_paths", "json"), "w") as f:
            json.dump(list(d["primPaths"]), f)

    def _write_bounding_box_2d_tight(self, d, p):
        self._write_bbox("2d_tight", d, p)

    def _write_bounding_box_2d_loose(self, d, p):
        self._write_bbox("2d_loose", d, p)

    def _write_bounding_box_3d(self, d, p):
        self._write_bbox("3d", d, p)

    def _write_camera_params(self, d, p):
        with open(p + self._f("camera_params", "json"), "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v) for k, v in d.items()}, f)

    def _write_pointcloud(self, d, p):
        np.save(p + self._f("pointcloud", "npy"), d["data"])
        np.save(p + self._f("pointcloud_rgb", "npy"), np.asarray(d["pointRgb"]).reshape(-1, 4))
        np.save(p + self._f("pointcloud_normals", "npy"), np.asarray(d["pointNormals"]).reshape(-1, 4))
        np.save(p + self._f("pointcloud_semantic", "npy"), d["pointSemantic"])
        np.save(p + self._f("pointcloud_instance", "npy"), d["pointInstance"])

    # -- batched producer from IsaacAnnotators ----------------------------------------------------------------

    def collect(self, ann, envs=None, prev_state=None, rp_names=None, occlusion: bool = True) -> dict:
        """Run the enabled annotators on the GPU for the batch and return Isaac's frame dict for the
        chosen envs (one render product each, named ``rp_names[e]`` or ``env_####``)."""
        n = ann.n
        envs = list(range(n)) if envs is None else [int(e) for e in envs]
        name = (lambda e: rp_names[e]) if rp_names else (lambda e: f"env_{e:04d}")
        en = self.enabled
        gpu = {}
        if "rgb" in en:
            gpu["rgb"] = ann.rgb()
        if "normals" in en:
            gpu["normals"] = ann.normals()
        if "distance_to_camera" in en:
            gpu["distance_to_camera"] = ann.distance_to_camera()
        if "distance_to_image_plane" in en:
            gpu["distance_to_image_plane"] = ann.distance_to_image_plane()
        infos = {}
        for k in ("semantic_segmentation", "instance_segmentation", "instance_id_segmentation"):
            if k in en:
                gpu[k], infos[k] = getattr(ann, k)()
        if "motion_vectors" in en:
            if prev_state is None:
                raise ValueError("motion_vectors needs prev_state")
            gpu["motion_vectors"] = ann.motion_vectors(prev_state)
        boxes = {}
        if "bounding_box_2d_tight" in en:
            boxes["bounding_box_2d_tight"] = ann.bounding_box_2d_tight(occlusion=occlusion)
        if "bounding_box_2d_loose" in en:
            boxes["bounding_box_2d_loose"] = ann.bounding_box_2d_loose(occlusion=occlusion)
        if "bounding_box_3d" in en:
            boxes["bounding_box_3d"] = ann.bounding_box_3d(occlusion=occlusion)
        cam = ann.camera_params() if "camera_params" in en else None
        pc = ann.pointcloud(self.pointcloud_include_unlabelled) if "pointcloud" in en else None
        torch.mps.synchronize()
        sel = torch.as_tensor(envs, device=ann.dev)
        cpu = {k: v[sel].cpu().numpy() for k, v in gpu.items()}
        out = {"annotators": {}, "trigger_outputs": {}}
        for k, arr in cpu.items():
            per = {}
            for i, e in enumerate(envs):
                d = {"data": arr[i]}
                if k in infos:
                    d.update(infos[k])
                per[name(e)] = d
            out["annotators"][k] = per
        for k, b in boxes.items():
            out["annotators"][k] = {}
            for e in envs:
                rec, info = ann.records(k, b, e)
                out["annotators"][k][name(e)] = {"data": rec, **info}
        if cam is not None:
            c = {k: (v[sel].cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in cam.items()}
            out["annotators"]["camera_params"] = {
                name(e): {k: (v[i].reshape(-1) if isinstance(v, np.ndarray) and v.ndim == 3 else (v[i] if isinstance(v, np.ndarray) else v))
                          for k, v in c.items()} for i, e in enumerate(envs)}
        if pc is not None:
            host = {k: v[sel].cpu().numpy() for k, v in pc.items()}
            out["annotators"]["pointcloud"] = {}
            for i, e in enumerate(envs):
                m = host["valid"][i]
                out["annotators"]["pointcloud"][name(e)] = {
                    "data": host["points"][i][m].astype(np.float32),
                    "pointRgb": host["rgb"][i][m].reshape(-1) if "rgb" in host else np.zeros(0, np.uint8),
                    "pointNormals": host["normals"][i][m].astype(np.float32).reshape(-1),
                    "pointSemantic": host["semantic"][i][m].astype(np.uint32),
                    "pointInstance": host["instance"][i][m].astype(np.uint32)}
        return out

    def write_batch(self, ann, envs=None, prev_state=None, rp_names=None, occlusion: bool = True):
        self.write(self.collect(ann, envs, prev_state, rp_names, occlusion))
