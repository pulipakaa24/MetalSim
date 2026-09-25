"""Dataset writers: run on the CPU from a batch of annotator tensors after one synchronization.

* ``NpzWriter`` (the BasicWriter before 2026-09-25; Isaac's layout is ``basic_writer.BasicWriter``):
  per-frame PNG (rgb), NPZ (depth, normals, segmentation, boxes, camera params).
* ``CocoWriter``: COCO detection JSON (images, categories, bbox annotations from 2-D tight boxes,
  per-instance PNG masks) — Isaac Replicator's CocoWriter fields.
* ``KittiWriter``: KITTI object format (``image_2/*.png``, ``label_2/*.txt`` with 2-D box and 3-D
  dimensions/location/rotation_y from the 3-D boxes).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


def _png(path, arr):
    import imageio.v2 as iio
    iio.imwrite(path, arr)


class _Base:
    def __init__(self, out_dir, class_names=None):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.frame = 0
        self.class_names = class_names or {}

    @staticmethod
    def _cpu(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class NpzWriter(_Base):
    def write(self, ann, mask=None):
        """``ann``: dict of annotator outputs (tensors with leading env dim). ``mask``: which envs to save."""
        data = {k: self._cpu(v) for k, v in ann.items() if v is not None and not isinstance(v, dict)}
        cam = {k: self._cpu(v) for k, v in ann.get("camera_params", {}).items()} if isinstance(ann.get("camera_params"), dict) else {}
        n = next(iter(data.values())).shape[0]
        envs = range(n) if mask is None else np.flatnonzero(self._cpu(mask))
        for e in envs:
            stem = os.path.join(self.out, f"{self.frame:06d}_{e:04d}")
            if "rgb" in data:
                _png(stem + "_rgb.png", data["rgb"][e])
            np.savez_compressed(stem + ".npz", **{k: v[e] for k, v in data.items() if k != "rgb"}, **{f"camera_{k}": v[e] for k, v in cam.items()})
        self.frame += 1


class CocoWriter(_Base):
    def __init__(self, out_dir, class_names=None):
        super().__init__(out_dir, class_names)
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
        self.images, self.annotations = [], []
        self.categories = {}

    def write(self, rgb, boxes2d, seg, slot_class, mask=None):
        """rgb (N,H,W,3), boxes2d (N,G,4), seg (N,H,W) instance ids (slot+1), slot_class (G,) class id per slot."""
        rgb, boxes2d, seg, slot_class = map(self._cpu, (rgb, boxes2d, seg, slot_class))
        n = rgb.shape[0]
        envs = range(n) if mask is None else np.flatnonzero(self._cpu(mask))
        for e in envs:
            image_id = len(self.images)
            fn = f"{self.frame:06d}_{e:04d}.png"
            _png(os.path.join(self.out, "images", fn), rgb[e])
            h, w = rgb[e].shape[:2]
            self.images.append({"id": image_id, "file_name": fn, "width": int(w), "height": int(h)})
            for s in range(boxes2d.shape[1]):
                b = boxes2d[e, s]
                if b[2] <= 0:
                    continue
                cid = int(slot_class[s])
                self.categories.setdefault(cid, {"id": cid, "name": str(self.class_names.get(cid, cid))})
                m = seg[e] == s + 1
                mfn = f"{self.frame:06d}_{e:04d}_{s:03d}.png"
                _png(os.path.join(self.out, "masks", mfn), (m * 255).astype(np.uint8))
                self.annotations.append({"id": len(self.annotations), "image_id": image_id, "category_id": cid,
                                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                                         "area": float(m.sum()), "iscrowd": 0, "segmentation_mask": mfn})
        self.frame += 1

    def close(self):
        with open(os.path.join(self.out, "annotations.json"), "w") as f:
            json.dump({"images": self.images, "annotations": self.annotations,
                       "categories": sorted(self.categories.values(), key=lambda c: c["id"])}, f)


class KittiWriter(_Base):
    def __init__(self, out_dir, class_names=None):
        super().__init__(out_dir, class_names)
        for d in ("image_2", "label_2", "calib"):
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    def write(self, rgb, boxes2d, boxes3d, slot_class, camera_params=None, mask=None):
        rgb, boxes2d, boxes3d, slot_class = map(self._cpu, (rgb, boxes2d, boxes3d, slot_class))
        n = rgb.shape[0]
        envs = range(n) if mask is None else np.flatnonzero(self._cpu(mask))
        for e in envs:
            stem = f"{self.frame:06d}_{e:04d}"
            _png(os.path.join(self.out, "image_2", stem + ".png"), rgb[e])
            lines = []
            for s in range(boxes2d.shape[1]):
                b = boxes2d[e, s]
                if b[2] <= 0:
                    continue
                c, hs, q = boxes3d[e, s, :3], boxes3d[e, s, 3:6], boxes3d[e, s, 6:10]
                yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
                name = str(self.class_names.get(int(slot_class[s]), int(slot_class[s])))
                # KITTI: type truncated occluded alpha bbox(4) dimensions(h w l) location(x y z) rotation_y
                lines.append(f"{name} 0 0 0 {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} "
                             f"{2*hs[2]:.3f} {2*hs[1]:.3f} {2*hs[0]:.3f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {yaw:.4f}")
            with open(os.path.join(self.out, "label_2", stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            if camera_params is not None:
                intr = self._cpu(camera_params["intrinsics"])[e]
                P = np.array([[intr[0], 0, intr[2], 0], [0, intr[1], intr[3], 0], [0, 0, 1, 0]])
                with open(os.path.join(self.out, "calib", stem + ".txt"), "w") as f:
                    f.write("P2: " + " ".join(f"{v:.6e}" for v in P.reshape(-1)) + "\n")
        self.frame += 1
