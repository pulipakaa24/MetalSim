"""WS6: randomizers, annotators and writers on the GPU lift scene."""
import json
import os

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.render.tier0 import Tier0Renderer, SEG_SLOT
from metalsim.replicator import Annotators, BasicWriter, CocoWriter, KittiWriter, Randomizer

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO101 = os.path.join(ROOT, "assets", "so101", "scene_box_rl.xml")


def test_randomize_annotate_write(tmp_path):
    model = mujoco.MjModel.from_xml_path(SO101)
    n = 8
    sim = BatchSim(model, n, options=BatchSimOptions(substeps=1))
    rend = Tier0Renderer(model, n, width=128, height=128, camera="base_cam", outputs=("rgb", "depth", "seg", "normal"),
                         seg_mode=SEG_SLOT, decimate_faces=2000)
    rnd = Randomizer(rend, seed=0)
    rnd.colors(); rnd.camera(); rnd.light(); rnd.ambient(); rnd.intrinsics(); rnd.backgrounds(); rnd.materials(slots=[rend.G - 1])
    torch.mps.synchronize()
    sim.synchronize()
    v = sim.forward()
    # randomization is torch work: order the render after it, then annotate
    import metalsim.interop.torch_bridge as tb
    from metalsim.interop import warp_metal as wm
    ev = wm.SharedEvent("metal:0", "rep-test"); tb.signal_event(ev, 1); sim.wait(ev, 1)
    v = sim.forward()
    vr = rend.render(sim, v); rend.after(vr)
    ann = Annotators(rend, sim)
    rgb, seg = ann.rgb(), ann.instance_segmentation()
    boxes = ann.bounding_box_2d_tight()
    boxes3d = ann.bounding_box_3d()
    sem = ann.semantic_segmentation()
    cam = ann.camera_params()
    torch.mps.synchronize(); rend.ctx.synchronize()
    # every env differs (colors/camera/background) and boxes enclose the instance masks
    r = rgb.cpu().numpy()
    assert not np.array_equal(r[0], r[1])
    b = boxes.cpu().numpy(); s = seg.cpu().numpy()
    for e in range(n):
        for slot in range(rend.G):
            m = s[e] == slot + 1
            if m.sum() == 0:
                assert b[e, slot, 2] <= 0
                continue
            ys, xs = np.nonzero(m)
            assert b[e, slot, 0] == xs.min() and b[e, slot, 1] == ys.min()
            assert b[e, slot, 2] == xs.max() + 1 and b[e, slot, 3] == ys.max() + 1
    assert sem.max().item() <= model.nbody and (sem.cpu().numpy() > 0).sum() > 0
    bb = boxes3d.cpu().numpy()
    assert np.isfinite(bb).all() and (bb[:, :, 3:6] > 0).all()
    # writers
    slot_class = torch.as_tensor(rend.tables.semantic[:, 1])
    out = str(tmp_path)
    BasicWriter(os.path.join(out, "basic")).write({"rgb": rgb, "depth": ann.depth(), "seg": seg, "boxes2d": boxes, "camera_params": cam})
    coco = CocoWriter(os.path.join(out, "coco"), class_names={int(i): mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in range(model.nbody)})
    coco.write(rgb, boxes, seg, slot_class); coco.close()
    KittiWriter(os.path.join(out, "kitti")).write(rgb, boxes, boxes3d, slot_class, camera_params=cam)
    j = json.load(open(os.path.join(out, "coco", "annotations.json")))
    assert len(j["images"]) == n and len(j["annotations"]) > n and len(j["categories"]) >= 2
    assert len(os.listdir(os.path.join(out, "kitti", "label_2"))) == n
    assert len([f for f in os.listdir(os.path.join(out, "basic")) if f.endswith("_rgb.png")]) == n
