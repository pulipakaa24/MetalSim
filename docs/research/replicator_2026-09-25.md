# Isaac Sim Replicator parity: annotators, randomizers, writers, semantics (2026-09-25)

Scope: what Isaac Sim 5.x's Replicator (`omni.replicator.core`) and Isaac Lab's event terms provide
for synthetic data and domain randomization, what MetalSim has, and the gap list. Low priority
workstream (WS6).

## Sources

- **[WHL]** `omni.replicator.core` **1.12.27** as bundled with Isaac Sim 5.1, read from NVIDIA's
  wheel `https://pypi.nvidia.com/isaacsim-extscache-kit/isaacsim_extscache_kit-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl`
  (`omni/replicator/core/scripts/annotators_default.py`, `writers_default/basicwriter.py`, `kitti.py`,
  `coco.py`, `tools.py`, `randomizer.py`, `modify.py`, `distribution.py`, `trigger.py`, `physics.py`).
- **[API]** https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/latest/source/extensions/omni.replicator.core/docs/API.html
- **[OLD]** annotator details page (archived): http://web.archive.org/web/20250327182255/https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html
- **[VIS]** https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/programmatic_visualization.html
- **[IL]** Isaac Lab, https://github.com/isaac-sim/IsaacLab (`source/isaaclab/isaaclab/envs/mdp/events.py`,
  `sensors/camera/camera.py`, `tiled_camera.py`, `camera_cfg.py`; v2.3.2 has the same event functions)
- **[ILDOC]** https://isaac-sim.github.io/IsaacLab/main/source/how-to/save_camera_output.html
- **[SEM]** https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/extensions/isaacsim.core.utils/python/impl/semantics.py
- **[WR]** https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/extensions/isaacsim.replicator.writers/python/scripts/writers/__init__.py
- **[TUT]** https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_getting_started.html
- **[PAIDF]** https://github.com/NVIDIA/paidf-simulation/blob/main/README.md (colorized id keys)

## 1. Annotators (Isaac)

| annotator | output | conventions | source |
|---|---|---|---|
| `rgb` (LdrColor) | uint8 (H,W,4) RGBA | | [API] |
| `distance_to_image_plane` | float32 (H,W) m | docs say "0 represents infinity", but NVIDIA's own code treats empty pixels as +inf (`nan_to_num(posinf=0)` in [VIS] and KittiWriter); Isaac Lab tests `torch.isinf` and has `depth_clipping_behavior` (default "none") | [API], [VIS], [WHL kitti.py], [IL camera.py] |
| `distance_to_camera` | float32 (H,W) m | as above; Isaac Lab sets values beyond far clip to inf | [API], [IL] |
| `normals` | float32 (H,W,4), xyz + unused w | frame **not stated** in the docs (Isaac Lab: "local surface normal vectors"); the RTX normal AOV is a world-space normal as far as we can tell (unverified) | [API], [IL] |
| `motion_vectors` | float32 (H,W,4) | image space; x positive for motion to the **left**, y positive for motion **up**; z, w unused; units not stated | [WHL annotators_default.py] |
| `semantic_segmentation` | uint32 (H,W) ids, or RGBA uint8 when `colorize=True` | `idToLabels` (id or `"(r, g, b, a)"` -> labels); labels inherited from ancestors, comma-joined | [API] |
| `instance_segmentation(_fast)` | uint32 (H,W) | descends to the lowest *labelled* prim; `idToLabels` id -> prim path, `idToSemantics` id -> labels | [API], [OLD] |
| `instance_id_segmentation(_fast)` | uint32 (H,W) | descends to leaf (render) prims regardless of labels; `idToLabels` id -> path | [OLD] |
| id conventions | | 0 `{"class":"BACKGROUND"}`, 1 `{"class":"UNLABELLED"}`; colorized keys `"(0, 0, 0, 0)"` / `"(0, 0, 0, 255)"` | [API], [PAIDF] |
| `bounding_box_2d_tight` / `_loose` | structured `semanticId<u4, x_min<i4, y_min<i4, x_max<i4, y_max<i4, occlusionRatio<f4` | tight = visible pixels only, fully occluded dropped; loose = whole object regardless of occlusion; info `idToLabels` (bbox-local semantic id -> labels), `bboxIds`, `primPaths`; occlusionRatio 0 visible .. 1 occluded, -1 for multi-mesh prims. **Whether `x_max` is inclusive is not documented** (Isaac's CocoWriter uses `x_max - x_min` as width) | [WHL], [API] |
| `bounding_box_3d` | structured `semanticId<u4, x_min..z_max<f4, transform<f4 (4,4), occlusionRatio<f4` | extents in the prim's local frame; returned regardless of occlusion; transform shown as a translation *row* (USD row-vector); the docs disagree on its direction ("local to world" [OLD] vs "World to local" docstring [WHL]) | [API], [OLD], [WHL] |
| `pointcloud` | (P,3) float32 world | info `pointRgb` (P*4 u8), `pointNormals` (P*4 f32), `pointSemantic`, `pointInstance`; labelled prims only unless `includeUnlabelled` | [API] |
| `skeleton_data` | dict | `numSkeletons`, `globalTranslations`, `localRotations`, `jointOcclusions`, `inView`, ... (animated skeletal meshes) | [API] |
| `camera_params` | dict | `cameraFocalLength`, `cameraFocusDistance`, `cameraFStop`, `cameraAperture`, `cameraApertureOffset`, `renderProductResolution`, `cameraModel`, `cameraViewTransform` (16 floats), `cameraProjection` (16 floats), `cameraNearFar`, `metersPerSceneUnit`, fisheye keys | [API] |
| `occlusion` | undocumented dtype | written as `occlusion_*.npy` | [WHL basicwriter.py] |

Isaac Lab's cameras (`Camera`, `TiledCamera`) expose `rgb`, `rgba`, `distance_to_camera`,
`distance_to_image_plane` (alias `depth`), `normals`, `motion_vectors`, `semantic_segmentation`,
`instance_segmentation_fast`, `instance_id_segmentation_fast` as (B,H,W,C) tensors, with the
`idToLabels`/`idToSemantics` dicts in `data.info[cam][name]`; no box annotators [IL camera.py, tiled_camera.py].

## 2. Writers

**BasicWriter** (1.12.27 source, [WHL basicwriter.py]). Parameters: `output_dir`, `semantic_types`
(default `["class"]`), one flag per annotator (`rgb`, `bounding_box_2d_tight`, `bounding_box_2d_loose`,
`semantic_segmentation`, `instance_id_segmentation`, `instance_segmentation`, `distance_to_camera`,
`distance_to_image_plane`, `bounding_box_3d`, `occlusion`, `normals`, `motion_vectors`,
`camera_params`, `pointcloud`, `pointcloud_include_unlabelled`, `skeleton_data`),
`image_output_format="png"`, `colorize_semantic_segmentation` / `colorize_instance_id_segmentation`
/ `colorize_instance_segmentation` (all True), `colorize_depth=False`, `frame_padding=4`,
`semantic_filter_predicate`, `use_common_output_dir=False`. File name suffix `{seq}{frame:0{pad}}`
(`seq` only with `on_time` triggers).

| annotator | files |
|---|---|
| rgb | `rgb_####.png` |
| normals | `normals_####.png` = `((n * 0.5 + 0.5) * 255).astype(uint8)` of the 4-channel data; no .npy |
| distance_to_camera / _image_plane | `distance_to_camera_####.npy` / `distance_to_image_plane_####.npy` (+ `.png` if `colorize_depth`) |
| semantic_segmentation | `semantic_segmentation_####.png` (RGBA if colorized, else raw uint32 ids) + `semantic_segmentation_labels_####.json` (`str(key)` -> labels) |
| instance_id_segmentation | `instance_id_segmentation_####.png` + `instance_id_segmentation_mapping_####.json` |
| instance_segmentation | `instance_segmentation_####.png` + `instance_segmentation_mapping_####.json` + `instance_segmentation_semantics_mapping_####.json` |
| bounding boxes | `bounding_box_{2d_tight,2d_loose,3d}_####.npy` (structured) + `_labels_####.json` + `_prim_paths_####.json` |
| motion_vectors / occlusion | `motion_vectors_####.npy` / `occlusion_####.npy` |
| camera_params | `camera_params_####.json` (arrays as lists) |
| pointcloud | `pointcloud_####.npy`, `pointcloud_rgb_`, `pointcloud_normals_`, `pointcloud_semantic_`, `pointcloud_instance_` |
| skeleton_data | `skeleton_####.json` |

Layout: one render product -> flat in `output_dir`; several -> `<rp_name>/<annotator>/<file>`
(or `<annotator>/<rp_name>_<file>` with `use_common_output_dir`). Annotator names ending `_fast` are
written without the suffix.

**Other writers**: `omni.replicator.core` registers BasicWriter, CocoWriter, FPSWriter, KittiWriter,
CosmosWriter [WHL `writers_default/__init__.py`]; Isaac Sim 5.1's `isaacsim.replicator.writers` adds
DataVisualizationWriter, DOPEWriter, PoseWriter, PytorchWriter, YCBVideoWriter [WR].
KittiWriter: `rgb/`, `object_detection/{frame}.txt`, `semantic/`, `instance_segmentation/`, `depth/`
(uint16 PNG, m x 256, inf -> 0); label lines carry class, truncated 0.00, occluded 0/1/2 (from the
tight/loose area ratio), alpha 0.00, the 2-D box, and **zeros for the 3-D fields** [WHL kitti.py].
CocoWriter: `coco_annotations_{id}.json`, `rgb_*.png`, panoptic PNGs [WHL coco.py].

## 3. Randomizers and semantics

- `rep.randomizer.*`: `scatter_2d`, `scatter_3d`, `materials`, `instantiate`, `rotation`, `texture`,
  `color`, `register` [WHL randomizer.py].
- `rep.modify.*`: `semantics`, `pose` (position/rotation/scale/look_at; camera randomization is
  `rep.modify.pose` on the camera), `pose_camera_relative`, `attribute`, `visibility`, `variant`,
  `material`, `projection_material`, `pose_orbit`, ... [WHL modify.py].
- `rep.distribution.*`: `uniform`, `normal`, `choice`, `sequence`, `combine`, `log_uniform`.
  `rep.trigger.*`: `on_frame`, `on_time`, `on_custom_event`, `on_condition`, ... [WHL].
- `rep.physics.*`: `rigid_body`, `collider`, `mass`, `drive_properties`, `physics_material`.
  `rep.create.light` with distributions for intensity/temperature/position [WHL].
- Semantics: Isaac Sim 5.1 writes `UsdSemantics.LabelsAPI` via
  `add_labels(prim, labels, instance_name="class")`; the older `Semantics.SemanticsAPI`
  (`semanticType`/`semanticData`, `add_update_semantics`) is deprecated, with
  `upgrade_prim_semantics_to_labels` for migration [SEM], [TUT].

**Isaac Lab event terms** (`isaaclab.envs.mdp.events`, [IL]); `_randomize_prop_by_op` gives
`operation` add/scale/abs and `distribution` uniform/log_uniform/gaussian:
`randomize_rigid_body_scale` (USD scale, prestartup), `randomize_rigid_body_material` (static/dynamic
friction, restitution, `num_buckets`, `make_consistent`), `randomize_rigid_body_mass`
(`recompute_inertia`, `min_mass`), `randomize_rigid_body_com`, `randomize_rigid_body_collider_offsets`,
`randomize_physics_scene_gravity`, `randomize_actuator_gains`, `randomize_joint_parameters`
(friction, armature, limits), `randomize_fixed_tendon_parameters`, `apply_external_force_torque`,
`push_by_setting_velocity`, `reset_root_state_uniform`, `reset_root_state_with_random_orientation`,
`reset_root_state_from_terrain`, `reset_joints_by_scale`, `reset_joints_by_offset`,
`reset_nodal_state_uniform`, `reset_scene_to_default`, `randomize_visual_texture_material`,
`randomize_visual_color` (both via Replicator, need `replicate_physics=False`).

Which shipped tasks use which: the locomotion velocity base config (and so the G1 task in
`assets/isaac/g1_velocity_env_cfg.py`) uses `physics_material`, `add_base_mass`, `base_com`,
`base_external_force_torque`, `reset_base` (= `reset_root_state_uniform`), `reset_robot_joints`
(= `reset_joints_by_scale`) and `push_robot` (= `push_by_setting_velocity`); the G1 config removes or
zeroes the mass/COM/push terms. The cartpole camera task has no event randomization (resets in
`_reset_idx`). Visual randomization in shipped tasks: only Franka stack visuomotor (dome light,
table and arm textures via task-local functions); Dexsuite randomizes object scale at prestartup;
Shadow Hand vision has none [IL, tag v2.3.2].

## 4. MetalSim before this round

`metalsim/replicator/` had: per-env GPU randomizers for instance colour, camera pose delta, camera
intrinsics, DR light direction/intensity, ambient, background images, per-slot roughness/metallic;
slot-level annotators (rgb, depth, raster normals, slot/body segmentation, per-slot 2-D tight boxes,
per-geom oriented 3-D boxes, intrinsics); writers: an NPZ "BasicWriter" (not Isaac's layout), a COCO
JSON writer and a KITTI writer. The renderer (tier 0/1) already exposes an id buffer in slot, geom or
body mode (`seg`), a world-frame raster normal buffer and linearized depth (distance to image plane,
0 for background); tier 2 has no id buffer. Physics DR: per-world model fields (`BatchSimOptions.per_world_fields`).
The scene layer writes `UsdSemantics.LabelsAPI` "class" labels (body name) on bodies and geoms.

## 5. Gap list and effort

| Isaac feature | MetalSim before | gap | effort |
|---|---|---|---|
| Semantics labels on prims, inheritance, BACKGROUND/UNLABELLED | body id as class | label table from rules / body names / USD LabelsAPI, instance = labelled prim | S (done) |
| semantic / instance / instance_id segmentation with `idToLabels`/`idToSemantics` | slot and body ids | Isaac id conventions and info dicts, from the id buffer | S (done) |
| bounding_box_2d_tight (per labelled instance, structured) | per slot | per instance, Isaac dtype, info | S (done) |
| bounding_box_2d_loose, occlusionRatio | none | need the unoccluded silhouette: one seg-only render per instance through a private renderer (no renderer change) | M (done; cost scales with instance count) |
| bounding_box_3d (instance extents in local frame + transform) | per geom | per instance in the body frame, USD row-vector transform | S (done) |
| distance_to_camera, inf background | depth only | derived from depth + intrinsics | S (done) |
| normals | raster world normals | Isaac 4-channel layout; depth-derived fallback for renderers without a normal output (tier 2) | S (done) |
| pointcloud (+rgb/normals/semantic/instance) | none | back-projection of depth | S (done) |
| motion_vectors | none | state-derived: rigid motion of the geom under each pixel, reprojected with the previous camera | S (done) |
| camera_params dict | intrinsics | view/projection matrices, aperture, near/far | S (done) |
| BasicWriter on-disk layout | NPZ | Isaac file names, colorized RGBA segmentation + JSON maps, structured .npy boxes, per-render-product folders | M (done) |
| KittiWriter / CocoWriter in Isaac's exact layout | own layouts | ours follow the KITTI/COCO formats, not Isaac's folder names | S (not done) |
| skeleton_data | none | needs skinned characters; not applicable to rigid MuJoCo scenes | n/a |
| occlusion annotator (per-pixel) | none | dtype undocumented in Isaac | S (not done) |
| Isaac Lab physics event terms (material, mass, COM, gains, joint params, gravity, external wrench, push, root/joint resets, EventManager modes) | per-world fields only; G1 task hard-codes its resets | torch/Warp implementations with Isaac's parameters | M (done) |
| `randomize_rigid_body_collider_offsets` | none | PhysX contact/rest offsets have no exact MuJoCo counterpart (`geom_margin`/`gap`) | S (not done) |
| `randomize_rigid_body_scale` | none | per-world `geom_size` is batchable, but the renderer's meshes are baked per slot | L (not done) |
| `randomize_fixed_tendon_parameters`, `reset_nodal_state_uniform` | none | tendons/deformables not in our task set | M (not done) |
| `randomize_visual_color` | `Randomizer.colors` | Isaac's parameters (`colors` list or r/g/b ranges) | S (done) |
| `randomize_visual_texture_material`, dome-light HDR randomization | background images, material roughness/metallic | per-env texture selection needs a renderer change (texture index per env/slot) | M (renderer) |
| `rep.randomizer.scatter_2d/3d`, `instantiate` | none | pose sampling on surfaces / volumes; object instantiation needs a model rebuild | M (not done) |
| Replicator graph / triggers (`on_frame`, `on_time`) | Python loop | not needed for batched data generation | n/a |
| tier 2 (path tracer) id buffer | none | segmentation from the path tracer needs a renderer output; use tier 0 for labels (same geometry) | M (renderer) |
