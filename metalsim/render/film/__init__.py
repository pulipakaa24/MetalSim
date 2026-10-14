"""metalsim's filming tool: one command films any registered task (``python -m metalsim.render.film <task>``).

Isaac Lab records videos inside the training loop (``--video``: ``VideoRecorderCfg`` grabbing the visualizer's frames
every ``video_interval`` steps, camera from the env's ``ViewerCfg``). metalsim splits that into file-based stages so
the expensive path-traced render runs offline, after the rollout, in the venv that has the renderer:

1. **record** (the task's venv): the task's adapter rolls out the policy; per frame the bodies (and MPM particles)
   are captured, then turned into the world poses of a render twin (``record.py``; Newton scenes through
   ``newton_bridge.NewtonRenderSource`` with the scene's visual meshes, MuJoCo scenes by forward kinematics).
2. **render** (main venv): tier-2 path tracing of each shot (``render.py``), particles as render grains.
3. **encode**: H.264 master + docs copy + stills (``encode.py``).
4. **doc**: ``docs/renders/<TASK>.md`` from the stage JSONs (``doc.py``): what is shown, timings, hardware, pins,
   assets with sources and licences, how to reproduce.

A task needs a ``FilmTask`` entry (``tasks.py``: camera as Isaac's ``ViewerCfg``, envs, look, assets) and a recorder
adapter (``adapters.py``: build the env, load the policy, step).
"""
