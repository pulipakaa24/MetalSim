"""Run scripts/particle_demo/render_anymal.py in-process with the given arguments, then report the acceleration
structures it built (instances per structure vs Metal's standard 2^24, extended-limits usage), their sizes, the Metal
memory allocated and the peak RSS, as one JSON line prefixed [ramp]. Grain chunks are set to 2^24 - 1 so every
structure stays strictly under the standard limit (main's default chunk is exactly 2^24)."""
import json
import os
import resource
import runpy
import sys
import time

import Metal

REPO = os.path.expanduser("~/metalsim-fresh/MetalSim")
sys.path.insert(0, REPO)
from metalsim.render import particles as _particles  # noqa: E402

_particles.InstanceStructure.CHUNK = (1 << 24) - 1
sys.argv = ["render_anymal.py"] + sys.argv[1:]
t0 = time.time()
g = runpy.run_path(os.environ.get("RENDER_SCRIPT", os.path.join(REPO, "scripts/particle_demo/render_anymal.py")), run_name="__main__")
r = g["r"]
ias = r._ias
EXT = Metal.MTLAccelerationStructureUsageExtendedLimits
top = int(ias.desc.instanceCount())
chunks = [int(c[0].instanceCount()) for c in ias.chunks] if ias.mode == "two_level" else []
ext = bool(ias.desc.usage() & EXT) or any(bool(c[0].usage() & EXT) for c in (ias.chunks if ias.mode == "two_level" else []))
mx = max([top] + chunks)
rec = dict(mode=ias.mode, chunk=_particles.InstanceStructure.CHUNK, grains=int(r.grains.n_grains), top_instances=top,
           chunk_instances=chunks, max_instances_per_structure=mx, standard_limit=1 << 24,
           within_standard=mx < (1 << 24), extended_usage=ext,
           structures_mib=round(ias.as_bytes / 2**20, 1), grain_structures_mib=round(getattr(ias, "grain_as_bytes", 0) / 2**20, 1),
           metal_allocated_mib=round(r.ctx.device.currentAllocatedSize() / 2**20, 1),
           peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
           wall_s=round(time.time() - t0, 1))
print("[ramp] " + json.dumps(rec), flush=True)
