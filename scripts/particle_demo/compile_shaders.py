"""Compile the renderer's Metal libraries without dispatching anything (a CPU-side syntax / type check).

    python scripts/particle_demo/compile_shaders.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import Metal  # noqa: E402
from metalsim.render import metal_context as mc  # noqa: E402

dev = Metal.MTLCreateSystemDefaultDevice()
ok = True
for name in ("pathtrace", "particles", "rt", "tier0"):
    src = mc._inline_includes((mc.SHADER_DIR / f"{name}.metal").read_text())
    if name == "tier0":   # with the raster grain shader appended, as particles.GrainRaster compiles it
        src += "\n" + mc._inline_includes((mc.SHADER_DIR / "particles_tier0.metal").read_text())
    opts = Metal.MTLCompileOptions.new()
    opts.setFastMathEnabled_(True)
    lib, err = dev.newLibraryWithSource_options_error_(src, opts, None)
    print(name, "OK" if lib is not None else f"FAILED {str(err)[:2000]}")
    ok = ok and lib is not None
sys.exit(0 if ok else 1)
