import warp as wp

wp.config.quiet = True


@wp.kernel(module="unique", module_options={"fast_math": True}, enable_backward=False)
def k(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] / 3.0


for mode in ("cuda", "relaxed", "off"):
    wp.config.metal_fast_math = mode
    k.module.mark_modified()
    print(mode, k.module.get_module_hash().hex()[:12], k.module.options.get("fast_math"))
