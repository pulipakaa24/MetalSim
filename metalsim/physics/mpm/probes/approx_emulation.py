"""MSL candidates for CUDA's --use_fast_math reciprocal / division / sqrt (metal_approx_probe.py scores them)."""

MSL_PRELUDE = ""

CANDIDATES_UNARY = {
    "rcp": {
        "fast::divide(1,x)": "metal::fast::divide(1.0f, x)",
        "precise::divide(1,x)": "metal::precise::divide(1.0f, x)",
        "cuda emulation (wp_cuda_rcp)": "wp_cuda_rcp(x)",
        "fast rcp + 1 NR fma": "float r = metal::fast::divide(1.0f, x); float e = metal::fma(-x, r, 1.0f); "
                               "return metal::fma(r, e, r);",
    },
    "sqrt": {
        "fast::sqrt": "metal::fast::sqrt(x)",
        "precise::sqrt": "metal::precise::sqrt(x)",
        "cuda emulation (wp_cuda_sqrt)": "wp_cuda_sqrt(x)",
        "x*fast::rsqrt": "x * metal::fast::rsqrt(x)",
        "1/fast::rsqrt (fast div)": "metal::fast::divide(1.0f, metal::fast::rsqrt(x))",
    },
}

CANDIDATES_DIV = {
    "fast::divide": "metal::fast::divide(a, b)",
    "precise::divide": "metal::precise::divide(a, b)",
    "a*fast rcp": "a * metal::fast::divide(1.0f, b)",
    "a*precise rcp": "a * metal::precise::divide(1.0f, b)",
    "cuda emulation (wp_cuda_div)": "wp_cuda_div(a, b)",
}
