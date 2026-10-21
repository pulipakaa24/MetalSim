"""Make the packed / 32-bit forms the only implementation in metal_cuda_fast_math.h."""
P = "_forks/warp/warp/native/metal_cuda_fast_math.h"
s = open(P).read()
# drop the unpacked tables (first clang-format block)
a = s.index("// clang-format off\nconstant uint wp_mufu_rcp_table")
b = s.index("// clang-format on\n", a) + len("// clang-format on\n")
s = s[:a] + s[b:]
# drop v1 functions (mul64 .. end of v1 sqrt)
a = s.index("inline ulong wp_mufu_mul64")
b = s.index("\n// Packed tables:")
s = s[:a] + s[b + 1:]
s = s.replace("// Packed tables: {C0, C1 | C2 << (16 rcp, 17 sqrt)}, one 8-byte load per evaluation.",
              "// Coefficient tables (sass-math's, packed): {C0, C1 | C2 << 16} for MUFU.RCP (m = 7), {C0, C1 | C2 << 17} for\n"
              "// MUFU.SQRT (m = 6, rows 64..127 for odd exponents); one 8-byte constant load per evaluation.")
s = s.replace("// MUFU.RCP with the packed table and the sum in 32-bit halves (only bits 32..56 of it are needed).",
              "// MUFU.RCP (rcp.approx.ftz.f32). The fixed-point sum C0 2^30 - C1 xl 2^17 + C2 (sq(2 xl) >> 2) + bias is kept\n"
              "// in 32-bit halves; the result fraction is bits 32..56 of it (renormalised by one bit below 2^56).")
s = s.replace("// MUFU.SQRT with the packed table and the sum in 32-bit halves (bits 35..57 needed).",
              "// MUFU.SQRT (sqrt.approx.ftz.f32): C0 2^33 + C1 xl 2^18 - C2 sq(xl) + bias in 32-bit halves, fraction = bits\n"
              "// 35..57.")
s = s.replace("wp_cuda_rcp2", "wp_cuda_rcp").replace("wp_cuda_div2", "wp_cuda_div").replace("wp_cuda_sqrt2", "wp_cuda_sqrt")
s = s.replace('''inline float wp_cuda_div(float a, float b)
{
    const float r = wp_cuda_rcp(b);''', '''// div.approx.ftz.f32: FMUL.FTZ(a, MUFU.RCP(b)) (Apple GPUs flush float32 denormal results). The quotient is rounded
// on its own: on the L4, Warp's x / z + y / z and x / z - y / z round both quotients (div.approx is one PTX
// instruction; measured), while the Metal compiler contracts a visible product into a following subtraction (even
// under `fp contract(off)` here, and through fma(a, r, -0)). fma(a, r, z) with z the zero carrying the product's
// sign is a * r rounded once, signed zeros included, and the compiler cannot reduce it to a contractible multiply.
inline float wp_cuda_div(float a, float b)
{
    const float r = wp_cuda_rcp(b);''')
assert s.count("inline float wp_cuda_rcp(") == 1 and s.count("inline float wp_cuda_sqrt(") == 1
assert s.count("inline float wp_cuda_div(") == 1
open(P, "w").write(s)
print("ok")
