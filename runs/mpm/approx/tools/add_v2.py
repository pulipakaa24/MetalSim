"""Add packed-table, 32-bit-carry variants (wp_cuda_rcp2 / wp_cuda_sqrt2) to metal_cuda_fast_math.h for timing."""
import re

P = "/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b/_forks/warp/warp/native/metal_cuda_fast_math.h"
s = open(P).read()
if "wp_cuda_rcp2" in s:
    raise SystemExit("already there")
tabs = {}
for name in ("rcp", "sqrt"):
    body = s.split(f"wp_mufu_{name}_table[384] = {{", 1)[1].split("};", 1)[0]
    v = [int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]+", body)]
    tabs[name] = [(v[3 * i], v[3 * i + 1], v[3 * i + 2]) for i in range(128)]
shift = {"rcp": 16, "sqrt": 17}
txt = "\n// Packed tables: {C0, C1 | C2 << (16 rcp, 17 sqrt)}, one 8-byte load per evaluation.\n// clang-format off\n"
for name, rows in tabs.items():
    ent = [f"{{0x{c0:07x}u, 0x{(c1 | (c2 << shift[name])):08x}u}}" for c0, c1, c2 in rows]
    lines = ",\n".join("    " + ", ".join(ent[i:i + 4]) for i in range(0, 128, 4))
    txt += f"constant uint2 wp_mufu_{name}_packed[128] = {{\n{lines}\n}};\n"
txt += "// clang-format on\n"
txt += r'''
// MUFU.RCP with the packed table and the sum in 32-bit halves (only bits 32..56 of it are needed).
inline float wp_cuda_rcp2(float x)
{
    const uint u = as_type<uint>(x);
    const uint sign = u & 0x80000000u;
    const uint ex = (u >> 23) & 0xFFu;
    const uint f = u & 0x7FFFFFu;
    if (ex == 0u)
        return as_type<float>(sign | 0x7F800000u);
    if (ex == 0xFFu)
        return f != 0u ? as_type<float>(0x7FFFFFFFu) : as_type<float>(sign);
    const uint xl = f & 0xFFFFu;
    const uint2 c = wp_mufu_rcp_packed[f >> 16];
    const uint c1 = c.y & 0xFFFFu, c2 = c.y >> 16;
    const uint y0 = xl & 0xFFu, y1 = xl >> 8;
    const uint tri7 = wp_mufu_fold4(wp_mufu_spread4(y0) * wp_mufu_spread4(y1));
    const uint q = ((y0 * y1 - tri7) << 9) + ((y1 * y1 - (y1 & 1u)) << 16);
    const uint t = c1 * xl;
    const uint a_lo = c.x << 30, b_lo = t << 17, c_lo = c2 * q;
    const uint l1 = a_lo - b_lo;
    const uint l2 = l1 + c_lo;
    const uint l3 = l2 + 0xCFCE0000u;
    const uint hi = (c.x >> 2) - (t >> 15) + metal::mulhi(c2, q) - uint(a_lo < b_lo) + uint(l2 < l1) + uint(l3 < l2);
    const bool top = (hi >> 24) != 0u;
    const uint frac = (top ? (hi >> 1) : hi) & 0x7FFFFFu;
    const int rexp = (top ? 127 : 126) - (int(ex) - 127);
    return as_type<float>(rexp >= 1 ? (sign | (uint(rexp) << 23) | frac) : sign);
}

inline float wp_cuda_div2(float a, float b)
{
    const float r = wp_cuda_rcp2(b);
    const float z = as_type<float>((as_type<uint>(a) ^ as_type<uint>(r)) & 0x80000000u);
    return metal::fma(a, r, z);
}

// MUFU.SQRT with the packed table and the sum in 32-bit halves (bits 35..57 needed).
inline float wp_cuda_sqrt2(float x)
{
    const uint u = as_type<uint>(x);
    const uint sign = u & 0x80000000u;
    const uint ex = (u >> 23) & 0xFFu;
    const uint f = u & 0x7FFFFFu;
    if (ex == 0u)
        return as_type<float>(sign);
    if (ex == 0xFFu)
        return (f != 0u || sign != 0u) ? as_type<float>(0x7FFFFFFFu) : x;
    if (sign != 0u)
        return as_type<float>(0x7FFFFFFFu);
    const int e = int(ex) - 127;
    const uint odd = uint(e) & 1u;
    const int half_e = (e - int(odd)) >> 1;
    const uint2 c = wp_mufu_sqrt_packed[(f >> 17) + (odd << 6)];
    const uint c1 = c.y & 0x1FFFFu, c2 = c.y >> 17;
    const uint xl = f & 0x1FFFFu;
    const uint x0 = xl & 0x1FFu, x1 = xl >> 9;
    const uint s0 = wp_mufu_spread4(x0 & 0xFFu), s1 = wp_mufu_spread4(x1);
    const uint tri8 = wp_mufu_fold4(s0 * s1) + ((metal::mulhi(s0, s1) & 15u) << 8) + ((x0 >> 8) & x1 & 1u) * 256u;
    const uint p = c2 * (x0 * x1 - tri8);       // weight 2^10
    const uint r = c2 * (x1 * x1 - (x1 & 1u));  // weight 2^18
    const uint m_lo = c1 * xl, m_hi = metal::mulhi(c1, xl);  // weight 2^18
    // s = c0 2^33 + m 2^18 - p 2^10 - r 2^18 + 0x3FFFC0000, as (hi, lo) 32-bit halves
    const uint m18_lo = m_lo << 18, m18_hi = (m_hi << 18) | (m_lo >> 14);
    const uint p_lo = p << 10, p_hi = p >> 22;
    const uint r_lo = r << 18, r_hi = r >> 14;
    const uint l1 = m18_lo - p_lo;
    const uint l2 = l1 - r_lo;
    const uint l3 = l2 + 0xFFFC0000u;
    const uint hi = (c.x << 1) + m18_hi - p_hi - r_hi + 3u - uint(m18_lo < p_lo) - uint(l1 < r_lo) + uint(l3 < l2);
    return as_type<float>((uint(127 + half_e) << 23) | ((hi >> 3) & 0x7FFFFFu));
}
'''
s = s.rstrip("\n") + "\n" + txt
open(P, "w").write(s)
print("ok")
