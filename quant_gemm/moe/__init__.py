"""quant_gemm.moe — Hygon DCU W4A16 MoE operators via tilelang (gfx936/gfx928).

海光 DCU 上的 per-group int4 (W4A16) MoE 算子, 基于 tilelang (架构无关 JIT),
替换 sglang fused_moe 的 Triton int4 GEMM kernel.

提供:
    - kernels.w4a16_grouped_gemm:  per-group int4 grouped GEMM kernel (tilelang)
    - quant.quantize_int4_per_group:    per-group int4 量化 (zp=8 对称, sglang 兼容)
    - quant.dequant_int4_per_group:     per-group int4 反量化 (参考实现)
    - moe.w4a16_fused_moe:        高层 API, 路由+2GEMM+silu+combine 完整 fused MoE
    - moe.moe_align_torch:        纯 torch 路由重排 (小 E / 测试用)

精度 (gfx936, M3 真实 shape E=128): vs 量化反量化 rel≈1.05%.
性能 (vs sglang tuned Triton int4): M=1 1.69x, M=2 2.66x, M=8 4.00x, M=32 4.32x.
"""

from .kernels import w4a16_grouped_gemm, GROUP
from .quant import quantize_int4_per_group, dequant_int4_per_group, DTYPE
from .moe import w4a16_fused_moe, tilelang_fused_moe_simple, moe_align_torch, build_route_tables

__all__ = [
    "w4a16_grouped_gemm",
    "quantize_int4_per_group",
    "dequant_int4_per_group",
    "w4a16_fused_moe",
    "tilelang_fused_moe_simple",
    "moe_align_torch",
    "build_route_tables",
    "GROUP",
    "DTYPE",
]
