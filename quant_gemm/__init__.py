"""quant_gemm — Hygon DCU W8A8 / W4A16 GEMM operators via tilelang.

海光 DCU (gfx936/gfx928) 上的量化 GEMM 算子库, 基于 tilelang (架构无关 JIT),
不依赖 lightop 汇编包 / aiter_ C 扩展.

W8A8 (per-channel int8 weight + dynamic int8 activation):
    - kernels.w8a8_per_channel_gemm:  Linear W8A8 GEMM kernel (tilelang)
    - kernels.moe_w8a8_grouped_gemm:  MoE grouped W8A8 GEMM kernel (tilelang)
    - quant.per_token_sym_int8:       per-token int8 量化
    - quant.per_channel_sym_int8:     per-channel int8 量化
    - w8a8_linear:  高层 API, 一次调用完成 量化+GEMM (自动选 block + kernel 缓存)
    - w8a8_moe:     高层 API, MoE grouped W8A8

W4A16 (per-group int4 weight + bf16 activation, MoE only):
    - moe.kernels.w4a16_grouped_gemm:      per-group int4 grouped GEMM kernel
    - moe.quant.quantize_int4_per_group:   per-group int4 量化 (zp=8 对称)
    - moe.w4a16_fused_moe:                 完整 fused MoE (路由+2GEMM+combine)

精度 (gfx936): W8A8 vs 量化反量化 0.28%; W4A16 MoE vs 量化反量化 1.05%.
"""

from .kernels import w8a8_per_channel_gemm, moe_w8a8_grouped_gemm
from .quant import per_token_sym_int8, per_channel_sym_int8
from .api import w8a8_linear, w8a8_moe, make_moe_route_table

# W4A16 MoE 子包 (导入即注册, 但延迟编译: kernel 首次调用才 JIT)
from . import moe
from .moe import (
    w4a16_grouped_gemm,
    quantize_int4_per_group,
    dequant_int4_per_group,
    w4a16_fused_moe,
    tilelang_fused_moe_simple,
)

__version__ = "0.1.0"

__all__ = [
    # W8A8
    "w8a8_per_channel_gemm",
    "moe_w8a8_grouped_gemm",
    "per_token_sym_int8",
    "per_channel_sym_int8",
    "w8a8_linear",
    "w8a8_moe",
    "make_moe_route_table",
    # W4A16 MoE
    "moe",
    "w4a16_grouped_gemm",
    "quantize_int4_per_group",
    "dequant_int4_per_group",
    "w4a16_fused_moe",
    "tilelang_fused_moe_simple",
    "__version__",
]
