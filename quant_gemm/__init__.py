"""quant_gemm — Hygon DCU W8A8 GEMM operators via tilelang.

海光 DCU (gfx936/gfx928) 上的 per-channel W8A8 (int8 weight + int8 activation)
GEMM 算子库, 基于 tilelang (架构无关 JIT), 不依赖 lightop 汇编包 / aiter_ C 扩展.

提供:
    - kernels.w8a8_per_channel_gemm:  Linear W8A8 GEMM kernel (tilelang)
    - kernels.moe_w8a8_grouped_gemm:  MoE grouped W8A8 GEMM kernel (tilelang)
    - quant.per_token_sym_int8:       per-token int8 量化
    - quant.per_channel_sym_int8:     per-channel int8 量化
    - w8a8_linear:  高层 API, 一次调用完成 量化+GEMM (自动选 block + kernel 缓存)
    - w8a8_moe:     高层 API, MoE grouped W8A8

精度 (gfx936): vs 量化反量化 0.28%, vs bf16 1.25%.
"""

from .kernels import w8a8_per_channel_gemm, moe_w8a8_grouped_gemm
from .quant import per_token_sym_int8, per_channel_sym_int8
from .api import w8a8_linear, w8a8_moe, make_moe_route_table

__version__ = "0.1.0"

__all__ = [
    "w8a8_per_channel_gemm",
    "moe_w8a8_grouped_gemm",
    "per_token_sym_int8",
    "per_channel_sym_int8",
    "w8a8_linear",
    "w8a8_moe",
    "make_moe_route_table",
    "__version__",
]
