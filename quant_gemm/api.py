"""高层 API: 一次调用完成 int8 量化 + W8A8 GEMM.

自动选 block size, 缓存编译过的 tilelang kernel (按 shape key), 避免重复 JIT.
"""

import math
import torch
import tilelang.language as T

from .kernels import w8a8_per_channel_gemm, moe_w8a8_grouped_gemm
from .quant import per_token_sym_int8, per_channel_sym_int8

__all__ = ["w8a8_linear", "w8a8_moe", "make_moe_route_table"]


# kernel 缓存: key -> compiled kernel (tilelang JIT 首次调用约 4s, 后续命中缓存)
_linear_cache = {}
_moe_cache = {}


def _align(x, a):
    return ((x + a - 1) // a) * a


def _pick_blocks(M, N, K):
    """选 block_M/N/K. 默认 128x128x64, M 小时用 64. 要求 N,K 对齐 128, M 对齐 block_M."""
    block_N = 128
    block_K = 64
    block_M = 128 if M >= 128 else 64
    return block_M, block_N, block_K


def w8a8_linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None,
                x_scale: torch.Tensor = None, w_scale: torch.Tensor = None,
                out_dtype: torch.dtype = None):
    """W8A8 Linear: out = x @ w.T * (x_scale * w_scale.T) + bias.

    两种用法:
      1) 传原始 bf16/fp16 x, w (未量化): 内部做 per-token/per-channel int8 量化
      2) 传已量化的 q_x(int8), q_w(int8) + x_scale, w_scale: 跳过量化直接 GEMM
         (此时 x/w 必须是 int8, 且 x_scale/w_scale 必须提供)

    Args:
        x: [M, K] bf16/fp16 (用法1) 或 [M,K] int8 (用法2)
        w: [N, K] bf16/fp16 (用法1) 或 [N,K] int8 (用法2)
        bias: 可选 [N]
        x_scale, w_scale: 用法2 必填, [M,1]/[N,1] f32
        out_dtype: 输出 dtype, 默认 x 的 dtype (用法1) 或 bf16 (用法2)
    Returns:
        out: [M, N]
    """
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw, f"K mismatch: x {K} vs w {Kw}"
    device = x.device

    # 用法2: 已量化
    if x.dtype == torch.int8:
        assert x_scale is not None and w_scale is not None, "用法2 需提供 x_scale/w_scale"
        q_x, q_w = x, w
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        # 用法1: 量化
        q_x, x_scale = per_token_sym_int8(x.float())
        q_w, w_scale = per_channel_sym_int8(w.float())
        q_x = q_x.to(device); x_scale = x_scale.to(device)
        q_w = q_w.to(device); w_scale = w_scale.to(device)
        if out_dtype is None:
            out_dtype = x.dtype

    # pad M/N/K 到 block 倍数 (右下 pad)
    block_M, block_N, block_K = _pick_blocks(M, N, K)
    Mp = _align(M, block_M)
    Np = _align(N, block_N)
    Kp = _align(K, block_K)
    need_pad = (Mp != M) or (Np != N) or (Kp != K)

    if need_pad:
        q_x = torch.nn.functional.pad(q_x, (0, Kp - K, 0, Mp - M))
        x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, Mp - M))
        q_w = torch.nn.functional.pad(q_w, (0, Kp - K, 0, Np - N))
        w_scale = torch.nn.functional.pad(w_scale, (0, 0, 0, Np - N))

    key = (Mp, Np, Kp, block_M, block_N, block_K, out_dtype)
    if key not in _linear_cache:
        _linear_cache[key] = w8a8_per_channel_gemm(
            Mp, Np, Kp, block_M, block_N, block_K,
            in_dtype=T.int8, accum_dtype=T.int32, out_dtype=out_dtype,
        )
    kernel = _linear_cache[key]
    out = kernel(q_x, q_w, x_scale, w_scale)  # [Mp, Np]

    if need_pad:
        out = out[:M, :N]
    if bias is not None:
        out = out + bias.to(out_dtype)
    return out


def make_moe_route_table(group_sizes, group_offsets, block_M):
    """预算 MoE 路由表 (Python 端).

    把 sglang 的 sorted_token_ids 路由表示转成 tilelang kernel 需要的
    block_to_expert/block_m_start/block_actual_rows (避开 kernel 内控制流).

    Args:
        group_sizes: list[int], 每 expert 的 token 数
        group_offsets: list[int], 每 expert 的 token 起始偏移
        block_M: M 方向 block 大小
    Returns:
        (block_to_expert, block_m_start, block_actual_rows): list 列表
        num_m_blocks: int
    """
    block_to_expert, block_m_start, block_actual_rows = [], [], []
    for e, (gs, go) in enumerate(zip(group_sizes, group_offsets)):
        n_blocks_e = math.ceil(gs / block_M)
        for b in range(n_blocks_e):
            block_to_expert.append(e)
            block_m_start.append(go + b * block_M)
            block_actual_rows.append(min(block_M, gs - b * block_M))
    return block_to_expert, block_m_start, block_actual_rows, len(block_to_expert)


def w8a8_moe(x_grouped: torch.Tensor, w: torch.Tensor,
             x_scale: torch.Tensor, w_scale: torch.Tensor,
             group_sizes, group_offsets,
             block_M=64, block_N=128, block_K=64,
             out_dtype: torch.dtype = torch.bfloat16):
    """W8A8 MoE grouped GEMM (单步, gate/up 或 down 通用).

    Args:
        x_grouped: [total_tokens, K] int8, 已按 expert 分组连续排列 (Python 端 gather)
        w: [E, N, K] int8 (per-channel)
        x_scale: [total_tokens, 1] f32 (per-token)
        w_scale: [E, N, 1] f32 (per-channel)
        group_sizes: list[int], 每 expert token 数
        group_offsets: list[int], 每 expert token 起始
        block_M/N/K: tile 块大小
        out_dtype: 输出 dtype
    Returns:
        out: [total_tokens, N] out_dtype
    """
    total_tokens, K = x_grouped.shape
    E, N, Kw = w.shape
    assert K == Kw
    device = x_grouped.device

    # pad total_tokens 到 block_M 倍数
    Mp = _align(total_tokens, block_M)
    if Mp != total_tokens:
        x_grouped = torch.nn.functional.pad(x_grouped, (0, 0, 0, Mp - total_tokens))
        x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, Mp - total_tokens))

    b2e, bms, bar, num_m_blocks = make_moe_route_table(group_sizes, group_offsets, block_M)
    block_to_expert = torch.tensor(b2e, device=device, dtype=torch.int32)
    block_m_start = torch.tensor(bms, device=device, dtype=torch.int32)
    block_actual_rows = torch.tensor(bar, device=device, dtype=torch.int32)

    # pad N,K 到 block 倍数
    Np = _align(N, block_N)
    Kp = _align(K, block_K)
    need_pad = (Np != N) or (Kp != K)
    if need_pad:
        w = torch.nn.functional.pad(w, (0, Kp - K, 0, Np - N, 0, 0))
        w_scale = torch.nn.functional.pad(w_scale, (0, 0, 0, Np - N, 0, 0))
        x_grouped = torch.nn.functional.pad(x_grouped, (0, Kp - K))
        x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, 0))

    key = (E, Mp, Np, Kp, num_m_blocks, block_M, block_N, block_K, out_dtype)
    if key not in _moe_cache:
        _moe_cache[key] = moe_w8a8_grouped_gemm(
            E, Mp, Np, Kp, num_m_blocks, block_M, block_N, block_K,
            in_dtype=T.int8, accum_dtype=T.int32, out_dtype=out_dtype,
        )
    kernel = _moe_cache[key]
    out = kernel(x_grouped, w, x_scale, w_scale, block_to_expert, block_m_start, block_actual_rows)
    if need_pad:
        out = out[:total_tokens, :N]
    return out
