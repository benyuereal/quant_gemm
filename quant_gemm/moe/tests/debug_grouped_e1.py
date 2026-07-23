#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位 grouped MoE 正确性 bug: 退化到 E=1 (单 expert), 路由表只有 expert 0.
对比 grouped(E=1) 输出 vs 单 expert GEMM 输出 vs deq 参考, 找出路由表引入的 bug."""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")

import torch
import tilelang
import tilelang.language as T
from tilelang.quantize import _tir_packed_to_unsigned_convert

# 复用主文件的算子和工具
import sys, os
# 相对路径定位包根 (quant_gemm_pkg/), 不依赖绝对路径, 换机器/换目录都能跑.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from quant_gemm.moe import (
    w4a16_grouped_gemm, quantize_int4_per_group, dequant_int4_per_group,
    build_route_tables, GROUP,
)
# 旧版单 expert kernel (zp=0), 仅此历史调试脚本用于对比, 不属于本包.
# 文件 test_tilelang_w4a16_gemm.py 需在 PYTHONPATH 或 /models 下, 缺失时本脚本无法跑对比.
try:
    from test_tilelang_w4a16_gemm import w4a16_gemm  # type: ignore
except ImportError as _e:
    w4a16_gemm = None
    print(f"[warn] 未找到旧版单 expert kernel (test_tilelang_w4a16_gemm), 单 expert 对比跳过: {_e}")

DEV = "cuda"
DTYPE = torch.bfloat16


def main():
    torch.manual_seed(0)
    M, N, K = 16, 256, 512  # 同单 expert 验证用的 shape
    E = 1
    block_M, block_N, num_stages, threads = 16, 64, 2, 256

    A = torch.randn(M, K, device=DEV, dtype=DTYPE) * 0.1
    W_bf = torch.randn(E, N, K, device=DEV, dtype=DTYPE) * 0.05   # [1, N, K]
    W_bf_2d = W_bf[0]                                             # [N, K] 单 expert 形状
    qW1, wS1 = quantize_int4_per_group(W_bf, GROUP)               # [1,N,K//2], [1,N,K//group]
    qW2, wS2 = quantize_int4_per_group(W_bf_2d, GROUP)            # [N,K//2], [N,K//group]

    # deq 参考
    W_deq = dequant_int4_per_group(qW2, wS2, GROUP)               # [N, K]
    deq_ref = A @ W_deq.T                                         # [M, N]

    print(f"shape: E={E} M={M} N={N} K={K} block_M={block_M} block_N={block_N}")
    print("=" * 60)

    # ---- 单 expert GEMM (已验证正确) ----
    k_single = w4a16_gemm(M, N, K, GROUP, block_M=block_M, block_N=block_N, num_stages=num_stages, threads=threads)
    out_single = k_single(A, qW2, wS2)
    err_single = (out_single.float() - deq_ref.float()).abs().max().item()
    print(f"[单expert]  max_diff vs deq = {err_single:.4e}")

    # ---- grouped GEMM, E=1, 路由表只有 expert 0 ----
    group_sizes = [M]   # 1 个 expert, M 个 token
    b2e, bms, bar = build_route_tables(group_sizes, block_M)
    num_m_blocks = len(b2e)
    total_pad = num_m_blocks * block_M
    if total_pad > M:
        A_pad = torch.cat([A, torch.zeros(total_pad - M, K, device=DEV, dtype=DTYPE)], dim=0)
    else:
        A_pad = A
    b2e_t = torch.tensor(b2e, device=DEV, dtype=torch.int32)
    bms_t = torch.tensor(bms, device=DEV, dtype=torch.int32)
    bar_t = torch.tensor(bar, device=DEV, dtype=torch.int32)

    print(f"  route: b2e={b2e} bms={bms} bar={bar} num_m_blocks={num_m_blocks} total_pad={total_pad}")

    k_grouped = w4a16_grouped_gemm(
        E, total_pad, N, K, num_m_blocks, block_M, block_N,
        group_size=GROUP, num_stages=num_stages, threads=threads,
    )
    nvb = torch.tensor([num_m_blocks], device=DEV, dtype=torch.int32)  # E=1 全部有效
    out_grouped = k_grouped(A_pad, qW1, wS1, b2e_t, bms_t, bar_t, nvb)   # [total_pad, N]
    out_grouped = out_grouped[:M]
    err_grouped = (out_grouped.float() - deq_ref.float()).abs().max().item()
    rel_grouped = (out_grouped.float() - deq_ref.float()).abs().mean().item() / max(deq_ref.abs().mean().item(), 1e-12)
    print(f"[grouped E=1] max_diff vs deq = {err_grouped:.4e}  rel={rel_grouped*100:.4f}%")

    # 直接对比两个算子输出
    diff_two = (out_grouped.float() - out_single.float()).abs().max().item()
    print(f"[grouped vs 单expert] max_diff = {diff_two:.4e}")

    # 逐行看差异分布
    per_row = (out_grouped.float() - deq_ref.float()).abs().mean(dim=1)
    print(f"  per-row mean abs err: {per_row.tolist()}")

    if err_grouped < 1e-2:
        print("\n✅ grouped E=1 正确 -> 路由表 OK, bug 在多 expert (cur_expert)")
    else:
        print("\n❌ grouped E=1 也错 -> bug 在路由表/m_start/scale 索引, 非多 expert")


if __name__ == "__main__":
    main()
