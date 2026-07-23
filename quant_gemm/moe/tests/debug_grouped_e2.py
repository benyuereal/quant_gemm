#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位多 expert bug: E=2, 但所有 token 路由到 expert 0 (cur_expert 恒=0).
W 是 [2,N,K//2]. 看 W[cur_expert=0, ...] 这个运行时索引是否正确取到 expert 0 的权重.
如果错 -> 是运行时维度索引问题; 如果对 -> 是 cur_expert 真值传递问题."""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")

import torch
import sys, os
# 相对路径定位包根 (quant_gemm_pkg/), 不依赖绝对路径, 换机器/换目录都能跑.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from quant_gemm.moe import (
    w4a16_grouped_gemm, quantize_int4_per_group, dequant_int4_per_group,
    build_route_tables, GROUP,
)

DEV = "cuda"
DTYPE = torch.bfloat16


def main():
    torch.manual_seed(0)
    M, N, K = 16, 256, 512
    E = 2  # 两个 expert, 但只用 expert 0
    block_M, block_N, num_stages, threads = 16, 64, 2, 256

    A = torch.randn(M, K, device=DEV, dtype=DTYPE) * 0.1
    # 两个 expert 不同权重, expert 0 用种子 0, expert 1 用种子 1
    W0 = torch.randn(N, K, device=DEV, dtype=DTYPE) * 0.05
    W1 = torch.randn(N, K, device=DEV, dtype=DTYPE) * 0.05
    W_bf = torch.stack([W0, W1], dim=0)  # [2, N, K]
    qW, wS = quantize_int4_per_group(W_bf, GROUP)  # [2,N,K//2], [2,N,K//group]

    # deq 参考 (只用 expert 0)
    W0_deq = dequant_int4_per_group(qW[0], wS[0], GROUP)
    deq_ref = A @ W0_deq.T

    # 路由: 全部 token 给 expert 0
    group_sizes = [M, 0]  # expert0=M, expert1=0
    b2e, bms, bar = build_route_tables(group_sizes, block_M)
    num_m_blocks = len(b2e)
    total_pad = num_m_blocks * block_M
    A_pad = torch.cat([A, torch.zeros(total_pad - M, K, device=DEV, dtype=DTYPE)], dim=0) if total_pad > M else A
    b2e_t = torch.tensor(b2e, device=DEV, dtype=torch.int32)
    bms_t = torch.tensor(bms, device=DEV, dtype=torch.int32)
    bar_t = torch.tensor(bar, device=DEV, dtype=torch.int32)

    print(f"shape: E={E} M={M} N={N} K={K}  route all->expert0")
    print(f"  route: b2e={b2e} bms={bms} bar={bar}")

    k = w4a16_grouped_gemm(E, total_pad, N, K, num_m_blocks, block_M, block_N,
                           group_size=GROUP, num_stages=num_stages, threads=threads)
    out = k(A_pad, qW, wS, b2e_t, bms_t, bar_t)[:M]
    err = (out.float() - deq_ref.float()).abs().max().item()
    rel = (out.float() - deq_ref.float()).abs().mean().item() / max(deq_ref.abs().mean().item(), 1e-12)
    print(f"[E=2 all->exp0] max_diff vs deq(exp0) = {err:.4e}  rel={rel*100:.4f}%")

    # 对比: 如果取错了取到 expert 1
    W1_deq = dequant_int4_per_group(qW[1], wS[1], GROUP)
    deq_ref1 = A @ W1_deq.T
    err1 = (out.float() - deq_ref1.float()).abs().max().item()
    print(f"  (对照) vs deq(exp1) max_diff = {err1:.4e}")

    if err < 1e-2:
        print("\n✅ E=2 all->exp0 正确 -> 运行时 cur_expert 索引 OK, bug 在多 expert 混合路由")
    else:
        print("\n❌ E=2 all->exp0 也错 -> 运行时 W[cur_expert] 索引有问题")


if __name__ == "__main__":
    main()
