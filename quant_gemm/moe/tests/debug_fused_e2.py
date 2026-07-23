#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位 fused MoE bug: 最小 E=2 混合路由完整 fused, 逐步打印中间量对比."""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")

import torch
import torch.nn.functional as F
import sys, os
# 相对路径定位包根 (quant_gemm_pkg/), 不依赖绝对路径, 换机器/换目录都能跑.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from quant_gemm.moe import (
    w4a16_grouped_gemm, quantize_int4_per_group, dequant_int4_per_group,
    build_route_tables, GROUP,
)
from sgl_kernel import silu_and_mul

DEV = "cuda"
DTYPE = torch.bfloat16


def main():
    torch.manual_seed(0)
    num_tokens, topk, K, N_inter, N_down, E = 8, 2, 512, 256, 512, 2
    N_gate_up = 2 * N_inter
    block_M, block_N, num_stages, threads = 16, 64, 2, 256

    hidden = torch.randn(num_tokens, K, device=DEV, dtype=DTYPE) * 0.1
    w1_bf = torch.randn(E, N_gate_up, K, device=DEV, dtype=DTYPE) * 0.05
    w2_bf = torch.randn(E, N_down, N_inter, device=DEV, dtype=DTYPE) * 0.05
    w1_q, w1_s = quantize_int4_per_group(w1_bf, GROUP)
    w2_q, w2_s = quantize_int4_per_group(w2_bf, GROUP)

    # 简单路由: token 0-3 -> exp0, 4-7 -> exp1, 每个只选1个expert (topk=2 但我们手工设)
    # 用 topk=1 简化: 每 token 1 个 expert
    topk = 1
    topk_idx = torch.tensor([[0],[0],[0],[0],[1],[1],[1],[1]], device=DEV, dtype=torch.int32)
    topk_w = torch.ones(num_tokens, topk, device=DEV, dtype=DTYPE)

    # deq 参考 (逐 token)
    w1_deq = dequant_int4_per_group(w1_q, w1_s, GROUP)  # [E, N_gate_up, K]
    w2_deq = dequant_int4_per_group(w2_q, w2_s, GROUP)  # [E, N_down, N_inter]
    deq_ref = torch.zeros(num_tokens, N_down, device=DEV, dtype=DTYPE)
    for t in range(num_tokens):
        e = int(topk_idx[t, 0])
        g = hidden[t] @ w1_deq[e].T
        gate, up = g[:N_inter], g[N_inter:]
        inter = F.silu(gate) * up
        deq_ref[t] = inter @ w2_deq[e].T

    # ---- 手工跑 fused 流程, 逐步对比 ----
    # 1. gather (按 expert 排序: exp0 的 4 个 token 在前, exp1 的 4 个在后)
    tok_sorted = torch.tensor([0,1,2,3,4,5,6,7], device=DEV, dtype=torch.int64)
    exp_sorted = torch.tensor([0,0,0,0,1,1,1,1], device=DEV, dtype=torch.int64)
    A_gathered = hidden[tok_sorted]  # [8, K]

    group_sizes = [4, 4]
    b2e, bms, bar = build_route_tables(group_sizes, block_M)
    num_m_blocks = len(b2e)
    total_pad = num_m_blocks * block_M
    print(f"route: b2e={b2e} bms={bms} bar={bar} num_m_blocks={num_m_blocks} total_pad={total_pad}")
    # block_M=16, 8 tokens -> 1 block, actual_rows=8. 但这一个 block 跨了 2 个 expert!
    # 这就是 bug 根源: 一个 block_M=16 的 block 覆盖了 exp0(4)+exp1(4), 但 block_to_expert 只能填一个 expert!

    if total_pad > 8:
        A_pad = torch.cat([A_gathered, torch.zeros(total_pad-8, K, device=DEV, dtype=DTYPE)], dim=0)
    else:
        A_pad = A_gathered
    b2e_t = torch.tensor(b2e, device=DEV, dtype=torch.int32)
    bms_t = torch.tensor(bms, device=DEV, dtype=torch.int32)
    bar_t = torch.tensor(bar, device=DEV, dtype=torch.int32)

    # grouped GEMM: 一个 block 内 cur_expert 单一, 但 block 跨 2 expert -> 用了错误的单一 expert 算所有 16 行
    k1 = w4a16_grouped_gemm(E, total_pad, N_gate_up, K, num_m_blocks, block_M, block_N,
                            group_size=GROUP, num_stages=num_stages, threads=threads)
    cache1 = k1(A_pad, w1_q, w1_s, b2e_t, bms_t, bar_t)  # [total_pad, N_gate_up]

    # deq 参考 (gather 顺序)
    cache1_ref = torch.zeros(total_pad, N_gate_up, device=DEV, dtype=DTYPE)
    for i in range(8):
        e = int(exp_sorted[i])
        cache1_ref[i] = A_pad[i] @ w1_deq[e].T
    err1 = (cache1[:8].float() - cache1_ref[:8].float()).abs().max().item()
    print(f"  [gemm1] cache1[:8] vs deq_ref max_diff = {err1:.4e}")

    # silu_and_mul
    cache_inter = silu_and_mul(cache1.view(-1, N_gate_up))  # [total_pad, N_inter]
    cache_inter_ref = torch.zeros(total_pad, N_inter, device=DEV, dtype=DTYPE)
    for i in range(8):
        g = cache1_ref[i]
        cache_inter_ref[i] = F.silu(g[:N_inter]) * g[N_inter:]
    err_inter = (cache_inter[:8].float() - cache_inter_ref[:8].float()).abs().max().item()
    print(f"  [silu] cache_inter[:8] vs ref max_diff = {err_inter:.4e}")

    # gemm2
    k2 = w4a16_grouped_gemm(E, total_pad, N_down, N_inter, num_m_blocks, block_M, block_N,
                            group_size=GROUP, num_stages=num_stages, threads=threads)
    cache_down = k2(cache_inter, w2_q, w2_s, b2e_t, bms_t, bar_t)  # [total_pad, N_down]
    cache_down_ref = torch.zeros(total_pad, N_down, device=DEV, dtype=DTYPE)
    for i in range(8):
        e = int(exp_sorted[i])
        cache_down_ref[i] = cache_inter_ref[i] @ w2_deq[e].T
    err2 = (cache_down[:8].float() - cache_down_ref[:8].float()).abs().max().item()
    print(f"  [gemm2] cache_down[:8] vs deq_ref max_diff = {err2:.4e}")

    # combine (scatter 回原 token 顺序)
    final = torch.zeros(num_tokens, N_down, device=DEV, dtype=DTYPE)
    final.index_add_(0, tok_sorted, cache_down[:8] * topk_w)
    err_final = (final.float() - deq_ref.float()).abs().max().item()
    print(f"  [final] vs deq_ref max_diff = {err_final:.4e}")

    print("\n>>> 诊断: block_M=16 但 exp0 只有4 token, 一个 block 跨了 2 expert,")
    print(">>> block_to_expert 只能填一个值 -> 后4行用了错误的 expert 权重!")
    print(">>> 解决: block_M 必须小到不超过任一 group 的 token 数, 或路由表按 token 粒度而非 block 粒度")


if __name__ == "__main__":
    main()
