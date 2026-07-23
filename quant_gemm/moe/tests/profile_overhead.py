#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分项 profiling: 把 tilelang_fused_moe_simple 一次调用拆成多块单独计时,
定位 ~8ms 固定开销来源.

M3 真实 shape: E=128 K=6144 shard_inter=1536 topk=4.
对比 M=1 (慢, 0.18x) vs M=32 (快, 3.24x), 看哪个子项是平坦的.
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ["SGLANG_USE_AITER"] = "0"

import sys, os
# 相对路径定位包根 (quant_gemm_pkg/), 不依赖绝对路径, 换机器/换目录都能跑.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import tilelang
from quant_gemm.moe import (
    w4a16_grouped_gemm, quantize_int4_per_group, GROUP, DTYPE,
)

DEV = "cuda"
E = 128
HIDDEN = 6144
SHARD_INTER = 1536
N_GATE_UP = 2 * SHARD_INTER
N_DOWN = HIDDEN
TOPK = 4
BLOCK_M = 16
BLOCK_N = 64
NUM_STAGES = 2
THREADS = 256


def make_inputs(num_tokens):
    torch.manual_seed(42)
    w1_bf = torch.randn(E, N_GATE_UP, HIDDEN, device=DEV, dtype=DTYPE) * 0.02
    w2_bf = torch.randn(E, N_DOWN, SHARD_INTER, device=DEV, dtype=DTYPE) * 0.02
    w1q, w1s = quantize_int4_per_group(w1_bf, GROUP)
    w2q, w2s = quantize_int4_per_group(w2_bf, GROUP)
    x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
    # 随机路由
    logits = torch.randn(num_tokens, E, device=DEV, dtype=DTYPE)
    tw, ti = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)
    ti = ti.to(torch.int32)
    return x, w1q, w1s, w2q, w2s, tw.to(torch.float32), ti


def bench(fn, num_warmup=20, num_iters=100):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / num_iters * 1000  # us/call


def profile_m(num_tokens):
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import dcu_moe_align_block_size
    from sgl_kernel import silu_and_mul

    x, w1q, w1s, w2q, w2s, tw, ti = make_inputs(num_tokens)
    K = HIDDEN
    N_inter = SHARD_INTER
    dev = DEV
    M_flat = num_tokens * TOPK

    # 预先编译两个 kernel (用真实 shape), 拿到 kernel 对象
    # 先跑一次完整流程拿到 total_pad / num_m_blocks
    sorted_ids, expert_ids, _ = dcu_moe_align_block_size(ti, BLOCK_M, E)
    sorted_ids = sorted_ids.to(torch.int64)
    total_pad = sorted_ids.numel()
    num_m_blocks = expert_ids.numel()
    total_gem = num_m_blocks * BLOCK_M
    valid = sorted_ids < M_flat
    safe_flat = sorted_ids.clamp(max=M_flat - 1)
    tok_idx = safe_flat // TOPK
    A_gathered = x[tok_idx]
    A_gathered = A_gathered * valid.unsqueeze(-1).to(DTYPE)
    if total_gem > total_pad:
        A_gathered = torch.cat([A_gathered, torch.zeros(total_gem - total_pad, K, device=dev, dtype=DTYPE)], dim=0)
        valid = torch.cat([valid, torch.zeros(total_gem - total_pad, dtype=torch.bool, device=dev)])
        safe_flat = torch.cat([safe_flat, torch.zeros(total_gem - total_pad, dtype=torch.int64, device=dev)])
        tok_idx = torch.cat([tok_idx, torch.zeros(total_gem - total_pad, dtype=torch.int64, device=dev)])
    total_pad = total_gem
    block_m_start = torch.arange(num_m_blocks, device=dev, dtype=torch.int32) * BLOCK_M
    block_actual_rows = torch.full((num_m_blocks,), BLOCK_M, device=dev, dtype=torch.int32)

    # 预编译 kernel (cache 命中后, 后续调用只是 lookup + invoke)
    kernel1 = w4a16_grouped_gemm(E, total_pad, N_GATE_UP, K, num_m_blocks,
                                 BLOCK_M, BLOCK_N, group_size=GROUP,
                                 num_stages=NUM_STAGES, threads=THREADS)
    kernel2 = w4a16_grouped_gemm(E, total_pad, N_DOWN, N_inter, num_m_blocks,
                                 BLOCK_M, BLOCK_N, group_size=GROUP,
                                 num_stages=NUM_STAGES, threads=THREADS)
    # 跑一次让编译完成
    _ = kernel1(A_gathered, w1q, w1s, expert_ids, block_m_start, block_actual_rows)
    _ = kernel2(torch.zeros(total_pad, N_inter, device=dev, dtype=DTYPE), w2q, w2s,
                expert_ids, block_m_start, block_actual_rows)
    torch.cuda.synchronize()

    print(f"\n===== M={num_tokens} total_pad={total_pad} num_m_blocks={num_m_blocks} =====")

    # ---- 分项计时 ----
    # 1. align
    t_align = bench(lambda: dcu_moe_align_block_size(ti, BLOCK_M, E))
    print(f"  1. dcu_moe_align_block_size : {t_align:8.1f} us")

    # 2. gather + pad (重建 A_gathered/valid/safe_flat/tok_idx)
    def _gather():
        sid, eid, _ = dcu_moe_align_block_size(ti, BLOCK_M, E)
        sid = sid.to(torch.int64)
        tp = sid.numel()
        nmb = eid.numel()
        tg = nmb * BLOCK_M
        v = sid < M_flat
        sf = sid.clamp(max=M_flat - 1)
        tki = sf // TOPK
        ag = x[tki]
        ag = ag * v.unsqueeze(-1).to(DTYPE)
        if tg > tp:
            ag = torch.cat([ag, torch.zeros(tg - tp, K, device=dev, dtype=DTYPE)], dim=0)
            v = torch.cat([v, torch.zeros(tg - tp, dtype=torch.bool, device=dev)])
            sf = torch.cat([sf, torch.zeros(tg - tp, dtype=torch.int64, device=dev)])
            tki = torch.cat([tki, torch.zeros(tg - tp, dtype=torch.int64, device=dev)])
        bms = torch.arange(nmb, device=dev, dtype=torch.int32) * BLOCK_M
        bar = torch.full((nmb,), BLOCK_M, device=dev, dtype=torch.int32)
        return ag, eid, bms, bar, v, sf, tki
    t_gather = bench(_gather)
    print(f"  2. align+gather+pad+tables  : {t_gather:8.1f} us")

    # 3. kernel1 解析开销 (调用 w4a16_grouped_gemm(...) 本身, 不执行)
    def _resolve1():
        return w4a16_grouped_gemm(E, total_pad, N_GATE_UP, K, num_m_blocks,
                                  BLOCK_M, BLOCK_N, group_size=GROUP,
                                  num_stages=NUM_STAGES, threads=THREADS)
    t_resolve1 = bench(_resolve1)
    print(f"  3. kernel1 解析(查cache)    : {t_resolve1:8.1f} us")

    # 4. kernel1 实际执行 (已预编译, 直接 invoke)
    def _exec1():
        return kernel1(A_gathered, w1q, w1s, expert_ids, block_m_start, block_actual_rows)
    t_exec1 = bench(_exec1)
    print(f"  4. kernel1 执行            : {t_exec1:8.1f} us")

    # 5. silu_and_mul
    cache1 = kernel1(A_gathered, w1q, w1s, expert_ids, block_m_start, block_actual_rows)
    cache1_view = cache1.view(-1, N_GATE_UP)
    t_silu = bench(lambda: silu_and_mul(cache1_view))
    print(f"  5. silu_and_mul            : {t_silu:8.1f} us")

    # 6. kernel2 解析开销
    def _resolve2():
        return w4a16_grouped_gemm(E, total_pad, N_DOWN, N_inter, num_m_blocks,
                                  BLOCK_M, BLOCK_N, group_size=GROUP,
                                  num_stages=NUM_STAGES, threads=THREADS)
    t_resolve2 = bench(_resolve2)
    print(f"  6. kernel2 解析(查cache)    : {t_resolve2:8.1f} us")

    # 7. kernel2 实际执行
    cache_inter = silu_and_mul(cache1_view)
    def _exec2():
        return kernel2(cache_inter, w2q, w2s, expert_ids, block_m_start, block_actual_rows)
    t_exec2 = bench(_exec2)
    print(f"  7. kernel2 执行            : {t_exec2:8.1f} us")

    # 8. combine (index_add)
    cache_down = kernel2(cache_inter, w2q, w2s, expert_ids, block_m_start, block_actual_rows)
    w_flat = tw.reshape(-1).to(DTYPE)
    w_per_row = torch.where(valid, w_flat[safe_flat], torch.zeros((), device=dev, dtype=DTYPE))
    def _combine():
        cdw = cache_down * w_per_row.unsqueeze(-1)
        fin = torch.zeros(num_tokens, N_DOWN, device=dev, dtype=DTYPE)
        fin.index_add_(0, tok_idx, cdw)
        return fin
    t_combine = bench(_combine)
    print(f"  8. combine (index_add)     : {t_combine:8.1f} us")

    total = t_align + (t_gather - t_align) + t_resolve1 + t_exec1 + t_silu + t_resolve2 + t_exec2 + t_combine
    print(f"  -- 分项合计 (含重复)       : {total:8.1f} us")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)} | tilelang: {tilelang.__version__}")
    for m in [1, 2, 4, 8, 32]:
        profile_m(m)


if __name__ == "__main__":
    main()
