#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐 op 计时 tilelang_fused_moe_simple 的 host-side Python 提交开销 (不 sync).
定位 ~790us host 开销在哪些 op, 指导合并/预分配优化.

测法: 每个 op 单独跑 N 次, 用 CPU perf_counter 测提交时间 (kernel 异步, 不 sync,
故测的是纯 Python + launch dispatch 开销). 最后 sync 一次排空.
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "3")
os.environ["SGLANG_USE_AITER"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import time
import torch
from quant_gemm.moe import w4a16_grouped_gemm, quantize_int4_per_group, GROUP, DTYPE
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import dcu_moe_align_block_size
from sgl_kernel import silu_and_mul
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", tp_size=1))

DEV = "cuda"
E = 128; HIDDEN = 6144; SHARD_INTER = 3072; N_GATE_UP = 2*SHARD_INTER; N_DOWN = HIDDEN; TOPK = 4
BLOCK_M = 16; BLOCK_N = 64; NUM_STAGES = 2; THREADS = 256


def time_op(fn, n=300):
    """测 host 提交时间 (不 sync). 返回 us/op."""
    for _ in range(20):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1e6


def main():
    torch.manual_seed(42)
    # 随机权重 (省内存)
    w1q = torch.randint(0, 256, (E, N_GATE_UP, HIDDEN//2), device=DEV, dtype=torch.uint8)
    w1s = torch.rand(E, N_GATE_UP, HIDDEN//GROUP, device=DEV, dtype=DTYPE)*0.01+1e-4
    w2q = torch.randint(0, 256, (E, N_DOWN, SHARD_INTER//2), device=DEV, dtype=torch.uint8)
    w2s = torch.rand(E, N_DOWN, SHARD_INTER//GROUP, device=DEV, dtype=DTYPE)*0.01+1e-4

    num_tokens = 1
    x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
    logits = torch.randn(num_tokens, E, device=DEV, dtype=DTYPE)
    tw, ti = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)
    ti = ti.to(torch.int32)
    tw32 = tw.to(torch.float32)

    K = HIDDEN; N_inter = SHARD_INTER; dev = DEV; M_flat = num_tokens*TOPK

    # 预跑一次拿形状
    sorted_ids, expert_ids, ntp = dcu_moe_align_block_size(ti, BLOCK_M, E)
    sorted_ids = sorted_ids.to(torch.int64)
    total_pad0 = sorted_ids.numel(); num_m_blocks = expert_ids.numel()
    ntp_val = int(ntp.item())
    total_gem = num_m_blocks * BLOCK_M
    nvb = torch.tensor([(ntp_val+BLOCK_M-1)//BLOCK_M], device=dev, dtype=torch.int32)

    # 预编译 kernel
    k1 = w4a16_grouped_gemm(E, total_gem, N_GATE_UP, K, num_m_blocks, BLOCK_M, BLOCK_N, group_size=GROUP, num_stages=NUM_STAGES, threads=THREADS)
    k2 = w4a16_grouped_gemm(E, total_gem, N_DOWN, N_inter, num_m_blocks, BLOCK_M, BLOCK_N, group_size=GROUP, num_stages=NUM_STAGES, threads=THREADS)
    # 预备 buffer
    valid = sorted_ids < M_flat
    safe_flat = sorted_ids.clamp(max=M_flat-1)
    tok_idx = safe_flat // TOPK
    A_gathered = x[tok_idx] * valid.unsqueeze(-1).to(DTYPE)
    if total_gem > total_pad0:
        A_gathered = torch.cat([A_gathered, torch.zeros(total_gem-total_pad0, K, device=dev, dtype=DTYPE)], 0)
        valid = torch.cat([valid, torch.zeros(total_gem-total_pad0, dtype=torch.bool, device=dev)])
        safe_flat = torch.cat([safe_flat, torch.zeros(total_gem-total_pad0, dtype=torch.int64, device=dev)])
        tok_idx = torch.cat([tok_idx, torch.zeros(total_gem-total_pad0, dtype=torch.int64, device=dev)])
    bms = torch.arange(num_m_blocks, device=dev, dtype=torch.int32)*BLOCK_M
    bar = torch.full((num_m_blocks,), BLOCK_M, device=dev, dtype=torch.int32)
    cache1 = k1(A_gathered, w1q, w1s, expert_ids, bms, bar, nvb)
    cache_inter = silu_and_mul(cache1.view(-1, N_GATE_UP))
    cache_down = k2(cache_inter, w2q, w2s, expert_ids, bms, bar, nvb)
    w_flat = tw32.reshape(-1).to(DTYPE)
    torch.cuda.synchronize()

    print(f"M={num_tokens} total_gem={total_gem} num_m_blocks={num_m_blocks} ntp_val={ntp_val}")
    print("=" * 60)
    total = 0.0
    rows = []

    def rec(name, fn):
        nonlocal total
        t = time_op(fn)
        rows.append((name, t)); total += t

    # 1. align
    rec("dcu_align", lambda: dcu_moe_align_block_size(ti, BLOCK_M, E))
    # 2. sorted_ids.to(int64)
    rec("sorted_ids.to(int64)", lambda: dcu_moe_align_block_size(ti, BLOCK_M, E)[0].to(torch.int64))
    # 3. valid = sorted < M_flat
    rec("valid = sorted<M", lambda: sorted_ids < M_flat)
    # 4. clamp
    rec("safe_flat = clamp", lambda: sorted_ids.clamp(max=M_flat-1))
    # 5. // topk
    rec("tok_idx = //topk", lambda: safe_flat // TOPK)
    # 6. gather
    rec("A_gathered = x[tok_idx]", lambda: x[tok_idx])
    # 7. mul mask
    rec("A *= mask", lambda: A_gathered * valid.unsqueeze(-1).to(DTYPE))
    # 8. cat pad (A_gathered)
    if total_gem > total_pad0:
        rec("cat A_gathered", lambda: torch.cat([A_gathered[:total_pad0], torch.zeros(total_gem-total_pad0, K, device=dev, dtype=DTYPE)], 0))
        rec("cat valid/safe/tok", lambda: (torch.cat([valid[:total_pad0], torch.zeros(total_gem-total_pad0, dtype=torch.bool, device=dev)]),
                                            torch.cat([safe_flat[:total_pad0], torch.zeros(total_gem-total_pad0, dtype=torch.int64, device=dev)]),
                                            torch.cat([tok_idx[:total_pad0], torch.zeros(total_gem-total_pad0, dtype=torch.int64, device=dev)])))
    # 9. arange + full
    rec("arange bms", lambda: torch.arange(num_m_blocks, device=dev, dtype=torch.int32)*BLOCK_M)
    rec("full bar", lambda: torch.full((num_m_blocks,), BLOCK_M, device=dev, dtype=torch.int32))
    rec("tensor nvb", lambda: torch.tensor([(ntp_val+BLOCK_M-1)//BLOCK_M], device=dev, dtype=torch.int32))
    # 10. kernel1 解析
    rec("k1 resolve", lambda: w4a16_grouped_gemm(E, total_gem, N_GATE_UP, K, num_m_blocks, BLOCK_M, BLOCK_N, group_size=GROUP, num_stages=NUM_STAGES, threads=THREADS))
    # 11. kernel1 执行
    rec("k1 exec", lambda: k1(A_gathered, w1q, w1s, expert_ids, bms, bar, nvb))
    # 12. view + silu
    rec("silu_and_mul", lambda: silu_and_mul(cache1.view(-1, N_GATE_UP)))
    # 13. kernel2 解析
    rec("k2 resolve", lambda: w4a16_grouped_gemm(E, total_gem, N_DOWN, N_inter, num_m_blocks, BLOCK_M, BLOCK_N, group_size=GROUP, num_stages=NUM_STAGES, threads=THREADS))
    # 14. kernel2 执行
    rec("k2 exec", lambda: k2(cache_inter, w2q, w2s, expert_ids, bms, bar, nvb))
    # 15. combine
    rec("w_flat = reshape+to", lambda: tw32.reshape(-1).to(DTYPE))
    rec("slice cd/sf/tok/valid", lambda: (cache_down[:ntp_val], safe_flat[:ntp_val], tok_idx[:ntp_val], valid[:ntp_val]))
    rec("w_flat[sf]", lambda: w_flat[safe_flat[:ntp_val]])
    rec("where(valid,...)", lambda: torch.where(valid[:ntp_val], w_flat[safe_flat[:ntp_val]], torch.zeros((), device=dev, dtype=DTYPE)))
    rec("cd * wpr", lambda: cache_down[:ntp_val] * torch.where(valid[:ntp_val], w_flat[safe_flat[:ntp_val]], torch.zeros((), device=dev, dtype=DTYPE)).unsqueeze(-1))
    rec("zeros final", lambda: torch.zeros(num_tokens, N_DOWN, device=dev, dtype=DTYPE))
    rec("index_add_", lambda: torch.zeros(num_tokens, N_DOWN, device=dev, dtype=DTYPE).index_add_(0, tok_idx[:ntp_val], cache_down[:ntp_val]))

    rows.sort(key=lambda r: -r[1])
    print(f"{'op':<28} {'us':>8}")
    print("-" * 40)
    for name, t in rows:
        print(f"{name:<28} {t:8.1f}")
    print("-" * 40)
    print(f"{'host 合计':<28} {total:8.1f}")


if __name__ == "__main__":
    main()
