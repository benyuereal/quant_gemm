#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3 真实 shape: tilelang W4A16 fused MoE 性能 vs sglang Triton fused_moe.

正确性基线: tilelang vs deq 反量化参考 (sglang int4 的 scale/group 语义与我们不同,
  直接喂同权重 sglang 会算错, 故 sglang 仅作性能基线 -- 它自洽 max_diff=0, 速度有效).
性能基线: sglang Triton (default + tuned config).

M3 参数 (intermediate=3072, TP=4 -> shard_inter=768? 实测产物 w1 N=3072=2*1536 -> shard_inter=1536):
  这里用 shard_inter=1536 (与 verify_moe_config.py / 产物 w1 N=3072 一致):
    E=128, topk=4, K=hidden=6144, shard_inter=1536
    w1 [E, 2*shard_inter, K//2] = [128, 3072, 3072] uint8
    w2 [E, K, shard_inter//2]   = [128, 6144, 768]  uint8
    scale [E, N, K//group] (group=128 unpacked, tilelang 期望)
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ["SGLANG_USE_AITER"] = "0"

import torch
import torch.nn.functional as F
from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_default_config, get_moe_configs,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", tp_size=1))

import sys, os
# 相对路径定位包根 (quant_gemm_pkg/), 不依赖绝对路径, 换机器/换目录都能跑.
# __file__ = .../quant_gemm_pkg/quant_gemm/moe/tests/bench_perf.py, 往上 3 级到 quant_gemm_pkg/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from quant_gemm.moe import (
    tilelang_fused_moe_simple, quantize_int4_per_group, dequant_int4_per_group, GROUP,
)

DEV = "cuda"
DTYPE = torch.bfloat16
E = 128
HIDDEN = 6144
SHARD_INTER = 1536
N_GATE_UP = 2 * SHARD_INTER
N_DOWN = HIDDEN
TOPK = 4
BLOCK_SHAPE = [0, 128]
DTYPE_STR = "int4_w4a16"


def make_quant_weights():
    """bf16 -> int4 (我们的 zp=8 对称量化). tilelang 用这套 (已验证 vs ref 1%).
    sglang 也喂同张量测速 (sglang 会算错但自洽, 速度有效)."""
    w1_bf = torch.randn(E, N_GATE_UP, HIDDEN, device=DEV, dtype=DTYPE) * 0.02
    w2_bf = torch.randn(E, N_DOWN, SHARD_INTER, device=DEV, dtype=DTYPE) * 0.02
    w1q, w1s = quantize_int4_per_group(w1_bf, GROUP)
    w2q, w2s = quantize_int4_per_group(w2_bf, GROUP)
    return w1q, w2q, w1s, w2s, w1_bf, w2_bf


def deq_ref(x, w1q, w1s, w2q, w2s, topk_w, topk_i):
    w1d = dequant_int4_per_group(w1q, w1s, GROUP)
    w2d = dequant_int4_per_group(w2q, w2s, GROUP)
    nt = x.shape[0]
    ref = torch.zeros(nt, N_DOWN, device=DEV, dtype=DTYPE)
    for t in range(nt):
        for k in range(TOPK):
            e = int(topk_i[t, k])
            g = x[t] @ w1d[e].T
            inter = F.silu(g[:SHARD_INTER]) * g[SHARD_INTER:]
            ref[t] += inter @ w2d[e].T * topk_w[t, k]
    return ref


def bench_fn(fn, num_warmup=20, num_iters=100):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / num_iters * 1000  # us/call


def main():
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"M3: E={E} K={HIDDEN} shard_inter={SHARD_INTER} N_gate_up={N_GATE_UP} N_down={N_DOWN} topk={TOPK}")
    print("=" * 70)

    w1q, w2q, w1s, w2s, _, _ = make_quant_weights()

    # sglang tuned configs (按 batch 索引的字典)
    N_cfg = SHARD_INTER // 2 // 2   # 384, 与 config 文件名一致
    try:
        tuned_configs = get_moe_configs(E, N_cfg, DTYPE_STR, BLOCK_SHAPE[0], BLOCK_SHAPE[1], False, False) or {}
    except Exception:
        tuned_configs = {}
    print(f"sglang tuned configs batch keys: {sorted(tuned_configs.keys()) if tuned_configs else 'none'}")

    for num_tokens in [1, 2, 4, 8, 16, 32]:
        torch.manual_seed(42)
        x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
        gating = torch.randn(num_tokens, E, device=DEV, dtype=DTYPE)
        to = select_experts(x, gating, TopKConfig(top_k=TOPK, renormalize=True))
        tw = to.topk_weights.clone(); ti = to.topk_ids.clone(); rl = to.router_logits.clone()

        # ---- 正确性: tilelang vs deq ref ----
        out_tl = tilelang_fused_moe_simple(x, w1q, w1s, w2q, w2s, tw.to(torch.float32), ti.to(torch.int32),
                                           block_M=16, block_N=64, num_stages=2, threads=256)
        if num_tokens <= 8:
            ref = deq_ref(x, w1q, w1s, w2q, w2s, tw, ti)
            err = (out_tl.float() - ref.float()).abs().max().item()
            rel = (out_tl.float() - ref.float()).abs().mean().item() / max(ref.abs().mean().item(), 1e-12)
            corr = f"max_diff={err:.2e} rel={rel*100:.3f}%"
        else:
            corr = "(skip ref, 大 batch)"

        # ---- 性能 ----
        def _tl():
            return tilelang_fused_moe_simple(x, w1q, w1s, w2q, w2s, tw.to(torch.float32), ti.to(torch.int32),
                                             block_M=16, block_N=64, num_stages=2, threads=256)
        t_tl = bench_fn(_tl)

        def _sgl(cfg):
            def run():
                to.topk_weights.copy_(tw); to.topk_ids.copy_(ti); to.router_logits.copy_(rl)
                with override_config(cfg):
                    return fused_moe(x, w1q, w2q, to, MoeRunnerConfig(inplace=True),
                                     use_int4_w4a16=True, w1_scale=w1s, w2_scale=w2s, block_shape=BLOCK_SHAPE)
            return run
        def_cfg = get_default_config(num_tokens, E, SHARD_INTER, HIDDEN, TOPK, DTYPE_STR, False, BLOCK_SHAPE)
        t_sgl_def = bench_fn(_sgl(def_cfg))
        if tuned_configs:
            tun_key = min(tuned_configs.keys(), key=lambda k: abs(k - num_tokens))
            tun_cfg = tuned_configs[tun_key]
        else:
            tun_cfg = def_cfg
        t_sgl_tun = bench_fn(_sgl(tun_cfg))

        speed_vs_def = t_sgl_def / t_tl
        speed_vs_tun = t_sgl_tun / t_tl
        print(f"M={num_tokens:3d} (pairs={num_tokens*TOPK:4d}): tilelang={t_tl:8.1f}us  "
              f"sglang-def={t_sgl_def:8.1f}us  sglang-tuned={t_sgl_tun:8.1f}us  "
              f"| vs def={speed_vs_def:.2f}x vs tuned={speed_vs_tun:.2f}x  | {corr}")


if __name__ == "__main__":
    main()
