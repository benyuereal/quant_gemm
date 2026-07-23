#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3 真实 shape: tilelang W4A16 fused MoE vs sglang Triton fused_moe.
同权重同输入同路由, 对比正确性 + 性能.

M3 参数 (单 TP, sglang 实际部署):
    E=128, topk=4, hidden=K=6144, shard_intermediate=1536
    w1 [E, 2*inter, K//2] = [128, 3072, 3072] uint8   (N_gate_up=3072, packed K)
    w2 [E, K, inter//2]   = [128, 6144, 768]  uint8   (N_down=6144, packed inter)
    scale [E, N, K//group] bf16, group=128
    decode: num_tokens=8 (2 req × 4 expert per token? 实际 num_tokens 个 token, 每 token topk=4)

用法:
    HIP_VISIBLE_DEVICES=2 python3 bench_tilelang_vs_sglang.py
    (设备号可用环境变量覆盖, 默认 2)
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")
os.environ["SGLANG_USE_AITER"] = "0"

import torch
from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_dtype_str, get_default_config, get_moe_configs,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", tp_size=1))

import sys, os
# 相对路径定位包根 (quant_gemm_pkg/), 不依赖绝对路径, 换机器/换目录都能跑.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from quant_gemm.moe import (
    tilelang_fused_moe_simple, quantize_int4_per_group, dequant_int4_per_group, GROUP,
)

DEV = "cuda"
DTYPE = torch.bfloat16

# M3 真实参数
E = 128
HIDDEN = 6144          # K
SHARD_INTER = 1536     # intermediate per partition
N_GATE_UP = 2 * SHARD_INTER   # 3072
N_DOWN = HIDDEN              # 6144
N_INTER = SHARD_INTER        # 1536
TOPK = 4
GROUP = 128
BLOCK_SHAPE = [0, 128]   # sglang per-group: block_n=0 (per-row N), block_k=128
DTYPE_STR = "int4_w4a16"


def make_weights(E, n_gate_up, n_down, K, n_inter, dtype=DTYPE):
    """构造 int4 W4A16 权重 (random). 布局严格匹配 sglang:
       w1 [E, n_gate_up, K//2], w2 [E, n_down, n_inter//2]
       scale [E, N, K//group] (per-group, block_n=1 即 per-row N)."""
    # 用我们的 quantize 从 bf16 生成, 保证 nibble 顺序和 scale 与 sglang kernel 期望一致
    w1_bf = torch.randn(E, n_gate_up, K, device=DEV, dtype=dtype) * 0.05
    w2_bf = torch.randn(E, n_down, n_inter, device=DEV, dtype=dtype) * 0.05
    w1_q, w1_s = quantize_int4_per_group(w1_bf, GROUP)   # [E, n_gate_up, K//2], [E, n_gate_up, K//group]
    w2_q, w2_s = quantize_int4_per_group(w2_bf, GROUP)   # [E, n_down, n_inter//2], [E, n_down, n_inter//group]
    return w1_q, w2_q, w1_s, w2_s, w1_bf, w2_bf


def run_sglang(x, w1, w2, w1s, w2s, topk_output, config):
    moe_runner_config = MoeRunnerConfig(inplace=True)
    # sglang int4 scale 布局是 [E, num_groups, N] (group 维中间, N 最内),
    # 我们内部用 [E, N, num_groups], 喂 sglang 前 transpose(1,2)
    w1s_sgl = w1s.transpose(1, 2).contiguous()
    w2s_sgl = w2s.transpose(1, 2).contiguous()
    with override_config(config):
        out = fused_moe(
            x, w1, w2, topk_output,
            moe_runner_config=moe_runner_config,
            use_int4_w4a16=True,
            w1_scale=w1s_sgl, w2_scale=w2s_sgl,
            block_shape=BLOCK_SHAPE,
        )
    return out


def bench_fn(fn, *args, num_warmup=20, num_iters=100, **kw):
    for _ in range(num_warmup):
        fn(*args, **kw)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_iters):
        fn(*args, **kw)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / num_iters * 1000  # us/call


def main():
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"M3: E={E} K={HIDDEN} inter={SHARD_INTER} N_gate_up={N_GATE_UP} N_down={N_DOWN} topk={TOPK} group={GROUP}")
    print(f"w1{tuple([E,N_GATE_UP,HIDDEN//2])} w2{tuple([E,N_DOWN,N_INTER//2])} "
          f"w1s{tuple([E,N_GATE_UP,HIDDEN//GROUP])} w2s{tuple([E,N_DOWN,N_INTER//GROUP])}")
    print("=" * 70)

    w1, w2, w1s, w2s, w1_bf, w2_bf = make_weights(E, N_GATE_UP, N_DOWN, HIDDEN, N_INTER)

    # 获取 sglang config (default + tuned)
    def_cfg = get_default_config(E, N_GATE_UP // 2, DTYPE_STR, BLOCK_SHAPE[0], BLOCK_SHAPE[1], False, False)
    try:
        tun_cfg = get_moe_configs(E, N_GATE_UP // 2, DTYPE_STR, BLOCK_SHAPE[0], BLOCK_SHAPE[1], False, False)
        if tun_cfg is None:
            tun_cfg = def_cfg
    except Exception as ex:
        print(f"  [warn] get_moe_configs failed: {ex}, 用 default")
        tun_cfg = def_cfg
    print(f"sglang default config: {def_cfg}")
    print(f"sglang tuned  config: {tun_cfg}")

    # 多个 batch size (decode 场景)
    for num_tokens in [1, 2, 8, 32]:
        print(f"\n----- num_tokens={num_tokens} (token-expert pairs={num_tokens*TOPK}) -----")
        torch.manual_seed(42)
        x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
        gating = torch.randn(num_tokens, E, device=DEV, dtype=DTYPE)
        topk_config = TopKConfig(top_k=TOPK, renormalize=True)
        topk_output = select_experts(x, gating, topk_config)
        topk_w = topk_output.topk_weights.clone()
        topk_i = topk_output.topk_ids.clone()
        router_logits = topk_output.router_logits.clone()

        # ---- 正确性: sglang vs tilelang ----
        out_sgl = run_sglang(x, w1, w2, w1s, w2s, topk_output, def_cfg)
        # 恢复 topk (sglang inplace 可能改)
        topk_output.topk_weights.copy_(topk_w)
        topk_output.topk_ids.copy_(topk_i)
        topk_output.router_logits.copy_(router_logits)

        out_tl = tilelang_fused_moe_simple(
            x, w1, w1s, w2, w2s,
            topk_w.to(torch.float32), topk_i.to(torch.int32),
            block_M=16, block_N=64, num_stages=2, threads=256,
        )

        max_diff = (out_sgl.float() - out_tl.float()).abs().max().item()
        rel = (out_sgl.float() - out_tl.float()).abs().mean().item() / max(out_sgl.abs().mean().item(), 1e-12)
        print(f"  [正确性] sglang vs tilelang: max_diff={max_diff:.4e}  rel={rel*100:.4f}%")

        # ---- 性能 ----
        # sglang (default + tuned)
        def _sgl_run():
            topk_output.topk_weights.copy_(topk_w)
            topk_output.topk_ids.copy_(topk_i)
            topk_output.router_logits.copy_(router_logits)
            return run_sglang(x, w1, w2, w1s, w2s, topk_output, def_cfg)
        t_sgl_def = bench_fn(lambda: _sgl_run())

        def _sgl_run_tun():
            topk_output.topk_weights.copy_(topk_w)
            topk_output.topk_ids.copy_(topk_i)
            topk_output.router_logits.copy_(router_logits)
            return run_sglang(x, w1, w2, w1s, w2s, topk_output, tun_cfg)
        t_sgl_tun = bench_fn(lambda: _sgl_run_tun())

        def _tl_run():
            return tilelang_fused_moe_simple(
                x, w1, w1s, w2, w2s,
                topk_w.to(torch.float32), topk_i.to(torch.int32),
                block_M=16, block_N=64, num_stages=2, threads=256,
            )
        t_tl = bench_fn(_tl_run)

        print(f"  [性能] sglang-default={t_sgl_def:.1f}us  sglang-tuned={t_sgl_tun:.1f}us  tilelang={t_tl:.1f}us")
        print(f"         tilelang vs default={t_sgl_def/t_tl:.2f}x  vs tuned={t_sgl_tun/t_tl:.2f}x")


if __name__ == "__main__":
    main()
