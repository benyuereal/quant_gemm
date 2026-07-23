#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 w4a16_fused_moe_aligned 可替换 sglang TritonRunnerCore.run 的 int4 分支.

模拟 sglang TritonRunnerCore.run 的输入 (已对齐的 sorted_token_ids/expert_ids/
num_tokens_post_padded, 由 sglang _prepare_fused_moe_run / dcu_moe_align_block_size 生成),
分别喂给:
  1) sglang _fused_moe_kernel_sequence (原始 Triton int4 路径)
  2) 我们的 w4a16_fused_moe_aligned (tilelang, 复用同样的已对齐输入)
对比输出. 若一致 (max_diff < 1e-2), 说明替换安全.

这是替换 sglang fused moe kernel 前的最终正确性门禁.
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "3")
os.environ["SGLANG_USE_AITER"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from quant_gemm.moe import w4a16_fused_moe_aligned, GROUP, DTYPE

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _fused_moe_kernel_sequence, _prepare_fused_moe_run,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import get_default_config
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", tp_size=1))

DEV = "cuda"
E = 128
HIDDEN = 6144
SHARD_INTER = 3072
N_GATE_UP = 2 * SHARD_INTER
N_DOWN = HIDDEN
TOPK = 4
BLOCK_SHAPE = [0, 128]
BLOCK_M = 16


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"维度: E={E} K={HIDDEN} shard_inter={SHARD_INTER} N_gate_up={N_GATE_UP} N_down={N_DOWN}")
    print("=" * 70)

    # 随机权重 (省内存, 正确性看 vs sglang 一致性)
    torch.manual_seed(42)
    w1q = torch.randint(0, 256, (E, N_GATE_UP, HIDDEN // 2), device=DEV, dtype=torch.uint8)
    w1s = torch.rand(E, N_GATE_UP, HIDDEN // GROUP, device=DEV, dtype=DTYPE) * 0.01 + 1e-4
    w2q = torch.randint(0, 256, (E, N_DOWN, SHARD_INTER // 2), device=DEV, dtype=torch.uint8)
    w2s = torch.rand(E, N_DOWN, SHARD_INTER // GROUP, device=DEV, dtype=DTYPE) * 0.01 + 1e-4

    all_ok = True
    for num_tokens in [1, 2, 4, 8]:
        torch.manual_seed(100 + num_tokens)
        x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
        gating = torch.randn(num_tokens, E, device=DEV, dtype=DTYPE)
        to = select_experts(x, gating, TopKConfig(top_k=TOPK, renormalize=True))
        tw = to.topk_weights.clone(); ti = to.topk_ids.clone()

        # ---- sglang 已对齐输入 (复用 _prepare_fused_moe_run 拿 sorted/expert/ntp + config) ----
        (config, down_config, down_moe_use_tma,
         sorted_token_ids, expert_ids, num_tokens_post_padded) = _prepare_fused_moe_run(
            x, w1q, w2q, ti,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=True, per_channel_quant=False, block_shape=BLOCK_SHAPE,
        )

        # ---- 1) sglang 原始 Triton int4 ----
        out_sgl = _fused_moe_kernel_sequence(
            x, w1q, w2q, tw, ti,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            config, down_config, down_moe_use_tma,
            b1=None, b2=None,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=True, per_channel_quant=False,
            w1_scale=w1s, w2_scale=w2s, w1_zp=None, w2_zp=None,
            a1_scale=None, a2_scale=None, block_shape=BLOCK_SHAPE,
            activation="silu", is_gated=True, no_combine=False, inplace=False,
            apply_router_weight_on_input=False, routed_scaling_factor=None,
            gemm1_alpha=None, gemm1_limit=None, filter_expert=False, hooks=None,
            swiglu_limit=None,
        )

        # ---- 2) 我们的 tilelang (复用同样已对齐输入) ----
        out_tl = w4a16_fused_moe_aligned(
            x, w1q, w1s, w2q, w2s, tw.to(torch.float32), ti,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            block_M=BLOCK_M, block_N=64, num_stages=2, threads=256,
        )

        d = (out_tl.float() - out_sgl.float()).abs()
        max_d = d.max().item()
        mean_d = d.mean().item()
        ok = max_d < 1e-2
        all_ok = all_ok and ok
        print(f"M={num_tokens}: tilelang vs sglang  max_diff={max_d:.4e}  mean={mean_d:.4e}  "
              f"{'✅ PASS' if ok else '❌ FAIL'}")

    print("=" * 70)
    if all_ok:
        print("✅ 全部 PASS: w4a16_fused_moe_aligned 可安全替换 sglang TritonRunnerCore.run int4 分支.")
    else:
        print("❌ 有 FAIL, 需排查.")


if __name__ == "__main__":
    main()
