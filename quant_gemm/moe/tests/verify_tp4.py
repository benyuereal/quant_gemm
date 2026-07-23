#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 tilelang W4A16 fused MoE 在 TP=4 (每卡 E=32) 语义下正确.

真实部署 TP=4: 128 expert 分到 4 卡, 每卡 E_local=32. sglang EP 下:
  - 每卡 w13 [32, N_gate_up, K//2], w2 [32, N_down, N_inter//2]
  - 每卡只处理路由到本卡 32 个 expert 的 token
  - dcu_align 的 E 参数 = w1.shape[0] = 32, expert_ids 值域 [0, 32)
  - sorted_token_ids 仍是 (token*topk+k) 扁平索引, 但只含本卡 expert 的对

本测试模拟 TP=4 单卡视角: 取 32 个 expert, 路由只指向这 32 个, 对比 tilelang vs sglang.
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "3")
os.environ["SGLANG_USE_AITER"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from quant_gemm.moe import w4a16_fused_moe_aligned, GROUP, DTYPE

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _fused_moe_kernel_sequence, _prepare_fused_moe_run,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", tp_size=1))

DEV = "cuda"
# TP=4: 每卡 E_local=32 (128/4). 维度同真实产物 (inter 不切).
E_LOCAL = 32
HIDDEN = 6144
SHARD_INTER = 3072
N_GATE_UP = 2 * SHARD_INTER
N_DOWN = HIDDEN
TOPK = 4
BLOCK_SHAPE = [0, 128]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"模拟 TP=4 单卡视角: E_local={E_LOCAL} K={HIDDEN} shard_inter={SHARD_INTER}")
    print("=" * 70)

    torch.manual_seed(42)
    w1q = torch.randint(0, 256, (E_LOCAL, N_GATE_UP, HIDDEN // 2), device=DEV, dtype=torch.uint8)
    w1s = torch.rand(E_LOCAL, N_GATE_UP, HIDDEN // GROUP, device=DEV, dtype=DTYPE) * 0.01 + 1e-4
    w2q = torch.randint(0, 256, (E_LOCAL, N_DOWN, SHARD_INTER // 2), device=DEV, dtype=torch.uint8)
    w2s = torch.rand(E_LOCAL, N_DOWN, SHARD_INTER // GROUP, device=DEV, dtype=DTYPE) * 0.01 + 1e-4

    all_ok = True
    for num_tokens in [1, 2, 4, 8, 16]:
        torch.manual_seed(100 + num_tokens)
        x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
        # 路由只指向本卡 32 个 expert (模拟 EP: 跨卡 expert 的 token 不在本卡处理)
        gating = torch.randn(num_tokens, E_LOCAL, device=DEV, dtype=DTYPE)
        to = select_experts(x, gating, TopKConfig(top_k=TOPK, renormalize=True))
        tw = to.topk_weights.clone(); ti = to.topk_ids.clone()   # ti 值域 [0, 32)

        # sglang 已对齐输入
        (config, down_config, down_moe_use_tma,
         sorted_token_ids, expert_ids, num_tokens_post_padded) = _prepare_fused_moe_run(
            x, w1q, w2q, ti,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=True, per_channel_quant=False, block_shape=BLOCK_SHAPE,
        )

        # 1) sglang Triton int4
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

        # 2) tilelang
        out_tl = w4a16_fused_moe_aligned(
            x, w1q, w1s, w2q, w2s, tw.to(torch.float32), ti,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            block_M=16, block_N=64, num_stages=2, threads=256,
        )

        d = (out_tl.float() - out_sgl.float()).abs()
        max_d = d.max().item()
        ok = max_d < 1e-2
        all_ok = all_ok and ok
        print(f"M={num_tokens}: tilelang vs sglang  max_diff={max_d:.4e}  "
              f"{'✅ PASS' if ok else '❌ FAIL'}")

    print("=" * 70)
    if all_ok:
        print("✅ TP=4 单卡视角 (E_local=32) 全 PASS: tilelang 兼容 EP, 可安全用于 TP=4 部署.")
    else:
        print("❌ 有 FAIL, 需排查 EP 兼容性.")


if __name__ == "__main__":
    main()
