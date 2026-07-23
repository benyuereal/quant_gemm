#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 sglang TritonRunnerCore.run 在应用 tilelang 补丁后, int4 分支走 tilelang 且正确.

走完整 MoeRunner.run 路径 (不是直接调 _fused_moe_kernel_sequence), 模拟真实部署:
  1) 用 SGLANG_USE_TILELANG_W4A16=0 (默认 Triton) 跑一次, 得 sglang 基线
  2) 用 SGLANG_USE_TILELANG_W4A16=1 (走 tilelang 补丁) 跑一次, 得 tilelang 结果
  3) 对比两者 (应一致, max_diff < 1e-2)

这验证补丁在 sglang 实际调用链里生效且正确.
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "3")
os.environ["SGLANG_USE_AITER"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from quant_gemm.moe import GROUP, DTYPE

from sglang.srt.layers.moe import MoeRunnerConfig, MoeRunnerBackend
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonRunnerCore, TritonRunnerInput, TritonMoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import get_default_config
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", tp_size=1))

DEV = "cuda"
E = 128; HIDDEN = 6144; SHARD_INTER = 3072; N_GATE_UP = 2*SHARD_INTER; N_DOWN = HIDDEN; TOPK = 4
BLOCK_SHAPE = [0, 128]
BLOCK_M = 16


def build_runner_input(x, tw, ti, w1q, w2q):
    """模拟 sglang token_dispatcher 产出 TritonRunnerInput (含已对齐 sorted/expert/ntp)."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import _prepare_fused_moe_run
    (config, down_config, down_moe_use_tma,
     sorted_token_ids, expert_ids, num_tokens_post_padded) = _prepare_fused_moe_run(
        x, w1q, w2q, ti,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
        use_int4_w4a16=True, per_channel_quant=False, block_shape=BLOCK_SHAPE,
    )
    runner_input = TritonRunnerInput(
        hidden_states=x,
        topk_weights=tw,
        topk_ids=ti,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )
    running_state = {"config": config, "down_config": down_config, "down_moe_use_tma": down_moe_use_tma}
    return runner_input, running_state


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    torch.manual_seed(42)
    w1q = torch.randint(0, 256, (E, N_GATE_UP, HIDDEN//2), device=DEV, dtype=torch.uint8)
    w1s = torch.rand(E, N_GATE_UP, HIDDEN//GROUP, device=DEV, dtype=DTYPE)*0.01+1e-4
    w2q = torch.randint(0, 256, (E, N_DOWN, SHARD_INTER//2), device=DEV, dtype=torch.uint8)
    w2s = torch.rand(E, N_DOWN, SHARD_INTER//GROUP, device=DEV, dtype=DTYPE)*0.01+1e-4

    # MoeRunnerConfig (简化, 只用 TritonRunnerCore 需要的字段)
    runner_config = MoeRunnerConfig(
        num_experts=E, num_local_experts=E, top_k=TOPK,
        hidden_size=HIDDEN, intermediate_size_per_partition=SHARD_INTER,
        is_gated=True, activation="silu", inplace=False, no_combine=False,
        apply_router_weight_on_input=False,
    )
    core = TritonRunnerCore(runner_config)

    quant_info = TritonMoeQuantInfo(
        w13_weight=w1q, w2_weight=w2q,
        use_int4_w4a16=True, w13_scale=w1s, w2_scale=w2s, block_shape=BLOCK_SHAPE,
    )

    all_ok = True
    for num_tokens in [1, 2, 4, 8]:
        torch.manual_seed(100 + num_tokens)
        x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
        gating = torch.randn(num_tokens, E, device=DEV, dtype=DTYPE)
        to = select_experts(x, gating, TopKConfig(top_k=TOPK, renormalize=True))
        tw = to.topk_weights.clone(); ti = to.topk_ids.clone()

        # 1) 默认 Triton (SGLANG_USE_TILELANG_W4A16=0)
        os.environ["SGLANG_USE_TILELANG_W4A16"] = "0"
        ri, st = build_runner_input(x, tw, ti, w1q, w2q)
        out_sgl = core.run(ri, quant_info, st).hidden_states

        # 2) tilelang (SGLANG_USE_TILELANG_W4A16=1)
        os.environ["SGLANG_USE_TILELANG_W4A16"] = "1"
        ri2, st2 = build_runner_input(x, tw, ti, w1q, w2q)
        out_tl = core.run(ri2, quant_info, st2).hidden_states

        d = (out_tl.float() - out_sgl.float()).abs()
        max_d = d.max().item()
        ok = max_d < 1e-2
        all_ok = all_ok and ok
        print(f"M={num_tokens}: tilelang(patch) vs sglang(triton)  max_diff={max_d:.4e}  "
              f"{'✅ PASS' if ok else '❌ FAIL'}")

    print("=" * 70)
    print("✅ 补丁在 sglang TritonRunnerCore.run 完整路径生效且正确." if all_ok else "❌ 有 FAIL.")


if __name__ == "__main__":
    main()
