#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端验证: 真实 M3 W4A16 产物权重 -> sglang 海光路径 weight 转换 -> 喂 tilelang 算子,
对比 sglang Triton fused_moe_kernel_gptq_awq 输出.

这是 "tilelang 算子可安全替换 sglang triton moe" 的最终判据:
  - 若 tilelang vs sglang 输出一致 (max_diff 接近 0), 布局/语义/nibble 顺序全兼容, 可替换.
  - 若不一致, 定位是 nibble 顺序还是 scale 语义问题.

真实产物 (单 expert, weight_shape 给的真值):
    w1 (gate): [3072, 6144] -> packed (3072, 768) int32   [N_gate=3072, K//8=768]
    w3 (up):   [3072, 6144] -> packed (3072, 768) int32   [N_up=3072, K//8]
    w2 (down): [6144, 3072] -> packed (6144, 384) int32   [N_down=6144, N_inter//8=384]
    scale: [N, K//group] bf16, group=128
  => shard_intermediate=3072, N_gate_up=2*3072=6144, N_inter=3072, K=hidden=6144, N_down=6144
  (此产物 TP=1 / inter 不切)

sglang 海光转换 (CompressedTensorsWNA16TritonMoE.process_weights_after_loading):
    w13: [E, K//8, N] int32 --transpose(1,2)--> [E, N, K//8] --view(uint8)--> [E, N, K//2] uint8
    w13_scale: [E, K//group, N] --transpose(1,2)--> [E, N, K//group]
  产物单 expert 是 [N, K//8] (已转置好, 无 E/K 在前), 直接 view(uint8) 即 [N, K//2].
  w13 = cat([gate, up], dim=N) -> [N_gate_up=6144, K//2].
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "3")
os.environ["SGLANG_USE_AITER"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.nn.functional as F
from safetensors import safe_open
from quant_gemm.moe import tilelang_fused_moe_simple, dequant_int4_per_group, GROUP

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import get_default_config
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", tp_size=1))

DEV = "cuda"
DTYPE = torch.bfloat16
PRODUCT = "/models/MiniMax/MiniMax-M3-w4a16-moe-only"
SHARD_FILE = "model-00010-of-00059.safetensors"   # 含 layer.10 experts
LAYER = 10
E = 128
HIDDEN = 6144
SHARD_INTER = 3072          # 真实产物 (TP=1, inter 不切)
N_GATE_UP = 2 * SHARD_INTER # 6144
N_INTER = SHARD_INTER       # 3072
N_DOWN = HIDDEN             # 6144
TOPK = 4
BLOCK_SHAPE = [0, 128]


def load_one_expert(expert_id):
    """读单个 expert 的 w1(gate)/w3(up)/w2(down) int32 packed + scale, 转 uint8, 组装 w13/w2.
    返回 w13_u8 [N_gate_up, K//2] uint8, w13_scale [N_gate_up, K//group] bf16,
           w2_u8 [N_down, N_inter//2], w2_scale [N_down, N_inter//group]."""
    f = os.path.join(PRODUCT, SHARD_FILE)
    pre = f"language_model.model.layers.{LAYER}.block_sparse_moe.experts.{expert_id}"
    with safe_open(f, framework="pt") as st:
        w1_p = st.get_tensor(f"{pre}.w1.weight_packed")   # (3072, 768) int32 = [N_gate, K//8]
        w1_s = st.get_tensor(f"{pre}.w1.weight_scale")    # (3072, 48) bf16 = [N_gate, K//group]
        w3_p = st.get_tensor(f"{pre}.w3.weight_packed")   # (3072, 768) = [N_up, K//8]
        w3_s = st.get_tensor(f"{pre}.w3.weight_scale")
        w2_p = st.get_tensor(f"{pre}.w2.weight_packed")   # (6144, 384) = [N_down, N_inter//8]
        w2_s = st.get_tensor(f"{pre}.w2.weight_scale")    # (6144, 24) = [N_down, N_inter//group]
    # int32 -> uint8 (sglang 海光 view(uint8)): [N, K//8] int32 -> [N, K//2] uint8
    w1_u8 = w1_p.contiguous().view(torch.uint8)
    w3_u8 = w3_p.contiguous().view(torch.uint8)
    w2_u8 = w2_p.contiguous().view(torch.uint8)
    # w13 = cat([gate, up], dim=0=N)  -> [N_gate_up, K//2]
    w13_u8 = torch.cat([w1_u8, w3_u8], dim=0).contiguous()
    w13_scale = torch.cat([w1_s, w3_s], dim=0).contiguous()
    assert w13_u8.shape == (N_GATE_UP, HIDDEN // 2), w13_u8.shape
    assert w13_scale.shape == (N_GATE_UP, HIDDEN // GROUP), w13_scale.shape
    assert w2_u8.shape == (N_DOWN, N_INTER // 2), w2_u8.shape
    assert w2_s.shape == (N_DOWN, N_INTER // GROUP), w2_s.shape
    return w13_u8, w13_scale, w2_u8, w2_s


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"真实维度: shard_inter={SHARD_INTER} N_gate_up={N_GATE_UP} N_inter={N_INTER} N_down={N_DOWN} K={HIDDEN}")
    print("=" * 70)

    # 读若干个不同 expert, 组装成 [E, ...] (用真实不同权重, 不是复制)
    num_real_experts = 8   # 读 8 个真实 expert, 复制/填充到 E=128
    w13_list, w13s_list, w2_list, w2s_list = [], [], [], []
    for e in range(num_real_experts):
        w13, w13s, w2, w2s = load_one_expert(e)
        w13_list.append(w13); w13s_list.append(w13s)
        w2_list.append(w2); w2s_list.append(w2s)
    print(f"读取 {num_real_experts} 个真实 expert 权重 OK")

    # 组装 [E, ...]: 8 个真实 expert 复制到 128 (每个真实 expert 填 16 个槽)
    reps = E // num_real_experts
    w13_E = torch.stack(w13_list * reps, dim=0)      # [E, N_gate_up, K//2]
    w13s_E = torch.stack(w13s_list * reps, dim=0)    # [E, N_gate_up, K//group]
    w2_E = torch.stack(w2_list * reps, dim=0)        # [E, N_down, N_inter//2]
    w2s_E = torch.stack(w2s_list * reps, dim=0)      # [E, N_down, N_inter//group]
    w13_E = w13_E.to(DEV); w13s_E = w13s_E.to(DEV); w2_E = w2_E.to(DEV); w2s_E = w2s_E.to(DEV)
    print(f"组装: w13_E {tuple(w13_E.shape)} {w13_E.dtype}, w2_E {tuple(w2_E.shape)}, "
          f"w13s_E {tuple(w13s_E.shape)}, w2s_E {tuple(w2s_E.shape)}")

    # 激活 + 随机路由
    num_tokens = 4
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DTYPE) * 0.1
    gating = torch.randn(num_tokens, E, device=DEV, dtype=DTYPE)
    to = select_experts(x, gating, TopKConfig(top_k=TOPK, renormalize=True))
    tw = to.topk_weights.clone(); ti = to.topk_ids.clone(); rl = to.router_logits.clone()
    print(f"路由 topk_ids: {ti.tolist()}")

    # ---- tilelang ----
    out_tl = tilelang_fused_moe_simple(x, w13_E, w13s_E, w2_E, w2s_E,
                                       tw.to(torch.float32), ti.to(torch.int32),
                                       block_M=16, block_N=64, num_stages=2, threads=256)

    # ---- sglang Triton int4 (同权重同布局) ----
    def _sgl():
        to.topk_weights.copy_(tw); to.topk_ids.copy_(ti); to.router_logits.copy_(rl)
        cfg = get_default_config(num_tokens, E, SHARD_INTER, HIDDEN, TOPK, "int4_w4a16", False, BLOCK_SHAPE)
        with override_config(cfg):
            return fused_moe(x, w13_E, w2_E, to, MoeRunnerConfig(inplace=True),
                             use_int4_w4a16=True, w1_scale=w13s_E, w2_scale=w2s_E, block_shape=BLOCK_SHAPE)
    out_sgl = _sgl()

    # ---- dequant 参考 (只反量化被路由到的 expert, 避免全 E 反量化 OOM) ----
    ref = torch.zeros(num_tokens, N_DOWN, device=DEV, dtype=DTYPE)
    used_experts = sorted(set(int(ti[t, k]) for t in range(num_tokens) for k in range(TOPK)))
    w13_deq_cache = {}
    w2_deq_cache = {}
    for e in used_experts:
        w13_deq_cache[e] = dequant_int4_per_group(w13_E[e], w13s_E[e], GROUP)   # [N_gate_up, K]
        w2_deq_cache[e] = dequant_int4_per_group(w2_E[e], w2s_E[e], GROUP)      # [N_down, N_inter]
    for t in range(num_tokens):
        for k in range(TOPK):
            e = int(ti[t, k])
            g = x[t] @ w13_deq_cache[e].T                  # [N_gate_up]
            inter = F.silu(g[:N_INTER]) * g[N_INTER:]      # [N_inter]
            ref[t] += inter @ w2_deq_cache[e].T * tw[t, k]
    del w13_deq_cache, w2_deq_cache
    torch.cuda.empty_cache()

    # ---- 对比 ----
    print("\n" + "=" * 70)
    print("对比 (真实 M3 产物权重, 8 真实 expert 复制到 E=128):")
    d_tl_sgl = (out_tl.float() - out_sgl.float()).abs()
    d_tl_ref = (out_tl.float() - ref.float()).abs()
    d_sgl_ref = (out_sgl.float() - ref.float()).abs()
    print(f"  tilelang vs sglang : max_diff={d_tl_sgl.max().item():.4e}  mean={d_tl_sgl.mean().item():.4e}")
    print(f"  tilelang vs deq_ref: max_diff={d_tl_ref.max().item():.4e}  rel={(d_tl_ref.mean()/ref.abs().mean()).item()*100:.4f}%")
    print(f"  sglang   vs deq_ref: max_diff={d_sgl_ref.max().item():.4e}  rel={(d_sgl_ref.mean()/ref.abs().mean()).item()*100:.4f}%")

    # 判据
    if d_tl_sgl.max().item() < 1e-2:
        print(f"\n  ✅ PASS: tilelang vs sglang max_diff={d_tl_sgl.max().item():.4e} < 1e-2, 布局/nibble/scale 全兼容, 可安全替换.")
    elif d_tl_ref.max().item() < d_sgl_ref.max().item() * 3:
        print(f"\n  ✅ PASS(近似): tilelang vs deq_ref 误差与 sglang 同量级, 两者都正确 (vs sglang diff={d_tl_sgl.max().item():.4e}).")
    else:
        print(f"\n  ❌ FAIL: tilelang vs sglang diff={d_tl_sgl.max().item():.4e} 过大, 需排查 nibble/scale.")


if __name__ == "__main__":
    main()
