#!/usr/bin/env python3
"""
MiniMax-M3 W4A16 量化脚本 (标准 model_free_ptq 接口)

方案:
    - 只量化 MoE expert (block_sparse_moe.experts.*.w1/w2/w3) 为 INT4
      (W4A16 preset: weights num_bits=4, strategy=group, group_size=128, symmetric)
    - 激活保持 bf16 (A16, input_activations=None)
    - 其余所有权重 (attention / shared_experts / dense mlp / norm / embed /
      lm_head / 视觉塔 / 路由 gate) 保持 bf16 不量化

为什么用 model_free_ptq 标准接口而不是手写脚本:
    - config.json 的 config_groups / tensor 命名 (.weight_packed + .weight_scale)
      / 打包格式全部由 compressed-tensors 库标准生成, sglang / vLLM 原生识别
    - 之前手写脚本生成的 tensor 命名 (weight 而非 weight_packed) + 单 group
      全声明 W4 但实际混合精度, 导致 sglang 报错
      "No compressed-tensors compatible scheme was found."

参考: llm-compressor/examples/model_free_ptq/minimax_m27_w4a16.py
    (同作者, 同 MoE 命名 block_sparse_moe.experts.\\d+.w[1-3], 已验证可被 sglang 加载)

用法:
    python3 minimax_m3_w4a16.py \
        --input-path /models/MiniMax/MiniMax-M3 \
        --output-path /models/MiniMax/MiniMax-M3-w4a16 \
        --max-workers 8

说明:
    - 数据无关 (data-free), 直接操作 safetensors 文件, 无需模型定义
    - max_workers 控制并行 shard 数, 建议设为 GPU 数
    - 预计输出 ~224 GB (MoE expert INT4 + group_scale + 其余 bf16), 压缩比 ~3.55x
"""

import argparse
import os

from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import W4A16

from llmcompressor import model_free_ptq


# ============================================================
# 量化目标: 只量 MoE expert
# ============================================================

# MiniMax-M3 expert tensor 命名:
#   language_model.model.layers.{N}.block_sparse_moe.experts.{E}.w1/w2/w3.weight
# 用正则匹配所有 expert 的 w1/w2/w3
EXPERT_TARGETS = [
    r"re:.*block_sparse_moe\.experts\.\d+\.w[1-3]$",
]

# 额外排除 (保险, 防止正则意外匹配; targets 已精确圈定 expert, 这里主要兜底)
# 注意: 路由 gate / norm / embed / lm_head / 视觉塔本就不在 EXPERT_TARGETS 里,
#       不会量化; ignore 是双保险
IGNORE_LAYERS = [
    r"re:.*block_sparse_moe\.gate(\.weight)?$",   # MoE 路由 gate (敏感, 不量化)
    r"re:.*block_sparse_moe\.e_score_correction_bias$",
    r"re:.*norm(\.weight)?$",                       # 所有 layernorm / rmsnorm
    r"re:.*embed_tokens.*",
    r"re:.*lm_head.*",
    r"re:.*multi_modal_projector.*",
    r"re:.*patch_merge_mlp.*",
    r"re:.*(vision|visual).*",
    r"re:.*mtp.*",
]


def main(input_path: str, output_path: str, max_workers: int):
    scheme = QuantizationScheme(
        **W4A16,
        targets=EXPERT_TARGETS,
    )

    print(f"[INFO] 量化方案: W4A16 (MoE expert -> INT4, group_size=128, symmetric)")
    print(f"[INFO]   weights: num_bits=4, strategy=group, group_size=128, symmetric=True")
    print(f"[INFO]   input_activations: None (保持 bf16)")
    print(f"[INFO]   targets: {EXPERT_TARGETS}")
    print(f"[INFO] 输入: {input_path}")
    print(f"[INFO] 输出: {output_path}")
    print(f"[INFO] 并行 workers: {max_workers}")

    model_free_ptq(
        model_stub=input_path,
        save_directory=output_path,
        scheme=scheme,
        ignore=IGNORE_LAYERS,
        max_workers=max_workers,
    )

    print(f"[INFO] Quantization complete! Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax-M3 W4A16 Quantization (model_free_ptq)")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Input model directory (original bf16 weights)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Number of parallel worker shards (建议 = GPU 数)")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.max_workers)
    print("Done!")
