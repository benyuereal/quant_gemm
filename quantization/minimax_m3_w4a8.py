#!/usr/bin/env python3
"""
MiniMax-M3 W4A8 (channel INT4 + INT8 activation) 量化脚本
数据无关 (data-free), 直接操作 safetensors 文件

用法:
    python3 minimax_m3_w4a8.py \
        --input-path /models/MiniMax/MiniMax-M3 \
        --output-path /models/MiniMax/MiniMax-M3-channel-int4-w4a8

说明:
    - 非 MoE 层 (dense MLP, attention) → per-channel INT8
    - MoE expert 层 (block_sparse_moe.experts.*.w1/w2/w3) → per-channel INT4
      使用 percentile 网格搜索 + 2×int4→int8 打包
    - 输出 config.json 使用 compressed-tensors 格式, vLLM 可直接加载
"""

import argparse
import json
import os
import re
import shutil
from glob import glob
from multiprocessing import Manager
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# ============================================================
# 量化函数
# ============================================================

def weight_quant_int8(tensor: torch.Tensor):
    """Per-channel symmetric INT8 量化"""
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def weight_quant_int4(tensor: torch.Tensor, percentile: float):
    """Per-channel INT4 量化 + 2×int4→int8 打包"""
    assert tensor.dim() == 2
    qmax = 7.0
    abs_value = torch.abs(tensor)
    sorted_matrix, _ = torch.sort(abs_value, dim=1)
    k = tensor.shape[1]
    index = int(k * percentile)
    index = min(index, k - 1)
    abs_max = sorted_matrix[:, index].reshape(-1, 1)
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -8, 7).to(torch.int8)
    dequant = quantized * scale
    # 2×int4→int8 packing
    n, k = quantized.size()
    new_shape = (n, k // 2)
    quantized_uint8 = quantized.to(torch.uint8)
    a = quantized_uint8[..., ::2]
    b = quantized_uint8[..., 1::2]
    b_4bit = b & 0x0F
    quantized_int4 = ((a & 0x0F) << 4) | b_4bit
    quantized_int4 = quantized_int4.contiguous().to(torch.int8)
    return quantized_int4, scale.to(torch.float32), dequant


def weight_quantint4_search_k(
    tensor: torch.Tensor,
    k_min: float = 0.97,
    k_max: float = 1.0,
    steps: int = 30,
    metric: str = "mse",
):
    """Percentile 网格搜索, 找到最佳截断阈值"""
    assert tensor.dim() == 2
    percentiles = torch.linspace(k_min, k_max, steps, device=tensor.device)
    best_error = float("inf")
    best_k = None
    best_quant = None
    best_scale = None

    for p in percentiles:
        p = float(p.item())
        q_int4, scale, dequant = weight_quant_int4(tensor, p)
        if metric == "mse":
            error = torch.mean((tensor - dequant) ** 2).item()
        elif metric == "l2":
            error = torch.norm(tensor.to(torch.float16) - dequant.to(torch.float16)).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if error < best_error:
            best_error = error
            best_k = p
            best_quant = q_int4
            best_scale = scale

    return best_quant, best_scale, best_k


# ============================================================
# MiniMax-M3 专用忽略列表
# ============================================================

IGNORE_LAYERS = [
    "re:.*norm.weight.*",
    "re:.*embed_tokens.*",
    "re:.*input_layernorm.*",
    "re:.*post_attention_layernorm.*",
    "re:.*self_attn.k_norm.*",
    "re:.*self_attn.q_norm.*",
    "re:.*block_sparse_moe.e_score_correction_bias.*",
    "re:.*block_sparse_moe.gate\\.weight$",  # MoE routing gate (敏感, 不量化)
    "re:.*multi_modal_projector.*",          # 视觉投影层
    "re:.*patch_merge_mlp.*",                # 图像 token 压缩模块
    "re:.*lm_head.*",
    "re:.*visual.*",
    "re:.*vision.*",
    "re:.*mtp.*",
    # 注意: shared_experts.gate_proj 是 SwiGLU 的 gate 投影, 应该被量化
    # MoE routing gate 已由 block_sparse_moe.gate 单独覆盖
]

# --moe-only 模式下额外忽略的非 MoE Linear (保留 bf16, 只量化 MoE expert).
# 覆盖: attention 投影 (q/k/v/o/index_q/index_k_proj), shared_experts, dense mlp.
# MoE expert (block_sparse_moe.experts.*.w1/w2/w3) 不在此列, 仍被量化.
MOE_ONLY_EXTRA_IGNORE = [
    # 注意: sglang 匹配 ignore 时用不带 .weight 的 module name, 故 .weight 要可选
    "re:.*self_attn\\.(q_proj|k_proj|v_proj|o_proj|index_q_proj|index_k_proj)(\\.weight)?$",
    "re:.*block_sparse_moe\\.shared_experts\\.(gate_proj|up_proj|down_proj)(\\.weight)?$",
    "re:.*mlp\\.(gate_proj|up_proj|down_proj)(\\.weight)?$",
]


def is_ignored(weight_name: str) -> bool:
    """检查权重名称是否在忽略列表中"""
    for pattern in IGNORE_LAYERS:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.search(regex, weight_name):
                return True
        else:
            if pattern in weight_name:
                return True
    return False


# ============================================================
# Worker 进程
# ============================================================

def worker(
    rank: int,
    world_size: int,
    safetensor_files: list,
    input_dir: str,
    output_dir: str,
    quant_type: str,
    shared_weight_map,
    moe_only: bool = False,
):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)

    local_files = safetensor_files[rank::world_size]

    for safetensor_file in tqdm(local_files, position=rank, desc=f"GPU {rank}"):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device=device)
        new_state_dict = {}

        for weight_name, weight in state_dict.items():
            if is_ignored(weight_name):
                new_state_dict[weight_name] = weight
                shared_weight_map[weight_name] = file_name
                continue

            # --moe-only: 只量化 MoE expert (block_sparse_moe.experts.*.w1/w2/w3),
            # 其他所有 Linear (attn/shared_experts/dense mlp) 保留 bf16 不量化
            if moe_only and ".block_sparse_moe.experts." not in weight_name:
                new_state_dict[weight_name] = weight
                shared_weight_map[weight_name] = file_name
                continue

            assert weight.dim() != 1, f"Cannot quant 1D layer: {weight_name}"

            # 处理 3D 张量 (fused MoE 专家)
            original_shape = None
            if weight.dim() == 3:
                E, N, K = weight.shape
                weight = weight.reshape(E * N, K)
                original_shape = (E, N, K)
            else:
                N = K = None

            if quant_type == "fp8":
                q, s = weight_quant_fp8(weight)
            else:
                # 第一步: 所有层 INT8 量化
                try:
                    q, s = weight_quant_int8(weight)
                except Exception as e:
                    new_state_dict[weight_name] = weight
                    shared_weight_map[weight_name] = file_name
                    continue

                # 第二步: MoE expert 层 INT4 量化
                # MiniMax-M3 expert 命名: block_sparse_moe.experts.*.w1/w2/w3
                if quant_type == "int4" and ".block_sparse_moe.experts." in weight_name:
                    K = K // 2 if K is not None else None
                    q, scale_int4, _ = weight_quantint4_search_k(q)
                    s = s * scale_int4 / 16

            # 恢复 3D 形状
            if original_shape is not None:
                E, N, K_orig = original_shape
                K_out = q.shape[-1]
                q = q.reshape(E, N, K_out)
                s = s.reshape(E, N, 1)

            new_state_dict[weight_name] = q
            new_scale_name = f"{weight_name}_scale"
            new_state_dict[new_scale_name] = s
            shared_weight_map[new_scale_name] = file_name
            shared_weight_map[weight_name] = file_name

        save_file(new_state_dict, os.path.join(output_dir, file_name))


# ============================================================
# 主函数
# ============================================================

def main(input_path: str, output_path: str, quant_type: str = "int4", moe_only: bool = False):
    assert quant_type in ("int4", "int8", "fp8"), f"Unsupported quant_type: {quant_type}"

    src_dir = Path(input_path)
    dst_dir = Path(output_path)
    dst_dir.mkdir(exist_ok=True)

    # 复制所有非 safetensors 文件
    for file in src_dir.rglob("*"):
        if file.is_file() and not file.name.endswith(".safetensors"):
            rel_path = file.relative_to(src_dir)
            (dst_dir / rel_path.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst_dir / rel_path)

    index_path = os.path.join(output_path, "model.safetensors.index.json")
    config_path = os.path.join(output_path, "config.json")

    if not os.path.exists(index_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing index or config: {index_path}, {config_path}")

    with open(index_path, "r") as f:
        model_index = json.load(f)

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    print(f"[INFO] Found {len(safetensor_files)} safetensor files")

    world_size = torch.cuda.device_count()
    assert world_size > 0, "No CUDA devices found"
    print(f"[INFO] Using {world_size} GPUs")

    manager = Manager()
    shared_weight_map = manager.dict()

    mp.spawn(
        worker,
        args=(
            world_size,
            safetensor_files,
            input_path,
            output_path,
            quant_type,
            shared_weight_map,
            moe_only,
        ),
        nprocs=world_size,
        join=True,
    )

    # 更新 weight map
    model_index["weight_map"] = dict(shared_weight_map)
    with open(index_path, "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"[INFO] Updated {index_path}")

    # 更新 config.json (compressed-tensors 格式, vLLM 兼容)
    with open(config_path, "r") as f:
        config = json.load(f)

    config.pop("quantization_config", None)

    # --moe-only: 把非 MoE 的 Linear 加入 ignore, sglang 加载时它们走 unquantized (bf16)
    effective_ignore = list(IGNORE_LAYERS)
    if moe_only:
        effective_ignore += MOE_ONLY_EXTRA_IGNORE

    if quant_type == "int4":
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "version": "0.14.0.1",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                        "group_size": None,
                        "actorder": None,
                        "block_structure": None,
                        "observer": "memoryless_minmax",
                        "observer_kwargs": {},
                        "scale_dtype": None,
                        "zp_dtype": None,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                        "group_size": None,
                        "actorder": None,
                        "block_structure": None,
                        "observer": None,
                        "observer_kwargs": {},
                        "scale_dtype": None,
                        "zp_dtype": None,
                    },
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": effective_ignore,
            "global_compression_ratio": None,
            "kv_cache_scheme": None,
            "sparsity_config": {},
            "transform_config": {},
        }
    else:
        qtype = "int" if quant_type == "int8" else "float"
        qformat = "int-quantized" if quant_type == "int8" else "float-quantized"
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": qformat,
            "quantization_status": "compressed",
            "version": "0.14.0.1",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": qtype,
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                        "group_size": None,
                        "observer": "memoryless_minmax",
                        "observer_kwargs": {},
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": qtype,
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                        "group_size": None,
                        "observer": None,
                        "observer_kwargs": {},
                    },
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": effective_ignore,
            "global_compression_ratio": None,
            "kv_cache_scheme": None,
            "sparsity_config": {},
            "transform_config": {},
        }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Updated {config_path}")
    print(f"[INFO] Quantization complete! Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax-M3 W4A8 Quantization")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Input model directory")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--quant-type", type=str, default="int4",
                        choices=["int4", "int8", "fp8"],
                        help="Quantization type")
    parser.add_argument("--moe-only", action="store_true",
                        help="Only quantize MoE experts (block_sparse_moe.experts.*.w1/w2/w3); "
                             "keep all other Linears (attn/shared_experts/dense mlp) as bf16. "
                             "MoE experts are 96.7%% of weights, so nearly all memory savings are kept.")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.quant_type, args.moe_only)
    print("Done!")