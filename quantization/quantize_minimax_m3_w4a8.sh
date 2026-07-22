#!/bin/bash
set -euo pipefail
# ============================================================
# MiniMax-M3  W4A8 (channel-int4-w4a8) 量化脚本
# 使用说明:  bash /models/quantize_minimax_m3_w4a8.sh
# 输出:      /models/MiniMax/MiniMax-M3-channel-int4-w4a8
# 输出格式:  compressed-tensors (vLLM 原生支持)
# 预估耗时:  60-120 分钟（取决于 GPU 性能）
# 磁盘需求:  ~100GB 剩余空间
# ============================================================

MODEL_DIR="/models/MiniMax/MiniMax-M3"
OUTPUT_DIR="/models/MiniMax/MiniMax-M3-channel-int4-w4a8"
SCRIPT_DIR="/models/llm-compressor/examples/one_click_quant"

# 检查输入目录
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[ERROR] 模型目录不存在: $MODEL_DIR"
    exit 1
fi

# 检查磁盘空间
AVAIL_GB=$(df -BG /models | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -lt 100 ]; then
    echo "[ERROR] 磁盘空间不足: 仅 ${AVAIL_GB}G 可用, 需要至少 100G"
    exit 1
fi

# 检查 GPU
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "[ERROR] 未检测到 GPU"
    exit 1
fi
echo "[INFO] 检测到 $GPU_COUNT 个 GPU"

# 切换到脚本目录（模板使用相对路径导入）
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo " MiniMax-M3  W4A8 量化启动"
echo " 输入:  $MODEL_DIR"
echo " 输出:  $OUTPUT_DIR"
echo "============================================"
echo ""

# 执行量化
# 算法说明:
#   - 使用专用 minimax_m3_w4a8.py 独立脚本 (不走 main.py/slimquant_ptq, 因后者
#     moe_pattern='.mlp.experts.' 匹配不到 MiniMax-M3 的 'block_sparse_moe.experts',
#     会导致 MoE expert 被错误地量化成 INT8 而非 INT4)
#   - quant_type=int4: 非 MoE 层 INT8 + MoE expert 层 INT4 (per-channel)
#   - 多 GPU 并行处理 shard 文件 (mp.spawn, 每个 DCU 处理一部分 shard)
# 忽略列表 (脚本内硬编码, 已验证覆盖全部 53 种 tensor 模式):
#   - lm_head, embed_tokens, norm, layernorm, qk_norm, index_qk_norm
#   - e_score_correction_bias (MoE 路由偏置, 1D)
#   - block_sparse_moe.gate (MoE 路由门控, 敏感不量化)
#   - multi_modal_projector, patch_merge_mlp (视觉投影/压缩)
#   - visual, vision (视觉塔, 整体不量化)
#   - mtp (MTP 模块)
#   注意: shared_experts.gate_proj/up_proj/down_proj 是 SwiGLU 权重, 正常量化 INT8
# 输出:
#   - config.json 使用 compressed-tensors 格式, vLLM/sglang 原生支持
#   - 层次: 非 MoE 层 → INT8, MoE expert 层 → INT4 (网格搜索最佳截断)

python3 "$SCRIPT_DIR/templates/minimax_m3_w4a8.py" \
    --input-path "$MODEL_DIR" \
    --output-path "$OUTPUT_DIR" \
    --quant-type int4

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[ERROR] 量化失败, 退出码: $EXIT_CODE"
    echo "建议排查:"
    echo "  1. 查看上方错误日志"
    echo "  2. 检查 GPU 显存是否充足"
    echo "  3. 检查磁盘空间: df -h /models"
    exit $EXIT_CODE
fi

echo ""
echo "============================================"
echo " 量化完成!"
echo " 输出目录: $OUTPUT_DIR"
echo "============================================"
echo ""

# 输出摘要
echo "[INFO] 输出内容:"
ls -lh "$OUTPUT_DIR" | head -5
echo "  ... (共 $(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | wc -l) 个 shard)"
echo ""

# 检查 config.json 中的 quantization_config
if [ -f "$OUTPUT_DIR/config.json" ]; then
    echo "[INFO] config.json 量化配置:"
    python3 -c "
import json
with open('$OUTPUT_DIR/config.json') as f:
    c = json.load(f)
qc = c.get('quantization_config', {})
if qc:
    print(f'  quant_method: {qc.get(\"quant_method\", \"N/A\")}')
    print(f'  format: {qc.get(\"format\", \"N/A\")}')
    if qc.get('config_groups'):
        g0 = qc['config_groups']['group_0']
        if g0.get('weights'):
            w = g0['weights']
            print(f'  weights: {w.get(\"num_bits\")}-bit {w.get(\"strategy\")}, symmetric={w.get(\"symmetric\")}')
        if g0.get('input_activations'):
            a = g0['input_activations']
            print(f'  activations: {a.get(\"num_bits\")}-bit {a.get(\"strategy\")}, dynamic={a.get(\"dynamic\")}')
else:
    print('  (未设置 quantization_config, 检查是否正常)')
"
fi

echo ""
echo "如需使用 vLLM 加载量化后模型:"
echo "  vllm serve $OUTPUT_DIR ..."