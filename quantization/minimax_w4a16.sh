#!/bin/bash
# MiniMax-M3 W4A16 moe-only 功能验证启动脚本 (4卡, 短上下文, 仅验证加载+forward)
# 产物: /models/MiniMax/MiniMax-M3-w4a16-moe-only (225GB, int4 group=128 naive无校准)
# TP=4: 128 expert / 4 = 32 整除; 每卡权重 ~62GB, 剩余给 KV/激活很少 -> 短上下文
#
# 用法:  bash /models/minimax_w4a16.sh
# 日志:  /models/sglang_w4a16.log
# 端口:  8081

set -e

# 1) 清理可能残留的 sglang 进程(只杀 sglang, 不影响下载器等)
pkill -9 -f "sglang serve" 2>/dev/null || true
sleep 3

# 2) 准备缓存目录(避免磁盘满/旧缓存干扰)
mkdir -p /models/.sglang_tmp /models/.torchinductor_cache /models/.triton_cache
rm -rf /models/.triton_cache/* /models/.torchinductor_cache/* /models/.sglang_tmp/* 2>/dev/null || true
rm -rf /tmp/torchinductor_root 2>/dev/null || true

# 3) 环境变量
export CUDA_VISIBLE_DEVICES=0,1,4,5              # 4张均衡空闲卡 (2/3/6/7被占)
export SGLANG_USE_AITER=0                         # 纯 Triton 路径
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/models/.sglang_tmp
export TORCHINDUCTOR_CACHE_DIR=/models/.torchinductor_cache
export TRITON_CACHE_DIR=/models/.triton_cache

MODEL_PATH="/models/MiniMax/MiniMax-M3-w4a16-moe-only"
LOG_FILE="/models/sglang_w4a16.log"

echo "[INFO] 启动 W4A16 moe-only 验证服务"
echo "[INFO] 模型:   ${MODEL_PATH}"
echo "[INFO] GPU:    \$CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] TP=4, context=512, cuda_graph_max_bs=8, port=8081"
echo "[INFO] 日志:   ${LOG_FILE}"
echo "[INFO] 前台运行, 日志同时输出到屏幕和文件; Ctrl+C 停止"

# 4) 前台启动, 日志同时输出到屏幕和文件(便于实时观察)
# 显存账(每卡67G): 权重57.25G + KV pool + cuda graph(~0.6G, bs≤8) + 激活
# 加载后剩4.72G, frac=0.97 让 KV pool 预算最大化, 但开cuda graph后KV被挤, 上下文调小到512
sglang serve \
    --model-path ${MODEL_PATH} \
    --mem-fraction-static 0.97 \
    --tp-size 4 \
    --dtype bfloat16 \
    --context-length 512 \
    --max-total-tokens 512 \
    --chunked-prefill-size 512 \
    --cuda-graph-max-bs 8 \
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --trust-remote-code \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 8081 2>&1 | tee ${LOG_FILE}
