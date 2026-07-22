#!/bin/bash
# ============================================
# SGLang MiniMax-M3 启动脚本 (W8A8 moe-only, 海光 DCU)
# 适用环境：容器内（已挂载 /data1/models 到 /models）
# 日志: /models/sglang_serve.log (每次启动覆盖)
# ============================================

# 1. 停残留 sglang 进程 + 清编译缓存 (改过 kernel 后必须清, 否则用旧编译结果)
pkill -9 -f sglang 2>/dev/null
sleep 3
mkdir -p /models/.sglang_tmp /models/.torchinductor_cache /models/.triton_cache
rm -rf /models/.triton_cache/* /models/.torchinductor_cache/* /models/.sglang_tmp/* 2>/dev/null
rm -rf /tmp/torchinductor_root 2>/dev/null

# 2. 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 若需指定卡可修改
export SGLANG_USE_AITER=0                       # 纯 Triton 路径
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片
# 根文件系统 (/ overlay) 可写空间小, 临时目录/编译缓存指到 /models (284G free)
export TMPDIR=/models/.sglang_tmp
export TORCHINDUCTOR_CACHE_DIR=/models/.torchinductor_cache
export TRITON_CACHE_DIR=/models/.triton_cache
MODEL_PATH="/models/MiniMax/MiniMax-M3-w8a8-moe-only"
LOG_FILE="/models/sglang_serve.log"

# 3. SGLang 启动命令 (日志覆盖写)
sglang serve \
    --model-path ${MODEL_PATH} \
    --mem-fraction-static 0.85 \
    --tp-size 8 \
    --dtype bfloat16 \
    --context-length 4096 \
    --max-total-tokens 4096 \
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --trust-remote-code \
    --skip-server-warmup \
    --disable-cuda-graph \
    --host 0.0.0.0 \
    --port 8080 > ${LOG_FILE} 2>&1
