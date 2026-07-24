#!/bin/bash
# install_hip_moe_w4a16.sh — 安装 HIP MoE W4A16 kernel patch 到 site-packages,
# 并在 sitecustomize 里启用自动加载 (sglang serve 启动即生效, 不依赖工作目录).
#
# 用法: bash sglang_patches/hip_moe_w4a16/install_hip_moe_w4a16.sh
# 卸载: 见脚本末尾注释.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# site-packages 路径 (取第一个存在的)
SP_DIR=$(python3 -c "import site; print(next(d for d in site.getsitepackages() if __import__('os').path.isdir(d)))")
DEST_DIR="$SP_DIR/hip_moe_w4a16"
SITECUSTOMIZE="/etc/python3.10/sitecustomize.py"
MARK="# hip_moe_w4a16 auto-load"

echo "[install] site-packages: $SP_DIR"
echo "[install] dest:          $DEST_DIR"

# 1. 拷贝 kernel + patch 到 site-packages
mkdir -p "$DEST_DIR"
cp -f "$SRC_DIR/hip_moe_w4a16_dcu.so" "$DEST_DIR/" 2>/dev/null || echo "[install] no precompiled .so (将现场编译 .hip)"
cp -f "$SRC_DIR/moe_w4a16_dcu.hip" "$DEST_DIR/"
cp -f "$SRC_DIR/hip_moe_w4a16_patch.py" "$DEST_DIR/"

# 2. 幂等追加 sitecustomize 自动加载
touch "$SITECUSTOMIZE"
if grep -qF "$MARK" "$SITECUSTOMIZE"; then
    echo "[install] sitecustomize 已含 hip_moe_w4a16 钩子, 跳过"
else
    cat >> "$SITECUSTOMIZE" <<EOF

# --- hip_moe_w4a16: MoE W4A16 decode kernel patch for Hygon DCU ---
$MARK
# sglang serve 启动自动加载; SGLANG_USE_HIP_MOE_W4A16=0 可禁用.
try:
    import os as _os, sys as _sys
    if _os.environ.get("SGLANG_USE_HIP_MOE_W4A16", "1") == "1":
        _d = "$DEST_DIR"
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        import hip_moe_w4a16_patch
except Exception as _e:
    pass
EOF
    echo "[install] 已追加 sitecustomize 钩子"
fi

echo "[install] 完成. 验证:"
echo "  python3 -c \"import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm; print(fm.invoke_fused_moe_kernel.__module__)\""
echo "  期望输出: hip_moe_w4a16_patch"
echo
echo "开关: SGLANG_USE_HIP_MOE_W4A16=0 禁用 (回退 sglang 原生 Triton, W4A16 仍正常)"
echo
echo "卸载:"
echo "  rm -rf $DEST_DIR"
echo "  编辑 $SITECUSTOMIZE 删除带 '$MARK' 的段落"
