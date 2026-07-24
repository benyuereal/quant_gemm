# HIP MoE W4A16 kernel — 海光 DCU 上的 W4A16 decode 加速

## 这是什么

一个手写的 HIP MoE W4A16 GEMM kernel, 通过 monkey-patch 替换 sglang 的
`invoke_fused_moe_kernel`, 在 **decode 阶段 (小 M, token 数 ≤ 512)** 用海光原生
`v_mmac` 矩阵核指令算 int4 权重的 MoE GEMM, 比 sglang 自带 Triton W4A16 kernel 快
(小 batch 时最多约 3x, 主要在 GEMM2). 大 M 自动回落原生 Triton, 行为不变.

## 这不是什么 (重要)

**它不是量化步骤.** 它消费一个已有的 W4A16 (int4 权重 / bf16 激活) MoE 量化产物,
加速其 decode 时的矩阵乘. 与 `quantization/` 里的量化脚本完全是两条线:

- 量化 (`quantization/minimax_m3_w4a16.py`): 离线把 MoE expert 权重压成 int4, 产出模型.
- 本 kernel: 推理时算这个 int4 GEMM 更快.

**没有它 W4A16 也能正常推理**, 只是 decode 慢 (走 sglang 原生 Triton 路径).
工作记录 26.1 末 "activation 仍为 bf16, sglang 走 W4A16 Triton 路径, 不需要自定义 kernel"
说的就是能跑通; 本 kernel 是在能跑通之后的**可选性能优化**.

## 目标架构

| 架构 | 路径 | 说明 |
|---|---|---|
| gfx928 (CDNA3/MI300) | `v_mmac` 快路径 | 性能收益所在 |
| gfx936 (CDNA2/MI200) 等 | 标量 fallback | `v_mmac` 仅 gfx928 有; 其他架构能编译加载、结果正确, 但走 `.hip` 源码 `#else` 标量分支, **无加速** (基本等同不用本 kernel). gfx936 真要加速需另写 `mfma` 版, 不在本组件范围. |

预编译 `hip_moe_w4a16_dcu.so` 是 fat binary, 同时打包 gfx928/gfx936, 两架构都能直接加载.

## 文件

| 文件 | 用途 |
|---|---|
| `moe_w4a16_dcu.hip` | kernel 源码 (HIP C++). 核心技巧: scratch LDS roundtrip — 反量化后写共享内存, 再 `ds_read_b64` 读回当 `half4_t`, 使数据落在 `v_mmac` 期望的寄存器布局. |
| `hip_moe_w4a16_dcu.so` | 预编译产物 (gfx928+gfx936 fat binary). 优先用这个, 免现场编译. |
| `hip_moe_w4a16_patch.py` | sglang monkey-patch. 路径三级查找, 不依赖工作目录. |
| `install_hip_moe_w4a16.sh` | 安装脚本: 装到 site-packages + 启用 sitecustomize 自动加载. |

## 安装 (推荐, 容器交付)

```bash
bash sglang_patches/hip_moe_w4a16/install_hip_moe_w4a16.sh
```

做三件事:
1. 把 `.so` / `.hip` / `_patch.py` 拷到 `<site-packages>/hip_moe_w4a16/` (路径固定, 不受工作目录影响).
2. 在 `/etc/python3.10/sitecustomize.py` 幂等追加 `import hip_moe_w4a16_patch` (受 `SGLANG_USE_HIP_MOE_W4A16` 开关保护).
3. 此后 `sglang serve` 启动自动加载, 无需任何 PYTHONPATH.

验证:
```bash
python3 -c "import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm; print(fm.invoke_fused_moe_kernel.__module__)"
# 期望: hip_moe_w4a16_patch
```

## 不安装也能用 (开发/调试)

```bash
export PYTHONPATH=/path/to/sglang_patches/hip_moe_w4a16:$PYTHONPATH
# 之后 import sglang 即生效 (patch 同目录有 .so)
```

## 开关与环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_USE_HIP_MOE_W4A16` | `1` | `0` 禁用本 patch, 回退 sglang 原生 Triton (W4A16 仍正常) |
| `HIP_MOE_KERNEL_DIR` | 自动 | 显式指定 `.so`/`.hip` 所在目录 |
| `HIP_MOE_OFFLOAD_ARCH` | rocminfo 探测 | 仅现场编译 `.hip` 时用的 offload-arch (有预编译 `.so` 时不用) |

## kernel 路径查找顺序

`hip_moe_w4a16_patch.py` 按以下顺序找 kernel:
1. `$HIP_MOE_KERNEL_DIR`
2. patch 文件自身所在目录 (repo 或 site-packages 安装位置)
3. `<site-packages>/hip_moe_w4a16/`

找到 `.so` 直接加载; 只有 `.hip` 则用 `torch.utils.cpp_extension.load` 现场编译.

## 卸载

```bash
rm -rf <site-packages>/hip_moe_w4a16
# 编辑 /etc/python3.10/sitecustomize.py 删除带 "# hip_moe_w4a16 auto-load" 的段落
```

## 重新编译 .so (可选)

源码改动后重编:
```bash
python3 -c "
from torch.utils.cpp_extension import load
mod = load(name='hip_moe_w4a16_dcu',
    sources=['sglang_patches/hip_moe_w4a16/moe_w4a16_dcu.hip'],
    extra_cuda_cflags=['-O3','--offload-arch=gfx928','-std=c++17','-ffast-math'],
    verbose=False, with_cuda=True)
print(mod.__file__)
"
# 把输出的 .so 覆盖回 sglang_patches/hip_moe_w4a16/hip_moe_w4a16_dcu.so
```
