# quant_gemm

海光 DCU (gfx936 / gfx928) 上的 **per-channel W8A8** (int8 weight + int8 activation) GEMM 算子库, 基于 [tilelang](https://github.com/tile-ai/tilelang) 实现 (架构无关 JIT).

## 为什么需要

- **lightop** (海光汇编库) 0.6.0 只有 gfx936/gfx938 的预编译产物, **gfx928 (K100) 无产物**.
- **aiter** per-channel a8w8 路径依赖未启用的 `aiter_` C 扩展.
- **tilelang** 海光定制版, 架构无关 JIT, 可自写 W8A8 算子, 自主可控.

本库把为 MiniMax-M3 W8A8 量化推理写的两个 tilelang kernel 打包, 供 sglang 适配层调用.

## 算子

| 算子 | 说明 | 精度 (gfx936) |
|---|---|---|
| `w8a8_per_channel_gemm` | Linear W8A8 GEMM | vs deq 0.28%, vs bf16 1.25% |
| `moe_w8a8_grouped_gemm` | MoE grouped W8A8 (多 expert + 路由表) | vs deq 0.28%, vs bf16 1.25% |

量化方案: per-channel int8 weight + per-token int8 activation (dynamic).
`C[m,n] = (sum_k A[m,k] * B[n,k]) * x_scale[m] * w_scale[n]`

## 安装

```bash
cd /models/quant_gemm_pkg
pip install -e .
# 或打 wheel: pip wheel . -w dist/
```

## 用法

### 高层 API (自动量化 + kernel 缓存)

```python
import torch
from quant_gemm import w8a8_linear, w8a8_moe

# Linear: 传 bf16, 内部自动 int8 量化
x = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.05
out = w8a8_linear(x, w)  # [256, 4096]

# MoE: 传已量化的 int8 + scale + 路由分组
# x_grouped: [total_tokens,K] int8 (已按 expert 分组连续排列)
# w: [E,N,K] int8, x_scale:[total_tokens,1], w_scale:[E,N,1]
out = w8a8_moe(x_grouped, w, x_scale, w_scale, group_sizes, group_offsets)
```

### 底层 kernel (已量化输入)

```python
from quant_gemm import w8a8_per_channel_gemm, per_token_sym_int8, per_channel_sym_int8
import tilelang.language as T

q_x, x_s = per_token_sym_int8(x.float())     # [M,K] int8, [M,1] f32
q_w, w_s = per_channel_sym_int8(w.float())   # [N,K] int8, [N,1] f32
kernel = w8a8_per_channel_gemm(M, N, K, 128, 128, 64, out_dtype=T.bfloat16)
out = kernel(q_x.cuda(), q_w.cuda(), x_s.cuda(), w_s.cuda())  # [M,N] bf16
```

## 测试

```bash
HIP_VISIBLE_DEVICES=0 python3 /models/test/test_quant_gemm_package.py
```

## tilelang 写法备忘

- 条件控制流用 `T.If`/`T.Else` (大写), 循环 `T.Serial` (非 `T.serial`)
- 避开 kernel 内控制流: 路由表 Python 端预算传入
- int8 GEMM: `T.gemm(A, B, C, transpose_B=True)` (int8×int8→int32)
- scale 注入用独立 float32 fragment, **不要写回 int32 累加器** (会截断)
- `T.copy` 支持动态切片但不支持间接 gather → token 须先按 expert 分组连续排列
- `@tilelang.jit(out_idx=[N])`: N 是输出 tensor 在参数列表的 0-based 位置

## gfx928 实测

tilelang 架构无关, gfx936 通过 ≈ gfx928 可 JIT. **需在 gfx928 实机验证** (Triton/tilelang 对 gfx928 MMAC lowering 偶有边界问题).
