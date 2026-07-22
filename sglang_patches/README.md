# sglang 补丁 — 海光 DCU W8A8 MoE 适配

本目录是对 sglang (dev 0.0.0.dev12695) 的补丁, 使其能在海光 DCU (gfx936/gfx928)
上加载并推理 **W8A8 量化 (只量化 MoE expert) 的 MiniMax-M3**.

**状态: BW100 (gfx936) 上 forward 已跑通**, chat/completions 请求成功, 输出连贯.

## 改动总览

| 文件 | 类型 | 改动 |
|---|---|---|
| `added/compressed_tensors_w8a8_int8_triton_moe.py` | **新增** | 海光 W8A8 MoE scheme, 复用 sglang 原生 Triton fused_moe kernel |
| `modified/compressed_tensors.py` | 改动 | ① MoE W8A8 海光分支 raise→return 新 scheme; ② Linear W8A8 海光下走 bf16 (moe-only) |
| `modified/schemes/__init__.py` | 改动 | 导出新 scheme |
| `modified/int8_kernel.py` | 改动 | `per_token_quant_int8` 的 round: `tl.extra.cuda.libdevice` → `tl.extra.hip.libdevice` |
| `modified/topk_sparse_prefill.py` | 改动 | sparse attn **prefill** kernel `num_stages=1` (BW100 共享内存 64K, stages≥2 超 65536) |
| `modified/topk_sparse_decode.py` | 改动 | sparse attn **decode** kernel `num_stages=1` (同上, decode 阶段也超) |

每个 `modified/*.py.patch` 是相对原始 sglang 的 diff, 可用 `patch -p1 < xxx.patch` 应用.
`modified/*.py` 是改后的完整文件, 可直接覆盖.

## 各改动详解

### 1. 新增 `compressed_tensors_w8a8_int8_triton_moe.py` — 海光 W8A8 MoE scheme

**背景**: sglang `compressed_tensors.py` 的 `get_moe_scheme` 在 `_is_dynamic_token_w8a8`
分支只对 NPU 返回 scheme, 海光 (非 NPU) 直接 `raise NotImplementedError`. 但 sglang
原生 `fused_moe_kernel` (Triton) 本身支持 `use_int8_w8a8` + `per_channel_quant`.

**做法**: 新写 `CompressedTensorsW8A8Int8TritonMoE`, 仿 `CompressedTensorsWNA16TritonMoE`
但权重 int8 (不 pack). `create_weights` 建 `[E,2*intermediate,hidden] int8` + `[E,N,1] f32`
per-channel scale; `get_triton_quant_info` 设 `use_int8_w8a8=True, per_channel_quant=True`.
激活 per-token int8 量化由 kernel 内部 `per_token_quant_int8` 完成 (dynamic).

**权重布局** (与 sglang fused_moe kernel 期望 `B.shape=[E,N,K]` 一致):
- `w13_weight [E, 2*intermediate, hidden] int8` (gate w1 + up w3 沿 N 拼接)
- `w2_weight [E, hidden, intermediate] int8`
- `w13_weight_scale [E, 2*intermediate, 1] f32`, `w2_weight_scale [E, hidden, 1] f32`

### 2. `compressed_tensors.py` — 分发接线 + Linear 走 bf16

**改动 A** (`get_moe_scheme`, `_is_dynamic_token_w8a8` 分支):
```python
elif _is_hip:
    return CompressedTensorsW8A8Int8TritonMoE(self)  # 原: raise NotImplementedError
```

**改动 B** (`get_linear_scheme`, 新增拦截):
```python
# 海光 moe-only: Linear W8A8 底层 int8_scaled_mm 不可用, 且只量化 MoE expert.
# 海光下把 Linear W8A8 dynamic-token 当不量化 (走 bf16).
if _is_hip and self._is_dynamic_token_w8a8(weight_quant, input_quant):
    weight_quant = None; input_quant = None
```
**为什么**: `qkv_proj`/`index_qkv_proj` 等融合 Linear 即使在 config ignore 里,
sglang 的 fused_mapping 匹配仍会分到 W8A8 scheme, 而 `int8_scaled_mm` 海光未注册
(`if _is_cuda` 守卫). moe-only 方案下 Linear 本就该 bf16, 直接海光拦截.

### 3. `schemes/__init__.py` — 导出新 scheme

加 `from .compressed_tensors_w8a8_int8_triton_moe import CompressedTensorsW8A8Int8TritonMoE`
和 `__all__` 条目.

### 4. `int8_kernel.py` — round 函数海光兼容

`per_token_quant_int8` 的 Triton kernel 原用 `tl.extra.cuda.libdevice.round`, 海光 HIP
报 `Implicit conversion of CUDA __nv_roundf ... dropped`. 改为:
```python
x_q = tl.extra.hip.libdevice.round(x_q).to(tl.int8)  # 海光镜像专用, 写死
```
**注意**: 此改动写死 `tl.extra.hip`, 仅适用于海光 HIP 镜像 (不在 CUDA 跑). 已单独验证
kernel 跑通, 误差 0.93%.

### 5. `topk_sparse_prefill.py` — sparse attn prefill 共享内存

MiniMax sparse attention **prefill** 的 `_gqa_share_sparse_fwd_kernel` autotune 含
`num_stages=2/3` config, 在 BW100 (共享内存上限 65536) 上需 69632 → `OutOfResources`.
实测 `num_stages=2` 仍超 (差 4KB), 必须降到 `num_stages=1`:
```python
configs=[
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),  # 原为 stages=2/3
]
```

### 6. `topk_sparse_decode.py` — sparse attn decode 共享内存

MiniMax sparse attention **decode** 的 `_gqa_share_sparse_decode_kernel` autotune
`for ns in [2, 3, 4, 5]`, 同样超 BW100 64K. 改为 `for ns in [1]`:
```python
configs=[
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in [4, 8]
    for ns in [1]  # 原为 [2, 3, 4, 5]
]
```

**注意**:
- `num_stages=1` 偏保守 (无 pipeline), 性能非最优, 但保证跑通. 后续可测算各 kernel 在 64K 内的最大 stages.
- `decode/flash_with_topk_idx.py` 和 `prefill/flash_with_topk_idx.py` 也有 `num_stages≥3`,
  本次未改 (forward 未触发). 若报同错同法处理.
- 改 stages 后**必须清 triton 缓存** (`/models/.triton_cache` 或 `~/.triton`), 否则用旧编译结果.

## 应用补丁

```bash
SG=/usr/local/lib/python3.10/dist-packages/sglang/srt
# 新增文件
cp added/compressed_tensors_w8a8_int8_triton_moe.py \
   $SG/layers/quantization/compressed_tensors/schemes/
# 改动文件 (直接覆盖, 或用 patch)
cp modified/compressed_tensors.py $SG/layers/quantization/compressed_tensors/
cp modified/schemes___init__.py $SG/layers/quantization/compressed_tensors/schemes/__init__.py
cp modified/int8_kernel.py $SG/layers/quantization/
cp modified/topk_sparse_prefill.py $SG/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py
cp modified/topk_sparse_decode.py $SG/layers/attention/minimax_sparse_ops/decode/topk_sparse.py
# 清 triton 缓存 (改过 kernel 后必须清)
rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root
```

## 启动

`/models/minimax.sh` (已配 W8A8 moe-only 模型 + 海光适配参数, 自动停残留+清缓存+日志重定向):
```bash
bash /models/minimax.sh
# 日志: /models/sglang_serve.log (覆盖写)
# 另开终端: tail -f /models/sglang_serve.log
```
等价的手动命令:
```bash
sglang serve --model-path /models/MiniMax/MiniMax-M3-w8a8-moe-only \
  --tp-size 8 --dtype bfloat16 --context-length 4096 --max-total-tokens 4096 \
  --attention-backend triton --mm-attention-backend triton_attn \
  --trust-remote-code --skip-server-warmup --disable-cuda-graph \
  --mem-fraction-static 0.85
```
环境变量: `SGLANG_USE_AITER=0` (纯 Triton), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`TMPDIR/TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR` 指到 /models (根文件系统可写空间小).

**测试**: `curl -N http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"default","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'`
