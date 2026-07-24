# sglang 补丁 — 海光 DCU W8A8 / W4A16 MoE 适配 + EAGLE3 投机解码

本目录是对 sglang (dev 0.0.0.dev12695) 的补丁, 使其能在海光 DCU (gfx936/gfx928)
上加载并推理 **W8A8 与 W4A16 量化 (只量化 MoE expert) 的 MiniMax-M3**, 并支持
**EAGLE3 投机解码** (搭配 `Inferact/MiniMax-M3-EAGLE3` draft head).

**状态:**
- **W8A8 moe-only**: BW100 (gfx936) 上 forward 已跑通, chat/completions 请求成功, 输出连贯.
- **W4A16 moe-only**: BW100 (gfx936) 上 sglang 加载成功 (`The server is fired up and ready to roll!`),
  走 `CompressedTensorsWNA16TritonMoE (ROCm)` 纯 Triton 路径. 补丁 1-6 为 W8A8, 补丁 7 为 W4A16.
- **EAGLE3**: 补丁 8 给 M3 VL 类补 EAGLE3 target 侧接口 (上游缺失, 见 docs/MiniMax-M3-EAGLE3-工作记录.md).

## 改动总览

| 文件 | 类型 | 改动 |
|---|---|---|
| `added/compressed_tensors_w8a8_int8_triton_moe.py` | **新增** | 海光 W8A8 MoE scheme, 复用 sglang 原生 Triton fused_moe kernel |
| `added/sitecustomize.py` | **新增** | transformers 5.6.0 注册 `minimax_m3_sparse` 自定义层类型 (sglang dev 已支持 M3, transformers 未跟进, 否则 config 校验报 `Unknown layer type`) |
| `modified/compressed_tensors.py` | 改动 | ① MoE W8A8 海光分支 raise→return 新 scheme; ② Linear W8A8 海光下走 bf16 (moe-only) |
| `modified/schemes/__init__.py` | 改动 | 导出新 scheme |
| `modified/int8_kernel.py` | 改动 | `per_token_quant_int8` 的 round: `tl.extra.cuda.libdevice` → `tl.extra.hip.libdevice` |
| `modified/topk_sparse_prefill.py` | 改动 | sparse attn **prefill** kernel `num_stages=1` (BW100 共享内存 64K, stages≥2 超 65536) |
| `modified/topk_sparse_decode.py` | 改动 | sparse attn **decode** kernel `num_stages=1` (同上, decode 阶段也超) |
| `modified/compressed_tensors_wNa16_moe.py` | 改动 | **W4A16**: `CompressedTensorsWNA16MoE.__init__` 兼容正则层名 target, 不再硬编码 `target_scheme_map["Linear"]` (否则 `KeyError: 'Linear'`) |
| `modified/minimax_m3_vl.py` | 改动 | **EAGLE3**: M3 VL 类 (`MiniMaxM3SparseForConditionalGeneration`) 补 `set_eagle3_layers_to_capture` (含 `setattr layer._is_layer_to_capture=True` 修复 aux 捕获链路) / `get_embed_and_head` / aux-aware forward. 原 VL 类无这些接口 (只在 text-only 类上), 量化产物加载 VL 类故 EAGLE3 启动即 AttributeError. 原文件备份 `sglang_backup/minimax_m3_vl.py`. |
| `modified/minimax_sparse_backend.py` | 改动 | **EAGLE3**: `init_forward_metadata` 兜底 EAGLE3 TARGET_VERIFY. TARGET_VERIFY 的 `is_extend()==True` (走 forward_extend) 但 `ForwardBatch.init_new` 把它归 decode 分支不填 extend 字段, sparse backend 取 `max(None)`/`.device` 崩. 兜底: `extend_seq_lens` 为 None 时用 `spec_info.draft_token_num` 一次性补全 `extend_seq_lens`/`extend_seq_lens_cpu`/`extend_prefix_lens`/`extend_prefix_lens_cpu`, 再算 `_max_seqlen_q`. 原文件备份 `sglang_backup/minimax_sparse_backend.py`. |

每个 `modified/*.py.patch` 是相对原始 sglang 的 diff, 可用 `patch -p1 < xxx.patch` 应用
(单文件回溯可用 `patch -p4 < xxx.py.patch`). `modified/*.py` 是改后的完整文件, 可直接覆盖.

> 注: EAGLE3 补丁已直接应用进 site-packages (子进程需继承, monkey-patch 仅主进程生效不够).
> 回滚: `cp /models/sglang_backup/minimax_m3_vl.py /usr/local/lib/python3.10/dist-packages/sglang/srt/models/minimax_m3_vl.py`.

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
SP=/usr/local/lib/python3.10/dist-packages
# 新增文件
cp added/compressed_tensors_w8a8_int8_triton_moe.py \
   $SG/layers/quantization/compressed_tensors/schemes/
# sitecustomize.py — 装到 site-packages 根目录 (非 sglang 子目录),
# Python 解释器启动时自动加载, 早于任何 import, 在模型 config 校验前完成补丁
cp added/sitecustomize.py $SP/sitecustomize.py
# 改动文件 (直接覆盖, 或用 patch)
cp modified/compressed_tensors.py $SG/layers/quantization/compressed_tensors/
cp modified/schemes___init__.py $SG/layers/quantization/compressed_tensors/schemes/__init__.py
cp modified/int8_kernel.py $SG/layers/quantization/
cp modified/topk_sparse_prefill.py $SG/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py
cp modified/topk_sparse_decode.py $SG/layers/attention/minimax_sparse_ops/decode/topk_sparse.py
# 清 triton 缓存 (改过 kernel 后必须清)
rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root
```

## 8. `sitecustomize.py` — 注册 `minimax_m3_sparse` 层类型

**背景**: sglang dev build 已支持 MiniMax-M3, 但配套的 transformers 5.6.0 还没把
`minimax_m3_sparse` 这个自定义 layer type 注册进 `ALLOWED_LAYER_TYPES`. 加载模型
config 时 `transformers.configuration_utils` 校验 layer type 不在白名单就抛
`ValueError: Unknown layer type <class '...minimax_m3_sparse'>`, 启动即失败.

**做法**: 一个 monkey-patch, 在任何模型 import 之前把该类型追加进白名单:
```python
import transformers.configuration_utils as _cu
if "minimax_m3_sparse" not in _cu.ALLOWED_LAYER_TYPES:
    _cu.ALLOWED_LAYER_TYPES = _cu.ALLOWED_LAYER_TYPES + ("minimax_m3_sparse",)
```

**为什么用 `sitecustomize.py` 而不是改 sglang 代码 / 改 config**:
- 必须在 `transformers` 被 import 之前生效, 改 sglang 代码时机太晚 (transformers 早被 import).
- 改量化产物的 config.json 不行 — `ALLOWED_LAYER_TYPES` 是 transformers 源码硬编码的白名单,
  config 里写什么都会被拒.
- `sitecustomize.py` 是 Python 解释器启动时自动加载的钩子 (只要所在目录在 `sys.path` 中),
  早于一切业务 import, 是注入这种"启动前 patch"的标准位置.

**部署位置 (交付容器必须注意)**:
- **装到 site-packages 根目录**: `/usr/local/lib/python3.10/dist-packages/sitecustomize.py`
  (上面应用补丁脚本已含). site-packages 永远在 `sys.path` 中, 容器内固定, 不受工作目录变化影响.
- **不要依赖 PYTHONPATH 指向工作目录** (如 `export PYTHONPATH=/workspace:$PYTHONPATH`):
  交付给用户的容器工作目录会变 (用户可能挂到别处), 这样 sitecustomize 加载不到, 补丁失效.
  容器化交付的正确做法是把补丁固化进 site-packages, 让工作目录可变.
- `try/except` 兜底: transformers 缺失 / 已注册 / 接口变动都不影响主流程.

**验证**:
```bash
python3 -c "import sitecustomize, transformers.configuration_utils as c; print('minimax_m3_sparse' in c.ALLOWED_LAYER_TYPES)"
# True
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

---

## W4A16 补丁

### 7. `compressed_tensors_wNa16_moe.py` — 兼容正则层名 target (W4A16)

**背景**: sglang `CompressedTensorsWNA16MoE.__init__` 硬编码
`self.quant_config.target_scheme_map["Linear"].get("weights")`, 期望 config 用模块类名
`targets:["Linear"]` (像社区 AWQ 模型). 但我们 moe-only 量化产物用**正则层名**做 target
(`re:.*\.mlp\.experts\.\d+\.(gate|up|down)_proj$`, 只量 MoE expert), 此时
`target_scheme_map` 的键是该正则字符串, 没有 `"Linear"` 键 → `KeyError: 'Linear'`.

**做法**: `"Linear"` 在则用它, 否则取 `target_scheme_map` 第一个 (通常也是唯一一个) scheme:
```python
_scheme_map = self.quant_config.target_scheme_map
if "Linear" in _scheme_map:
    config = _scheme_map["Linear"].get("weights")
else:
    config = next(iter(_scheme_map.values())).get("weights")
```
hip 上 `CompressedTensorsWNA16TritonMoE` 继承此类, 同样生效.

**为什么不用 `targets:["Linear"]` + ignore 排除**: `"Linear"` 会声明所有 Linear 都量化,
moe-only 下漏排一个非 expert Linear (attn/dense mlp/shared) 就会被当 int4 加载但权重是 bf16 → 报错.
正则层名 target 语义精确 (只量 expert), 此补丁让 sglang 接受这种写法.

**配套的 config.json 调整 (非 sglang 代码, 是量化产物配置)**:
- `targets` 必须用 **sglang 层名** (`mlp.experts.*`), 不是 HF 权重名 (`block_sparse_moe.experts.*`).
  sglang 加载时通过 weight_loader 映射 `w1→gate_proj`/`block_sparse_moe→mlp`,
  compressed-tensors 的 target/ignore 匹配用**映射后的 sglang 名**.
- `ignore` 要覆盖所有非 expert Linear (`self_attn.*` / `mlp.shared_experts.*` /
  `mlp.(gate|up|down|gate_up)_proj` / `mlp.gate` / norm / embed / lm_head / vision / mtp),
  否则报 `Unable to find matching target for ...`.

详见工作文档 `docs/MiniMax-M3-量化工作记录.md` 第二十六章 26.5 (三个坑) / 26.6 (本补丁).
