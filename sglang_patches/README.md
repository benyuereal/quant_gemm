# sglang 补丁 — 海光 DCU W8A8 / W4A16 MoE 适配 + EAGLE3 投机解码

本目录是对 sglang (dev 0.0.0.dev12695) 的补丁, 使其能在海光 DCU (gfx936/gfx928)
上加载并推理 **W8A8 与 W4A16 量化 (只量化 MoE expert) 的 MiniMax-M3**, 并支持
**EAGLE3 投机解码** (搭配 `Inferact/MiniMax-M3-EAGLE3` draft head).

**状态:**
- **W8A8 moe-only**: BW100 (gfx936) 上 forward 已跑通, chat/completions 请求成功, 输出连贯.
- **W4A16 moe-only**: BW100 (gfx936) 上 sglang 加载成功 (`The server is fired up and ready to roll!`),
  走 `CompressedTensorsWNA16TritonMoE (ROCm)` 纯 Triton 路径. 补丁 1-6 为 W8A8, 补丁 7 为 W4A16.
- **EAGLE3**: 补丁 8 给 M3 VL 类补 EAGLE3 target 侧接口 (上游缺失, 见 docs/MiniMax-M3-EAGLE3-工作记录.md). 补丁 9 修复 EAGLE3 V2 draft extend 在 triton backend eager 路径走错分支致 VMFault 崩溃. 补丁 10 修复 EAGLE3 target verify 在 cuda graph replay 下 sparse attention 越界 (见下).

## 改动总览

| 文件 | 类型 | 改动 |
|---|---|---|
| `added/compressed_tensors_w8a8_int8_triton_moe.py` | **新增** | 海光 W8A8 MoE scheme, 复用 sglang 原生 Triton fused_moe kernel |
| `modified/compressed_tensors.py` | 改动 | ① MoE W8A8 海光分支 raise→return 新 scheme; ② Linear W8A8 海光下走 bf16 (moe-only) |
| `modified/schemes/__init__.py` | 改动 | 导出新 scheme |
| `modified/int8_kernel.py` | 改动 | `per_token_quant_int8` 的 round: `tl.extra.cuda.libdevice` → `tl.extra.hip.libdevice` |
| `modified/topk_sparse_prefill.py` | 改动 | sparse attn **prefill** kernel `num_stages=1` (BW100 共享内存 64K, stages≥2 超 65536) |
| `modified/topk_sparse_decode.py` | 改动 | sparse attn **decode** kernel `num_stages=1` (同上, decode 阶段也超) |
| `modified/compressed_tensors_wNa16_moe.py` | 改动 | **W4A16**: `CompressedTensorsWNA16MoE.__init__` 兼容正则层名 target, 不再硬编码 `target_scheme_map["Linear"]` (否则 `KeyError: 'Linear'`) |
| `modified/minimax_m3_vl.py` | 改动 | **EAGLE3**: M3 VL 类 (`MiniMaxM3SparseForConditionalGeneration`) 补 `set_eagle3_layers_to_capture` (含 `setattr layer._is_layer_to_capture=True` 修复 aux 捕获链路) / `get_embed_and_head` / aux-aware forward. 原 VL 类无这些接口 (只在 text-only 类上), 量化产物加载 VL 类故 EAGLE3 启动即 AttributeError. 原文件备份 `sglang_backup/minimax_m3_vl.py`. |
| `modified/minimax_sparse_backend.py` | 改动 | **EAGLE3**: ① `init_forward_metadata` 兜底 TARGET_VERIFY 的 extend 字段缺失 (is_extend()==True 但 init_new 归 decode 分支不填, 取 max(None)/.device 崩); ② `init_forward_metadata_capture/replay_cuda_graph` 的 TARGET_VERIFY 分支: `_max_seqlen_k` 用 context_len 固定上界 (方案C, 为全 cuda graph 方案铺路; 当前方案A 下 target verify 走 eager 不经过此路径, 作死代码保留). 原文件备份 `sglang_backup/minimax_sparse_backend.py`. |
| `modified/triton_backend.py` | 改动 | **EAGLE3 V2 修复 (补丁9)**: `TritonAttnBackend.init_forward_metadata` (eager 路径) 加 `is_draft_extend_v2()` 分支. 原代码用 `is_draft_extend()` (默认 `include_v2=False`) 不匹配 DRAFT_EXTEND_V2, 导致 V2 draft extend 落入普通 EXTEND 分支, 用错误 KV 布局构建 kv_indices → HIP VMFault (显存越界) 崩 draft worker. cuda graph 路径用 `is_draft_extend(include_v2=True)` 本就正确, 故仅 eager (bs 不在 cuda-graph 列表或关 cg) 必崩. 镜像 graph 路径构建 attn args. 原文件备份 `sglang_backup/triton_backend.py.bak`. |
| `modified/eagle_info_v2.py` | 改动 | **EAGLE3 target verify eager 修复 (补丁10, 方案A)**: `prepare_for_v2_verify` 对 MiniMax sparse attention 模型强制 `can_run_cuda_graph=False`, 让 target verify 走 eager. 配合 `cuda_graph_runner.py` (补丁11) 双保险. 原因: sparse attention 的 score tensor 按 max_seqlen_k 动态分配, cuda graph capture 用 dummy seq_lens=1 固化成小尺寸, replay 时真实 seq_lens~190 越界 → VMFault. 强制 eager 让 sparse 用真实 seq_lens. draft/draft_extend 仍走 cuda graph 不受影响. 环境变量 `EAGLE3_VERIFY_EAGER=1` 控制 (默认开). 原文件备份 `sglang_backup/eagle_info_v2.py.bak`. |
| `modified/cuda_graph_runner.py` | 改动 | **EAGLE3 target verify eager 修复 (补丁11, 方案A)**: `CudaGraphRunner.can_run` 对 TARGET_VERIFY + MiniMax sparse backend 返回 False. 因 `model_runner.forward` 独立调 `can_run` 决定走 graph, 仅改 `prepare_for_v2_verify` 不够 (会因 `self.raw_num_token` 未初始化 AttributeError). 原文件备份 `sglang_backup/cuda_graph_runner.py.bak`. |

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
# 新增文件
cp added/compressed_tensors_w8a8_int8_triton_moe.py \
   $SG/layers/quantization/compressed_tensors/schemes/
# 改动文件 (直接覆盖, 或用 patch)
cp modified/compressed_tensors.py $SG/layers/quantization/compressed_tensors/
cp modified/schemes___init__.py $SG/layers/quantization/compressed_tensors/schemes/__init__.py
cp modified/int8_kernel.py $SG/layers/quantization/
cp modified/topk_sparse_prefill.py $SG/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py
cp modified/topk_sparse_decode.py $SG/layers/attention/minimax_sparse_ops/decode/topk_sparse.py
# EAGLE3 (补丁 8-11)
cp modified/minimax_m3_vl.py $SG/models/minimax_m3_vl.py
cp modified/minimax_sparse_backend.py $SG/layers/attention/minimax_sparse_backend.py
cp modified/triton_backend.py $SG/layers/attention/triton_backend.py
cp modified/eagle_info_v2.py $SG/speculative/eagle_info_v2.py
cp modified/cuda_graph_runner.py $SG/model_executor/cuda_graph_runner.py
# 清 triton 缓存 (改过 kernel 后必须清; triton_backend.py 只改 Python 控制流可不清)
rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root
```

## 启动

W8A8 + EAGLE3 启动脚本在 `../quantization/minimax_w8a8_eagle3.sh` (已配 W8A8 moe-only 模型 + EAGLE3 + 海光适配参数, 自动停残留+清缓存+日志重定向):
```bash
bash ../quantization/minimax_w8a8_eagle3.sh
# 日志: /models/sglang_w8a8_eagle3.log (tee)
# 另开终端: tail -f /models/sglang_w8a8_eagle3.log
```
等价的手动命令:
```bash
sglang serve --model-path /models/MiniMax/MiniMax-M3-w8a8-moe-only \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /models/Inferact/MiniMax-M3-EAGLE3 \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --tp-size 8 --dtype bfloat16 --context-length 16384 --max-total-tokens 24576 \
  --cuda-graph-max-bs 16 \
  --attention-backend triton --mm-attention-backend triton_attn \
  --trust-remote-code --mem-fraction-static 0.93 \
  --host 0.0.0.0 --port 8081
```
环境变量: `SGLANG_USE_AITER=0` (纯 Triton), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`TMPDIR/TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR` 指到 /models (根文件系统可写空间小).

**测试**: `curl -N http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"default","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'`

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

---

## EAGLE3 V2 修复

### 9. `triton_backend.py` — DRAFT_EXTEND_V2 eager 路径走错分支致 VMFault

**症状**: W8A8/W4A16 + EAGLE3 服务连续推理后, accept rate 恒定 1.0 (重复生成) → 输出乱码/`\x00`
→ `VMFault HSA QUEUE ANALYSIS` (HIP 显存越界) → SIGABRT (exit -6) 崩 draft worker. 单请求不复现,
连续评测/warmup 必崩. 关 cuda graph 也崩 (此前误判为 cuda graph buffer 复用 / W8A8 量化精度问题).

**根因**: `TritonAttnBackend.init_forward_metadata` (eager 路径) 用
`forward_mode.is_draft_extend()` 判断 draft extend 分支, 但该函数默认 `include_v2=False`,
**不包含 `DRAFT_EXTEND_V2`**:
```python
def is_draft_extend(self, include_v2: bool = False):
    return self == ForwardMode.DRAFT_EXTEND or (
        include_v2 and self == ForwardMode.DRAFT_EXTEND_V2  # 默认 False
    )
```
于是 EAGLE3 V2 的 draft extend 在 eager 模式下不匹配任何专门分支, 落入 `else` (普通 EXTEND)
分支, 用 `extend_prefix_lens` 按错误 KV 布局构建 `kv_indices` → 越界访问 → VMFault.

**关键对照**: cuda graph 捕获路径 `init_forward_metadata_capture_cuda_graph` 正确用了
`is_draft_extend(include_v2=True)`, 故开 cuda graph 的 batch 不崩, 走 eager 的 (bs 不在
`cuda_graph_bs` 列表, 或 `--disable-cuda-graph`) 必崩 — 这解释了所有现象.

**做法**: 在 `init_forward_metadata` 的 `is_draft_extend()` 分支后新增 `is_draft_extend_v2()`
分支, 逻辑镜像 cuda graph 路径 (`init_forward_metadata_capture_cuda_graph` 的 V2 分支):
```python
elif forward_batch.forward_mode.is_draft_extend_v2():
    num_tokens_per_bs = self.num_draft_tokens
    qo_indptr = self.qo_indptr[: bs + 1]
    qo_indptr[: bs + 1] = torch.arange(0, bs * num_tokens_per_bs + 1,
        step=num_tokens_per_bs, dtype=torch.int32, device=self.device)
    kv_indptr = self.kv_indptr[: bs + 1]
    kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
    kv_indptr = kv_indptr[: bs + 1]
    kv_indices = torch.empty(int(forward_batch.seq_lens.sum().item()),
        dtype=torch.int64, device=self.device)
    create_flashinfer_kv_indices_triton[(bs,)](
        self.req_to_token, forward_batch.req_pool_indices,
        forward_batch.seq_lens, kv_indptr, None, kv_indices,
        self.req_to_token.stride(0))
    custom_mask = None; mask_indptr = None
    max_extend_len = num_tokens_per_bs
    num_kv_splits = None; attn_logits = None; attn_lse = None
```
`seq_lens` 此处已是 prefix+draft (`prepare_for_extend_to_fill_draft_kvcache` 设的),
`num_draft_tokens` 是每请求 draft token 数. V2 的 spec_info (`EagleDraftInput` + V2 mixin)
没有 V1 的 `generate_attn_arg_prefill`, 故不能复用 V1 分支, 必须直接构建.

**验证**: 修复后 (关 cuda graph, 全 eager) warmup 不再崩; 单请求 88s/2513 字符无重复;
串行 5 请求 5/5; **并发 8 请求 8/8**; VMFault/Aborted = 0. batch size 1→9→8→6→1 全程稳定.

**应用**:
```bash
cp modified/triton_backend.py $SG/layers/attention/triton_backend.py
# 无需清 triton 缓存 (未改 Triton kernel, 只改 Python 控制流)
```

---

## EAGLE3 target verify cuda graph VMFault 修复 (方案A)

### 10-11. `eagle_info_v2.py` + `cuda_graph_runner.py` — target verify 走 eager

**症状**: W8A8 + EAGLE3 + **开 cuda graph + 并发**请求时, 崩在 target verify forward:
- accept rate 恒 1.0 → 乱码 → `VMFault HSA QUEUE ANALYSIS` (HIP 显存越界) → SIGABRT
- 单请求不崩, 关 cuda graph 并发 8 路不崩, 仅"开 cuda graph + 并发"崩
- 崩点 (EAGLE_PROBE + HIP_LAUNCH_BLOCKING 定位): `verify.target_forward.start` 之后, 即 target model 的 sparse attention forward

**根因**: MiniMax sparse attention 的 `minimax_sparse_prefill` 内部分配 `score` tensor:
```python
max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)  # block_size_k=128
score = torch.full((num_heads, total_q, max_seqblock_k), -inf, ...)
```
`max_seqlen_k` 来自 `sparse_backend._max_seqlen_k`, 由 `init_forward_metadata_*_cuda_graph` 算:
- **capture 时**: dummy `seq_lens=[1,1,...]` (fill_value) → `max_seqlen_k = 1+4 = 5` → `max_seqblock_k = 1` → score 第三维=1
- **replay 时**: 真实 `seq_lens=[186,188,...]` → 需要 `max_seqblock_k = 2` → kernel 写 score[*,*,1] 越界

cuda graph 把 capture 时的小 score tensor 固化, replay 时真实 KV 更长, kernel 写超出 score 边界 → VMFault.

**为什么 eager 不崩**: eager 走 `init_forward_metadata`, 每次按真实 seq_lens 算 `max_seqlen_k`, score tensor 大小匹配.
**为什么并发才崩**: 单请求 seq_lens 也 >1, 但单请求开 cuda graph 同样会崩 (之前只测过关 cg 的单请求). 并发是稳定触发条件.

**方案A (采用)**: 强制 target verify 走 eager, draft/draft_extend 仍走 cuda graph.
- `eagle_info_v2.py::prepare_for_v2_verify`: 对 MiniMax sparse backend 强制 `can_run_cuda_graph=False`
- `cuda_graph_runner.py::can_run`: 对 TARGET_VERIFY + MiniMax sparse backend 返回 False (双保险, 因 `model_runner.forward` 独立调 `can_run`; 仅改 `prepare_for_v2_verify` 会因 `self.raw_num_token` 未初始化报 AttributeError)
- 环境变量 `EAGLE3_VERIFY_EAGER=1` (默认开), =0 可关闭

**性能**: 比"全 cuda graph (理想但不崩)"慢约 20% (target verify 占 70%, eager 比 graph 慢 ~30%); 比"关 cuda graph + EAGLE3"快约 10% (draft 两阶段仍走 graph). EAGLE3 投机加速完全保留.

**方案C (保留死代码, 为方案D铺路)**: `minimax_sparse_backend.py` 的 capture/replay 用 `max_seqlen_k = context_len` (固定上界 16384), 让 score tensor 固定大小. 单独不够 (sparse kernel 还有其他动态依赖), 但解决了 score tensor 这一项, 为未来全 cuda graph 方案 (方案D) 留基础. 方案A 下 target verify 走 eager 不经此路径, 作死代码保留.

**验证 (方案A, 开 cuda graph max-bs 16, 关 HIP_LAUNCH_BLOCKING)**:
- 并发 3 路: 3/3 成功, 无 nulls, VMFault=0, 服务存活 ✓

**应用**:
```bash
cp modified/eagle_info_v2.py $SG/speculative/eagle_info_v2.py
cp modified/cuda_graph_runner.py $SG/model_executor/cuda_graph_runner.py
cp modified/minimax_sparse_backend.py $SG/layers/attention/minimax_sparse_backend.py
# 无需清 triton 缓存 (只改 Python 控制流)
```

**待办 (方案D, 全 cuda graph)**: 让 sparse attention 的所有动态 tensor (score/o/topk_idx) 在 capture/replay 间一致, 使 target verify 也能走 cuda graph. 基于方案C 的固定上界思路继续解决剩余动态依赖. 见 `cases/eagle3_cuda_graph_vmfault/`.
