# Case: HumanEval index=4 重复生成（两个叠加 bug）

## 现象

evalscope 跑 HumanEval 连续评测到 **index=4**（第 5 题，`mean_absolute_deviation`）时:

- accept rate 恒定 1.0（draft 永远被接受 = 重复循环）
- 输出乱码 / `\x00` 空字节
- 最终 `VMFault HSA QUEUE ANALYSIS`（HIP 显存越界）→ SIGABRT 崩 draft/target worker
- 单发 `he4`（HumanEval 第 4 题）不复现（干净 KV），连续评测必复现
- 复现时 cuda graph: True

这是**两个叠加的 bug** 触发同一现象，分两轮解决。两个 bug 都表现为「连续评测到 index=4 时重复/乱码/崩溃」，但根因不同，分别在「关 cuda graph」和「开 cuda graph」配置下暴露。

## 误判历程（第一轮，全部排除）

第一轮定位走了一串弯路，所有怀疑方向都被实验排除:

- ❌ **W8A8 量化精度**: 单发 he4 输出正确，主模型埋点 probe 全程 0（无数值异常）
- ❌ **cuda graph buffer 复用**: 关 cuda graph 也崩
- ❌ **NaN/Inf 数值崩溃**: probe 全程 0，是物理越界不是数值问题
- ✅ **EAGLE3「数学上不改变推理结果」是对的**，但 draft worker 本身有 bug

排除完上面这些，才把注意力锁定到 EAGLE3 draft worker 的 attention backend 控制流上。

## 第一轮：补丁9（triton_backend.py DRAFT_EXTEND_V2 修复）

### 触发配置

关 cuda graph（全 eager）下连续评测到 index=4 必崩。

### 根因

sglang `triton_backend.py` 的 `TritonAttnBackend.init_forward_metadata`（eager 路径）用
`forward_mode.is_draft_extend()` 判断 draft extend 分支，但该函数默认 `include_v2=False`，
**不包含 `DRAFT_EXTEND_V2`**:
```python
def is_draft_extend(self, include_v2: bool = False):
    return self == ForwardMode.DRAFT_EXTEND or (
        include_v2 and self == ForwardMode.DRAFT_EXTEND_V2  # 默认 False
    )
```
于是 EAGLE3 V2 的 draft extend 在 eager 模式下不匹配任何专门分支，落入 `else`（普通 EXTEND）
分支，用 `extend_prefix_lens` 按错误 KV 布局构建 `kv_indices` → 越界访问 → VMFault。

### 关键对照

cuda graph 捕获路径 `init_forward_metadata_capture_cuda_graph` 正确用了
`is_draft_extend(include_v2=True)`，所以开 cuda graph 的 batch 不崩，走 eager 的
（bs 不在 `cuda_graph_bs` 列表，或 `--disable-cuda-graph`）必崩 — 这解释了所有现象:
- 单发 he4 不复现：干净 KV，错布局刚好没越界
- 连续评测必复现：前几题把 KV 撑长后，错误布局命中越界

### 修复

`init_forward_metadata` 加 `is_draft_extend_v2()` 专用分支，逻辑镜像 cuda graph 路径
（`init_forward_metadata_capture_cuda_graph` 的 V2 分支）。V2 的 spec_info
（`EagleDraftInput` + V2 mixin）没有 V1 的 `generate_attn_arg_prefill`，故不能复用 V1
分支，必须直接构建 attn args。

### 验证

关 cuda graph（全 eager）下:
- warmup 不崩
- 单请求正常
- 串行 5 路 5/5
- 并发 8 路 8/8
- VMFault=0

## 第二轮：补丁10-11（target verify cuda graph VMFault 修复，方案A）

### 触发配置

补丁9 修好关 cg 的问题后，**开 cuda graph + 并发**又崩（同样的 index=4 连续评测场景，在开 cg 下复现）。

- 开 cuda graph + EAGLE3 + 并发 3 路 → VMFault 崩
- 崩在 target verify forward（不是 draft）
- 单请求不崩，关 cg 并发 8 路不崩，仅「开 cg + 并发」崩

### 根因

MiniMax sparse attention 的 `minimax_sparse_prefill` 内部分配 `score` scratch tensor:
```python
max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)  # block_size_k=128
score = torch.full((num_heads, total_q, max_seqblock_k), -inf, ...)
```
`max_seqlen_k` 由 `sparse_backend.init_forward_metadata_*_cuda_graph` 计算:

| 阶段 | seq_lens | max_seqlen_k | max_seqblock_k | score 第三维 |
|---|---|---|---|---|
| capture | `[1,1,...]` (dummy fill_value) | 1+4=5 | ceil(5/128)=1 | 1 |
| replay | `[186,188,...]` (真实) | 190+4=194 | ceil(194/128)=2 | 需 2 |

cuda graph 把 capture 时的小 score tensor（第三维=1）固化。replay 时真实 KV 更长，kernel 写
`score[*,*,1]` 越界 → VMFault。

**为什么 eager 不崩**: eager 走 `init_forward_metadata`，每次按真实 seq_lens 算 `max_seqlen_k`，
score tensor 大小匹配。
**为什么并发才崩**: 单请求 seq_lens 也 >1，并发是稳定触发条件。

### 修复（方案A）

强制 target verify 走 eager，draft/draft_extend 仍走 cuda graph:
- `eagle_info_v2.py::prepare_for_v2_verify`: 对 MiniMax sparse backend 强制 `can_run_cuda_graph=False`
- `cuda_graph_runner.py::can_run`: 对 TARGET_VERIFY + MiniMax sparse backend 返回 False（双保险，因
  `model_runner.forward` 独立调 `can_run`；仅改 `prepare_for_v2_verify` 会因 `self.raw_num_token`
  未初始化报 AttributeError）
- 环境变量 `EAGLE3_VERIFY_EAGER=1`（默认开），=0 可关闭

**性能**: 比「全 cuda graph（理想但不崩）」慢约 20%（target verify 占 70%，eager 比 graph 慢 ~30%）；
比「关 cuda graph + EAGLE3」快约 10%（draft 两阶段仍走 graph）。EAGLE3 投机加速完全保留。

### 验证

开 cuda graph max-bs 16:
- 并发 3 路 3/3 成功
- VMFault=0

## 涉及文件

| 文件 | 补丁 | 修改 |
|---|---|---|
| `sglang/srt/layers/attention/triton_backend.py` | 补丁9 | eager 路径加 `is_draft_extend_v2()` 分支，镜像 cuda graph 路径 |
| `sglang/srt/speculative/eagle_info_v2.py` | 补丁10 | `prepare_for_v2_verify` 对 MiniMax sparse backend 强制 eager |
| `sglang/srt/model_executor/cuda_graph_runner.py` | 补丁11 | `can_run` 对 TARGET_VERIFY + MiniMax sparse backend 返回 False（双保险） |

备份在 `sglang_backup/`（`triton_backend.py.bak` / `eagle_info_v2.py.bak` / `cuda_graph_runner.py.bak`）。

详见 `../../sglang_patches/README.md` 补丁 9 / 10-11。

## 总结

index=4 重复问题是两个叠加 bug 的触发场景:

1. **补丁9** 解决了 eager 路径的 DRAFT_EXTEND_V2 走错分支（关 cg 下崩）— draft worker 的 attention
   metadata 构建问题
2. **补丁10-11** 解决了 target verify 的 sparse attention 在 cuda graph 下的 score tensor 越界
   （开 cg 下崩）— target model 的 sparse attention buffer 固化问题

两个 bug 都表现为「连续评测到 index=4 时重复/乱码/崩溃」，但根因不同: 一个在 draft worker
的 KV 布局构建（控制流分支判断漏了 V2），一个在 target model 的 sparse attention buffer
（cuda graph capture/replay 尺寸不一致）。单发不复现、连续评测必复现，是因为前几题把 KV 撑长后
才命中越界边界。

第二轮（补丁10-11）的完整定位过程（EAGLE_PROBE + HIP_LAUNCH_BLOCKING 链路埋点、score tensor
越界量化分析、方案A/C/D 对比）见 `../eagle3_cuda_graph_vmfault/README.md`。
