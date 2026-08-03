# Case: EAGLE3 + cuda graph 并发 VMFault 崩溃

## 现象

W8A8 moe-only + EAGLE3 + **开 cuda graph + 并发**请求时:
- accept rate 恒定 1.0 (重复生成) → 输出乱码 → `VMFault HSA QUEUE ANALYSIS` (HIP 显存越界) → SIGABRT 崩
- 崩在 target verify forward (target model 的 sparse attention)
- 触发条件: 开 cuda graph + 并发 (3 路即可). 单请求不崩, 关 cuda graph 并发 8 路不崩

## 对照矩阵

| 配置 | 并发 | 结果 |
|---|---|---|
| 关 cuda graph + EAGLE3 | 8 路 | ✅ 全通过 |
| 开 cuda graph + EAGLE3 | 3 路 | ❌ VMFault 崩 |
| 开 cuda graph + EAGLE3 | 单请求 | ✅ 不崩 (但未充分测) |

问题精确锁定在「EAGLE3 + cuda graph + 并发」组合.

## 根因

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

cuda graph 把 capture 时的小 score tensor (第三维=1) 固化. replay 时真实 KV 更长, kernel 写 `score[*,*,1]` 越界 → VMFault.

**为什么 eager 不崩**: eager 走 `init_forward_metadata`, 每次按真实 seq_lens 算 max_seqlen_k, score tensor 大小匹配.
**为什么并发才崩**: 单请求 seq_lens 也 >1, 并发是稳定触发条件 (单请求开 cg 可能也崩, 未充分测).

## 定位过程 (EAGLE_PROBE + HIP_LAUNCH_BLOCKING)

1. 加链路埋点 (`eagle_worker_v2.py` 的 draft/verify/draft_extend 阶段边界 + tensor 检查)
2. `HIP_LAUNCH_BLOCKING=1` 让 GPU kernel 同步执行, VMFault 立即在出错 kernel 触发
3. 崩点定位: `verify.target_forward.start` 之后无 `verify.target_forward.end` → 崩在 target verify forward
4. `sparse.fwd_extend` probe 显示 capture 时 `maxk=5`, replay 时需要 `maxk=194` → score tensor 越界
5. 排除 draft_token 越界 (tolist 确认 max=200059=`<mm:think>`, 合法; 之前误判 200100 是 probe `:.3e` 格式化精度)

## 解决方案

### 方案A (采用): target verify 走 eager

强制 target verify 走 eager, draft/draft_extend 仍走 cuda graph:
- `eagle_info_v2.py::prepare_for_v2_verify`: 对 MiniMax sparse backend 强制 `can_run_cuda_graph=False`
- `cuda_graph_runner.py::can_run`: 对 TARGET_VERIFY + MiniMax sparse backend 返回 False (双保险)
- 环境变量 `EAGLE3_VERIFY_EAGER=1` (默认开)

**性能**: 比全 cuda graph 慢约 20% (target verify 占 70%, eager 比 graph 慢 ~30%); 比关 cg 快约 10% (draft 两阶段仍 graph). EAGLE3 投机加速完全保留.

**验证**: 开 cuda graph max-bs 16, 并发 3 路 3/3 成功, VMFault=0.

### 方案C (保留死代码, 为方案D铺路)

`minimax_sparse_backend.py` 的 capture/replay 用 `max_seqlen_k = context_len` (固定上界 16384), 让 score tensor 固定大小. 单独不够 (sparse kernel 还有其他动态依赖: o/topk_idx 等), 但解决 score tensor 这一项. 方案A 下走 eager 不经此路径, 作死代码保留, 为方案D 留基础.

### 方案D (待办, 全 cuda graph)

让 sparse attention 的所有动态 tensor (score/o/topk_idx) 在 capture/replay 间一致, 使 target verify 也能走 cuda graph. 基于方案C 的固定上界思路继续解决剩余动态依赖. 工作量大, 需逐个分析 sparse kernel 的动态 shape 依赖并预分配固定 buffer.

## 涉及文件

| 文件 | 修改 | 备份 |
|---|---|---|
| `sglang/srt/speculative/eagle_info_v2.py` | 方案A: prepare_for_v2_verify 强制 eager | `sglang_backup/eagle_info_v2.py.bak` |
| `sglang/srt/model_executor/cuda_graph_runner.py` | 方案A: can_run 对 sparse+verify 返回 False | `sglang_backup/cuda_graph_runner.py.bak` |
| `sglang/srt/layers/attention/minimax_sparse_backend.py` | 方案C: capture/replay 用固定上界 (死代码) | `sglang_backup/minimax_sparse_backend.py` |

详见 `../../sglang_patches/README.md` 补丁 10-11.
