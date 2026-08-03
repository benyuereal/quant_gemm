# EAGLE3 TARGET_VERIFY 全 cuda graph VMFault — 链路分析

## 核心结论（根因链）

**"没开 EAGLE3 时 cuda graph 好好的"** 的根本原因：

sglang 的 cuda graph 只对 `ForwardMode.is_cuda_graph()` 返回 True 的模式启用：
```python
def is_cuda_graph(self):
    return (self == DECODE or self == TARGET_VERIFY or self == IDLE or self == DLLM_EXTEND)
```
- **普通 EXTEND（prefill）不在内** → 普通 prefill 永远走 eager，sparse prefill kernel 的 graph-unsafe 缺陷从未暴露
- **TARGET_VERIFY 在内** → EAGLE3 的 verify 走 cuda graph

而 MiniMax sparse attention 有两套 kernel：
- `minimax_sparse_decode`：**graph-safe**，注释 "grid independent of seq_len for cuda graph"，固定 chunk 数
- `minimax_sparse_prefill`：**graph-unsafe**，grid 依赖 `batch_size * num_heads` 和 `max_seqlen_q`，有动态间接寻址

**EAGLE3 是第一个让 sparse prefill kernel 进入 cuda graph 的场景**。普通推理 target model 只做 decode（走 graph-safe 的 decode kernel）；EAGLE3 让 target model 多做 verify，verify 走 forward_extend → sparse prefill kernel → graph-unsafe 缺陷暴露 → VMFault。

## EAGLE3 V2 一个 decode step 的三阶段

```
1. draft()        draft model 生成 num_draft_tokens(4) 个候选 (DRAFT_DECODE)
2. verify()       target model 验证候选 (TARGET_VERIFY)  ← VMFault 在这
3. draft_extend() 用 verify 接受的 token 生成下一批 draft (DRAFT_EXTEND_V2)
```

## verify 阶段数据流（prepare_for_v2_verify → init_new → forward_batch_generation）

`prepare_for_v2_verify`（eagle_info_v2.py:261）构造 verify_forward_batch：

| 字段 | 值 | 备注 |
|------|-----|------|
| input_ids | draft_token `[bs*4]` | 真实候选 token |
| forward_mode | TARGET_VERIFY | |
| seq_lens | batch.seq_lens (**只含 prefix**) | scheduler verify 后才加 accept_lens |
| out_cache_loc | assign_extend_cache_locs(seq_lens, seq_lens+4) | 从 req_to_token[req, prefix:prefix+4] 读 slot |
| positions | clamp_position(seq_lens) 或 spec_info.positions | EAGLE3 特有: spec_info.positions=None 时走 clamp |
| extend_seq_lens | **None** | target_verify 走 init_new 的 decode 分支(forward_batch_info.py:557) |
| extend_prefix_lens | **None** | 同上 |
| req_pool_indices | batch.req_pool_indices | 真实 |
| spec_info | EagleVerifyInput(draft_token_num=4) | |

**关键差异（EAGLE3 verify vs 普通 prefill）**：
1. `seq_lens` 只含 prefix（普通 prefill 是 prefix+extend）
2. `extend_seq_lens`/`extend_prefix_lens` 是 None（普通 prefill 由 init_new 构造）→ sparse backend 的 forward_extend 必须现场物化
3. 走 cuda graph（普通 prefill 走 eager）

## sparse prefill kernel 在 graph 下的动态张量审计

`forward_extend`（minimax_sparse_backend.py:312）传给 `minimax_sparse_prefill` 的动态量：

| 量 | capture 时 | replay 时 | graph-safe? |
|----|-----------|----------|-------------|
| cu_seqlens | [0,4,8,...,bs*4] (extend_seq_lens.cumsum) | 同 capture (forward_extend 不执行, tensor 固定) | ✓ 固定 |
| seq_lens | dummy [1,1,..]+4=[5,5,..] | 真实 prefix + 4 (buffers.seq_lens + 固定 extend_seq_lens) | ✓ 真实 |
| prefix_lens | dummy [1,1,..] | 真实 prefix (buffers.seq_lens view) | ✓ 真实 |
| slot_ids (req_pool_indices) | [0,0,..] | 真实 (buffers.req_pool_indices, [raw_bs:bs] padding=0) | ✓ 真实 |
| _max_seqlen_k | upper=16384 (Fix C) | upper=16384 | ✓ 固定 |
| score tensor | (num_heads, bs*4, 128) | 同 capture (torch.full 在 capture 分配) | ✓ 固定大小 |
| total_q (q.shape[0]) | bs*4 | 同 capture | ✓ 固定 |

## 已排除的假设

1. **padding 请求越界**: capture_bs=[1,2,3,4,5,6,7,8,10,12,14,16] 含 3, raw_bs=3 无 padding 仍崩
2. **score 张量越界**: make_block_ptr + boundary_check 保护写入
3. **req_to_token 第0行 padding**: slot 0 是有效地址 (ReqToTokenPool 注释明确为 padding 行)

## 复现现场（决定性证据）

```
bs=1, seq_lens=[703], req_pool_indices=[2]            → 不崩
bs=3, seq_lens=[188,187,190], req_pool_indices=[4,5,6] → 崩 (VMFault 0x7f...afd000)
```
无 padding，真实请求，仅 batch_size 不同。

## 当前怀疑（待 OOB 探针验证）

sparse prefill kernel 间接寻址：
```triton
slots = req_to_token[sid, pos]   # sid=req_pool_indices[i], pos in [0, seq_len)
k = k_cache[slots]               # 若 slots >= max_slots → OOB → VMFault
```
verify 时 seq_len = prefix + 4。若 `req_to_token[req, prefix:prefix+4]`（draft token 范围）有 ≥max_slots 的无效 slot id → k_cache OOB。

draft 阶段 `prepare_for_decode` 用 `assign_req_to_token_pool_func` 预填 `req_to_token[req, cur_kv_len:nxt_kv_len]`，`nxt = max(cur, kv_committed_len + 2*alloc_len_per_decode)`，`alloc_len_per_decode = max(3*1, 4) = 4`，所以 nxt ≈ seq_lens+7，应覆盖 [seq_lens, seq_lens+4]。**但若 kv_committed_len 落后 seq_lens 多步，或 alloc 间隙，可能漏填**。

OOB 探针（EAGLE3_VERIFY_PROBE=1）在 replay 前直接检查 `req_to_token[req, :prefix+4]` 的 slot 范围：
- 若 "OOB DETECTED" → 修 req_to_token draft 范围填充（Fix D 方向）
- 若 "slots OK" → 排除间接寻址，查其他动态量（topk_idx 写入、o 输出 tensor）

## Fix 方案谱系

- **Fix A** (当前临时方案): EAGLE3_VERIFY_EAGER=1, verify 走 eager, draft/draft_extend 仍 cuda graph. 慢 ~20%.
- **Fix C** (已合入, 死代码): _max_seqlen_k 用 context_len 上界固定 score 大小. Fix A 下不生效, 留作 Fix D 基础.
- **Fix D** (目标): 全 cuda graph verify. 需让 sparse prefill kernel 所有动态量 graph-safe. 待 OOB 定位真凶后实现.
