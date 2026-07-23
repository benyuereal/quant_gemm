"""tilelang W4A16 fused MoE 高层 API (MiniMax-M3 MoE expert 完整实现).

替换 sglang fused_moe 里的两个 int4 GEMM kernel, 路由/重排/combine 复用 sglang 基础设施:
    1. 路由重排:  sglang dcu_moe_align_block_size (GPU kernel, ~25us)
    2. gemm1:     tilelang w4a16_grouped_gemm (gate_up, [total, K] @ [E, N_gate_up, K]_int4)
    3. silu_and_mul: sgl_kernel
    4. gemm2:     tilelang w4a16_grouped_gemm (down, [total, N_inter] @ [E, N_down, N_inter]_int4)
    5. combine:   topk_weights 加权 index_add (只对前 ntp_val 有效行)

输入权重布局 (sglang W4A16, 与量化产物一致):
    w1       [E, N_gate_up, K//2]        uint8  (int4 packed)
    w1_scale [E, N_gate_up, K//group]    bf16
    w2       [E, N_down, N_inter//2]     uint8
    w2_scale [E, N_down, N_inter//group] bf16

激活 bf16 (不量化), 路由后 token 按 expert 分组连续排列.

性能 (gfx936, M3 真实 shape E=128 K=6144 shard_inter=1536 topk=4, vs sglang tuned Triton):
    M=1  (decode): 1.69x   M=2: 2.66x   M=4: 3.76x   M=8: 4.00x   M=32: 4.32x
正确性: vs 量化反量化参考 rel≈1.05%, max_diff≈3e-4 (全 batch PASS).
"""

import math
import torch

from .kernels import w4a16_grouped_gemm, GROUP

__all__ = ["w4a16_fused_moe", "tilelang_fused_moe_simple", "moe_align_torch", "build_route_tables"]

DTYPE = torch.bfloat16


def build_route_tables(group_sizes, block_M):
    """group_sizes: list[E] 每 expert 的 token 数.
    返回 (block_to_expert, block_m_start, block_actual_rows) list (供纯 torch 路径用)."""
    block_to_expert, block_m_start, block_actual_rows = [], [], []
    for e, gs in enumerate(group_sizes):
        go = sum(group_sizes[:e])
        n_blocks_e = math.ceil(gs / block_M)
        for b in range(n_blocks_e):
            block_to_expert.append(e)
            block_m_start.append(go + b * block_M)
            block_actual_rows.append(min(block_M, gs - b * block_M))
    return block_to_expert, block_m_start, block_actual_rows


def moe_align_torch(topk_ids, topk_weights, block_M, E):
    """纯 torch 路由重排 (等价 sglang moe_align_block_size, 但避开其小 batch/小 E kernel bug).
    返回 (tok_idx[total_pad], expert_ids[num_m_blocks], w_per_row[total_pad], valid[total_pad], total_pad).
    每 expert 的 (token,k) 对连续排列, 各自 pad 到 block_M 倍数, block 严格不跨 expert.
    用于 dcu_moe_align_block_size 不可用 (小 E) 或需要可控 weights_per_row 的场景."""
    num_tokens, topk = topk_ids.shape
    M_flat = num_tokens * topk
    dev = topk_ids.device
    flat_exp = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_exp, stable=True)         # 排序后扁平索引
    exp_sorted = flat_exp[order]
    counts = torch.bincount(exp_sorted, minlength=E)
    blocks_per_expert = (counts + block_M - 1) // block_M
    total_pad = int(blocks_per_expert.sum().item()) * block_M

    sorted_ids = torch.full((total_pad,), M_flat, dtype=torch.int64, device=dev)
    w_per_row = torch.zeros(total_pad, dtype=topk_weights.dtype, device=dev)
    w_flat = topk_weights.reshape(-1)
    expert_ids = torch.zeros(int(blocks_per_expert.sum().item()), dtype=torch.int32, device=dev)
    pos = 0
    bpos = 0
    for e in range(E):
        c = int(counts[e].item())
        nb = int(blocks_per_expert[e].item())
        if c > 0:
            idxs = order[exp_sorted == e]
            n = idxs.numel()
            sorted_ids[pos:pos + n] = idxs
            w_per_row[pos:pos + n] = w_flat[idxs]
        expert_ids[bpos:bpos + nb] = e
        pos += nb * block_M
        bpos += nb
    valid = sorted_ids < M_flat
    tok_idx = sorted_ids.clamp(max=M_flat - 1) // topk
    return tok_idx, expert_ids, w_per_row, valid, total_pad


def tilelang_fused_moe_simple(
    hidden_states,            # [num_tokens, K] bf16
    w1, w1_scale,             # w1 [E, N_gate_up, K//2] uint8, w1_scale [E, N_gate_up, K//group]
    w2, w2_scale,             # w2 [E, N_down, N_inter//2] uint8, w2_scale [E, N_down, N_inter//group]
    topk_weights, topk_ids,   # [num_tokens, topk] fp32 / int32
    block_M=16, block_N=64, num_stages=2, threads=256,
):
    """tilelang W4A16 fused MoE.

    路由重排用 sglang dcu_moe_align_block_size (GPU kernel, ~25us),
    两个 GEMM 用 tilelang W4A16, silu_and_mul / combine 复用 sgl_kernel + index_add.
    小 batch 性能关键: kernel 跳过 padding 块 (num_valid_blocks), M=1 时 117/121 块是 pad.

    注意: dcu_moe_align_block_size 在小 E (如 E<8) 时 expert_ids 输出异常,
    该路径仅适用于真实部署规模 (E=128). 小 E 测试用 moe_align_torch 路径.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import dcu_moe_align_block_size
    from sgl_kernel import silu_and_mul

    num_tokens, K = hidden_states.shape
    E, N_gate_up, _ = w1.shape
    N_inter = N_gate_up // 2
    N_down = w2.shape[1]
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    M_flat = num_tokens * topk

    # ---- 1. 路由重排 (sglang GPU kernel) ----
    # 返回 sorted_ids [total_pad], expert_ids [num_m_blocks], num_tokens_post_padded [1].
    # 注意: expert_ids 只有前 ceil(num_tokens_post_padded/block_M) 个有效, 后面是未初始化垃圾
    # (dcu_align 只写有效 block). 我们用 num_valid_blocks 让 kernel 跳过这些垃圾块.
    sorted_ids, expert_ids, num_tokens_post_padded = dcu_moe_align_block_size(topk_ids, block_M, E)
    sorted_ids = sorted_ids.to(torch.int64)
    total_pad = sorted_ids.numel()
    num_m_blocks = expert_ids.numel()

    # 有效 block 数 = ceil(num_tokens_post_padded / block_M). 超出的 block 是 padding, 跳过.
    ntp_val = int(num_tokens_post_padded.item()) if torch.is_tensor(num_tokens_post_padded) else int(num_tokens_post_padded)
    num_valid_blocks = (ntp_val + block_M - 1) // block_M
    nvb = torch.tensor([num_valid_blocks], device=dev, dtype=torch.int32)

    # ---- 2. gather 激活 (pad 位置读 0) ----
    valid = sorted_ids < M_flat
    safe_flat = sorted_ids.clamp(max=M_flat - 1)
    tok_idx = safe_flat // topk
    A_gathered = hidden_states[tok_idx]
    A_gathered = A_gathered * valid.unsqueeze(-1).to(A_gathered.dtype)
    # dcu_align 的 sorted_ids 长度可能不是 block_M 整数倍, 但 num_m_blocks*block_M 是.
    # pad A_gathered 到 num_m_blocks*block_M, 保证 GEMM 不越界.
    total_gem = num_m_blocks * block_M
    if total_gem > total_pad:
        A_gathered = torch.cat(
            [A_gathered, torch.zeros(total_gem - total_pad, K, device=dev, dtype=DTYPE)], dim=0,
        )
    # combine 用的 w_per_row / tok_idx 也补齐到 total_gem (pad 行权重 0, 不影响 index_add)
    if total_gem > total_pad:
        valid = torch.cat([valid, torch.zeros(total_gem - total_pad, dtype=torch.bool, device=dev)])
        safe_flat = torch.cat([safe_flat, torch.zeros(total_gem - total_pad, dtype=torch.int64, device=dev)])
        tok_idx = torch.cat([tok_idx, torch.zeros(total_gem - total_pad, dtype=torch.int64, device=dev)])
    total_pad = total_gem

    # ---- 3. 路由表: 每 block 满 block_M (pad 行靠 gather 读 0) ----
    block_m_start = torch.arange(num_m_blocks, device=dev, dtype=torch.int32) * block_M
    block_actual_rows = torch.full((num_m_blocks,), block_M, device=dev, dtype=torch.int32)

    # ---- 4. gemm1 (gate_up) ----
    kernel1 = w4a16_grouped_gemm(
        E, total_pad, N_gate_up, K, num_m_blocks,
        block_M, block_N, group_size=GROUP,
        num_stages=num_stages, threads=threads,
    )
    cache1 = kernel1(A_gathered, w1, w1_scale, expert_ids, block_m_start, block_actual_rows, nvb)

    # ---- 5. silu_and_mul ----
    cache_inter = silu_and_mul(cache1.view(-1, N_gate_up))     # [total_pad, N_inter]

    # ---- 6. gemm2 (down) ----
    kernel2 = w4a16_grouped_gemm(
        E, total_pad, N_down, N_inter, num_m_blocks,
        block_M, block_N, group_size=GROUP,
        num_stages=num_stages, threads=threads,
    )
    cache_down = kernel2(cache_inter, w2, w2_scale, expert_ids, block_m_start, block_actual_rows, nvb)

    # ---- 7. combine: w_per_row[i] = topk_weights.flatten()[sorted_ids[i]], pad 位置 0 ----
    # 只对前 ntp_val 行做 (ntp_val = 填充到 block_M 整数倍的有效区域边界).
    # 注意 [:ntp_val] 内仍含 padding 行 (sorted_ids == M_flat), 用 valid mask 置 0 权重.
    w_flat = topk_weights.reshape(-1).to(DTYPE)
    cd_valid = cache_down[:ntp_val]                          # [ntp_val, N_down]
    sf_valid = safe_flat[:ntp_val]                           # [ntp_val]
    tok_valid = tok_idx[:ntp_val]                            # [ntp_val]
    valid_v = valid[:ntp_val]                                # [ntp_val]
    w_per_row = torch.where(valid_v, w_flat[sf_valid], torch.zeros((), device=dev, dtype=DTYPE))
    cache_down_weighted = cd_valid * w_per_row.unsqueeze(-1)
    final = torch.zeros(num_tokens, N_down, device=dev, dtype=DTYPE)
    final.index_add_(0, tok_valid, cache_down_weighted)
    return final


def w4a16_fused_moe_aligned(
    hidden_states,            # [num_tokens, K] bf16
    w1, w1_scale,             # w1 [E, N_gate_up, K//2] uint8, w1_scale [E, N_gate_up, K//group]
    w2, w2_scale,             # w2 [E, N_down, N_inter//2] uint8, w2_scale [E, N_down, N_inter//group]
    topk_weights, topk_ids,   # [num_tokens, topk] fp32 / int32
    sorted_token_ids,         # [total_pad] int32 (sglang 已对齐, -1 或 num_tokens*topk 表 pad)
    expert_ids,               # [num_m_blocks] int32 (每 block 的 expert, 仅前 nvb 个有效)
    num_tokens_post_padded,   # [1] int32 (有效 token 数, 填充到 block_M 倍数边界)
    block_M=16, block_N=64, num_stages=2, threads=256,
):
    """tilelang W4A16 fused MoE, 接收 sglang 已对齐的路由输入 (复用 sorted_token_ids 等,
    避免重复 align). 用于替换 sglang TritonRunnerCore.run 的 int4 分支.

    与 tilelang_fused_moe_simple 的区别: 不内部调 dcu_moe_align_block_size, 直接用传入的
    sorted_token_ids/expert_ids/num_tokens_post_padded (sglang token_dispatcher 已算好).

    输入语义 (sglang):
      sorted_token_ids[i] = 该 row 对应的 (token*topk+k) 扁平索引, pad 行 = num_tokens*topk
      expert_ids[b] = block b 的 expert id (仅前 ceil(ntp/block_M) 个有效, 后面未初始化)
      num_tokens_post_padded = 有效 token 总数 (填充到 block_M 倍数)
    """
    from sgl_kernel import silu_and_mul

    num_tokens, K = hidden_states.shape
    E, N_gate_up, _ = w1.shape
    N_inter = N_gate_up // 2
    N_down = w2.shape[1]
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    M_flat = num_tokens * topk

    sorted_ids = sorted_token_ids.to(torch.int64)
    total_pad = sorted_ids.numel()
    num_m_blocks = expert_ids.numel()

    ntp_val = int(num_tokens_post_padded.item()) if torch.is_tensor(num_tokens_post_padded) else int(num_tokens_post_padded)
    num_valid_blocks = (ntp_val + block_M - 1) // block_M
    nvb = torch.tensor([num_valid_blocks], device=dev, dtype=torch.int32)
    # C/A 行数: 防 nvb*block_M > total_pad 越界 (通常 total_pad 已够大)
    total_rows = max(total_pad, num_valid_blocks * block_M)

    # gather 激活: sorted_ids 是 (token*topk+k) 扁平索引, tok = sorted_ids // topk
    valid = sorted_ids < M_flat
    safe_flat = sorted_ids.clamp(max=M_flat - 1)
    tok_idx = safe_flat // topk
    A_gathered = hidden_states[tok_idx]
    A_gathered = A_gathered * valid.unsqueeze(-1).to(DTYPE)
    if total_rows > total_pad:
        A_gathered = torch.cat(
            [A_gathered, torch.zeros(total_rows - total_pad, K, device=dev, dtype=DTYPE)], dim=0,
        )

    block_m_start = torch.arange(num_m_blocks, device=dev, dtype=torch.int32) * block_M
    block_actual_rows = torch.full((num_m_blocks,), block_M, device=dev, dtype=torch.int32)

    kernel1 = w4a16_grouped_gemm(
        E, total_rows, N_gate_up, K, num_m_blocks,
        block_M, block_N, group_size=GROUP,
        num_stages=num_stages, threads=threads,
    )
    cache1 = kernel1(A_gathered, w1, w1_scale, expert_ids, block_m_start, block_actual_rows, nvb)
    cache_inter = silu_and_mul(cache1.view(-1, N_gate_up))

    kernel2 = w4a16_grouped_gemm(
        E, total_rows, N_down, N_inter, num_m_blocks,
        block_M, block_N, group_size=GROUP,
        num_stages=num_stages, threads=threads,
    )
    cache_down = kernel2(cache_inter, w2, w2_scale, expert_ids, block_m_start, block_actual_rows, nvb)

    # combine: 只对前 ntp_val 行 (含 pad 行用 valid mask 置 0 权重)
    w_flat = topk_weights.reshape(-1).to(DTYPE)
    cd_valid = cache_down[:ntp_val]
    sf_valid = safe_flat[:ntp_val]
    tok_valid = tok_idx[:ntp_val]
    valid_v = valid[:ntp_val]
    w_per_row = torch.where(valid_v, w_flat[sf_valid], torch.zeros((), device=dev, dtype=DTYPE))
    cache_down_weighted = cd_valid * w_per_row.unsqueeze(-1)
    final = torch.zeros(num_tokens, N_down, device=dev, dtype=DTYPE)
    final.index_add_(0, tok_valid, cache_down_weighted)
    return final


# 别名: 与 W8A8 的 w8a8_moe 命名对齐
w4a16_fused_moe = tilelang_fused_moe_simple
