"""tilelang W4A16 grouped GEMM kernel for Hygon DCU (gfx936/gfx928).

MoE expert 的 per-group int4 权重 + bf16 激活 GEMM. 替换 sglang fused_moe 里的
`fused_moe_kernel_gptq_awq` (Triton int4) 两个 GEMM:
    gemm1 (gate_up): [total_tokens, K] @ [E, N_gate_up, K]_int4 -> [total_tokens, N_gate_up]
    gemm2 (down):    [total_tokens, N_inter] @ [E, N_down, N_inter]_int4 -> [total_tokens, N_down]

权重布局 (sglang W4A16, 与量化产物一致):
    W       [E, N, K//2]        uint8  (int4 packed, 2/byte, nibble: k 偶数低 4 位)
    w_scale [E, N, K//group]    bf16   (per-group, group=128)
    deq = (q - 8) * scale,  q ∈ [0,15]  (zp=8 对称, 与 sglang int4_w4a16 一致)

性能关键: kernel 内 `if bx < num_valid_blocks[0]` 跳过 padding 块 (与 sglang
`if pid_m*BLOCK_M >= num_tokens_post_padded: return` 等价). 小 batch (M=1) 时
121 个 block 仅 4 个有效, 跳过 117 个垃圾块, 否则 97% 算力浪费.

精度 (gfx936, M3 真实 shape E=128): vs 量化反量化参考 rel≈1.05%, max_diff≈3e-4.
"""

import tilelang
import tilelang.language as T
from tilelang.quantize import _tir_packed_to_unsigned_convert

__all__ = ["w4a16_grouped_gemm", "GROUP", "MAX_TRANSACTION_BITS"]

GROUP = 128
MAX_TRANSACTION_BITS = 128


@tilelang.jit(out_idx=[7])
def w4a16_grouped_gemm(
    E, total_tokens, N, K, num_m_blocks,
    block_M, block_N,
    group_size=GROUP,
    in_dtype=T.bfloat16, accum_dtype=T.float32, out_dtype=T.bfloat16,
    storage_dtype=T.uint8, num_stages=2, threads=256,
):
    """W4A16 grouped GEMM (gate_up 或 down 通用).

    输出 C [total_tokens, N] bf16.
    block_K = group_size (每 K group 一个 scale, 索引简单).
    路由通过 block_to_expert/block_m_start/block_actual_rows 表传入 (同 W8A8 框架).
    num_tokens_post_padded: 有效 token 数 (sglang _prepare_fused_moe_run 输出). kernel 内算
        num_valid_blocks = ceildiv(ntp, block_M), 超出的 padding 块直接跳过 GEMM (与 sglang
        `if pid_m*BLOCK_M >= num_tokens_post_padded: return` 等价). 传 tensor 不在 host 算,
        避免 .item() (cuda graph capture 兼容). 小 batch 性能关键: M=1 时 121 个 block 只有
        4 个有效, 跳过 117 个垃圾块.

    Args:
        E: expert 数.
        total_tokens: A/C 的行数 (sorted+pad 后, = num_m_blocks*block_M).
        N: 输出维 (gate_up=N_gate_up, down=N_down).
        K: 输入维 (gate_up=hidden, down=N_inter).
        num_m_blocks: M 方向 block 数 (= grid.x). 编译期常量, 需为可能的最大值.
        block_M: M block 大小 (tilelang 要求 divisible by 16, 最小 16).
        block_N: N block 大小.
        group_size: per-group 量化组大小 (block_K = group_size).
    """
    assert K % group_size == 0
    block_K = group_size
    num_elems_per_byte = 2  # int4
    num_bits = 4
    storage_nbit = 8
    storage_type = "uint"
    num_k_groups = K // group_size
    num_n_blocks = T.ceildiv(N, block_N)

    from tvm import DataType
    local_size = MAX_TRANSACTION_BITS // DataType(in_dtype).bits  # bf16 -> 8
    local_size_compressed = local_size // num_elems_per_byte       # 4

    @T.prim_func
    def main(
        A: T.Tensor((total_tokens, K), in_dtype),                       # grouped 激活 [total_tokens, K] bf16
        W: T.Tensor((E, N, K // num_elems_per_byte), storage_dtype),    # packed int4 [E, N, K//2]
        w_scale: T.Tensor((E, N, num_k_groups), in_dtype),              # per-group [E, N, K//group]
        block_to_expert: T.Tensor((num_m_blocks,), T.int32),
        block_m_start: T.Tensor((num_m_blocks,), T.int32),
        block_actual_rows: T.Tensor((num_m_blocks,), T.int32),
        num_tokens_post_padded: T.Tensor((1,), T.int32),               # 有效 token 数, kernel 内算 nvb 跳过 pad 块
        C: T.Tensor((total_tokens, N), out_dtype),
    ):
        with T.Kernel(num_m_blocks, num_n_blocks, threads=threads) as (bx, by):
            # 跳过 padding 块 (与 sglang `if pid_m*BLOCK_M >= num_tokens_post_padded: return` 等价).
            # num_valid_blocks = ceildiv(num_tokens_post_padded, block_M), kernel 内算, 避免 host .item() (cuda graph 兼容).
            if bx * block_M < num_tokens_post_padded[0]:
                A_shared = T.alloc_shared((block_M, block_K), in_dtype)
                W_shared = T.alloc_shared((block_N, block_K // num_elems_per_byte), storage_dtype)
                B_local = T.alloc_local([local_size_compressed], storage_dtype)
                B_dequant_local = T.alloc_local([local_size], in_dtype)
                W_deq_shared = T.alloc_shared((block_N, block_K), in_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_partial = T.alloc_fragment((block_M, block_N), accum_dtype)
                ws_local = T.alloc_fragment((block_N,), in_dtype)

                tx = T.get_thread_binding()

                cur_expert = block_to_expert[bx]
                m_start = block_m_start[bx]
                actual_rows = block_actual_rows[bx]

                T.clear(C_local)
                for kg in T.Pipelined(num_k_groups, num_stages=num_stages):
                    # load A 的这个 K group
                    T.copy(A[m_start, kg * block_K], A_shared)
                    # load W 的这个 K group (packed), 本 expert 本 n_block
                    T.copy(W[cur_expert, by * block_N, kg * block_K // num_elems_per_byte], W_shared)

                    # 线程级 int4 unpack: W_shared [block_N, block_K//2] uint8 -> W_deq_shared [block_N, block_K] bf16
                    for i in T.serial(block_N * (block_K // num_elems_per_byte) // (threads * local_size_compressed)):
                        for v in T.vectorized(0, local_size_compressed):
                            index = i * threads * local_size_compressed + tx * local_size_compressed + v
                            vi = index // (block_K // num_elems_per_byte)
                            vj = index % (block_K // num_elems_per_byte)
                            B_local[v] = W_shared[vi, vj]
                        for v in T.serial(0, local_size):
                            B_dequant_local[v] = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                                num_bits,
                                B_local[v // num_elems_per_byte],
                                v % num_elems_per_byte,
                                dtype=in_dtype,
                            ) - T.cast(8.0, in_dtype)
                        for v in T.vectorized(0, local_size):
                            index = i * threads * local_size + tx * local_size + v
                            vi = index // block_K
                            vj = index % block_K
                            W_deq_shared[vi, vj] = B_dequant_local[v]

                    # 这个 K group 的 partial GEMM (无 scale)
                    T.clear(C_partial)
                    T.gemm(A_shared, W_deq_shared, C_partial, transpose_B=True)

                    # load 这个 group 的 per-N scale (本 expert, 本 n_block)
                    for j in T.Parallel(block_N):
                        ws_local[j] = w_scale[cur_expert, by * block_N + j, kg]

                    # 乘 scale 累加: C_local += C_partial * ws_local
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_partial[i, j] * T.cast(ws_local[j], accum_dtype)

                # 写回, 边界外填 0
                for i, j in T.Parallel(block_M, block_N):
                    if i < actual_rows:
                        C[m_start + i, by * block_N + j] = T.cast(C_local[i, j], out_dtype)
                    else:
                        C[m_start + i, by * block_N + j] = T.cast(0.0, out_dtype)

    return main
