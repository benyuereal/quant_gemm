"""tilelang W8A8 GEMM kernels for Hygon DCU (gfx936/gfx928).

在海光 DCU 上, lightop 汇编包无 gfx928 产物, aiter per-channel 路径依赖未启用的
aiter_ C 扩展, 故用 tilelang (海光定制版, 架构无关 JIT) 自写 W8A8 算子。

本模块提供两个 per-channel W8A8 GEMM kernel:
    - w8a8_per_channel_gemm:  Linear W8A8 (单矩阵)
    - moe_w8a8_grouped_gemm:  MoE grouped W8A8 (多 expert, 路由表)

量化方案: per-channel int8 weight + per-token int8 activation (dynamic).
    C[m,n] = (sum_k A[m,k] * B[n,k]) * x_scale[m] * w_scale[n]

精度 (gfx936 实测): vs 量化反量化 0.28%, vs bf16 1.25%.
gfx928 待实测 (架构无关, 应可 JIT).
"""

import tilelang
import tilelang.language as T

__all__ = ["w8a8_per_channel_gemm", "moe_w8a8_grouped_gemm"]


@tilelang.jit(out_idx=[4])
def w8a8_per_channel_gemm(M, N, K, block_M, block_N, block_K,
                          in_dtype=T.int8, accum_dtype=T.int32,
                          out_dtype=T.bfloat16, num_stages=3, threads=128):
    """per-channel W8A8 Linear GEMM. C = (A @ B.T) * x_scale * w_scale.T

    Args:
        M, N, K: 矩阵维度. 需 M 对齐 block_M, N 对齐 block_N, K 对齐 block_K.
        block_M/N/K: tile 块大小.
    Returns:
        tilelang kernel, 调用 kernel(A, B, x_scale, w_scale) -> C
        A: [M,K] int8 (per-token 量化), x_scale: [M,1] f32
        B: [N,K] int8 (per-channel 量化), w_scale: [N,1] f32
        C: [M,N] bf16
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        x_scale: T.Tensor((M, 1), T.float32),
        w_scale: T.Tensor((N, 1), T.float32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_scaled = T.alloc_fragment((block_M, block_N), T.float32)
            xs_local = T.alloc_fragment((block_M,), T.float32)
            ws_local = T.alloc_fragment((block_N,), T.float32)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            for i in T.Parallel(block_M):
                xs_local[i] = x_scale[by * block_M + i, 0]
            for j in T.Parallel(block_N):
                ws_local[j] = w_scale[bx * block_N + j, 0]

            for i, j in T.Parallel(block_M, block_N):
                C_scaled[i, j] = T.cast(C_local[i, j], T.float32) * xs_local[i] * ws_local[j]

            T.copy(C_scaled, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[7])
def moe_w8a8_grouped_gemm(E, total_tokens, N, K, num_m_blocks,
                          block_M, block_N, block_K,
                          in_dtype=T.int8, accum_dtype=T.int32, out_dtype=T.bfloat16,
                          num_stages=2, threads=256):
    """per-channel W8A8 MoE grouped GEMM (单步, gate/up 或 down 通用).

    多个 expert, 每个 expert 处理一组 (按路由分组后连续排列的) token.
    路由通过预算的 block_to_expert/block_m_start/block_actual_rows 表传入,
    避开 kernel 内控制流 (与 sglang expert_ids[pid_m] 一致).

    Args:
        E: expert 数. total_tokens: 总 token 数 (须 pad 到 block_M 倍数).
        N, K: 权重 [E,N,K] 的 N,K. num_m_blocks: M 方向 block 数.
        block_M/N/K: tile 块大小.
    Returns:
        tilelang kernel, 调用 kernel(A, W, x_scale, w_scale, block_to_expert,
        block_m_start, block_actual_rows) -> C
        A: [total_tokens,K] int8 (已按 expert 分组连续排列), x_scale: [total_tokens,1] f32
        W: [E,N,K] int8 (per-channel), w_scale: [E,N,1] f32
        block_to_expert: [num_m_blocks] int32, 每 m_block 的 expert
        block_m_start: [num_m_blocks] int32, 每 m_block 的 token 起始
        block_actual_rows: [num_m_blocks] int32, 每 m_block 有效 token 数
        C: [total_tokens,N] bf16
    """
    num_n_blocks = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((total_tokens, K), in_dtype),
        W: T.Tensor((E, N, K), in_dtype),
        x_scale: T.Tensor((total_tokens, 1), T.float32),
        w_scale: T.Tensor((E, N, 1), T.float32),
        block_to_expert: T.Tensor((num_m_blocks,), T.int32),
        block_m_start: T.Tensor((num_m_blocks,), T.int32),
        block_actual_rows: T.Tensor((num_m_blocks,), T.int32),
        C: T.Tensor((total_tokens, N), out_dtype),
    ):
        with T.Kernel(num_m_blocks, num_n_blocks, threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            W_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_scaled = T.alloc_fragment((block_M, block_N), T.float32)
            xs_local = T.alloc_fragment((block_M,), T.float32)
            ws_local = T.alloc_fragment((block_N,), T.float32)

            cur_expert = block_to_expert[bx]
            m_start = block_m_start[bx]
            actual_rows = block_actual_rows[bx]

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(
                    W[cur_expert, by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K],
                    W_shared,
                )
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            for i in T.Parallel(block_M):
                xs_local[i] = x_scale[m_start + i, 0]
            for j in T.Parallel(block_N):
                ws_local[j] = w_scale[cur_expert, by * block_N + j, 0]

            for i, j in T.Parallel(block_M, block_N):
                C_scaled[i, j] = T.cast(C_local[i, j], T.float32) * xs_local[i] * ws_local[j]

            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = T.cast(C_scaled[i, j], out_dtype)
                else:
                    C[m_start + i, by * block_N + j] = T.cast(0.0, out_dtype)

    return main
