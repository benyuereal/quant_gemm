// MoE W4A16 GEMM for Hygon DCU — scratch LDS roundtrip for correct v_mmac B fragment
//
// Key insight: v_mmac expects B operand in a specific vector register layout.
// Constructing half4_t from scalars doesn't match this layout.
// Fix: dequant → write to scratch LDS → read back via ds_read_b64 as half4_t.
// The ds_read_b64 instruction correctly places data in the expected register.
//
// Origin: evolved from an earlier packed-uint8 B in LDS, on-the-fly dequant variant.
// Target: gfx928 (CDNA3/MI300) uses v_mmac fast path; other gfx9xx fall back to
//         scalar (compiles & loads but no speedup — v_mmac is gfx928-only).

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

static inline hipStream_t get_stream() {
    return c10::hip::getCurrentHIPStream().stream();
}

typedef float float4_t __attribute__((ext_vector_type(4)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float4_t mmac_f16(half4_t a, half4_t b, float4_t c) {
#if __gfx928__
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0"
        : "+v"(c) : "v"(a), "v"(b));
#else
    for (int i = 0; i < 4; i++) c[i] += (float)a[i] * (float)b[i];
#endif
    return c;
}

constexpr int BM = 16;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / 64;

constexpr int LDS_A_STRIDE = BK + 4;       // 68
constexpr int LDS_B_STRIDE = BK / 2 + 8;   // 40 (multiple of 8)

__global__ __launch_bounds__(NUM_THREADS)
void moe_gemm_w4a16_dcu_kernel(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    const __half* __restrict__ B_scale,
    const uint8_t* __restrict__ B_zp,
    const float* __restrict__ topk_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded,
    const int N, const int K, const int EM, const int num_valid_tokens,
    const int stride_am, const int stride_ak,
    const int stride_be, const int stride_bk, const int stride_bn,
    const int stride_cm, const int stride_cn,
    const int stride_bse, const int stride_bsk, const int stride_bsn,
    const int stride_bze, const int stride_bzk, const int stride_bzn,
    const int group_size, const bool mul_routed_weight, const int top_k,
    const bool has_zp
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;

    const int num_n_blocks = (N + BN - 1) / BN;
    const int block_id = blockIdx.x;
    const int block_m_id = block_id / num_n_blocks;
    const int block_n_id = block_id % num_n_blocks;

    const int expert_id = expert_ids[block_m_id];
    if (expert_id < 0) return;

    const int m_start = block_m_id * BM;
    const int n_start = block_n_id * BN;
    const int warp_n_offset = warp_id * 16;

    __shared__ __half smem_A[BM * LDS_A_STRIDE];
    __shared__ uint8_t smem_B[BN * LDS_B_STRIDE];
    // Scratch buffer: each thread writes 4 fp16 values, reads back as half4_t
    __shared__ __half smem_scratch[NUM_THREADS * 4];

    float4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint8_t* b_expert = B + expert_id * stride_be;
    const __half* s_expert = B_scale + expert_id * stride_bse;
    const uint8_t* zp_expert = has_zp ? (B_zp + expert_id * stride_bze) : nullptr;

    const int num_k_iters = K / BK;
    const int b_load_n = tid / 4;
    const int b_load_k_off = (tid & 3) * 8;
    const int a_load_base = tid * 4;

    for (int ki = 0; ki < num_k_iters; ki++) {
        const int k_base = ki * BK;

        // --- Load A to LDS ---
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = a_load_base + i;
            int lm = idx / BK;
            int lk = idx % BK;
            int gm = m_start + lm;
            __half val = __float2half(0.0f);
            if (gm < EM) {
                int tok = sorted_token_ids[gm];
                if (tok < num_valid_tokens)
                    val = A[(tok / top_k) * stride_am + (k_base + lk) * stride_ak];
            }
            smem_A[lm * LDS_A_STRIDE + lk] = val;
        }

        // --- Load B (packed uint8) to LDS ---
        {
            int gn = n_start + b_load_n;
            int sbase = b_load_n * LDS_B_STRIDE + b_load_k_off;
            if (gn < N) {
                const uint8_t* baddr = b_expert + gn * stride_bn + (k_base/2 + b_load_k_off) * stride_bk;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    smem_B[sbase + j] = baddr[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) smem_B[sbase + j] = 0;
            }
        }
        __syncthreads();

        // --- Compute using v_mmac ---
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // Read A fragment from LDS (same as v5, verified correct)
            int a_row = lane_id % 16;
            int a_k_base = kk + (lane_id / 16) * 4;
            half4_t a_frag;
            const __half* a_ptr = &smem_A[a_row * LDS_A_STRIDE + a_k_base];
            a_frag = *reinterpret_cast<const half4_t*>(a_ptr);

            // Read packed B from LDS
            int b_col = warp_n_offset + (lane_id % 16);
            int b_k = kk + (lane_id / 16) * 4;
            int b_byte = b_k / 2;

            uint8_t b0 = smem_B[b_col * LDS_B_STRIDE + b_byte];
            uint8_t b1 = smem_B[b_col * LDS_B_STRIDE + b_byte + 1];

            int q0 = b0 & 0xF;
            int q1 = (b0 >> 4) & 0xF;
            int q2 = b1 & 0xF;
            int q3 = (b1 >> 4) & 0xF;

            // Scale and zp
            int k_global = k_base + kk;
            int k_group = k_global / group_size;
            int gn = n_start + b_col;

            float scale_f = __half2float(s_expert[gn * stride_bsn + k_group * stride_bsk]);

            float zp_f;
            if (has_zp) {
                int zp_n = gn / 2;
                uint8_t zp_byte = zp_expert[zp_n * stride_bzn + k_group * stride_bzk];
                zp_f = (float)((zp_byte >> ((gn & 1) * 4)) & 0xF);
            } else {
                zp_f = 8.0f;
            }

            // Dequant to float, convert to __half
            float d0 = ((float)q0 - zp_f) * scale_f;
            float d1 = ((float)q1 - zp_f) * scale_f;
            float d2 = ((float)q2 - zp_f) * scale_f;
            float d3 = ((float)q3 - zp_f) * scale_f;

            // Write dequanted values to scratch LDS (each thread's own slot)
            __half* my_scratch = &smem_scratch[tid * 4];
            my_scratch[0] = __float2half(d0);
            my_scratch[1] = __float2half(d1);
            my_scratch[2] = __float2half(d2);
            my_scratch[3] = __float2half(d3);

            // Read back as half4_t — ds_read_b64 correctly formats for v_mmac
            half4_t b_frag = *reinterpret_cast<const half4_t*>(my_scratch);

            acc = mmac_f16(a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    // --- Write output (same as v5) ---
    int out_row_local = lane_id % 16;
    int out_col_group = lane_id / 16;
    int out_row_global = m_start + out_row_local;

    if (out_row_global < EM) {
        int tok = sorted_token_ids[out_row_global];
        if (tok < num_valid_tokens) {
            float moe_w = mul_routed_weight ? topk_weights[tok] : 1.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int nc = n_start + warp_n_offset + out_col_group + i * 4;
                if (nc < N) {
                    C[out_row_global * stride_cm + nc * stride_cn] =
                        __float2half(acc[i] * moe_w);
                }
            }
        }
    }
}

void moe_gemm_w4a16_dcu_forward(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    torch::Tensor B_scale, torch::Tensor B_zp,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t N, int64_t K, int64_t EM, int64_t num_valid_tokens,
    bool mul_routed_weight, int64_t top_k, int64_t group_size,
    bool has_zp, int64_t block_m
) {
    const int num_m_blocks = expert_ids.size(0);
    int grid = num_m_blocks * ((N + BN - 1) / BN);
    auto stream = get_stream();

    hipLaunchKernelGGL(moe_gemm_w4a16_dcu_kernel,
        dim3(grid), dim3(NUM_THREADS), 0, stream,
        (const __half*)A.data_ptr(), (const uint8_t*)B.data_ptr(),
        (__half*)C.data_ptr(), (const __half*)B_scale.data_ptr(),
        has_zp ? (const uint8_t*)B_zp.data_ptr() : nullptr,
        (const float*)topk_weights.data_ptr(),
        (const int*)sorted_token_ids.data_ptr(),
        (const int*)expert_ids.data_ptr(),
        (const int*)num_tokens_post_padded.data_ptr(),
        (int)N,(int)K,(int)EM,(int)num_valid_tokens,
        (int)A.stride(0),(int)A.stride(1),
        (int)B.stride(0),(int)B.stride(2),(int)B.stride(1),
        (int)C.stride(C.dim()-2),(int)C.stride(C.dim()-1),
        (int)B_scale.stride(0),(int)B_scale.stride(2),(int)B_scale.stride(1),
        has_zp?(int)B_zp.stride(0):0, has_zp?(int)B_zp.stride(2):0, has_zp?(int)B_zp.stride(1):0,
        (int)group_size, mul_routed_weight, (int)top_k, has_zp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_gemm_w4a16_dcu_forward", &moe_gemm_w4a16_dcu_forward,
          "MoE W4A16 GEMM for Hygon DCU (scratch LDS roundtrip for v_mmac)");
}
