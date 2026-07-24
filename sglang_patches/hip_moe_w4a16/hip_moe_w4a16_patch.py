"""HIP MoE W4A16 kernel integration patch for sglang (Hygon DCU).

Monkey-patches sglang's invoke_fused_moe_kernel to use a hand-written HIP
kernel for W4A16 MoE in the small-M (decode) regime, where it is faster than
the Triton kernel (up to ~3x on GEMM2). For larger M it falls back to the
original Triton kernel.

What it is NOT: this is *not* a quantization step. It consumes an existing
W4A16 (int4-weight, bf16-activation) MoE quantization product and accelerates
its decode-time GEMM on Hygon DCU. Without it, W4A16 still runs correctly via
sglang's native Triton path — just slower in decode.

Target arch:
  - gfx928 (CDNA3/MI300): v_mmac fast path (the speedup).
  - other gfx9xx (e.g. gfx936/CDNA2): compiles & loads but runs the scalar
    fallback in the .hip source (no v_mmac) — functionally correct, no speedup.

Auto-loading: this module is imported by sitecustomize (see install script),
so `sglang serve` picks it up without any PYTHONPATH pointing at a working
directory. Toggle with SGLANG_USE_HIP_MOE_W4A16=0 to disable.

Path resolution for the kernel (.so / .hip):
  1. $HIP_MOE_KERNEL_DIR                (explicit override)
  2. this file's own directory          (repo / pip-installed location)
  3. <site-packages>/sglang_v6c/        (legacy install dir, kept for compat)
The precompiled .so is preferred; if absent, the .hip is JIT-compiled with
torch.utils.cpp_extension.load (offload arch from $HIP_MOE_OFFLOAD_ARCH or
auto-detected from rocminfo).
"""
import os
import logging

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_USE_HIP_MOE_W4A16", "1") == "1"
_THRESHOLD_EM = 512  # use the HIP kernel when sorted_token_ids.shape[0] <= this
_MODULE = None
_scale_cache = {}


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def _sitepkg_dir():
    try:
        import site
        for d in site.getsitepackages():
            if os.path.isdir(d):
                return d
    except Exception:
        pass
    return None


def _kernel_dir():
    """Resolve the directory holding the .so / .hip kernel."""
    candidates = []
    env = os.environ.get("HIP_MOE_KERNEL_DIR")
    if env:
        candidates.append(env)
    candidates.append(_here())
    sp = _sitepkg_dir()
    if sp:
        candidates.append(os.path.join(sp, "hip_moe_w4a16"))
    for c in candidates:
        if c and (os.path.exists(os.path.join(c, "hip_moe_w4a16_dcu.so"))
                  or os.path.exists(os.path.join(c, "moe_w4a16_dcu.hip"))):
            return c
    # fall back to first candidate even if empty (lets _load report a clear error)
    return candidates[0] if candidates else ""


def _detect_offload_arch():
    """Pick the first gfx9xx from rocminfo; default gfx928."""
    try:
        import subprocess
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Name:") and "gfx9" in line:
                return line.split()[-1]
    except Exception:
        pass
    return "gfx928"


def _load_kernel():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    kdir = _kernel_dir()
    so_path = os.path.join(kdir, "hip_moe_w4a16_dcu.so")
    hip_path = os.path.join(kdir, "moe_w4a16_dcu.hip")
    try:
        if os.path.exists(so_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("hip_moe_w4a16_dcu", so_path)
            _MODULE = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_MODULE)
            logger.info("hip_moe_w4a16 loaded precompiled .so from %s", so_path)
        elif os.path.exists(hip_path):
            from torch.utils.cpp_extension import load
            arch = os.environ.get("HIP_MOE_OFFLOAD_ARCH") or _detect_offload_arch()
            _MODULE = load(
                name="hip_moe_w4a16_dcu",
                sources=[hip_path],
                extra_cuda_cflags=["-O3", f"--offload-arch={arch}", "-std=c++17", "-ffast-math"],
                verbose=False, with_cuda=True,
            )
            logger.info("hip_moe_w4a16 JIT-compiled .hip for %s from %s", arch, hip_path)
        else:
            logger.warning("hip_moe_w4a16: no .so/.hip found under %s; patch disabled.", kdir)
            _MODULE = False
    except Exception as e:
        logger.warning("hip_moe_w4a16: failed to load kernel: %s; patch disabled.", e)
        _MODULE = False
    return _MODULE


def _get_fp16_scale(B_scale):
    import torch
    if B_scale.dtype == torch.float16:
        return B_scale
    ptr = B_scale.data_ptr()
    if ptr not in _scale_cache:
        _scale_cache[ptr] = B_scale.to(torch.float16)
    return _scale_cache[ptr]


def _apply_patch():
    if not _ENABLED:
        return
    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels as fmtk
    except ImportError:
        return

    kernel = _load_kernel()
    if not kernel:
        return

    original_invoke = fmtk.invoke_fused_moe_kernel

    def patched_invoke_fused_moe_kernel(
        A, B, bias, C, A_scale, B_scale, B_zp,
        topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight, top_k, config, compute_type,
        use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
        per_channel_quant, block_shape=None,
        no_combine=False, a_use_tma=False, b_use_tma=False,
        c_sorted=False, filter_expert=True,
        fuse_sum_all_reduce=False, router_topk=1,
        fuse_add_to_output=False, add_output_mask=None,
    ):
        use_hip = (
            use_int4_w4a16
            and not use_fp8_w8a8 and not use_int8_w8a8 and not use_int8_w8a16
            and bias is None
            and not a_use_tma and not b_use_tma
            and not fuse_sum_all_reduce and not fuse_add_to_output
            and B_scale is not None and B_scale.ndim == 3
            and (B_zp is None or B_zp.ndim == 3)
            and topk_ids.numel() <= _THRESHOLD_EM
            and block_shape is not None and block_shape[1] > 0
        )

        if use_hip:
            import torch
            EM = sorted_token_ids.shape[0]
            N = B.shape[1]
            K = B.shape[2]
            num_valid_tokens = topk_ids.numel()
            group_size = block_shape[1]
            has_zp = B_zp is not None
            BM = 16

            A_fp16 = A if A.dtype == torch.float16 else A.to(torch.float16)
            Bs_fp16 = _get_fp16_scale(B_scale)
            if C.dtype == torch.float16:
                C_fp16 = C
            else:
                C_fp16 = torch.empty_like(C, dtype=torch.float16)

            kernel.moe_gemm_w4a16_dcu_forward(
                A_fp16, B, C_fp16, Bs_fp16,
                B_zp if has_zp else Bs_fp16,
                topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                N, K, EM, num_valid_tokens,
                mul_routed_weight, top_k, group_size, has_zp, BM,
            )

            if C.dtype != torch.float16:
                C.copy_(C_fp16)
            return

        return original_invoke(
            A, B, bias, C, A_scale, B_scale, B_zp,
            topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight, top_k, config, compute_type,
            use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
            per_channel_quant, block_shape,
            no_combine, a_use_tma, b_use_tma,
            c_sorted, filter_expert,
            fuse_sum_all_reduce, router_topk,
            fuse_add_to_output, add_output_mask,
        )

    fmtk.invoke_fused_moe_kernel = patched_invoke_fused_moe_kernel
    # Also patch the reference imported at module level in fused_moe.py
    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
        fm.invoke_fused_moe_kernel = patched_invoke_fused_moe_kernel
    except Exception:
        pass
    logger.info("hip_moe_w4a16 patch applied (threshold EM<=%d).", _THRESHOLD_EM)


_apply_patch()
