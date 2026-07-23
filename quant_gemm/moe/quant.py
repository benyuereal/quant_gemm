"""per-group int4 量化工具 (sglang W4A16 兼容, zp=8 对称).

与 sglang int4_w4a16 语义一致: deq = (q - 8) * scale, q ∈ [0,15].
    scale = absmax / 7,  q = round(w / scale) + 8,  clamp [0,15].
nibble 顺序: k 偶数放低 4 位, k 奇数放高 4 位 (与 sglang b_shifter = (k%2)*4 一致).
"""

import torch

from .kernels import GROUP

__all__ = ["quantize_int4_per_group", "dequant_int4_per_group", "GROUP"]

DTYPE = torch.bfloat16


def quantize_int4_per_group(w, group_size=GROUP):
    """w: [..., K] bf16 -> (q_byte [..., K//2] uint8, scale [..., K//group] bf16).

    对称量化 zp=8 (与 sglang int4_w4a16 一致): deq = (q - 8) * scale, q∈[0,15].
    scale = absmax/7, q = round(w/scale) + 8, clamp [0,15].
    nibble 顺序: k 偶数低 4 位, k 奇数高 4 位 (与 sglang b_shifter=(k%2)*4 一致).
    """
    orig_shape = w.shape
    K = orig_shape[-1]
    assert K % group_size == 0
    w_flat = w.float().reshape(-1, K)
    M = w_flat.shape[0]
    w_g = w_flat.reshape(M, K // group_size, group_size)
    amax = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax / 7.0).to(DTYPE)                         # [M, K//group, 1]
    q = torch.round(w_g / scale.float().clamp(min=1e-8)) + 8.0   # 对称, 中心 8
    q = q.clamp(0, 15).to(torch.uint8).reshape(M, K)
    q_byte = (q[:, 1::2] << 4) | q[:, 0::2]                # [M, K//2]  k 偶数低 4 位
    scale_out = scale.squeeze(-1).contiguous()             # [M, K//group]
    q_byte = q_byte.reshape(*orig_shape[:-1], K // 2).contiguous()
    scale_out = scale_out.reshape(*orig_shape[:-1], K // group_size).contiguous()
    return q_byte, scale_out


def dequant_int4_per_group(q_byte, scale, group_size=GROUP):
    """反量化参考 (zp=8 对称). q_byte [..., K//2] uint8, scale [..., K//group] -> [..., K] bf16.

    deq = (q - 8) * scale.
    """
    orig_shape = q_byte.shape
    K2 = orig_shape[-1]
    K = K2 * 2
    qb = q_byte.reshape(-1, K2)
    sb = scale.reshape(-1, K // group_size)
    M = qb.shape[0]
    q_low = (qb & 0x0F).to(torch.float32)
    q_high = (qb >> 4).to(torch.float32)
    q = torch.stack([q_low, q_high], dim=-1).reshape(M, K)
    s = sb.unsqueeze(-1).expand(M, K // group_size, group_size).reshape(M, K)
    deq = ((q - 8.0) * s).to(DTYPE)
    return deq.reshape(*orig_shape[:-1], K)
