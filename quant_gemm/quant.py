"""int8 量化工具函数 (per-token / per-channel 对称量化).

与 quant_gemm.kernels 的 W8A8 GEMM 配套使用:
    q_x, x_scale = per_token_sym_int8(x)       # 激活 per-token
    q_w, w_scale = per_channel_sym_int8(w)     # 权重 per-channel
    out = kernel(q_x, q_w, x_scale, w_scale)   # C = (A@B.T)*x_scale*w_scale.T
"""

import torch

__all__ = ["per_token_sym_int8", "per_channel_sym_int8"]


def per_token_sym_int8(x: torch.Tensor):
    """per-token 对称 int8 量化. x: [M,K] -> (q [M,K] int8, scale [M,1] f32).

    每个 token (行) 一个 scale, 对称量化到 [-127,127].
    """
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = (absmax / 127.0).to(torch.float32)
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale


def per_channel_sym_int8(w: torch.Tensor):
    """per-output-channel 对称 int8 量化. w: [N,K] -> (q [N,K] int8, scale [N,1] f32).

    每个输出通道 (行) 一个 scale, 对称量化到 [-127,127].
    与 compressed-tensors W8A8 channel strategy / GLM-5.1 量化产物布局一致.
    """
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = (absmax / 127.0).to(torch.float32)
    q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale
