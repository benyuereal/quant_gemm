# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors W8A8 (int8 weight + int8 dynamic activation) MoE scheme
for ROCm/HIP (Hygon DCU).

sglang 原生 fused_moe Triton kernel 已支持 use_int8_w8a8 + per_channel_quant
(见 moe_runner/triton_utils/fused_moe_triton_kernels.py), 但 compressed_tensors
的 get_moe_scheme 在 _is_dynamic_token_w8a8 分支只对 NPU 返回 scheme, 海光(非 NPU)
直接 raise NotImplementedError.

本 scheme 填补海光缺口: 复用 sglang 原生 Triton MoE runner, 只负责
create_weights (int8 权重 + per-channel scale) + get_triton_quant_info
(use_int8_w8a8=True, per_channel_quant=True). 激活 per-token int8 量化由
kernel 内部 per_token_quant_int8 完成 (dynamic), 故无需 a_scale.

权重布局 (与 sglang fused_moe kernel 期望一致, B.shape = [E, N, K]):
    w13_weight       [E, 2*intermediate, hidden] int8   (gate w1 + up w3 沿 N 拼接)
    w2_weight        [E, hidden, intermediate]  int8
    w13_weight_scale [E, 1, 2*intermediate]     f32     (per-channel, N 维)
    w2_weight_scale  [E, 1, hidden]             f32
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW8A8Int8TritonMoE"]


class CompressedTensorsW8A8Int8TritonMoE(CompressedTensorsMoEScheme):
    """ROCm/HIP W8A8 MoE scheme using sglang native Triton fused_moe kernel.

    int8 per-channel weight + dynamic per-token int8 activation.
    Reuses sglang MoeRunner(TRITON) + TritonMoeQuantInfo(use_int8_w8a8=True).
    """

    def __init__(self, quant_config: "CompressedTensorsConfig") -> None:
        self.quant_config = quant_config
        target = (
            "MoEGMM" if "MoEGMM" in quant_config.target_scheme_map else "Linear"
        )
        if target in quant_config.target_scheme_map:
            wq = quant_config.target_scheme_map[target]["weights"]
            self.strategy = (
                wq.strategy.value if hasattr(wq.strategy, "value") else str(wq.strategy)
            )
            self.symmetric = wq.symmetric
        else:
            self.strategy = "channel"
            self.symmetric = True
        assert self.symmetric, "Only symmetric quantization is supported for W8A8 MoE"
        assert self.strategy == "channel", (
            f"W8A8 MoE int8 kernel only supports channel strategy, got {self.strategy}"
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # 权重: [E, 2*intermediate, hidden] int8 (w13=gate+up), [E, hidden, intermediate] int8 (w2)
        # 与 NPUCompressedTensorsW8A8Int8DynamicMoE 布局一致, sglang MoE weight_loader 按此布局切.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # per-channel scale: [E, N, 1] f32 (与 NPU 版一致, N=output channel 维)
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # 关键: per-channel scale 须标 CHANNEL, weight_loader 才能正确加载
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # dynamic activation quant: 无静态 input scale
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 权重已在 weight_loader 中按 [E, N, K] int8 + [E, N, 1] f32 scale 加载.
        # 仅保证 contiguous + dtype.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.contiguous(), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.contiguous(), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.float().contiguous(), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.float().contiguous(), requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def get_triton_quant_info(self, layer):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            block_shape=None,  # per-channel (非 block-wise)
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported for W8A8 MoE."

        quant_info = self.get_triton_quant_info(layer)
        return self.runner.run(dispatch_output, quant_info)
