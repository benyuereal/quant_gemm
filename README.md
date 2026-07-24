# MiniMax-M3 量化推理 (海光 DCU)

MiniMax-M3 在海光 DCU (gfx936 / gfx928) 上的 **W8A8 / W4A16 moe-only 量化** + **sglang 适配** + **EAGLE3 投机解码** 的完整链路.

实际推理走 sglang 原生 Triton kernel (W8A8 复用 `fused_moe_kernel` 的 `use_int8_w8a8` + `per_channel_quant`, W4A16 走 `CompressedTensorsWNA16TritonMoE` ROCm 路径), 本仓库只提供**量化产物生成脚本**和**让 sglang 能在海光 DCU 上加载这些产物所需的补丁**.

## 项目结构

```
quant-eagle3-hygon/
├── sglang_patches/      # sglang 补丁 (海光 W8A8/W4A16 MoE 适配 + EAGLE3)
│   ├── added/           #   新增 W8A8 MoE scheme + sitecustomize (注册 minimax_m3_sparse)
│   ├── modified/        #   改动文件 + .patch (compressed_tensors/int8_kernel/sparse attn/wNa16_moe/eagle3)
│   ├── hip_moe_w4a16/   #   W4A16 MoE decode 加速 kernel (源码 .hip + 预编译 .so + patch + install) — 非量化, 可选性能优化
│   ├── tests/           #   EAGLE3 verify 路径字段补全单测
│   └── README.md        #   补丁总览 + 应用方法
├── quantization/        # 量化脚本
│   ├── minimax_m3_w4a8.py        # W8A8 量化主脚本 (--quant-type int8 --moe-only)
│   ├── minimax_m3_w4a16.py       # W4A16 量化脚本 (model_free_ptq, 只量 MoE expert)
│   ├── quantize_minimax_m3_w4a8.sh  # 8 卡并行封装
│   ├── minimax.sh                # W8A8 sglang 启动脚本
│   ├── minimax_w4a16.sh          # W4A16 sglang 启动脚本
│   └── glm5.1-...config.json     # 参考 config
└── docs/
    ├── MiniMax-M3-量化工作记录.md   # 完整工作记录 (W4A8 探索 → W8A8 → W4A16, 26 章)
    └── MiniMax-M3-EAGLE3-工作记录.md  # EAGLE3 投机解码工作记录
```

## 完整流程

**W8A8:**
1. **量化**: `bash quantization/quantize_minimax_m3_w4a8.sh` → 412GB W8A8 moe-only 产物
2. **打 sglang 补丁**: 按 `sglang_patches/README.md` 应用补丁 1-6
3. **启动**: `bash quantization/minimax.sh` → sglang serve on :8080
4. **测试**: `curl http://127.0.0.1:8080/v1/chat/completions ...`

**W4A16:**
1. **量化**: `python3 quantization/minimax_m3_w4a16.py --input-path ... --output-path ... --max-workers 4` → 225GB W4A16 moe-only 产物
2. **打 sglang 补丁**: 补丁 1-6 (海光兼容) + 补丁 7 (W4A16 KeyError Linear)
3. **启动**: `bash quantization/minimax_w4a16.sh` → sglang serve on :8081

**EAGLE3 投机解码** (搭配 `Inferact/MiniMax-M3-EAGLE3` draft head, MiniMax-M3 无 MTP 权重故用 EAGLE3): 见 `sglang_patches/README.md` 补丁 8 + `docs/MiniMax-M3-EAGLE3-工作记录.md`.

详见 `docs/MiniMax-M3-量化工作记录.md`.

## 状态

- ✅ W8A8 moe-only: BW100 (gfx936) 量化 + sglang 适配 + forward 跑通, chat/completions 返回连贯中文
- ✅ W4A16 moe-only: BW100 (gfx936) 量化 (225G) + sglang 加载成功 (`CompressedTensorsWNA16TritonMoE` ROCm 路径), forward/精度待测
- ✅ EAGLE3: M3 VL 类补 target 侧接口 + sparse backend verify 字段兜底, cuda graph 下 4x 加速
- ⚙️ hip_moe_w4a16: W4A16 MoE **decode 加速** kernel (可选, 非量化步骤). gfx928 走 `v_mmac` 快路径 (小 batch 最多 ~3x), gfx936 走标量 fallback (无加速). 关闭 (`SGLANG_USE_HIP_MOE_W4A16=0`) 不影响 W4A16 正常推理.
- ⏳ gfx928 (K100): 待实测
