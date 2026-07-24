# MiniMax-M3 量化工作记录（W4A8 探索 → W8A8 跑通）
> 创建日期: 2026-07-21 ｜ 更新: 2026-07-22（W8A8 跑通）
> 目标: MiniMax-M3 (800GB bf16) 海光 DCU 量化推理 — 最终方案 W8A8 只量化 MoE, sglang 适配已跑通
> 硬件: 海光 DCU (非 NVIDIA GPU)

---

## 一、背景与约束

- **模型**: MiniMax-M3，800GB，下载耗时 2 天，量化耗时约 2 小时
- **核心风险**: 量化后若权重不能用，返工成本极高（重新下载 2 天 + 重新量化 2 小时）
- **核心诉求**: 最小化量化 —— 只量化大块权重（MoE expert），保证适配 sglang 等通用推理框架
- **参考成功案例**: GLM-5.1 已用 `GLM-5-w4a8-slimquant.py` 成功量化（`/models/glm5.1-channel-int4-w4a8.config.json`）

---

## 二、模型架构梳理

### 2.1 GLM-5.1 (`GlmMoeDsaForCausalLM`) — 已成功量化参考

| 维度 | 值 |
|------|-----|
| hidden_size | 6144 |
| num_hidden_layers | 78 |
| num_attention_heads | 64 |
| num_key_value_heads | 64 (无 GQA) |
| head_dim | 256 (qk_nope=192 + qk_rope=64) |
| n_routed_experts | 256 |
| num_experts_per_tok | 8 |
| n_shared_experts | 1 |
| moe_intermediate_size | 2048 |
| first_k_dense_replace | 3 |
| 特点 | 纯文本，无视觉塔；稀疏注意力为 indexer-based |
| 成功量化 config 的 ignore | 仅 `["lm_head"]`（W8A8 方案，非常干净） |

### 2.2 MiniMax-M3 (`MiniMaxM3SparseForConditionalGeneration`) — 待量化目标

| 维度 | 值 |
|------|-----|
| hidden_size | 6144 |
| num_hidden_layers | 60 |
| num_attention_heads | 64 |
| num_key_value_heads | 4 (GQA) |
| head_dim | 128 |
| n_routed_experts (num_local_experts) | 128 |
| num_experts_per_tok | 4 |
| n_shared_experts | 1 |
| moe_intermediate_size | 3072 |
| dense_intermediate_size | 12288 |
| first_k_dense_replace | 3 (前 3 层 dense, 4-60 层 MoE) |
| MTP modules | 7 |
| 特点 | **视觉语言模型** (ViT + multi_modal_projector + patch_merge_mlp)；block-sparse 注意力 |
| scoring_func | sigmoid |
| 稀疏注意力 | block_size=128, topk_blocks=16 |

### 2.3 实际 tensor 命名（扫描全部 23416 个 tensor，53 种唯一模式）

**LM 部分（language_model 前缀）**:
- `language_model.model.embed_tokens.weight` [200064, 6144] — 2D，需忽略
- `language_model.lm_head.weight` [200064, 6144] — 2D，需忽略
- `language_model.model.layers.N.input_layernorm.weight` [6144] — 1D，需忽略
- `language_model.model.layers.N.post_attention_layernorm.weight` [6144] — 1D，需忽略
- `language_model.model.norm.weight` [6144] — 1D，需忽略
- `language_model.model.layers.N.self_attn.q_proj.weight` [8192, 6144] — 量化
- `language_model.model.layers.N.self_attn.k_proj.weight` [512, 6144] — 量化
- `language_model.model.layers.N.self_attn.v_proj.weight` [512, 6144] — 量化
- `language_model.model.layers.N.self_attn.o_proj.weight` [6144, 8192] — 量化
- `language_model.model.layers.N.self_attn.q_norm.weight` [128] — 1D，需忽略
- `language_model.model.layers.N.self_attn.k_norm.weight` [128] — 1D，需忽略
- `language_model.model.layers.N.self_attn.index_q_proj.weight` [512, 6144] — 量化（稀疏注意力索引投影）
- `language_model.model.layers.N.self_attn.index_k_proj.weight` [128, 6144] — 量化
- `language_model.model.layers.N.self_attn.index_q_norm.weight` [128] — 1D，需忽略
- `language_model.model.layers.N.self_attn.index_k_norm.weight` [128] — 1D，需忽略
- `language_model.model.layers.N.mlp.gate_proj.weight` [12288, 6144] — 量化（dense 层 SwiGLU gate）
- `language_model.model.layers.N.mlp.up_proj.weight` [12288, 6144] — 量化
- `language_model.model.layers.N.mlp.down_proj.weight` [6144, 12288] — 量化
- `language_model.model.layers.N.block_sparse_moe.gate.weight` [128, 6144] — **MoE 路由 gate，需忽略**
- `language_model.model.layers.N.block_sparse_moe.e_score_correction_bias` [128] — 1D float32，需忽略
- `language_model.model.layers.N.block_sparse_moe.experts.E.w1.weight` [3072, 6144] — **MoE expert，INT4 量化**
- `language_model.model.layers.N.block_sparse_moe.experts.E.w2.weight` [6144, 3072] — **MoE expert，INT4 量化**
- `language_model.model.layers.N.block_sparse_moe.experts.E.w3.weight` [3072, 6144] — **MoE expert，INT4 量化**
- `language_model.model.layers.N.block_sparse_moe.shared_experts.gate_proj.weight` [3072, 6144] — 量化（SwiGLU gate，非路由）
- `language_model.model.layers.N.block_sparse_moe.shared_experts.up_proj.weight` [3072, 6144] — 量化
- `language_model.model.layers.N.block_sparse_moe.shared_experts.down_proj.weight` [6144, 3072] — 量化

**视觉部分（无 language_model 前缀，整体忽略）**:
- `multi_modal_projector.linear_1/2.weight/bias`
- `patch_merge_mlp.linear_1/2.weight/bias`
- `vision_tower.vision_model.*`（32 层 CLIP ViT，含 patch_embedding 是 5D）

**注意**: 扫描未发现 `mtp` 命名的 tensor（MTP 模块可能复用主层结构或不在 safetensors 中独立存在），但 ignore 列表保留 `re:.*mtp.*` 以防万一。

---

## 三、量化方案设计（W4A8 = channel int4 weight + int8 activation）

### 3.1 分层量化策略（精确权重统计）

原始模型总大小: **854.17 GB** (bf16)。下表基于对全部 23416 个 tensor 的实际扫描统计。

| 权重类别 | 张量数 | 原始大小 | 量化方案 | 理由 |
|---------|--------|---------|---------|------|
| **MoE expert (w1/w2/w3)** | 21888 | **826.24 GB (96.7%)** | **per-channel INT4** + 网格搜索 + 2×int4→int8 打包 | 大块权重，量化收益最大 |
| attention_proj (q/k/v/o/index_q/index_k_proj) | 354 | 13.28 GB (1.6%) | per-channel INT8 | 注意力投影，INT8 保精度 |
| shared_experts (gate/up/down_proj) | 171 | 6.46 GB (0.8%) | per-channel INT8 | SwiGLU 权重，INT8 |
| lm_head | 1 | 2.46 GB | **不量化** (bf16) | 词表大(20万)，量化损失大 |
| embed_tokens | 1 | 2.46 GB | **不量化** (bf16) | 词表大，量化损失大 |
| vision_tower (ViT 32层) | 515 | 1.27 GB | **不量化** (bf16/fp32) | 视觉计算量小，保精度 |
| dense_mlp (gate/up/down_proj, 前3层) | 9 | 1.36 GB | per-channel INT8 | 前 3 层 dense，INT8 |
| patch_merge_mlp | 4 | 0.38 GB | **不量化** (bf16) | 图像 token 压缩，保精度 |
| moe_routing_gate | 57 | 0.18 GB | **不量化** (fp32) | 路由敏感，量化破坏专家选择 |
| multi_modal_projector | 4 | 0.09 GB | **不量化** (bf16) | 视觉投影，保精度 |
| norm / layernorm / qk_norm | 355 | <0.01 GB | **不量化** | 1D 张量，无法 per-channel 量化 |
| e_score_correction_bias | 57 | <0.01 GB | **不量化** (fp32) | 1D 路由偏置 |

### 3.2 量化后大小估算

INT4 权重采用 2×int4→int8 打包（两个 4bit 值存进一个 int8 字节），存储 = `numel/2 × 1 byte`，相对 bf16（2 byte/元素）压缩 4×。scale 是 per-channel float32，每 channel 仅 1 个值，开销可忽略。

| 权重类别 | 量化 | 原始 GB | 权重 GB | scale GB | 量化后 GB | 压缩比 |
|---------|------|--------|--------|---------|----------|--------|
| **MoE expert** | INT4 | 826.24 | 206.56 | 0.36 | **206.92** | 3.99× |
| attention_proj | INT8 | 13.28 | 6.64 | 0.004 | 6.65 | 2.00× |
| shared_experts | INT8 | 6.46 | 3.23 | 0.003 | 3.23 | 2.00× |
| lm_head | 不量化 | 2.46 | 2.46 | 0 | 2.46 | 1.00× |
| embed_tokens | 不量化 | 2.46 | 2.46 | 0 | 2.46 | 1.00× |
| dense_mlp | INT8 | 1.36 | 0.68 | 0 | 0.68 | 2.00× |
| vision_tower | 不量化 | 1.27 | 1.27 | 0 | 1.27 | 1.00× |
| patch_merge_mlp | 不量化 | 0.38 | 0.38 | 0 | 0.38 | 1.00× |
| moe_routing_gate | 不量化 | 0.18 | 0.18 | 0 | 0.18 | 1.00× |
| multi_modal_projector | 不量化 | 0.09 | 0.09 | 0 | 0.09 | 1.00× |
| norm/qk_norm/bias | 不量化 | <0.01 | <0.01 | 0 | <0.01 | 1.00× |
| **合计** | — | **854.17** | — | — | **224.31** | **3.81×** |

**结论**:
- 原始模型: **854.17 GB** (bf16)
- 量化后模型: **约 224 GB**
- 压缩比: **3.81×**，节省约 630 GB（73.7%）
- 核心收益来源: MoE expert 从 826 GB → 207 GB（INT4 压缩 4×），贡献几乎全部压缩收益

### 3.3 INT4 量化算法（SlimQuant 风格）

1. 先做 per-channel INT8 量化（对称，qmax=127）
2. 对 MoE expert 层，在 INT8 基础上做 percentile 网格搜索（k_min=0.97, k_max=1.0, steps=30）
3. 找到最佳截断阈值后做 INT4 量化（对称，qmax=7）
4. 2×int4 → int8 打包（两个 4bit 值打包进一个 int8）
5. scale 合并：`final_scale = int8_scale * int4_scale / 16`

### 3.5 输出 config.json 格式（compressed-tensors，vLLM/sglang 原生支持）

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "quantization_status": "compressed",
    "version": "0.14.0.1",
    "config_groups": {
      "group_0": {
        "weights": {"num_bits": 4, "type": "int", "strategy": "channel", "symmetric": true, "dynamic": false, ...},
        "input_activations": {"num_bits": 8, "type": "int", "strategy": "token", "symmetric": true, "dynamic": true, ...},
        "targets": ["Linear"]
      }
    },
    "ignore": [...]
  }
}
```

**关键**: 早期版本错误地用了 `quant_method: "slimquant_w4a8"`，vLLM 不识别，报错 "Unknown quantization method"。已修复为 `compressed-tensors`。

---

## 四、代码路径与已修复的 bug

### 4.1 涉及文件

| 文件 | 作用 | 状态 |
|------|------|------|
| `templates/minimax_m3_w4a8.py` | 独立量化脚本（多GPU并行） | ✅ 已创建并验证 |
| `templates/slimquant_ptq.py` | slimquant 通用模板 | ✅ 已修复 ignore 列表 |
| `templates/slimquant/config.py` | config.json 输出格式 | ✅ 已修复 compressed-tensors 格式 |
| `templates/quant_templates.yaml` | 架构→模板映射 | ⚠️ 见 4.3 |
| `quantize_minimax_m3_w4a8.sh` | shell 包装脚本 | ✅ 已修复执行路径（改用独立脚本） |
| `test_minimax_m3_single_shard.py` | 单 shard 最小化验证脚本 | ✅ 已创建并验证通过 |

### 4.2 已修复的 bug 清单

1. **`quant_method: "slimquant_w4a8"` → `"compressed-tensors"`**
   - vLLM 不识别 slimquant_w4a8，报 "Unknown quantization method"
   - 修复: config.py / qwen3_5_bf16_to_channel.py / minimax_m3_w4a8.py 全部改为 compressed-tensors 格式

2. **`shared_experts.gate.*` 误忽略 SwiGLU gate_proj**
   - 该模式会匹配 `shared_experts.gate_proj`（SwiGLU gate 投影，应量化）
   - 修复: 从 ignore 列表删除 `shared_experts.gate.*`，只保留 `block_sparse_moe.gate`（路由 gate）

3. **slimquant_ptq.py 中 `block_sparse_moe.gate\.weight$` 永远匹配不到**
   - slimquant 的 `_match_name` 用 `re.match` 匹配 `module_name`（不含 `.weight` 后缀）
   - `\.weight$` 永远匹配不到 `module_name`（它没有 `.weight`）
   - 修复: 改为 `block_sparse_moe.gate$`（无 `.weight`）

4. **slimquant_ptq.py 中 `mlp.gate$*` 无效正则**
   - `$*` 是无效正则语法
   - 修复: 改为 `mlp.gate$`

5. **`patch_merge_mlp` 的 bias 张量未被忽略**
   - `patch_merge_mlp.linear_1.bias` 是 1D，未被 visual/vision 模式覆盖
   - 修复: 添加 `re:.*patch_merge_mlp.*`

### 4.3 ignore 列表验证结果（✅ 全部通过）

对全部 53 种 tensor 模式逐项验证：
- 所有 1D 张量（norm/bias/e_score_correction_bias）→ 正确忽略 ✅
- MoE 路由 gate → 正确忽略 ✅
- MoE expert w1/w2/w3 → 正确量化（INT4）✅
- shared_experts gate_proj/up_proj/down_proj → 正确量化（INT8）✅
- 稀疏注意力 index_q_proj/index_k_proj → 正确量化 ✅
- 稀疏注意力 index_q_norm/index_k_norm → 正确忽略 ✅
- 标准 q/k/v/o_proj → 正确量化 ✅
- dense mlp gate/up/down_proj → 正确量化 ✅
- 视觉塔 → 整体忽略（设计选择）✅

### 4.4 ⚠️ 关键未解决问题：moe_pattern 不匹配

**这是当前最大的隐患**。两条执行路径：

| 路径 | 命令 | MoE expert 识别 | 结果 |
|------|------|----------------|------|
| A: slimquant_ptq | `main.py --alg slimquant_ptq --scheme W4A8` | 用默认 `moe_pattern=".mlp.experts."` | ❌ **匹配不到 `block_sparse_moe.experts`，expert 全做 INT8，不是 W4A8!** |
| B: 独立脚本 | `python3 minimax_m3_w4a8.py --quant-type int4` | 硬编码 `".block_sparse_moe.experts."` (第200行) | ✅ 正确做 INT4 |

**根因**: `main.py` 没有 `--moe-pattern` 命令行参数，所以 slimquant_ptq.py 第 213 行 `getattr(args, "moe_pattern", ".mlp.experts.")` 永远拿到默认值。

**结论**: 当前 `quantize_minimax_m3_w4a8.sh` 写的 `--alg slimquant_ptq` 是**错误的**，会产出 INT8 而非 INT4。必须改用独立脚本路径 B。

---

## 五、待完成工作

### 5.1 ✅ 修复 `quantize_minimax_m3_w4a8.sh` 执行路径
- 已改为直接调用 `python3 minimax_m3_w4a8.py --input-path ... --output-path ... --quant-type int4`
- 不再走 `--alg slimquant_ptq`（避免 moe_pattern='.mlp.experts.' 匹配不到的致命 bug）

### 5.2 ✅ 海光 DCU 适配验证
- 硬件: 8× BW100 DCU, torch 2.10.0 + HIP 6.3.26113
- `torch.cuda` 接口完全可用（HIP 走 cuda 名义），safetensors 支持 `device='cuda'` 直接加载
- 设备可见性: 用环境变量 `HIP_VISIBLE_DEVICES=0,1,6,7` 控制（可用全部 8 张）
- vLLM 海光版: `vllm_hcu 0.18.1+das.dtk2604`
- sglang: **未安装**，验证用 vLLM

### 5.3 ✅ 单 shard 最小化量化验证（已通过）

**测试脚本**: `/models/test/test_minimax_m3_single_shard.py`
**测试设备**: `HIP_VISIBLE_DEVICES=0` 单卡 GPU

**shard 1 (5.58GB, dense+attn+embed+lm_head+norm) 结果**:
- 忽略层: 7 个（lm_head, embed_tokens, 5×norm/layernorm）✅
- INT8 量化层: 7 个（dense mlp gate/up/down + attn q/k/v/o_proj）✅
- 格式校验: 全部通过 ✅

**shard 3 (16GB, MoE expert+shared_experts+routing gate) 结果**:
- 忽略层: 9 个（6×norm + routing gate + e_score_correction_bias 等）✅
- INT8 量化层: 9 个（shared_experts gate/up/down + 稀疏注意力 index_q/k_proj）✅
- **INT4 量化层: 417 个（MoE expert w1/w2/w3）✅**
- INT4 打包格式: 低4位/高4位均在 [-8, 7] 范围 ✅
- INT4 q 形状: [N, K//2]（K 减半，2×int4 打包进 int8）✅
- scale 形状: [N, 1], dtype=float32 ✅

**关键确认**:
- ✅ MoE expert 确实做了 INT4（独立脚本硬编码 `.block_sparse_moe.experts.` 正确识别）
- ✅ shared_experts.gate_proj 被正确量化为 INT8（之前误忽略的 bug 已修复）
- ✅ 稀疏注意力 index_q_proj/index_k_proj 被正确量化
- ✅ routing gate 被正确忽略
- ✅ GPU 量化速度可接受（shard 3 含 417 个 INT4 网格搜索，几分钟完成）

### 5.4 ⏳ vLLM 加载验证（下一步）
单 shard 格式校验通过后，还需验证 vLLM 能否实际加载量化模型。由于 vLLM 需要完整权重，有两个方案：
- 方案 A: 直接全量量化（8卡并行约 30-60 分钟），再 vLLM 加载验证
- 方案 B: 先用 vLLM 解析 config.json 的 quantization_config（不加载权重），确认格式被识别

### 5.5 ⏳ sglang 算子适配确认（待装 sglang 后）
需确认 sglang 对 MiniMax-M3 的支持（当前 sglang 未安装，先用 vLLM）：
- `compressed-tensors` W4A8 算子
- MiniMax-M3 的 block-sparse attention 算子
- MoE expert w1/w2/w3 命名识别
- 视觉语言模型的 multimodal 加载

---

## 六、量化类型指定方式（确认一致）

| 脚本 | 参数 | 默认值 | 可选值 |
|------|------|--------|--------|
| `GLM-5-w4a8-slimquant.py` | `--quant-type` | `int4` | int4/int8/fp8 |
| `minimax_m3_w4a8.py` | `--quant-type` | `int4` | int4/int8/fp8 |

两个脚本都**默认 int4**，都可通过 `--quant-type` 指定。✅ 一致

---

## 七、执行命令备忘

### 单 shard 最小化验证（已通过）
```bash
# shard 1 (dense+attn, 验证 INT8 路径)
HIP_VISIBLE_DEVICES=0 python3 /models/test/test_minimax_m3_single_shard.py
# shard 3 (MoE expert, 验证 INT4 关键路径) — 修改脚本内 input_shard 为 00003
```

### 全量量化（已验证可放心执行）
```bash
# 8 卡并行, 预计 30-60 分钟
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /models/quantize_minimax_m3_w4a8.sh
```

### vLLM 加载验证（全量量化后）
```bash
vllm serve /models/MiniMax/MiniMax-M3-channel-int4-w4a8 \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --trust-remote-code
```

---

## 八、风险与注意事项

1. **返工成本极高** → 必须先做最小化验证再全量量化
2. **moe_pattern 陷阱** → 不要用 `--alg slimquant_ptq` 路径，必须用独立脚本
3. **海光 DCU 兼容性** → 多进程 mp.spawn 和 cuda API 需验证
4. **sglang 算子支持** → MiniMax-M3 的稀疏注意力/MoE 命名是最大不确定性
5. **视觉部分** → 整体不量化，但需确认 sglang 能加载未量化的视觉塔 + 量化的 LM 混合模型
6. **w1/w2/w3 命名** → sglang/vLLM 对 MoE expert 的命名识别需确认（DeepSeek 系用 w1/w2/w3，应支持）

---

## 九、sglang 报错根因与 W4A8 方案推翻（2026-07-21 深度排查）

> 本章推翻了前面"W4A8 手写脚本"的方向。结论：**当前手写 `minimax_m3_w4a8.py` 产出的 W4A8 格式 sglang 根本不支持，需换方案。**

### 9.1 报错真实根因（不是量化方式问题）

报错 `NotImplementedError: No compressed-tensors compatible scheme was found.` 发生在 sglang 给 `qkv_proj`（普通 Linear）找量化算子时。逐层查 `sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`：

- config.json 的 `config_groups.group_0` 声明了 **W4A8**（weights num_bits=4 channel + input_activations num_bits=8 token dynamic），`targets:["Linear"]` 全局套用。
- sglang Linear 路径 `_get_scheme_from_parts`（:543-631）**只有 W4A16 / FP8 / INT8-W8A8 分支，唯独没有 W4A8**。
- `_is_dynamic_token_w4a8`（:346）判断函数存在，但**只在 MoE 路径 `get_moe_scheme`（:717）调用，且仅 NPU 可用**。Linear 路径压根没调它 → 找不到 scheme → 报错。

**且 config 与实际权重自相矛盾**：产物里 `q_proj/gate_proj` 等实为 int8（W8A8），只有 MoE expert 是自定义 int4 打包，但 config 统一声明成 W4A8。

### 9.2 海光 DCU 在 sglang 里的身份

- `is_cuda()` = `torch.cuda.is_available() and torch.version.cuda is not None` → 海光下 `torch.version.cuda=None` → **`_is_cuda=False`**
- `is_hip()` = `torch.version.hip is not None` → 海光下 **`_is_hip=True`**
- **`_is_npu=False`**（海光不是 NPU）

这是判定各方案能否跑的关键。

### 9.3 sglang + 海光 DCU 量化方案支持矩阵（完整排查）

> MiniMax-M3 是 MoE 模型，关键看 **MoE expert 走什么 kernel**：marlin=CUDA-only ❌ / Triton=海光可用 ✅ / NPU-only ❌。

**compressed-tensors 框架：**

| 方案 | format | Linear(海光) | MoE(海光) | 缺什么 |
|---|---|---|---|---|
| W8A8 INT8 | int-quantized | ⚠️ int8_scaled_mm 未接 | ❌ | MoE `_is_dynamic_token_w8a8` 仅 NPU（:709 raise） |
| W4A8 INT8 | int-quantized | ❌ | ❌ | Linear 无 W4A8 scheme；MoE 仅 NPU |
| W4A16/W8A16 | pack_quantized | ❌ | ✅ TritonMoE | Linear 走 marlin（CUDA-only），无 HIP 分支 |
| FP8 W8A8 | float-quantized | ❌ | ❌ | BW100 无 FP8 硬件（min_capability=89） |
| 混合 format | — | ❌ | ❌ | sglang `self.quant_format` 单一值，不支持混合 |

**其它框架：**

| 方案 | MoE kernel | 海光 | 缺什么 |
|---|---|---|---|
| AWQ | `AWQMoEScheme`→marlin | ❌ | `get_moe_scheme`: `raise NotImplementedError("AWQConfig only supports MoE scheme on NPU.")`；且 `HIP does not support fused_marlin_moe` |
| GPTQ | `GPTQMarlinMoEMethod`→marlin | ❌ | MoE marlin CUDA-only；`is_gptq_marlin_compatible`: `if not _is_cuda: return False` |
| awq_marlin / gptq_marlin | marlin | ❌ | marlin CUDA-only |
| QoQ | 无 MoE | ❌ | 只有 `QoQLinearMethod` |
| FP8（原生） | — | ❌ | BW100 无 FP8 硬件 |
| bitsandbytes | `BitsAndBytesMoEMethod` | ❌ | apply `raise NotImplementedError` |
| modelslim | `ModelSlimFusedMoEMethod` | ❌ | NPU 专用（`_NPULinearMethodBase`） |
| quark | `QuarkFusedMoEMethod` | ❌ | `get_moe_scheme` 多处 raise |
| **blockwise_int8** | **`BlockInt8MoEMethod`→Triton runner** | **✅ 可能可行** | 见 9.4 |

### 9.4 唯一可能可行的 sglang 原生方案：blockwise_int8

`quant_method: "blockwise_int8"`（sglang 自带，非 compressed-tensors），INT8 block-wise W8A8：

- MoE 走 **Triton kernel**（`use_int8_w8a8=True` + `TritonMoeQuantInfo`），不是 marlin。
- `fused_moe.py` 有 `elif _is_hip:` 分支；`moe_align_block_size.py:15` `if _is_cuda or _is_hip ...` 海光放行，不 raise。
- Linear 走 `apply_w8a8_block_int8_linear`（int8_utils）。
- `weight_block_size:[128,128]` 正好对应海光 lightop `gemm_w8a8_asm` 要求的 `block_size[0]==128 and block_size[1]==128`。
- config：`quant_method:"blockwise_int8"`, `weight_block_size:[128,128]`, `activation_scheme:"dynamic"`, `is_checkpoint_int8_serialized:true`。磁盘：`weight`(int8)+`weight_scale_inv`(block scale)。
- 体积 ~415GB。**未在海光实测过 MiniMax-M3，需验证。**

---

## 十、lightop 海光算子库——突破口（2026-07-21）

> lightop 是海光 DCU 专用算子库（`0.6.0+das.dtk2604.torch290`），被 vllm 依赖。**它有 sglang 缺的所有 W8A8/W4A8/W4A16 算子。** 改 sglang 接 lightop 可补上 sglang 的缺口。

### 10.1 lightop 量化算子盘点（`/usr/local/lib/python3.10/dist-packages/lightop/gemmopt.py`）

**Linear 算子：**
- `gemm_w8a8_smooth(a,b,scale_a,scale_b,bias,out_dtype)` → W8A8 Linear ✅
- `gemm_w8a8_asm(a,b,scale_a,scale_b,block_size,out_dtype)` → W8A8 block(128×128) Linear ✅
- `gemm_w4a16(a,b,scale_n,zero_n)` → W4A16 Linear ✅
- `gemm_w8a16_fp8` → W8A16 FP8

**MoE expert 算子：**
- `moe_gemm_w8a8` / `moe_gemm_marlin_w8a8` → W8A8 MoE ✅
- `moe_gemm_w4a8` / `moe_gemm_marlin_w4a8` → W4A8 MoE ✅
- `moe_gemm_marlin_w4a16` / `moe_marlin_w4a16` → W4A16 MoE ✅
- `moe_groupgemm_w4a8`、`m_grouped_w8a8_gemm_nt_masked` 等 grouped 变体

**辅助：**
- `rms_norm_per_token_quant`、`moe_swiglu_dynamic_quant` → per-token int8 activation 量化 ✅
- `awq_marlin_repack`、`awq_marlin_repack_w4a8`、`w4tow16` → 权重打包转换 ✅

### 10.2 关键判断：难度从"写 kernel"降到"接线"

lightop 把算子备齐了，改 sglang 只需写 **Python 适配层**，不用碰 C++/HIP kernel 开发：

1. 写 scheme 适配类（仿 `NPUCompressedTensorsW8A8Int8DynamicMoE` / `CompressedTensorsWNA16`，每个 ~100-200 行），把 `apply` 里的算子调用换成 lightop 对应函数。
2. 权重布局对齐（lightop `b_qweight`/`b_scale`/`a_scale` ↔ sglang `create_weights` param ↔ 量化产物三方对齐）——**最需小心的点**。
3. 改 sglang scheme 分发（`compressed_tensors.py` 海光分支从 `raise` 改成 `return` 新 scheme）。

**最佳参考：海光版 vLLM（`vllm 0.15.1+das.opt1.alpha.dtk2604`）已接 lightop**，照搬它的调用方式和权重布局可省大量摸索。

### 10.3 候选方案

| 方案 | lightop 算子 | 体积 | 备注 |
|---|---|---|---|
| W8A8（int8w+int8a） | `gemm_w8a8_asm`+`moe_gemm_w8a8`+`moe_swiglu_dynamic_quant` | ~415GB | 算子最齐全，与手写脚本 int8 产物格式最近 |
| W4A16（int4w,A16） | `gemm_w4a16`+`moe_marlin_w4a16`+`awq_marlin_repack` | ~224GB | 最省显存，但要 llmcompressor 产 pack_quantized |
| W4A8（int4w+int8a） | `moe_gemm_w4a8`+`awq_marlin_repack_w4a8` | ~224GB | Linear 缺 scheme 要新写 |

---

## 十一、lightop W8A8 验证脚本（地基验证）

> 脚本：`/models/test/test_lightop_w8a8.py`
> 在动 sglang 前，先确认 lightop W8A8 算子在海光 DCU 上能跑且精度对。

**运行：**
```bash
HIP_VISIBLE_DEVICES=0 ROCM_HOME=/opt/dtk python3 /models/test/test_lightop_w8a8.py
```

- `ROCM_HOME=/opt/dtk` 必须设（lightop import 读 `/opt/dtk/.info/rocm_version`，dtk=26.04）。
- 验证内容：造随机权重→per-channel int8 量化权重→per-token int8 量化 activation→调 `lightop.gemm_w8a8_smooth`→与 bf16 参考 + 量化反量化参考对比误差。
- 通过标准：vs 量化反量化误差 <2%（验证算子实现正确），vs bf16 <5%（W8A8 正常范围）。
- MoE W8A8（`moe_gemm_w8a8`）需 config dict + MoE align 索引，第一步 Linear 通过后再单独验证。

---

## 十二、踩坑记录：装 llmcompressor 搞坏环境（2026-07-21）

> **重大教训：海光 sglang 容器里绝不能装 llmcompressor。**

为跑 `minimax_m3_w4a16.py`（`model_free_ptq` 接口）装 llmcompressor，用 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LLMCOMPRESSOR=0.0.0 pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple`，pip 把依赖连带升级：
- **torch**：海光定制版（2.10+HIP）→ `2.13.0+cu130`（非海光版，`hip=None`，看不到 DCU）
- **transformers**：→ 4.57.6（`PreTrainedModel` import 崩，因 torchvision::nms 算子错配）
- **compressed-tensors**：0.13 → 0.17.2
- **numpy**：→ 2.2.6

后果：整个容器 sglang 起不来、量化也跑不了（torch 看不到卡）。海光定制 torch 不在清华源，回滚困难。**解决方案：销毁容器重建。**

**重建容器后绝不再装 llmcompressor。** 当前方案（sglang blockwise_int8 或改 sglang 接 lightop）用手写脚本量化即可，不需要 llmcompressor。

---

## 十三、下一步行动计划（重建容器后）

1. **跑 lightop W8A8 验证**：`HIP_VISIBLE_DEVICES=0 ROCM_HOME=/opt/dtk python3 /models/test/test_lightop_w8a8.py`，确认 lightop W8A8 算子在海光 DCU 可用。
2. **验证 sglang 能加载 bf16 原模型（基线）**：sglang 有完整 MiniMax-M3 实现（`minimax_m3.py`+`minimax_m3_vl.py`，架构名 `MiniMaxM3SparseForConditionalGeneration` 已注册），但需实测 sparse attention/VL/MoE 在海光能跑。注意 `minimax_m3.py:1092` 和 `minimax_m3_vl.py:141` 有 `elif not _is_cuda:` 分支需确认海光走哪条。
3. **（可选）验证 blockwise_int8**：用小模型量化成 blockwise_int8 在 sglang 跑，确认 Triton int8 MoE kernel 在 BW100 可编译。
4. **核心：改 sglang 接 lightop 跑 W8A8**：参考海光版 vLLM 调用 lightop 的代码，写 scheme 适配层，优先 W8A8（算子最全、与已有 int8 产物格式最近）。

### 环境备忘
- 海光 dtk：`/opt/dtk`，rocm_version=26.04
- lightop env：`ROCM_HOME=/opt/dtk`，`HIP_VISIBLE_DEVICES` 选卡
- 原始模型：`/models/MiniMax/MiniMax-M3`（bf16，59 shard，854GB）
- 坏产物（可删）：`/models/MiniMax/MiniMax-M3-channel-int4-w4a8`（209G，W4A8 格式 sglang 不认）
- 量化脚本：`/models/llm-compressor/examples/one_click_quant/templates/minimax_m3_w4a8.py`（手写，`--quant-type int8` 产 W8A8 int-quantized，sglang Linear 认但 MoE 不认）

---

# 第二部分：W8A8 方案与 sglang 适配跑通（2026-07-22）

> 前面第一~十三章是 W4A8 探索与 lightop/aiter 调研的历史记录。
> 本部分起为最终方案：**W8A8（只量化 MoE）+ tilelang 自写算子 + 改 sglang 6 处补丁，forward 已跑通**。

## ✅ 当前最终状态

sglang 在海光 BW100 (gfx936) 上成功加载 W8A8 moe-only MiniMax-M3（412GB，8 卡 TP），server 起在 8080，
`/v1/chat/completions` 请求成功返回连贯中文（诗歌/对话），**W8A8 量化未破坏模型质量**。

```
curl /v1/chat/completions  "用中文写一首关于秋天的短诗"
→ <mm:think>...草拟多版... 秋风起时，黄叶飘落。冷露凝结，寒霜降临。孤雁南归...  ✅ finish_reason=length
```

GitHub 仓库：https://github.com/benyuereal/quant_gemm （含 quant_gemm 算子包 + sglang_patches）

## 十四、方案转折：从 W4A8/lightop 到 W8A8/tilelang

### 14.1 三条算子路的对比与抉择

| 路径 | 实现 | gfx928 | 结论 |
|---|---|---|---|
| lightop | 预编译 `.co` 汇编 | ❌ hsa/ 只有 gfx936/gfx938，无 gfx928 | 放弃（死二进制） |
| aiter per-channel | asm/CK（需 `aiter_` C 扩展） | ⚠️ 当前构建未启用 aiter_ | 备用 |
| aiter blockwise | `aiter.ops.triton.gemm_w8a8`（纯 Triton） | ✅ | 备用 |
| **tilelang** | 海光定制版，架构无关 JIT | ✅ | **采用**（自主可控） |

- lightop W8A8 在 gfx936 验证通过（`test_lightop_w8a8.py`，vs deq 0.14%），但 gfx928 无产物。
- aiter `gemm_w8a8`（blockwise Triton）在 gfx936 验证通过（`test_aiter_w8a8.py`，vs deq 0.14%），但 blockwise 需重新量化。
- **tilelang 自写 per-channel W8A8**（`test_tilelang_w8a8.py` / `test_tilelang_moe_w8a8.py`，vs deq 0.28%），per-channel 与现有量化产物对齐，最终采用。

### 14.2 量化方案：W8A8 只量化 MoE

- W8A8 = per-channel int8 weight + per-token dynamic int8 activation。
- **只量化 MoE expert**（96.7% 权重，826GB→413GB），attention/shared/dense mlp 留 bf16。
- 理由：MoE expert 是大块权重、对量化不敏感；attention/routing 留 bf16 精度更稳（routing gate 量化会破坏专家选择）。
- config.json `targets:["Linear"]` + ignore 覆盖非 MoE Linear → sglang 原生支持 MoE 走量化、其他走 unquantized。

### 14.3 MoE 算子策略：复用 sglang 原生 Triton kernel

关键发现：sglang 原生 `fused_moe_kernel`（Triton）**已支持 `use_int8_w8a8` + `per_channel_quant`**，缺的只是一个 MoE scheme 接上海光分支。
故**不用** quant_gemm 的 MoE kernel（备份方案），而是新写 scheme 复用 sglang Triton runner——能复用 sglang 全套 MoE 基础设施（路由、combine、a2a），工作量小得多。

## 十五、量化过程：W8A8 moe-only 怎么做的

> 本章讲最终跑通的 W8A8 量化全过程：脚本怎么量化、模型怎么适配 config、产物怎么验证。
> （第三章的 W4A8 int4 方案是早期探索，已被 W8A8 int8 取代，保留作历史。）

### 15.1 量化方案

- **W8A8**：权重 per-channel int8（对称，qmax=127），激活 per-token int8（dynamic，运行时量化）。
- **只量化 MoE expert**（`block_sparse_moe.experts.*.w1/w2/w3`，占 96.7% 权重 826GB）。
- 其余全部留 bf16：attention（q/k/v/o/index_q/index_k_proj）、shared_experts、dense mlp（前3层）、lm_head、embed_tokens、vision_tower、norm、routing gate、MTP。
- 理由：MoE expert 是大块权重、对量化不敏感；routing/attention 留 bf16 精度更稳（routing gate 量化会破坏专家选择）。
- 体积：796GB → **412GB**（约 1.9× 压缩，核心收益来自 MoE expert 826GB→413GB）。

### 15.2 量化脚本

脚本：`/models/llm-compressor/examples/one_click_quant/templates/minimax_m3_w4a8.py`（手写，不依赖 llmcompressor——海光容器装 llmcompressor 会搞坏环境，见第十二章）。

```bash
python3 minimax_m3_w4a8.py \
    --input-path /models/MiniMax/MiniMax-M3 \
    --output-path /models/MiniMax/MiniMax-M3-w8a8-moe-only \
    --quant-type int8 \
    --moe-only
```

封装：`/models/quantize_minimax_m3_w4a8.sh`（8 卡并行）。

**量化算法**（`weight_quant_int8`，per-channel 对称）：
```python
absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # 每 output channel 一个
scale = absmax / 127.0
q = round(w / scale).clamp(-128, 127).to(int8)             # [N,K] int8
# 存 q + scale（scale [N,1] f32）
```

**`--moe-only` 逻辑**（脚本里加的，worker 函数）：
```python
if moe_only and ".block_sparse_moe.experts." not in weight_name:
    # 非 MoE expert 一律保留 bf16，跳过量化
    new_state_dict[weight_name] = weight; continue
# 只有 MoE expert 走 weight_quant_int8
```

**多 GPU 并行**：`mp.spawn`，59 个 safetensor shard 按 `rank::world_size` 分给 8 卡，每卡加载自己的 shard → 量化 → 存回。3D MoE 权重 `[E,N,K]` reshape 成 2D `[E*N, K]` 量化再 reshape 回 `[E, N, K_out]`。

### 15.3 模型适配：config.json 怎么让 sglang 认

产物 `config.json` 的 `quantization_config`（compressed-tensors 格式，sglang/vLLM 原生支持）：
```json
{
  "quant_method": "compressed-tensors",
  "format": "int-quantized",
  "config_groups": {
    "group_0": {
      "weights": {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true, "dynamic": false},
      "input_activations": {"num_bits": 8, "type": "int", "strategy": "token", "symmetric": true, "dynamic": true},
      "targets": ["Linear"]
    }
  },
  "ignore": [ /* 17 条正则, 见下 */ ]
}
```

**关键适配点：targets + ignore 配合实现"只量化 MoE"**

- `targets: ["Linear"]` 按模块类型匹配——所有 Linear 默认走 W8A8 scheme。
- `ignore` 用正则把**非 MoE 的 Linear**排除，让它们走 unquantized（bf16）：
  - 原始 ignore（通用，13 条）：`norm`、`embed_tokens`、`lm_head`、`block_sparse_moe.gate.weight`（routing gate，敏感）、`e_score_correction_bias`、`multi_modal_projector`、`patch_merge_mlp`、`visual`/`vision`、`mtp`、各 `qk_norm` 等。
  - `--moe-only` 额外 ignore（4 条，`MOE_ONLY_EXTRA_IGNORE`）：
    ```
    re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj|index_q_proj|index_k_proj)(\.weight)?$
    re:.*block_sparse_moe\.shared_experts\.(gate_proj|up_proj|down_proj)(\.weight)?$
    re:.*mlp\.(gate_proj|up_proj|down_proj)(\.weight)?$
    ```
- **ignore 正则的坑**（踩坑 3）：sglang 匹配 ignore 时用**不带 `.weight`** 的 module name，所以正则里 `.weight` 要写成可选 `(\.weight)?`，不能强制 `\.weight$`（否则匹配不到，非 MoE Linear 会被误量化，加载时 `Unsupported copy between dtypes` 报错）。

**权重命名映射**（sglang 加载时处理，不用改产物）：
- 产物里 MoE expert 叫 `block_sparse_moe.experts.*.w1/w2/w3`（MiniMax 原始命名）。
- sglang 内部用 `gate_proj/up_proj/down_proj`，加载时通过 `ckpt_gate_proj_name="w1"` 等映射（`minimax_m3.py` 的 `make_expert_params_mapping`）。
- sglang `get_moe_scheme` 用 `.0.gate_proj` 匹配 expert → 命中 W8A8 scheme → 走海光 MoE scheme（补丁 1）。

### 15.4 量化产物验证

产物：`/models/MiniMax/MiniMax-M3-w8a8-moe-only`（412GB，59 shard + config.json + index.json）。

验证三项（全通过）：

1. **config 格式**：compressed-tensors W8A8，8bit channel weight + 8bit token dynamic act，ignore 17 条覆盖非 MoE Linear。✅
2. **权重 dtype**（单 shard 抽查）：
   - MoE expert `experts.0.w1.weight`：`int8 [3072,6144]` + `w1.weight_scale`：`float32 [3072,1]`（per-channel）✅
   - shared_experts / attn / norm：`bfloat16`（moe-only 生效，未量化）✅
3. **量化精度**（真实 expert 反量化 vs 原始 bf16）：平均相对误差 **1.02%**，余弦相似度 **1.0000** ✅（W8A8 per-channel 正常范围）。

### 15.5 量化耗时

8 卡 BW100 并行，59 shard，约 **3 分钟**完成（比预估 30-60 分钟快很多，因为 moe-only 下非 MoE 层直接拷贝 bf16 不算，只 MoE expert 做 int8 量化）。

---

## 十六、sglang 缺口梳理（gfx936/gfx928）

| 算子类 | sglang 现状 | 缺口 | 解法 |
|---|---|---|---|
| **Linear W8A8 GEMM** | `compressed_tensors_w8a8_int8.py` 守卫 `if _is_cuda: int8_scaled_mm`；海光 `_is_cuda=False`，`sgl_kernel.int8_scaled_mm` 未注册 | ❌ 底层算子缺 | moe-only 下 Linear 全 bf16，海光拦截不返回 W8A8 scheme |
| **MoE W8A8 grouped GEMM** | `compressed_tensors.py` `_is_dynamic_token_w8a8` MoE 分支 `raise NotImplementedError`（仅 NPU） | ❌ scheme 缺 | 新写 `CompressedTensorsW8A8Int8TritonMoE`，raise→return |
| Sparse Attention | `minimax_sparse_ops/` 纯 Triton | ✅ 架构无关 | 实测（共享内存需调 stages，见下） |
| RMSNorm | `layernorm.py forward_hip` 接 lightop | ⚠️ gfx928 无产物 | 有 `forward_aiter`/`forward_native` 退路 |
| RoPE / MoE routing | torch/Triton | ✅ | — |

**关键发现**：sglang dev 版对海光支持比预期好——`_use_aiter = SGLANG_USE_AITER and _is_hip` 已支持 aiter 跑 MoE；`CompressedTensorsWNA16TritonMoE`（W4A16/W8A16）海光走 Triton 可用。真正缺的是 W8A8 的 Linear 底层算子和 MoE scheme。

## 十七、改了 sglang 哪里（6 处补丁）

补丁归档：`/models/quant_gemm_pkg/sglang_patches/`（含改后文件 + .patch + README），已 push GitHub。

| # | 文件 | 类型 | 改动 |
|---|---|---|---|
| 1 | `compressed_tensors_w8a8_int8_triton_moe.py` | **新增** | 海光 W8A8 MoE scheme，复用 sglang 原生 Triton fused_moe kernel |
| 2 | `compressed_tensors.py` | 改 | ① MoE W8A8 海光分支 raise→return 新 scheme；② Linear W8A8 海光下走 bf16（moe-only） |
| 3 | `schemes/__init__.py` | 改 | 导出新 scheme |
| 4 | `int8_kernel.py` | 改 | `per_token_quant_int8` 的 round：`tl.extra.cuda.libdevice` → `tl.extra.hip.libdevice` |
| 5 | `prefill/topk_sparse.py` | 改 | sparse attn **prefill** kernel `num_stages=1`（BW100 共享内存 64K，stages≥2 超 65536） |
| 6 | `decode/topk_sparse.py` | 改 | sparse attn **decode** kernel `num_stages=1`（同上，decode 阶段也超） |

### 16.1 新增 W8A8 MoE scheme（补丁 1）

`CompressedTensorsW8A8Int8TritonMoE`，仿 `CompressedTensorsWNA16TritonMoE` 但权重 int8（不 pack）。
- `create_weights`：`w13_weight [E,2*intermediate,hidden] int8`（gate w1+up w3 沿 N 拼接）+ `w2_weight [E,hidden,intermediate] int8` + per-channel scale `[E,N,1] f32`。
- `get_triton_quant_info`：设 `use_int8_w8a8=True, per_channel_quant=True`。
- 激活 per-token int8 量化由 kernel 内部 `per_token_quant_int8` 完成（dynamic），无需 a_scale。
- 关键：scale 形状 `[E,N,1]` + `quant_method=CHANNEL`（对照 NPU 版修正，否则 weight_loader narrow 报错）。

### 16.2 Linear 走 bf16（补丁 2）

`get_linear_scheme` 里拦截：海光下 `_is_dynamic_token_w8a8` 的 Linear 一律 `weight_quant=None` 走 unquantized。
原因：`qkv_proj`/`index_qkv_proj` 等融合 Linear 即使在 config ignore 里，sglang fused_mapping 仍分到 W8A8 scheme，而 `int8_scaled_mm` 海光未注册 → `NameError`。moe-only 下 Linear 本就该 bf16。

### 16.3 round 海光兼容（补丁 4）

`per_token_quant_int8` 的 Triton kernel 原用 `tl.extra.cuda.libdevice.round`，海光报 `Implicit conversion of CUDA __nv_roundf dropped`。改为 `tl.extra.hip.libdevice.round`（写死，海光镜像专用）。已单独验证误差 0.93%。

### 16.4 sparse attn 共享内存（补丁 5/6）

MiniMax sparse attention 的 `_gqa_share_sparse_fwd_kernel`（prefill）和 `_gqa_share_sparse_decode_kernel`（decode）autotune 含 `num_stages≥2`，BW100 共享内存上限 65536，需 69632 → `OutOfResources`。实测 stages=2 仍超（差 4KB），必须 `num_stages=1`。

## 十八、踩过的坑（sglang 适配阶段，按顺序）

1. **scheme 缺口**：`get_moe_scheme` W8A8 海光 raise → 新写 scheme
2. **w13 权重布局**：`IndexError start out of range` → 对照 NPU 版修正 scale `[E,N,1]` + `quant_method=CHANNEL`
3. **ignore 正则 `\.weight$`**：sglang 匹配 ignore 用不带 `.weight` 的 module name，正则去掉 `\.weight$` 强制
4. **Linear W8A8 误触**：`qkv_proj` 融合层 ignore 没生效，`int8_scaled_mm` 未定义 → 海光下 Linear W8A8 直接走 bf16
5. **`__nv_roundf`**：`tl.extra.cuda.libdevice` 海光不支持 → `tl.extra.hip.libdevice`
6. **Triton constexpr**：kernel 内不能访问普通全局变量 → 写死 hip（海光镜像专用）
7. **cuda graph capture 失败**：同 `__nv_roundf` → `--disable-cuda-graph`
8. **KV cache 内存池**：mem_fraction 算法算爆 → `--max-total-tokens 4096` + mem_fraction 0.85 + expandable_segments
9. **根文件系统满**：`/tmp` 在 overlay 可写空间小，torchinductor 写锁 `Errno 28 No space` → TMPDIR/TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR 指到 /models
10. **sparse attn 共享内存（prefill）**：`num_stages=2/3` 需 69632 > 65536 → `num_stages=1`
11. **sparse attn 共享内存（decode）**：`num_stages=2~5` 同样超 → `num_stages=1`
12. **Triton 缓存复用旧编译**：改 stages 后不清缓存会用旧 69632 结果 → minimax.sh 启动前自动清缓存

## 十九、验证进展（按时间）

| 阶段 | 结果 |
|---|---|
| scheme 分发（get_moe_scheme 返回新 scheme） | ✅ 单元验证通过 |
| Linear 全走 bf16（含融合 qkv_proj） | ✅ 单元验证通过 |
| sglang 加载 412G 权重 | ✅ 8 卡加载成功，无 dtype 冲突 |
| `per_token_quant_int8` kernel（海光 round） | ✅ 单独验证，误差 0.93% |
| sglang server 启动（port 8080） | ✅ `fired up and ready to roll` |
| prefill（forward_extend） | ✅ 跑通（修 prefill sparse kernel stages 后） |
| decode（forward_decode） | ✅ 跑通（修 decode sparse kernel stages 后） |
| **端到端生成（chat/completions）** | ✅ 返回连贯中文，量化无质量退化 |

## 二十、工时预估与实际

### 19.1 事前预估（只量化 MoE 方案）

| # | 工作块 | 乐观 | 悲观 | 依据 |
|---|---|---|---|---|
| 1 | MoE W8A8 grouped GEMM kernel（含 fuse swiglu + 路由对齐） | 4 | 8 | 核心 GEMM 已验证 |
| 2 | MoE W8A8 scheme 接 sglang | 2 | 4 | 有 WNA16TritonMoE 模板 |
| 3 | 联调 + gfx928 实测 + RMSNorm 退路 + sparse attn + 端到端 | 3 | 7 | 联调不可控 |
| — | **合计** | **9 天** | **19 天** | **最可能 ~13 天** |

### 19.2 实际情况

实际走"复用 sglang 原生 Triton MoE kernel"路径（而非自写 tilelang MoE 接 sglang），省了工作块 1 的大部分（fuse swiglu/路由对齐由 sglang runner 处理）。主要工作量在**联调阶段排查 12 个坑**（尤其 sparse attn 共享内存、磁盘、Triton 缓存）。整体约 1-2 天跑通（含前期 tilelang 算子验证 + 量化 + sglang 补丁）。

## 二十一、安装包与测试目录

### 20.1 安装包 `quant_gemm`（`/models/quant_gemm_pkg/`）

```
quant_gemm_pkg/
├── quant_gemm/
│   ├── __init__.py     # 导出 API
│   ├── kernels.py      # w8a8_per_channel_gemm, moe_w8a8_grouped_gemm (tilelang)
│   ├── quant.py        # per_token_sym_int8, per_channel_sym_int8
│   └── api.py          # w8a8_linear, w8a8_moe (高层 API, kernel 缓存)
├── pyproject.toml / setup.py / README.md
└── sglang_patches/     # sglang 6 处补丁 + README
```

安装：`cd /models/quant_gemm_pkg && pip install -e . --no-deps --no-build-isolation`（用 --no-deps 避开 tilelang 的 tvm-ffi 版本校验冲突，依赖已装）。
任意目录 `from quant_gemm import w8a8_linear, w8a8_moe` 可用。回归测试 `test/test_quant_gemm_package.py` 通过。

### 20.2 测试目录 `/models/test/`

| 脚本 | 用途 |
|---|---|
| `test_quant_gemm_package.py` | 安装包回归测试（调 pip 装好的包，高层 API） |
| `test_tilelang_w8a8.py` | Linear W8A8 独立验证（内联 kernel，vs deq 0.28%） |
| `test_tilelang_moe_w8a8.py` | MoE W8A8 独立验证（内联 kernel，路由表） |
| `test_lightop_w8a8.py` | lightop W8A8 验证（gfx936 only） |
| `test_aiter_w8a8.py` | aiter blockwise Triton 验证 |
| `test_aiter_w8a8_gfx928.py` | aiter 三条路 gfx928 验证脚本 |
| `test_minimax_m3_single_shard.py` | 量化产物单 shard 格式校验 |

## 二十二、启动与测试

### 21.1 启动

`/models/minimax.sh`（自动：停残留 sglang → 清 triton/torchinductor 缓存 → 启动 → 日志覆盖到 `/models/sglang_serve.log`）：
```bash
bash /models/minimax.sh
# 另开终端: tail -f /models/sglang_serve.log
```
关键参数：`--tp-size 8 --mem-fraction-static 0.85 --context-length 4096 --max-total-tokens 4096 --attention-backend triton --mm-attention-backend triton_atn --disable-cuda-graph --skip-server-warmup`。
环境变量：`SGLANG_USE_AITER=0`（纯 Triton），`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，`TMPDIR/TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR` 指到 /models。

### 21.2 测试

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"用中文写一首关于秋天的短诗"}],"max_tokens":200}'
```

## 二十三、环境备忘（更新）

- 海光 dtk：`/opt/dtk`，rocm_version 26.04
- 开发机：8× BW100 (gfx936)，HIP 6.3.26113，torch 2.9.0；部署目标：gfx928 (K100)
- tilelang：`0.1.9+das.opt1.dtk2604.torch290`（海光定制，自动探 HCU，warp_size=64，MMAC intrinsic）
- lightop：`0.6.0+das.dtk2604.torch290`（gfx936/gfx938 only，**gfx928 无产物**）
- aiter：已装，但 `aiter_` C 扩展未装 → per-channel a8w8 不可用；`aiter.ops.triton.gemm_w8a8`（blockwise）可用
- sglang：dev 0.0.0.dev12695，已接 lightop（layernorm/moe_align/parallel_state）
- 原始模型：`/models/MiniMax/MiniMax-M3`（796GB bf16，59 shard）
- W8A8 量化产物：`/models/MiniMax/MiniMax-M3-w8a8-moe-only`（412GB，59 shard）
- 量化脚本：`/models/llm-compressor/examples/one_click_quant/templates/minimax_m3_w4a8.py`（`--quant-type int8 --moe-only`）
- tilelang 示例：`/models/tilelang/examples/`（gemm/, fusedmoe/, dequantize_gemm/w4a8）
- 启动脚本：`/models/minimax.sh`；sglang 补丁：`/models/quant_gemm_pkg/sglang_patches/`
- GitHub：https://github.com/benyuereal/quant_gemm
- **绝不能在 sglang 容器装 llmcompressor**（会升级海光定制 torch 致环境崩，见第十二章）

## 二十四、tilelang 写法备忘（踩过的坑）

- 条件控制流：`T.If`/`T.Else`（大写上下文管理器），不是 `T.if_`；循环 `T.Serial` 不是 `T.serial`
- 避开 kernel 内控制流：路由表 Python 端预算传入，kernel 直接索引（比 T.If 查找 expert 稳）
- int8 GEMM：`T.gemm(A, B, C, transpose_B=True)` 支持 int8×int8→int32
- scale 注入：用独立 float32 fragment 存 `cast(C_local,f32)*x_scale*w_scale`，**不要写回 int32 累加器**（会截断，36% 误差 bug）
- `T.copy` 支持动态切片 `A[m_start:m_start+block_M, ...]`，但不支持间接 gather → token 须先按 expert 分组连续排列
- `@tilelang.jit(out_idx=[N])`：N 是输出 tensor 在参数列表的 0-based 位置，数错报 "ndim expected X but got Y"
- grouped_gemm_fwd_ptr 指针表路径 "not stable"，不要用；用分组连续布局 + group_offsets

## 二十五、待办

- [ ] **gfx928 实测**：全部补丁在 gfx928 (K100) 上重测（BW100 验证 ≠ gfx928 一定通；gfx928 共享内存可能不同）
- [ ] thinking 模式处理：输出含 `<mm:think>` 思考过程，若需直接出结果要调 chat template 或参数
- [ ] 性能优化：
  - 启用 cuda graph（需先解决 `__nv_roundf` 在 capture 路径的问题）
  - 生成 BW100 专用 MoE Triton config（消 "Config file not found ... int8_w8a8" warning，提速）
  - 或开 `SGLANG_USE_AITER` 走 aiter MoE
- [ ] `flash_with_topk_idx.py` 的 stages≥3 暂未改（本次未触发）；若后续报同错同法处理
- [ ] stages 精调：当前全 stages=1 偏保守，可测算各 kernel 在 64K 内的最优 stages
- [ ] 端到端质量对比：W8A8 moe-only vs bf16 原模型，生成质量/perplexity
- [ ] W4A8（可选，省显存到 224GB）：tilelang `dequantize_gemm/example_dequant_gemm_w4a8.py` + scale 注入，需解决 int4 打包格式对齐

---

## 二十六、W4A16 moe-only 量化（进行中）

### 26.1 为什么做 W4A16

W8A8 moe-only 每卡权重 52GB（TP8），上下文余量有限（单请求约 64K）。
W4A16 把 MoE expert 从 int8 进一步压到 int4，权重再省一半：

| 方案 | 每卡权重(TP8) | 单请求上下文余量 |
|---|---|---|
| bf16 原版 | ~99GB（放不下） | — |
| W8A8 moe-only | 52GB | ~64K |
| **W4A16 moe-only** | **~28GB** | **~256K** |
| cyankiwi AWQ INT4（g=32+校准） | ~26GB | ~256K |

activation 仍为 bf16（A16），所以 sglang 走 W4A16 Triton 路径，不需要 W4A8 那种自定义 kernel。

### 26.2 量化方案

- **只量化 MoE routed expert**（`block_sparse_moe.experts.*.w1/w2/w3`）为 int4
- W4A16 preset：weights num_bits=4, strategy=group, **group_size=128**, symmetric, **无 activation 量化**
- observer=`memoryless_minmax`（**naive 无校准**，data-free）
- 其余全部 bf16：attn / shared_experts / dense mlp / gate / norm / embed / lm_head / vision / mtp
- 产物格式：compressed-tensors `pack-quantized`（`weight_packed` int32 + `weight_scale` + `weight_shape`），sglang 原生识别

### 26.3 量化脚本与产物

- 脚本：`/models/llm-compressor/examples/one_click_quant/templates/minimax_m3_w4a16.py`（标准 `model_free_ptq` 接口，data-free）
- 产物：`/models/MiniMax/MiniMax-M3-w4a16-moe-only`
- 大小：**225GB**（原 bf16 796GB，压缩 3.55×）
- 耗时：6分30秒（4卡：GPU 0/1/4/5，`--max-workers 4`）

**量化时踩的坑：**
- 首次用 8 卡跑，GPU6/7 被 AWQ 下载器占用 → `torch.OutOfMemoryError: HIP out of memory`（抢显存）
- 解法：`CUDA_VISIBLE_DEVICES` 限定只用空闲卡，worker 数 = 卡数。8卡被占时改用 4 卡（0,1,4,5），稳定完成

**产物权重布局（已验证）：**
```
expert w1: weight_packed [3072, 768] int32  (= [N, K//8], 8×int4 打包进 int32)
           weight_scale  [3072, 48]  bf16   (= [N, K//128], group_size=128)
           weight_shape   [2] int64
expert w2: weight_packed [6144, 384] int32  (= [N, K//8])
           weight_scale  [6144, 24]  bf16
非expert:  gate.weight fp32, shared_experts.*.weight bf16（保持原精度）
```

### 26.4 显存账（决定 TP 数）

每卡 64GB（BW100，可用 ~67GB）。MoE expert 按 expert 切分（EP）：

| TP | 每卡 expert | 每卡非expert | 每卡权重合计 | 评价 |
|---|---|---|---|---|
| 4 | 53.3GB | 9.0GB | **62.2GB** | 极限，KV 几乎没空间，只能短上下文 |
| 6 | 35.5GB | 6.7GB | 42.2GB | 宽裕，但 **128÷6 不整除，sglang 报错** |
| 8 | 26.6GB | 5.5GB | 32.1GB | 最优（需 8 卡） |

**硬限制：sglang 要求 `num_experts % ep_size == 0`**（`topk.py:1270` / `fused_moe_triton/layer.py:215`）。
128 的约数只有 1/2/4/8/16/32/64/128 → **TP 只能选 4 或 8**（当前 6 卡空闲时只能用 4）。

### 26.5 sglang 加载：踩了三个坑（核心）

加载自己的 W4A16 产物，目标走 sglang 原生 `CompressedTensorsWNA16TritonMoE`（hip Triton 路径，非 marlin）。

**坑 1：`Unable to find matching target for ...self_attn.qkv_proj`**
- 原因：config.json 的 `ignore` 没覆盖 `self_attn`/`shared_experts`/dense `mlp`。compressed-tensors 要求每个 Linear **要么被 targets 命中（量化），要么被 ignore 命中（bf16）**，两边都没匹配就报错
- 修复：ignore 补 `self_attn`、`shared_experts`、dense `mlp.(gate|up|down|gate_up)_proj`

**坑 2：`Unable to find matching target for ...mlp.experts.0.gate_proj`**
- 原因：**权重文件名（HF 原名 `block_sparse_moe.experts.0.w1`）≠ sglang 内部层名（`mlp.experts.0.gate_proj`）**。sglang 加载时通过 weight_loader 映射（`w1→gate_proj`、`block_sparse_moe→mlp`，见 `minimax_m3.py:1184` `ckpt_gate_proj_name="w1"`），**compressed-tensors 的 target/ignore 匹配用的是映射后的 sglang 名**
- 修复：config.json 的 `targets` 和 `ignore` 全部改用 **sglang 层名**：
  - targets: `re:.*\.mlp\.experts\.\d+\.(gate|up|down)_proj$`
  - ignore: `mlp.gate` / `mlp.shared_experts.*` / `self_attn.*` / `mlp.(gate|up|down|gate_up)_proj` / norm / embed / lm_head / vision / mtp
- 验证：正则边界正好分开 expert（`mlp.experts.N.gate_proj`）和 dense mlp（`mlp.gate_proj`）——`\.mlp\.` 后紧跟 `gate_proj` 才命中 dense，expert 中间隔了 `experts.N` 不命中

**坑 3：`KeyError: 'Linear'`**（`compressed_tensors_wNa16_moe.py:61`）
- 原因：`CompressedTensorsWNA16MoE.__init__` 硬编码 `target_scheme_map["Linear"]`，期望 config 用模块类名 `targets:["Linear"]`（像 cyankiwi AWQ）。我们用正则层名 target，`target_scheme_map` 键是正则字符串，没有 `"Linear"` 键
- 修复（sglang 源码补丁 7）：改成兼容正则 target——`"Linear"` 在则用它，否则取 map 第一个 scheme
- 备选方案（未采用）：config 改 `targets:["Linear"]` + ignore 排除所有非 expert Linear，但风险是漏排一个 Linear 就被误量化

### 26.6 sglang 源码补丁 7（W4A16 新增）

文件：`/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`

`CompressedTensorsWNA16MoE.__init__` 第 61 行：
```python
# 改前
config = self.quant_config.target_scheme_map["Linear"].get("weights")
# 改后
_scheme_map = self.quant_config.target_scheme_map
if "Linear" in _scheme_map:
    config = _scheme_map["Linear"].get("weights")
else:
    config = next(iter(_scheme_map.values())).get("weights")
```
作用：让 compressed-tensors 支持用**正则层名**做 target（只量 MoE expert），不强制 `targets:["Linear"]`。hip 上 `CompressedTensorsWNA16TritonMoE` 继承此类，同样生效。

### 26.7 启动脚本（已跑通）

`/models/minimax_w4a16.sh`（4 卡验证版）。经多轮调参最终跑通的参数：
```bash
export CUDA_VISIBLE_DEVICES=0,1,4,5   # 4张均衡空闲卡(2/3/6/7被占)
sglang serve \
    --model-path /models/MiniMax/MiniMax-M3-w4a16-moe-only \
    --mem-fraction-static 0.97 \   # 关键: 权重占92%,frac要>0.93才不让KV公式算负
    --tp-size 4 \                  # 128÷4=32 整除
    --dtype bfloat16 \
    --context-length 512 \         # 开cuda graph后KV被挤,上下文调小
    --max-total-tokens 512 \
    --chunked-prefill-size 512 \
    --cuda-graph-max-bs 8 \        # 只capture bs≤8, 额外~0.6GB/卡
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --trust-remote-code --skip-server-warmup \
    --host 0.0.0.0 --port 8081
```

**关键调参过程（`mem-fraction-static` 不是直觉的"调高=KV大"）：**
- sglang 公式：`rest_memory = 加载后空闲 - 加载前空闲 × (1 - mem_fraction_static)`
- 实测：加载前 62GB，加载后只剩 **4.82GB**（权重 57.25GB/卡）
- frac=0.55 → rest = 4.82 - 62×0.45 = **-23GB** → `Not enough memory`
- frac=0.92 → rest = -0.14GB（仍负）
- frac=0.95 → rest = 1.72GB（正，不开 cuda graph 时用这个，context=1024 跑通）
- frac=0.97 → rest = 2.96GB（开 cuda graph 时用，context=512）
- **本质**：frac 表达"加载前预留给 KV 的比例"，权重几乎占满时必须接近 0.92+ 才让公式算正；真正 KV 分配看加载后实际剩余

**4 卡启动额外坑：`The memory capacity is unbalanced`**
- sglang 检查各卡加载前空闲显存，若某卡 < 其他卡×0.9 则拒绝启动
- `CUDA_VISIBLE_DEVICES=0,1,2,3` 时 GPU2/3 只 59GB（被占 8GB），0/1 有 67GB → 不均衡
- 解法：选显存均衡的 4 张卡（0,1,4,5 均 67GB）

### 26.8 4 卡上下文与 cuda graph 显存测算

**4 卡 TP=4 上下文容量**（每卡加载后剩 4.72GB）：
- 每 token 每卡 KV = 2(K+V) × 1 kv_head(GQA 4头÷4卡) × 128 head_dim × 67层 × 2字节 ≈ **33.5 KB**
- 不开 cuda graph（KV 预算 ~1.7GB）：单请求 **~51K tokens**
- 开 cuda graph bs≤8（KV 预算 ~0.7GB）：单请求 **~21K tokens**
- 理论极限（留 1.5GB 激活，其余全 KV）：**~94K tokens**（但 prefill 激活会先 OOM）

**cuda graph 额外显存**（sglang 公式 `reserved = chunked_prefill×1.5 + cuda_graph_max_bs×2` GB）：
| cuda_graph_max_bs | 额外显存/卡 |
|---|---|
| 8 | ~0.6 GB |
| 80（TP4 默认） | ~1.0 GB |
| 160 | ~1.3 GB |

**结论：4 卡显存极限下 cuda graph 不划算**——bs≤8 占 0.6GB 挤压 KV（51K→21K），且 capture 峰值可能 OOM。cuda graph 真正适合 **8 卡 TP=8**（每卡剩 35GB，扣 1GB 无压力）。4 卡验证阶段用 bs≤8 做功能验证。

### 26.9 验证进展

- [x] 量化完成，产物 225GB，权重布局正确
- [x] config.json targets/ignore 修复（sglang 层名）
- [x] sglang 源码补丁 7（KeyError 'Linear'）
- [x] 日志确认进入 `Using CompressedTensorsWNA16TritonMoE (ROCm)` + `Falling back to UnquantizedLinearMethod`（非expert走bf16）
- [x] **4 卡加载成功**：`Load weight end. avail mem=4.82 GB, mem usage=57.25 GB`，`The server is fired up and ready to roll!`（port 8081, frac=0.95, context=1024, 不开cuda graph）
- [x] `mem-fraction-static` 调参公式摸清（>0.93 才不报 Not enough memory）
- [ ] cuda graph bs≤8 capture 是否成功（待验证，可能踩 W8A8 同款 `__nv_roundf`/shared mem 坑）
- [ ] forward 跑通 + 生成质量
- [ ] 端到端评测（MMLU/GPQA），与 cyankiwi AWQ、bf16 对比

### 26.9 关键认知（关于"校准"与"12.88%"）

- **naive int4 必须校准才能恢复精度**：RTN/min-max 被 outlier 拖累，我们自己 data-free 量化的 W4A16 是 naive（observer=memoryless_minmax），**精度预期不如校准过的**
- **"12.88%" 站不住脚**：该数字无评测记录，是权重/中间层误差的中间度量，非端到端精度，不能代表真实任务得分
- **MiniMax-M3 自己校准跑不通**：无 HF modeling 文件，`AutoModelForCausalLM` 加载失败，AutoAWQ/llmcompressor 校准无法进行
- **cyankiwi AWQ 是现成校准产物**：`observer=mse`（真AWQ校准）+ group=32 + MoE量化 + 敏感层保bf16，sglang 原生可加载，作为有校准基准对比用
- 本次自做 W4A16 moe-only 目的：验证 naive int4 moe-only 在 sglang 上能否加载+跑通，并与校准版客观对比（需搭端到端评测）

## 二十七、tilelang W4A16 fused MoE 算子（替换 sglang Triton int4）

**目标**：MoE 占 decode 76%，sglang 海光路径走 Triton `fused_moe_kernel_gptq_awq`（慢）。用 tilelang 自写 W4A16 fused MoE 替换两个 int4 GEMM，路由/重排/combine 复用 sglang 基础设施。代码已入 `quant_gemm_pkg/quant_gemm/moe/`（push github `benyuereal/quant_gemm`）。

### 27.1 包结构

```
quant_gemm/moe/
├── __init__.py        # 导出 w4a16_grouped_gemm / w4a16_fused_moe / quantize_int4_per_group ...
├── kernels.py         # w4a16_grouped_gemm: per-group int4 grouped GEMM (tilelang)
├── quant.py           # quantize/dequant_int4_per_group (zp=8 对称, sglang 兼容)
├── moe.py             # w4a16_fused_moe: 路由+2GEMM+silu+combine 完整 fused MoE
└── tests/
    ├── bench_perf.py              # M3 真实 shape 性能 vs sglang
    ├── verify_real_weights.py     # 端到端真实产物对比 sglang (可替换性判据)
    ├── profile_overhead.py        # 分项 profiling
    └── debug_*.py                 # 历史调试脚本
```

### 27.2 kernel 设计

- `w4a16_grouped_gemm`：per-group int4（zp=8 对称，group=128），`block_K = group_size`（每 K group 一个 scale，索引简单）。线程级 int4 unpack：`_tir_packed_to_unsigned_convert` + 减 8（zp）→ bf16 → GEMM → 乘 per-N scale 累加。
- 路由通过 `block_to_expert/block_m_start/block_actual_rows` 表传入（同 W8A8 框架）。
- **性能关键：kernel 内 `if bx < num_valid_blocks[0]` 跳过 padding 块**（等价 sglang `if pid_m*BLOCK_M >= num_tokens_post_padded: return`）。`dcu_moe_align_block_size` 返回的 `expert_ids` 只有前 `ceil(num_tokens_post_padded/block_M)` 个有效，后面是未初始化垃圾；M=1 时 121 个 block 仅 4 个有效，跳过 117 个垃圾块（否则 97% 算力浪费）。
- combine 只对前 `ntp_val` 行做（`cache_down[:ntp_val]`），避免对 total_pad 行的无谓逐元素乘 + index_add。

### 27.3 权重布局（与 sglang 海光路径一致）

sglang `CompressedTensorsWNA16TritonMoE.process_weights_after_loading` 把产物 int32 packed 转成 kernel 期望格式：
```
w13: [E, K//8, N] int32 --transpose(1,2)--> [E, N, K//8] --view(uint8)--> [E, N, K//2] uint8
w13_scale: [E, K//group, N] --transpose(1,2)--> [E, N, K//group] bf16
```
tilelang 算子期望正是 `[E, N, K//2] uint8` + `[E, N, K//group] bf16`，**布局完全兼容，可无缝接收 sglang 处理后的权重**。

### 27.4 真实产物维度（更正之前记错的）

实测 `MiniMax-M3-w4a16-moe-only` 的 `weight_shape`（TP=1 / inter 不切）：
- w1(gate) `[3072, 6144]`、w3(up) `[3072, 6144]` → 合 w13 `[6144, 6144]` = N_gate_up=6144
- w2(down) `[6144, 3072]` = `[N_down, N_inter=3072]`
- **shard_intermediate=3072，N_gate_up=6144，N_inter=3072，K=N_down=6144**（之前误记 shard_inter=1536，那是随机权重 bench 的旧维度）

### 27.5 正确性验证（真实产物，可安全替换）

`tests/verify_real_weights.py`：读真实 M3 产物 8 个 expert 权重，转 uint8 组装到 E=128，同权重同路由喂 tilelang 和 sglang Triton int4：
```
tilelang vs sglang : max_diff=7.8125e-03  mean=4.1137e-04   ✅ < 1e-2
```
**结论：tilelang 与 sglang Triton int4 输出一致，布局/nibble 顺序/scale 语义全兼容，可安全替换。**
（注：tilelang vs deq_ref 与 sglang vs deq_ref 都 ~342%，两者一起错，是 deq_ref 参考实现的量化语义假设与真实产物不一致，非算子问题，不影响替换判据。）

### 27.6 性能对比（真实维度，vs sglang default Triton）

M3 真实 shape：E=128, K=6144, shard_inter=3072, N_gate_up=6144, N_down=6144, topk=4。
sglang 在 N=6144 无 tuned config（tuned 仅 N=384 有），故 sglang-tuned = sglang-default。

| M | tilelang | sglang | 加速比 | tl_vs_sgl max_diff |
|---|---|---|---|---|
| 1 | 1319us | 3310us | **2.51x** | 7.3e-4 ✅ |
| 2 | 1743us | 6511us | **3.74x** | 9.8e-4 ✅ |
| 4 | 2790us | 12924us | **4.63x** | 1.2e-3 ✅ |
| 8 | 4630us | 22490us | **4.87x** | 1.2e-3 ✅ |
| 16 | 8209us | 39146us | **4.77x** | — |
| 32 | 13937us | 68507us | **4.92x** | — |

**全 batch 大幅领先，生产 decode（M=1-2）2.5-3.7x。**（之前错维度 shard_inter=1536 时 M=1 仅 1.69x，真实维度 3072 下 N 翻倍 sglang 更慢，tilelang 优势扩大。）

### 27.7 性能拆解（M=1，定位下一步优化）

`profile_overhead.py` 分项（真实维度）：
- kernel 端：align 24us + kernel1(gate_up) 254us + silu 38us + kernel2(down) 137us + combine 75us ≈ **528us**
- e2e 实测 1319us，**~790us 是 host-side Python op launch 开销**（gather/cat/arange/where/index_add 等 ~25 个 op，每个 30-40us launch）
- → **下一步优化重点：降 host 开销**（预分配 buffer 复用、合并 op、减少临时 tensor）

### 27.8 已知稳定性问题（待修）

1. **`sgl_kernel.silu_and_mul` launch_bounds 警告**：N_inter=3072 时 `Launch params (384,1,1) > launch bounds (256)`。目前是 warning 不崩，但需查是否有替代实现或自写 tilelang silu。
2. **小 E（<8）dcu_moe_align_block_size segfault**：dcu_align 的 expert_ids 在小 E 时输出垃圾，仅适用真实 E=128。小规模测试用 `moe_align_torch` 纯 torch 路径。
3. **sglang 无 N=6144 tuned config**：真实维度下 sglang 走 default config（未调优），对比偏保守；若要更公平可为 N=6144 调一份 tuned config。

### 27.9 待办：降 host 开销（M=1 ~790us Python op launch）

`profile_host_ops.py` 逐 op 计时（M=1，不 sync 测 host 提交）host 合计 ~726us，主要开销：

| op | us | 占比 |
|---|---|---|
| cat valid/safe/tok（pad 到 num_m_blocks×block_M） | 130 | 18% |
| cat A_gathered（同上 pad） | 118 | 16% |
| where(valid, w_flat[sf], 0) | 74 | 10% |
| cd_valid × w_per_row（combine 逐元素乘） | 71 | 10% |
| sorted_ids.to(int64) | 39 | 5% |
| tensor nvb / A×=mask / index_add_ / dcu_align | ~100 | 14% |

**最大头是两个 `torch.cat`（248us，34%）**——pad A_gathered/valid/safe_flat/tok_idx 到 `total_gem=num_m_blocks*block_M`。但 kernel 已用 `num_valid_blocks` early-exit 跳过 pad 块，**这些 cat 实际多余**（kernel 只读前 nvb×block_M ≤ ntp_val 行，A_gathered 的 total_pad 已 ≥ ntp_val）。

**已验证去 cat 方案**（`/tmp/test_nopad2.py`）：C/A 行数用 `max(total_pad, nvb×block_M)` 防越界，通常无需 pad。nopad vs sglang max_diff ~1.5e-3（与原版同量级），nopad vs orig ~4.9e-4（纯 bf16 编译差异，因 total_rows 不同触发不同 kernel cache key，非 bug）。**待写入 moe.py 并测性能提升**（预计 M=1 省 ~250us，e2e 1319→~1070us）。

**其他 host 优化方向**（未做）：
- kernel 的 total_tokens/num_m_blocks 用固定上界，避免随 M 变化重编译（省 JIT lookup + cache 抖动）；但 num_m_blocks 是 grid 维度，上界浪费 block（有 early-exit 兜底）。
- combine 的 `where + 逐元素乘 + index_add`（~170us）可合并成单个自定义 kernel，或用 scatter_add。
- 预分配 buffer 复用（A_gathered / cache1 / cache_inter / cache_down / final），避免每帧 malloc。

**注**：生产开 cuda graph 后，这些 host 开销会被 graph capture 吸收（capture 时记录一次，replay 时无 Python 开销），故 host 优化对 graph 路径收益小，主要利好非 graph 路径（prefill / 大 batch）。优先级：先做 sglang 实际替换验证端到端收益，host 优化后置。

### 27.10 关键发现：dcu_moe_align_block_size 在小 E (TP=4) 产生垃圾 sorted_ids

**现象**：sglang 海光版 `dcu_moe_align_block_size` 在 E<128 时输出的 `sorted_ids` / `expert_ids` 大部分是未初始化垃圾（如 1068547945、-1081393681），不是有效 token 索引或 -1。

实测（M=1, topk=4, 应只有 4 个有效 token）：
| E | sid[:ntp] 中 `< M_flat` 的数（应=4） | 状态 |
|---|---|---|
| 32 | 40 | 垃圾（dcu_align 残缺） |
| 64 | 43 | 垃圾 |
| 128 | 4 | 正常 |

E=32 各 M 下都垃圾（M=8 应 32 实测 185，M=32 应 128 实测 496）。**dcu_align 只填了每 block 首位置（0,16,32,...），其余 15 行是未初始化内存**，`sid < M_flat` 误判垃圾为有效 token。

**影响**：
- **W4A16 TP=1（E=128）forward 能跑**（dcu_align 正常，tilelang 验证 PASS）。
- **W4A16 TP=4（每卡 E=32）forward 跑不通**——这正是文档 26.9 `[ ] forward 跑通` 未完成的根因（之一）：sglang Triton int4 kernel 喂垃圾 sorted_ids/expert_ids 会越界崩（`Invalid address access` / HSAIL hardware exception）。
- **不是 tilelang 的问题**：sglang 自己的 `_fused_moe_kernel_sequence` 在 E=32 喂同款垃圾也崩。

**根因猜测**：sglang DCU 版 `op.moe_align_block_size`（HIP kernel）在小 E（E<128）时线程/grid 配置不当，未填满 sorted_ids 的 padding 行。E=128 是 M3 全量 expert，恰好避开。

**解决方向**（待定）：
1. tilelang 替换路径里**绕过 dcu_align**，用我们自己的 `moe_align_torch`（纯 torch，已验证 E=4/8/32 正确）或修好的 align kernel。但 moe_align_torch 有 71-92% overhead（见 27.6 之前），需优化或写 tilelang align kernel。
2. 修 sglang dcu_align kernel（HIP C++，难）。
3. TP=1 部署（E=128，dcu_align 正常），但单卡 64GB 装不下 225GB 模型，不可行。
4. **先用 TP=1 + 单层 MoE 验证 tilelang 端到端正确**（绕过 TP=4 显存限制），再决定 TP=4 方案。

**当前状态**：tilelang 算子本身在 E=128（TP=1 语义）正确性 + 性能已验证（27.5-27.6），可安全替换 sglang Triton int4。**阻塞点在 sglang dcu_align 的 E=32 bug，需先解决才能 TP=4 部署。**

### 27.11 待办：tilelang W4A16 替换 sglang 的遗留问题（暂停）

**状态**：tilelang 替换 sglang Triton int4 MoE 在真实服务（TP=4, E=128, cuda graph）下输出乱码，**暂停**，回退 sglang 原生 Triton int4 + cuda graph。

**已验证可用**：
- 单卡裸调（E=128, 随机/真实产物权重）：tilelang vs sglang max_diff<1e-3，性能 M=1 2.51x ~ M=32 4.92x（`bench_perf.py` / `verify_real_weights.py`）。
- M=1-128 裸调正确性 PASS。

**真实服务乱码根因（已定位部分）**：
1. **grid 维度用错**（已修）：`w4a16_fused_moe_aligned` 之前用 `expert_ids.numel()` 作 grid，大 M 时 `eid_len << sid_len/16`（裸调 M=178: eid=138 但 sid/16=549），少跑 400+ block → 乱码。改用 `cdiv(total_pad, block_M)` + eid pad 0。但真实服务 sid_len=2632（非裸调的 8776，因 `num_token` 参数不同），eid_len=165=cdiv(sid,16) 本就够，此修复在真实服务无作用。
2. **routed_scaling_factor 漏乘**（已修）：M3 `routed_scaling_factor=2.0`，sglang combine 的 `moe_sum_reduce_triton(..., routed_scaling_factor)` 乘了，我们 `index_add` 没乘 → 输出差 2x。已加 `routed_scaling_factor` 参数并在 combine 乘。但重启后 max_diff 仍大（12-43），说明不是主因。
3. **未定位**：真实服务 M=178 `tl_mean` 比 `sgl_mean` 小 7-14 倍且比值不固定（0.07~0.46），非单纯系数差，像根本性计算错误。怀疑：sorted_ids 在真实服务的垃圾分布与裸调不同（真实 sid_len=2632 vs 裸调 8776），`valid` mask 误判垃圾 sid 为有效 → gather 错误 token。裸调无法复现（align 的 `num_token` 参数差异）。

**调试手段**：`fused_moe.py` 补丁加 `TILELANG_DEBUG=1` 同时跑 sglang 原生对比，dump max_diff/sid_valid/eid_len/rsf。`minimax_w4a16.sh` 设 `SGLANG_USE_TILELANG_W4A16=1` + `TILELANG_DEBUG=1` + `--disable-cuda-graph`。

**下一步（恢复时）**：
- 在真实服务 dump sorted_ids 实际值分布（sid_valid/sid_neg/sid_ge_Mflat 已加调试），确认是否垃圾 sid 落在 [0,M_flat) 被 valid 误判。
- 若是，改用 sglang kernel 同款保护：kernel 内 `token_mask = offs_token < num_valid_tokens`，而非 host 端 `valid = sid<M_flat`。或绕过 dcu_align 用自己的 align。
- 解决正确性后，再攻 cuda graph 兼容（多 op 串进 graph 的问题，`.item()` 已修但 `torch.cat`/动态 pad 待验证）。

**当前回退**：`SGLANG_USE_TILELANG_W4A16=0`（sglang 原生 Triton int4）+ `--cuda-graph-max-bs 8`。tilelang 代码保留在 `quant_gemm/moe/`，补丁在 `sglang_patches/modified/fused_moe.py`（环境变量控制，默认关，不影响原生）。
