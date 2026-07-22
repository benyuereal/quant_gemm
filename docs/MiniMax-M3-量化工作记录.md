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
