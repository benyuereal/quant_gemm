# MiniMax-M3 EAGLE3 投机解码工作记录
> 创建日期: 2026-07-23
> 目标: 在海光 DCU (gfx928) 上给 W4A16 moe-only 量化的 MiniMax-M3 接 EAGLE3 投机解码
> 关联: `MiniMax-M3-量化工作记录.md` (量化本体) ｜ sglang dev12695 (DCU dtk2604 build)

---

## 一、背景: MTP vs EAGLE3

最初需求是"给 MiniMax-M3 开 MTP"。澄清后明确:

| | MTP | EAGLE3 |
|---|---|---|
| Draft 来源 | 模型自带的 MTP head 权重 | 独立训练的 draft head checkpoint |
| 是否需训练 | 否(权重随模型发布) | 是(SpecForge/TorchSpec) |
| M3 是否有 | **没有** — config 有 `num_mtp_modules:7` 但 safetensors 里 0 个 `model.mtp.layers.*` 权重 | 需外部 draft |

**结论**: M3 这个 checkpoint 没有自带 MTP 权重(config 里 `num_mtp_modules:7` 只是架构能力声明),原生 MTP 路线走不通。只能用 EAGLE3(独立训练的 draft head)。参考的 MiniMax-M2.5 文档(`thoughtworks/MiniMax-M2.5-Eagle3`)用的也是 EAGLE3,不是 MTP。

---

## 二、Draft head 来源与结构

### 2.1 模型
- **仓库**: `Inferact/MiniMax-M3-EAGLE3` (modelscope), 已下到 `/models/Inferact/MiniMax-M3-EAGLE3`
- **文件**: `model.safetensors` (6.5GB), `config.json`
- **训练目标**: `MiniMaxAI/MiniMax-M3-MXFP8` (FP8 精度!) — 用 TorchSpec 训
- **aux hidden layers**: (2, 30, 57) + final layer

### 2.2 架构 (`LlamaForCausalLMEagle3`)

| 维度 | 值 | 与 M3 target 对齐? |
|---|---|---|
| hidden_size | 6144 | ✅ 一致 |
| vocab_size | 200064 | ✅ 一致 |
| num_hidden_layers | 1 (单层 draft) | — |
| num_attention_heads | 64 | — |
| intermediate_size | 18432 | — |
| fc_norm | true (3 个 aux 各一个 RMSNorm) | — |
| norm_output | true | — |
| draft_vocab_size | 200064 (与 target 同, 不裁剪) | — |
| tie_word_embeddings | false (运行时复用 target 的 embed/lm_head) | ✅ |

### 2.3 实际权重键 (17 个)
```
embed_tokens.weight        # 运行时被 target 覆盖
fc.weight                  # 3 个 aux hidden 拼接后投影
fc_norm.0/1/2.weight       # 3 个 aux 各一个 norm
layers.0.{input_layernorm, post_attention_layernorm, hidden_norm}.weight
layers.0.self_attn.{q,k,v,o}_proj.weight
layers.0.mlp.{gate,up,down}_proj.weight
lm_head.weight             # 运行时被 target 覆盖
norm.weight
```
结构与 sglang `srt/models/llama_eagle3.py` 的 `LlamaForCausalLMEagle3` 完全匹配(`fc_norm`/`norm_output`/`draft_vocab_size` 字段均覆盖)。

### 2.4 README 性能 (target=MXFP8, vLLM, TP=4, num_spec_tokens=3, greedy topk=1)
- GSM8K/HumanEval/MATH500: 平均接受长度 ~3.5, 接受率 ~84%
- MT-Bench/低熵合成数据: 接受长度 ~2.7, 接受率 ~57%
- **注意**: 这些是 MXFP8 target 的数字。我们用 W4A16 moe-only target, 精度不同, 接受率会偏低。

---

## 三、量化模型与 sglang 的 EAGLE3 支持核查

### 3.1 量化模型 (W4A16 moe-only) 与 draft 对齐
| 检查项 | draft 期望 | W4A16 产物 | 结论 |
|---|---|---|---|
| hidden_size | 6144 | 6144 | ✅ |
| vocab_size | 200064 | 200064 | ✅ |
| architectures | M3 | `MiniMaxM3SparseForConditionalGeneration` (VL 类) | ✅ |
| tie_word_embeddings | false | false | ✅ |
| MoE 量化范围 | — | moe-only (embed/lm_head 原精度) | ✅ 共享无精度错配 |

### 3.2 sglang EAGLE3 target 侧接口 (核查 minimax_m3.py / minimax_m3_vl.py)

sglang 里 M3 有**两个模型类**:

| 类 | 文件 | EAGLE3 接口 |
|---|---|---|
| `MiniMaxM3SparseForCausalLM` (text-only) | minimax_m3.py | **有** `set_eagle3_layers_to_capture` / `get_embed_and_head` / aux-aware forward |
| `MiniMaxM3SparseForConditionalGeneration` (VL) | minimax_m3_vl.py | **没有** — 这就是量化模型加载的类 |

draft 侧: sglang `llama_eagle3.py` 能加载该 draft; spec_registry 里 EAGLE3/NEXTN 均为保留算法; server_args 有完整 `--speculative-*` flag。

---

## 四、踩坑: sglang M3 EAGLE3 两处缺口 (上游 bug)

### 缺口 1: VL 类缺 EAGLE3 接口 (直接崩溃)
首次启动报错:
```
AttributeError: 'MiniMaxM3SparseForConditionalGeneration' object
has no attribute 'set_eagle3_layers_to_capture'
```
原因: `model_runner.init_aux_hidden_state_capture()` (model_runner.py:904) 调
`model.set_eagle3_layers_to_capture()`, 但 VL 类没这个方法 (只在 text-only 类上)。
量化模型 config.architectures = VL 类, 所以加载 VL 类 → 崩溃。

### 缺口 2: M3 aux 捕获链路本身是断的 (更隐蔽)
即便补上接口, `MiniMaxM3Model.forward` (minimax_m3.py:1016-1020) 用
`getattr(layer, "_is_layer_to_capture", False)` 判断要不要把 `aux_hidden_states`
列表传进 decoder layer。但**全 sglang 没有任何代码把 `layers_to_capture` 转成
layer 上的 `_is_layer_to_capture=True`**。

对比: `qwen3_moe.py:939` 的 `set_eagle3_layers_to_capture` 里有
`setattr(self.layers[layer_id], "_is_layer_to_capture", True)` —— M3 缺这一步。
即 text-only 类的 EAGLE3 路径也是断的 (设了 `layers_to_capture` 但 layer 没被标记 →
`aux_hidden_states` 永远空 → 返回纯 hidden, 不是元组)。

> 这说明 sglang 上游对 M3 的 EAGLE3 支持是半成品: 接口写了, 但 aux 捕获链路没接通。

---

## 五、修复: 直接改 site-packages (沿用 W4A16 patch 方式)

### 5.1 备份与改动位置
- **原文件备份**: `/models/sglang_backup/minimax_m3_vl.py` (与 W4A16 patch 的 backup 同目录)
- **改动文件**: `/usr/local/lib/python3.10/dist-packages/sglang/srt/models/minimax_m3_vl.py`
- **曾尝试 monkey-patch** (`sglang_patches/added/minimax_m3_vl_eagle3.py`), 但 sglang TP=4 起 4 个 scheduler 子进程, monkey-patch 只在主进程生效, 子进程仍是原类 → 报错不变。故改为直接改文件 (子进程 fork 后继承改动)。

### 5.2 改动 (全部针对 VL 类 `MiniMaxM3SparseForConditionalGeneration`)

| 改动 | 内容 | 依据 |
|---|---|---|
| 补 `set_eagle3_layers_to_capture` | 转发给 `self.model`, 设 `self.model.layers_to_capture`, 并对每个待捕获 layer `setattr(_is_layer_to_capture=True)` (修复缺口 2) | 仿 qwen3_moe.py:939 |
| 补 `get_embed_and_head` | `return self.model.embed_tokens.weight, self.lm_head.weight` | 仿 text-only 类 minimax_m3.py:1130 |
| 补 `capture_aux_hidden_states` 属性 | set_eagle3 时设 True, 默认 False | — |
| 重写 `forward` | capture 模式下解包 `general_mm_embed_routine` 返回的 `(hidden, aux)` 元组, 把 aux 传给 logits_processor | 仿 text-only 类 minimax_m3.py:1147-1154 |

**默认 aux layers**: `set_eagle3_layers_to_capture(None)` → `[2, num_layers//2, num_layers-3]`
= `[2, 30, 57]` (N=60), 与 draft 训练用的 (2,30,57) 一致。draft config 无 `eagle_config`
字段 → model_runner 走 except → `layer_ids=None` → 用默认 → 自动对齐, 无需改 draft config。

### 5.3 影响范围
- 仅 `capture_aux_hidden_states=True` (开了 EAGLE3) 时走新 forward 路径
- 平时 (普通 W4A16 推理) `capture_aux_hidden_states=False`, 新 forward 行为与原 forward 完全一致 (都调 `general_mm_embed_routine` + `logits_processor`, 只是多一个 `getattr` 判断)
- **不改 site-packages 原文件**, sglang 升级不丢 patch; 去掉 import 即完全回滚

### 5.4 已验证 (不实际起服务)
- ✅ patch import 生效, VL 类有了 `set_eagle3_layers_to_capture` / `get_embed_and_head`
- ✅ mock 验证: layer 2/30/57 被标记 `_is_layer_to_capture=True`, layer 0 未标记; 显式 layer_ids 走 +1 偏移
- ✅ sglang 参数解析: EAGLE3 归一化 num_steps=3 / topk=1 → num_draft_tokens=4 (=steps+1) 正确
- ✅ 幂等: 重复 import 不报错

### 5.5 未验证 (需实际起服务, 加载 225G ~3min)
- DCU 上 draft worker 实际加载 (可能还有 DCU 特定 cuda graph / attention backend 问题)
- 实际接受率 (预期因 MXFP8 vs W4A16 精度差异偏低)
- 实际加速比 (W4A16 瓶颈是 MoE int4 GEMM, spec decode 可能加速比打折 — 见 `minimax-m3-perf-bottleneck`)

### 5.7 实测踩坑 (启动后)

跑通 patch 后, 启动又依次撞到两个 sglang 上游与 EAGLE3 的兼容 bug, 均已修 (直接改 site-packages + 备份):

**坑 A: `MiniMaxM3Model` 无 `.config` 属性**
`set_eagle3_layers_to_capture` 里取 `self.model.config.num_hidden_layers` 报 `AttributeError`.
原因: text-only 类 `self.config` 是 text_config, 但 VL 类 `self.config` 是顶层 VL config (text 在 `self.config.text_config`), 且 `MiniMaxM3Model` 构造时不保留 `.config` 引用.
修法: 改成 `self.config.text_config.num_hidden_layers` (VL 类自身代码也是这么取的).

**坑 B: EAGLE3 TARGET_VERIFY 时 sparse backend 取 `max(None)` 崩**
`minimax_sparse_backend.py:init_forward_metadata` 用 `mode.is_extend()` 判断, TARGET_VERIFY 的 `is_extend()==True`, 故进 extend 分支取 `forward_batch.extend_seq_lens_cpu`. 但 `ForwardBatch.init_new` 把 TARGET_VERIFY 归到 **decode 分支** (`if is_decode() or is_target_verify()`), 不填 `extend_seq_lens_cpu` → 保持 None → `max(None)` 报 `TypeError: 'NoneType' object is not iterable`.
根因: sparse backend 用 `is_extend()` 分类, init_new 用 `is_target_verify()` 分类, 两者对 TARGET_VERIFY 的归类不一致 (sglang 上游 bug, 非 MiniMax 专属, 但只有 sparse backend 的模型触发).

**坑 C: 同根 — `forward_extend` 用 `extend_seq_lens.device` 崩**
修完坑 B 后, `forward_extend` (line 216) 又取 `forward_batch.extend_seq_lens.device` → None 崩. 同根: TARGET_VERIFY 走 decode 分支, **所有** extend 字段 (`extend_seq_lens` tensor / `extend_seq_lens_cpu` list / `extend_prefix_lens`) 都没填. 且基类 `AttentionBackend.forward` 也用 `is_decode()` 分发, TARGET_VERIFY → `forward_extend`, 所以所有 backend 在 verify 都走 extend 路径, 只是多数 backend 不直接读这些字段所以没崩.
修法 (坑 B+C 合一): 在 `init_forward_metadata` 里, 若 `extend_seq_lens is None`, 用 `spec_info.draft_token_num` 一次性补全 `extend_seq_lens` (tensor) / `extend_seq_lens_cpu` (list) / `extend_prefix_lens` / `extend_prefix_lens_cpu`. 改动文件: `minimax_sparse_backend.py`, 备份 `sglang_backup/minimax_sparse_backend.py`.

**坑 D: verify 时 seq_lens 语义 — 输出乱码/复读的真正根因**
修完坑 B/C 服务能跑, accept rate 正常 (0.3~1.0), 但**输出乱码 + 复读** (`<mm:think>...user has been thinking...`), 而纯 W4A16 同 prompt 正常.
根因: sparse backend `forward_extend` 给 kernel 的契约是 `seq_lens = prefix + extend`, `prefix_lens = prefix`. 但 verify 路径里:
- `prepare_for_v2_verify` **不给 `batch.seq_lens` 加 draft** (对比 `prepare_for_extend_to_fill_draft_kvcache` 会 `seq_lens += draft`). verify 后 scheduler 用 `seq_lens + accept_lens` 更新, 所以 verify 时 `seq_lens` 必须保持 **prefix** (不含 draft).
- 标准 backend (triton/flashinfer) 不直接读 `extend_seq_lens`, 用 `spec_info.positions`, 所以没暴露此问题. M3 sparse backend 直接读, 就错.
- 我第一版补 `extend_prefix_lens = seq_lens - draft` 是**错的** (基于"seq_lens 已加 draft"的错误假设), 导致 prefix 少算 draft, attention 位置错 → accept 的 token 写错位置 → 乱码/复读.

正确修法 (对照 `prepare_for_extend_to_fill_draft_kvcache` 的自洽三元组):
- `extend_prefix_lens = seq_lens` (verify 时 seq_lens 就是 prefix, 不减 draft)
- `extend_seq_lens = draft_token_num`
- kernel 的 `seq_lens` = `prefix + draft` (在 `forward_extend` 里用 `prefix_lens + extend_seq_lens` 重建, 不改 `forward_batch.seq_lens` 以免破坏后续 `+accept_lens`)
- `_max_seqlen_k` = `max(prefix + draft)`
- 用 `seq_lens >= prefix + extend` 判断是 normal extend (seq_lens 已含 extend, 不重建) 还是 verify (重建), 两种情况都对.

> 这几个坑 fork (tails-mpt) 也没修 — fork 做的是 M2 (text-only, 无 VL 类问题) 且不用 M3 的 sparse backend. 属于 M3 + EAGLE3 + sparse attention 三者组合的独有缺口.

---
参考 `https://github.com/tails-mpt/sglang` (MiniMax-M2.5 Eagle3 的 fork) 的相关 commit:

| fork commit | 改动 | 对我们 W4A16 场景 | 我们状态 |
|---|---|---|---|
| `b5927f9a7c` feat: MiniMax-M2.5 Eagle3 support | minimax_m2.py 默认层 [2,30,57]→[1,30,58] + logits_processor aux→bf16 cast | 默认层不适用 (M3=60层, draft 用 (2,30,57)); aux cast 见下行 | 默认层用 [2,30,57] ✅ |
| `b5927f9a7c` / `9049265fb2` logits_processor aux→bf16 cast | aux_hidden_states 拼接后 cast 到 bf16 (FP8 target 产生 float32 aux) | **可能需要** — W4A16 MoE dequant 可能产生 float32 aux, 跑起来若报 dtype mismatch 再加 | ⏳ 暂未加, 待实测 |
| `9049265fb2` llama.py set_embed→bf16 cast | 共享 embed/lm_head 时 cast 到 bf16 (FP8 target embed fp16/f32) | 不触发 — 我们 embed/lm_head 本就是 bf16 (moe-only 不量化) | 不需要 |
| `ea6c44888b` / `b3e73aba38` llama_eagle3 FC/embeds dtype cast | FP8 aux float32 vs bf16 FC 的 dtype 修复 | 不触发 — 我们 aux 是 bf16 | 不需要 |
| `0675f9531c` final-layer aux capture | 循环外捕获 final layer hidden | **不该加** — draft FC in_features=18432=3×6144, 只吃 3 个 aux (2,30,57); 加 final 会变 4 个 → FC 维度不匹配报错 | 正确未加 ✅ |
| `b3e73aba38` multi-layer EAGLE3 (raise→warning) | 允许 num_hidden_layers>1 | 不触发 — 我们 draft num_hidden_layers=1 | 不需要 |
| **M3 VL 类缺接口** (fork 也没做 M3) | set_eagle3_layers_to_capture / get_embed_and_head / aux-aware forward + `_is_layer_to_capture` setattr | **必须** — M3 VL 类缺这些, fork 只做了 M2 (text-only 已有接口) | ✅ 已补 (5.2) |

**结论**: fork 的 EAGLE3 修复大部分针对 **FP8 target** 的 dtype 问题 (float32 aux/embed), 我们 W4A16 (bf16) 多数不触发。唯一可能需要的 `logits_processor aux→bf16 cast` 留待实测 — 若启动后 draft FC 报 `expected mat1 and mat2 to have the same dtype` 再加 (条件触发, 零风险)。M3 VL 类的接口缺失是 fork 也没覆盖的 (fork 只做 M2), 我们已补齐。

---

## 六、启动脚本

**位置**: `/models/minimax_w4a16_eagle3.sh` (端口 8081)

### 6.1 关键设计
- **用 `python3 -c` 包装启动**, 不用 `sglang serve` CLI:
  ```bash
  python3 -c "
  import minimax_m3_vl_eagle3  # EAGLE3 patch (import 即生效)
  from sglang.launch_server import main
  main()
  " --model-path ... --speculative-algorithm EAGLE3 ...
  ```
  确保 patch 在 sglang 加载模型前生效。
  - 注意: `prepare_server_args` (server_args.py:7683) 的 parser 是 `prog="sglang serve"`,
    直接 `parser.parse_args(argv)`, **不需要 `serve` 子命令前缀** (那是 `sglang` CLI 的子命令,
    不是 launch_server 的)。所以 `--model-path` 直接起头。
- `PYTHONPATH` 加上 patch 目录: `/models/quant_gemm_pkg/sglang_patches/added`

### 6.2 EAGLE3 配置 (README 验证过的 greedy 配置)
| 参数 | 值 | 说明 |
|---|---|---|
| `--speculative-algorithm` | EAGLE3 | |
| `--speculative-draft-model-path` | `/models/Inferact/MiniMax-M3-EAGLE3` | M3 不在 DeepSeek MTP 自动复用 model_path 的列表, 必须显式给 |
| `--speculative-num-steps` | 3 | README `num_speculative_tokens=3` (树深度) |
| `--speculative-eagle-topk` | 1 | greedy; topk=1 在 DCU 上最稳 |
| `--speculative-num-draft-tokens` | 不设 | topk=1 时 sglang 自动改成 steps+1=4 (server_args.py:3802) |
| `--mem-fraction-static` | 0.93 | 比纯 W4A16 (0.97) 降, 给 draft (~1.6G/卡) + spec 调度留余量 |
| `--disable-cuda-graph` | 是 | DCU 上 EAGLE draft graph 未验证, 先 eager 跑通 |

### 6.3 与纯 W4A16 脚本 (`minimax_w4a16.sh`) 的差异
- 加 EAGLE3 三参数 + draft path
- mem-fraction 0.97 → 0.93
- 启动方式 `sglang serve` → `python3 -c` 包装 (为加载 patch)
- 端口 8081 (与原脚本同, 互斥)

---

## 七、待办 / 风险

1. **实测 DCU 加载**: 起 `minimax_w4a16_eagle3.sh`, 看日志确认 draft worker 加载 + spec 调度生效 (关注 `Mean accept length` / `draft accept rate` 及有无 DCU 特定报错)
2. **接受率风险**: draft 为 MXFP8 target 训, 我们 W4A16 target 精度不同, 接受率可能 <60%。若太低需用我们的 W4A16 target 重训 draft (TorchSpec/SpecForge 流程, 成本高)
3. **加速比风险**: W4A16 瓶颈是 MoE int4 GEMM (16.9 tok/s), EAGLE3 每个 verify step 也调 MoE, 加速比可能远低于 README 的 1.7-2.1x, 甚至 MoE 调度开销大时更低。需对比 baseline 实测
4. **text-only 类 EAGLE3 仍断**: 本 patch 只修 VL 类。若将来用 text-only M3 (`MiniMaxM3SparseForCausalLM`), 其 `set_eagle3_layers_to_capture` 同样缺 setattr layer 那步, 需另补 (但当前量化产物用 VL 类, 暂不需要)

---

## 八、文件清单

| 文件 | 作用 |
|---|---|
| `/models/Inferact/MiniMax-M3-EAGLE3/` | draft head (下载) |
| `/models/quant_gemm_pkg/sglang_patches/added/minimax_m3_vl_eagle3.py` | EAGLE3 monkey-patch |
| `/models/minimax_w4a16_eagle3.sh` | 启动脚本 (端口 8081) |
| `/models/sglang_w4a16_eagle3.log` | 运行日志 |
