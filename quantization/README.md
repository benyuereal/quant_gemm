# MiniMax-M3 量化脚本（W8A8 / W4A16 moe-only）

本目录含 MiniMax-M3 量化的全部脚本与参考文件。量化方案详见 `docs/MiniMax-M3-量化工作记录.md` 第十五章。

## 文件

| 文件 | 用途 |
|---|---|
| `minimax_m3_w4a8.py` | 量化主脚本（手写，不依赖 llmcompressor）。支持 `--quant-type int4/int8/fp8/w4a16` + `--moe-only` |
| `minimax_m3_w4a16.py` | **W4A16 量化脚本**（标准 `model_free_ptq` 接口，data-free，只量 MoE expert → int4 group=128） |
| `quantize_minimax_m3_w4a8.sh` | 8 卡并行量化封装（调上面的 .py） |
| `minimax.sh` | sglang 启动脚本（W8A8，加载量化后模型，含清缓存 + 日志重定向） |
| `minimax_w4a16.sh` | **W4A16 启动脚本**（4 卡验证版，含 cuda graph bs≤8 + mem-fraction 调参） |
| `glm5.1-channel-int4-w4a8.config.json` | 参考的 GLM-5.1 W8A8 config（量化格式参照对象） |

## 方案对比

两种方案都是 **只量化 MoE expert（96.7% 权重），其余 bf16**：

| 方案 | MoE 权重 | MoE 激活 | 显存 | sglang 海光 kernel | sglang scheme | 状态 |
|---|---|---|---|---|---|---|
| **W8A8 moe-only** | int8 | int8 (dynamic) | 412G | `use_int8_w8a8` ✅ | 补的 `W8A8Int8TritonMoE` | ✅ forward 跑通 |
| **W4A16 moe-only** | int4 | bf16 (不量化) | 225G | `use_int4_w4a16` ✅ | 原生 `WNA16TritonMoE` + 补丁7 | ✅ sglang 加载成功 |

W4A16 比 W8A8 更省显存（225G vs 412G），sglang 原生 scheme（hip 走 `CompressedTensorsWNA16TritonMoE` 纯 Triton），仅需补丁 7（兼容正则层名 target，否则 `KeyError: 'Linear'`）。精度：W4A16 为 naive 无校准（`memoryless_minmax`），真实精度待端到端评测（之前的"12.88%"是中间误差非端到端，不可信；MiniMax-M3 无 modeling 文件，自己校准跑不通）。

## W8A8 moe-only 量化（已跑通）

```bash
python3 minimax_m3_w4a8.py \
    --input-path /models/MiniMax/MiniMax-M3 \
    --output-path /models/MiniMax/MiniMax-M3-w8a8-moe-only \
    --quant-type int8 \
    --moe-only
```
产出 412G，compressed-tensors W8A8（per-channel int8 weight + per-token dynamic int8 act）。

## W4A16 moe-only 量化

```bash
python3 minimax_m3_w4a16.py \
    --input-path /models/MiniMax/MiniMax-M3 \
    --output-path /models/MiniMax/MiniMax-M3-w4a16-moe-only \
    --max-workers 4        # = GPU 数; 8卡被占时用4卡
```
产出 **225G**（6分30秒/4卡），compressed-tensors W4A16（per-group int4 weight group=128 pack_quantized + bf16 激活，naive `memoryless_minmax` 无校准）。
sglang 加载用原生 `CompressedTensorsWNA16TritonMoE`（hip Triton 路径），需补丁 7（`compressed_tensors_wNa16_moe.py`，兼容正则层名 target）。
**产物 config.json 的 targets/ignore 必须用 sglang 层名**（`mlp.experts.*` 而非 HF 名 `block_sparse_moe.experts.*`），详见工作文档 26.5。

## sglang 启动

```bash
# W8A8 (8卡):
bash minimax.sh
# W4A16 (4卡验证):
bash minimax_w4a16.sh
# 自动: 停残留 sglang → 清 triton/torchinductor 缓存 → 启动 → 日志 /models/sglang_w4a16.log
# 测试: curl -s http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" \
#        -d '{"model":"default","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```
W8A8 需 6 处补丁，W4A16 额外需补丁 7（见 `../sglang_patches/`）。

## 参数说明

`minimax_m3_w4a8.py`:
- `--quant-type int8`：W8A8（per-channel int8 weight）。`int4`=W4A8（MoE expert int4），`fp8`=FP8。
- `--moe-only`：只量化 MoE expert（`block_sparse_moe.experts.*.w1/w2/w3`，96.7% 权重），attention/shared/dense mlp 留 bf16。不加则全量化。
- `--input-path` / `--output-path`：原始模型 / 量化产物目录。

## 注意

- 脚本里的 `/models/...` 路径是本机（容器）路径，换环境需改 `MODEL_DIR`/`OUTPUT_DIR`/`MODEL_PATH`。
- **绝不能在 sglang 容器装 llmcompressor**（会升级海光定制 torch 致环境崩，见工作记录第十二章）。本脚本手写量化，不依赖 llmcompressor。
- 量化产物需配合 sglang 补丁才能在海光 DCU 上跑（`../sglang_patches/`）。
