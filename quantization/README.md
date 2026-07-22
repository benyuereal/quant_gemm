# MiniMax-M3 W8A8 量化脚本

本目录含 MiniMax-M3 量化的全部脚本与参考文件。量化方案详见 `docs/MiniMax-M3-量化工作记录.md` 第十五章。

## 文件

| 文件 | 用途 |
|---|---|
| `minimax_m3_w4a8.py` | 量化主脚本（手写，不依赖 llmcompressor）。支持 `--quant-type int4/int8/fp8` + `--moe-only` |
| `quantize_minimax_m3_w4a8.sh` | 8 卡并行量化封装（调上面的 .py） |
| `minimax.sh` | sglang 启动脚本（加载量化后模型，含清缓存 + 日志重定向） |
| `glm5.1-channel-int4-w4a8.config.json` | 参考的 GLM-5.1 W8A8 config（量化格式参照对象） |

## W8A8 moe-only 量化（最终方案）

```bash
# 1. 量化（8 卡并行，约 3 分钟）
bash quantize_minimax_m3_w4a8.sh
# 或直接调 python:
python3 minimax_m3_w4a8.py \
    --input-path /models/MiniMax/MiniMax-M3 \
    --output-path /models/MiniMax/MiniMax-M3-w8a8-moe-only \
    --quant-type int8 \
    --moe-only
```

产出 `/models/MiniMax/MiniMax-M3-w8a8-moe-only`（412GB），格式 compressed-tensors W8A8（per-channel int8 weight + per-token dynamic int8 act），只量化 MoE expert，其余 bf16。

## sglang 启动

```bash
bash minimax.sh
# 自动: 停残留 sglang → 清 triton/torchinductor 缓存 → 启动 → 日志 /models/sglang_serve.log
# 测试: curl -N http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
#        -d '{"model":"default","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```
sglang 需要 6 处补丁（见 `../sglang_patches/`），否则海光 DCU 上 W8A8 MoE 会 raise / kernel 报错。

## 参数说明

`minimax_m3_w4a8.py`:
- `--quant-type int8`：W8A8（per-channel int8 weight）。`int4`=W4A8（MoE expert int4），`fp8`=FP8。
- `--moe-only`：只量化 MoE expert（`block_sparse_moe.experts.*.w1/w2/w3`，96.7% 权重），attention/shared/dense mlp 留 bf16。不加则全量化。
- `--input-path` / `--output-path`：原始模型 / 量化产物目录。

## 注意

- 脚本里的 `/models/...` 路径是本机（容器）路径，换环境需改 `MODEL_DIR`/`OUTPUT_DIR`/`MODEL_PATH`。
- **绝不能在 sglang 容器装 llmcompressor**（会升级海光定制 torch 致环境崩，见工作记录第十二章）。本脚本手写量化，不依赖 llmcompressor。
- 量化产物需配合 sglang 补丁才能在海光 DCU 上跑（`../sglang_patches/`）。
