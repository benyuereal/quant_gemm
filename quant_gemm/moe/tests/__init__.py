"""quant_gemm.moe 测试与基准脚本.

- test_correctness.py:  功能正确性 (小规模 E=4/8 + M3 真实 shape)
- bench_perf.py:        M3 真实 shape 性能 vs sglang tuned Triton int4
- profile_overhead.py:  分项 profiling (定位开销来源)
- bench_tilelang_vs_sglang.py: tilelang vs sglang 端到端对比
- debug_*.py:           历史调试脚本 (定位路由表/多 expert bug, 已归档)
"""
