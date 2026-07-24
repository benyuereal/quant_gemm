"""Patch transformers to accept minimax_m3_sparse layer type for MiniMax-M3.

sglang dev build supports MiniMax-M3 but transformers 5.6.0 has not yet
registered the custom layer type in ALLOWED_LAYER_TYPES. This sitecustomize
module is loaded automatically via PYTHONPATH and patches the tuple before
any model config validation runs.
"""
try:
    import transformers.configuration_utils as _cu
    if "minimax_m3_sparse" not in _cu.ALLOWED_LAYER_TYPES:
        _cu.ALLOWED_LAYER_TYPES = _cu.ALLOWED_LAYER_TYPES + ("minimax_m3_sparse",)
except Exception:
    pass