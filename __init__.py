"""
ComfyUI Attention Optimizer

Automatically benchmark and optimize the attention mechanism in diffusion models.
Attention is the most expensive operation in transformer-based models (40-70% of generation time).
This plugin benchmarks all available backends (SDPA, Flash Attention, SageAttention, xFormers, Comfy Kitchen)
and applies the fastest one for your GPU.

Nodes:
- AttentionOptimizer: Main node - benchmarks, caches results, and auto-applies optimal backend
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
