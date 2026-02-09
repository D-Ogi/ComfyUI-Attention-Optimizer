"""
ComfyUI Attention Optimizer

Automatically benchmark and optimize the attention mechanism in diffusion models.
The attention operation is the most expensive part of transformer-based models (40-70% of generation time).
This plugin benchmarks all available backends and applies the fastest one for your GPU.

Supported attention backends:
- basic: Basic einsum-based attention
- sub_quad: Sub-quadratic attention (memory efficient)
- split: Split attention (for low VRAM)
- pytorch: PyTorch's scaled_dot_product_attention (SDPA)
- xformers: xFormers memory efficient attention
- sage_auto: SageAttention auto mode
- sage_cuda: SageAttention CUDA (qk_int8_pv_fp16)
- sage_triton: SageAttention Triton (qk_int8_pv_fp16)
- sage_fp8_cuda: SageAttention CUDA fp8
- sage_fp8_cuda_fast: SageAttention CUDA fp8 optimized
- sage3: SageAttention 3 (Blackwell GPUs)
- flash: Flash Attention
"""
import time
import torch
from . import cache


# ============================================================================
# Backend availability checks
# ============================================================================

def check_xformers_available():
    try:
        import xformers
        import xformers.ops
        return True
    except ImportError:
        return False


def check_sage_available():
    try:
        from sageattention import sageattn
        return True
    except ImportError:
        return False


def check_sage_cuda_available():
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        return True
    except ImportError:
        return False


def check_sage_triton_available():
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        return True
    except ImportError:
        return False


def check_sage_fp8_available():
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        return True
    except ImportError:
        return False


def check_sage3_available():
    try:
        from sageattn3 import sageattn3_blackwell
        return True
    except ImportError:
        return False


def check_flash_available():
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False


def check_triton_available():
    try:
        import triton
        return True, triton.__version__
    except ImportError:
        return False, None


def get_available_backends():
    """Get list of all available attention backends."""
    backends = ["basic", "sub_quad", "split", "pytorch"]

    if check_xformers_available():
        backends.append("xformers")

    # SageAttention variants
    if check_sage_available():
        backends.append("sage_auto")
    if check_sage_cuda_available():
        backends.append("sage_cuda")
    if check_sage_triton_available():
        backends.append("sage_triton")
    if check_sage_fp8_available():
        backends.append("sage_fp8_cuda")
        backends.append("sage_fp8_cuda_fast")

    if check_sage3_available():
        backends.append("sage3")

    if check_flash_available():
        backends.append("flash")

    return backends


def get_backend_info():
    """Get detailed info about available backends."""
    info = {
        "triton_available": False,
        "triton_version": None,
        "cuda_version": None,
        "sage_version": None,
    }

    triton_ok, triton_ver = check_triton_available()
    info["triton_available"] = triton_ok
    info["triton_version"] = triton_ver

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda

    if check_sage_available():
        try:
            import sageattention
            info["sage_version"] = getattr(sageattention, '__version__', 'unknown')
        except:
            pass

    return info


# ============================================================================
# Validation functions - test backends directly to detect silent fallbacks
# ============================================================================

def validate_sage_auto(q, k, v):
    from sageattention import sageattn
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    out = sageattn(q_t, k_t, v_t, is_causal=False, tensor_layout="NHD")
    return out is not None and out.numel() > 0


def validate_sage_cuda(q, k, v):
    from sageattention import sageattn_qk_int8_pv_fp16_cuda
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    out = sageattn_qk_int8_pv_fp16_cuda(q_t, k_t, v_t, is_causal=False, pv_accum_dtype="fp32", tensor_layout="NHD")
    return out is not None and out.numel() > 0


def validate_sage_triton(q, k, v):
    from sageattention import sageattn_qk_int8_pv_fp16_triton
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    out = sageattn_qk_int8_pv_fp16_triton(q_t, k_t, v_t, is_causal=False, tensor_layout="NHD")
    return out is not None and out.numel() > 0


def validate_sage_fp8(q, k, v, fast=False):
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    accum = "fp32+fp16" if fast else "fp32+fp32"
    out = sageattn_qk_int8_pv_fp8_cuda(q_t, k_t, v_t, is_causal=False, pv_accum_dtype=accum, tensor_layout="NHD")
    return out is not None and out.numel() > 0


def validate_sage3(q, k, v):
    from sageattn3 import sageattn3_blackwell
    if q.shape[-1] >= 256 or q.shape[2] <= 1024:
        return False
    out = sageattn3_blackwell(q, k, v, is_causal=False)
    return out is not None and out.numel() > 0


def validate_flash(q, k, v):
    try:
        from flash_attn import flash_attn_func
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q_t, k_t, v_t, dropout_p=0.0, causal=False)
        return out is not None and out.numel() > 0
    except Exception:
        return False


def validate_xformers(q, k, v):
    import xformers.ops
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    out = xformers.ops.memory_efficient_attention(q_t, k_t, v_t)
    return out is not None and out.numel() > 0


def validate_backend(backend, q, k, v):
    """Validate that a backend works without silent fallback."""
    try:
        if backend == "sage_auto":
            return validate_sage_auto(q, k, v), None
        elif backend == "sage_cuda":
            return validate_sage_cuda(q, k, v), None
        elif backend == "sage_triton":
            return validate_sage_triton(q, k, v), None
        elif backend == "sage_fp8_cuda":
            return validate_sage_fp8(q, k, v, fast=False), None
        elif backend == "sage_fp8_cuda_fast":
            return validate_sage_fp8(q, k, v, fast=True), None
        elif backend == "sage3":
            return validate_sage3(q, k, v), None
        elif backend == "flash":
            return validate_flash(q, k, v), None
        elif backend == "xformers":
            return validate_xformers(q, k, v), None
        else:
            return True, None
    except Exception as e:
        return False, str(e)[:100]


# ============================================================================
# Attention wrappers — used for benchmarking and as override functions.
#
# Each wrapper follows the ComfyUI optimized_attention signature:
#   (q, k, v, heads, mask=None, skip_reshape=False, **kwargs)
#
# In the non-skip_reshape path we use view(b, -1, heads, dim_head) so that
# each tensor infers its own sequence length via -1. This correctly handles
# cross-attention where q has a different seq_len than k/v.
# ============================================================================

def get_attention_function(backend):
    """Get the attention function for a backend."""
    from comfy.ldm.modules.attention import wrap_attn
    from comfy.ldm.modules import attention as comfy_attn

    # Basic backends from ComfyUI — already have @wrap_attn
    if backend == "basic":
        return comfy_attn.attention_basic
    elif backend == "sub_quad":
        return comfy_attn.attention_sub_quad
    elif backend == "split":
        return comfy_attn.attention_split
    elif backend == "pytorch":
        return comfy_attn.attention_pytorch
    elif backend == "xformers":
        return comfy_attn.attention_xformers
    elif backend == "flash":
        from flash_attn import flash_attn_func

        @wrap_attn
        def flash_attn_wrapper(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            if skip_reshape:
                q_t = q.transpose(1, 2).contiguous()
                k_t = k.transpose(1, 2).contiguous()
                v_t = v.transpose(1, 2).contiguous()
                out = flash_attn_func(q_t, k_t, v_t, dropout_p=0.0, causal=False)
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.transpose(1, 2).contiguous()
            else:
                b, _, dim_total = q.shape
                dim_head = dim_total // heads
                q = q.view(b, -1, heads, dim_head)
                k = k.view(b, -1, heads, dim_head)
                v = v.view(b, -1, heads, dim_head)
                out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.reshape(b, -1, heads * dim_head)
        return flash_attn_wrapper

    # SageAttention variants
    elif backend == "sage_auto":
        from sageattention import sageattn
        sage_func = sageattn

        @wrap_attn
        def sage_auto_attn(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            in_dtype = v.dtype
            if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            if skip_reshape:
                b, _, _, dim_head = q.shape
                tensor_layout = "HND"
            else:
                b, _, dim_total = q.shape
                dim_head = dim_total // heads
                q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
                tensor_layout = "NHD"
            out = sage_func(q, k, v, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
            if tensor_layout == "HND":
                if skip_output_reshape:
                    return out
                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            else:
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.reshape(b, -1, heads * dim_head)
        return sage_auto_attn

    elif backend == "sage_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        sage_func = sageattn_qk_int8_pv_fp16_cuda

        @wrap_attn
        def sage_cuda_attn(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            in_dtype = v.dtype
            if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            if skip_reshape:
                b, _, _, dim_head = q.shape
                tensor_layout = "HND"
            else:
                b, _, dim_total = q.shape
                dim_head = dim_total // heads
                q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
                tensor_layout = "NHD"
            out = sage_func(q, k, v, is_causal=False, pv_accum_dtype="fp32", tensor_layout=tensor_layout).to(in_dtype)
            if tensor_layout == "HND":
                if skip_output_reshape:
                    return out
                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            else:
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.reshape(b, -1, heads * dim_head)
        return sage_cuda_attn

    elif backend == "sage_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        sage_func = sageattn_qk_int8_pv_fp16_triton

        @wrap_attn
        def sage_triton_attn(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            in_dtype = v.dtype
            if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            if skip_reshape:
                b, _, _, dim_head = q.shape
                tensor_layout = "HND"
            else:
                b, _, dim_total = q.shape
                dim_head = dim_total // heads
                q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
                tensor_layout = "NHD"
            out = sage_func(q, k, v, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
            if tensor_layout == "HND":
                if skip_output_reshape:
                    return out
                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            else:
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.reshape(b, -1, heads * dim_head)
        return sage_triton_attn

    elif backend == "sage_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        sage_func = sageattn_qk_int8_pv_fp8_cuda

        @wrap_attn
        def sage_fp8_attn(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            in_dtype = v.dtype
            if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            if skip_reshape:
                b, _, _, dim_head = q.shape
                tensor_layout = "HND"
            else:
                b, _, dim_total = q.shape
                dim_head = dim_total // heads
                q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
                tensor_layout = "NHD"
            out = sage_func(q, k, v, is_causal=False, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout).to(in_dtype)
            if tensor_layout == "HND":
                if skip_output_reshape:
                    return out
                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            else:
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.reshape(b, -1, heads * dim_head)
        return sage_fp8_attn

    elif backend == "sage_fp8_cuda_fast":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        sage_func = sageattn_qk_int8_pv_fp8_cuda

        @wrap_attn
        def sage_fp8_fast_attn(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            in_dtype = v.dtype
            if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            if skip_reshape:
                b, _, _, dim_head = q.shape
                tensor_layout = "HND"
            else:
                b, _, dim_total = q.shape
                dim_head = dim_total // heads
                q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
                tensor_layout = "NHD"
            out = sage_func(q, k, v, is_causal=False, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout).to(in_dtype)
            if tensor_layout == "HND":
                if skip_output_reshape:
                    return out
                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            else:
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.reshape(b, -1, heads * dim_head)
        return sage_fp8_fast_attn

    elif backend == "sage3":
        from sageattn3 import sageattn3_blackwell

        @wrap_attn
        def sage3_attn(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            in_dtype = v.dtype
            if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            if skip_reshape:
                out = sageattn3_blackwell(q, k, v, is_causal=False).to(in_dtype)
                if skip_output_reshape:
                    return out
                b = out.shape[0]
                return out.transpose(1, 2).reshape(b, -1, heads * q.shape[-1])
            else:
                b, _, dim_total = q.shape
                dim_head = dim_total // heads
                q = q.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous()
                k = k.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous()
                v = v.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous()
                out = sageattn3_blackwell(q, k, v, is_causal=False).to(in_dtype)
                if skip_output_reshape:
                    return out
                return out.permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
        return sage3_attn

    return None


# ============================================================================
# Benchmark functions
# ============================================================================

def benchmark_backend(backend, q, k, v, heads, num_iterations=10):
    """Benchmark a specific backend."""
    import warnings
    import sys
    from io import StringIO

    attn_func = get_attention_function(backend)
    if attn_func is None:
        raise RuntimeError(f"Backend {backend} not available")

    # Suppress warnings during benchmark
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Warmup
            for _ in range(3):
                _ = attn_func(q, k, v, heads, skip_reshape=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = attn_func(q, k, v, heads, skip_reshape=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            return (time.perf_counter() - start) / num_iterations * 1000
    finally:
        sys.stderr = old_stderr


def run_benchmark(head_dim=128, seq_len=4096, num_heads=24, batch_size=1):
    """Run benchmark for all available backends."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    results = {}
    available = get_available_backends()

    # All possible backends in order
    all_backends = [
        "basic", "sub_quad", "split", "pytorch", "xformers",
        "sage_auto", "sage_cuda", "sage_triton", "sage_fp8_cuda", "sage_fp8_cuda_fast",
        "sage3", "flash"
    ]

    # Store backend info
    results["_info"] = get_backend_info()

    for backend in all_backends:
        if backend not in available:
            results[backend] = {
                "time_ms": float('inf'),
                "available": False,
                "error": "Not installed",
                "impl": "N/A"
            }
            continue

        # Determine implementation type
        impl = "pytorch"
        if "triton" in backend:
            impl = "triton"
        elif "cuda" in backend or backend in ["flash", "sage3"]:
            impl = "cuda"
        elif backend == "sage_auto":
            impl = "auto"
        elif backend == "xformers":
            impl = "cuda/triton"

        # Validate first
        valid, error = validate_backend(backend, q, k, v)
        if not valid:
            results[backend] = {
                "time_ms": float('inf'),
                "available": False,
                "error": error or "Validation failed",
                "impl": impl,
                "validated": True
            }
            continue

        # Benchmark
        try:
            time_ms = benchmark_backend(backend, q, k, v, num_heads)
            results[backend] = {
                "time_ms": round(time_ms, 3),
                "available": True,
                "impl": impl,
                "validated": True
            }
        except Exception as e:
            results[backend] = {
                "time_ms": float('inf'),
                "available": False,
                "error": str(e)[:80],
                "impl": impl,
                "validated": True
            }

    # Calculate speedups relative to pytorch
    baseline = results.get("pytorch", {}).get("time_ms", float('inf'))
    if baseline == float('inf'):
        baseline = results.get("basic", {}).get("time_ms", 1.0)

    for backend in results:
        if backend.startswith("_"):
            continue
        data = results[backend]
        if data.get("available") and data.get("time_ms", float('inf')) < float('inf') and baseline > 0:
            data["speedup"] = round(baseline / data["time_ms"], 2)
        else:
            data["speedup"] = 0.0

    # Find best backend
    best = "pytorch"
    best_time = baseline
    for backend, data in results.items():
        if backend.startswith("_"):
            continue
        if data.get("available") and data.get("time_ms", float('inf')) < best_time:
            best = backend
            best_time = data["time_ms"]

    results["_best"] = best
    results["_best_speedup"] = round(baseline / best_time, 2) if best_time > 0 else 1.0
    results["_best_time"] = round(best_time, 3)

    # Cleanup
    del q, k, v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def apply_backend(backend, model):
    """Apply attention backend via model's optimized_attention_override (per-model, reversible)."""
    try:
        attn_func = get_attention_function(backend)
        if not attn_func:
            return False

        def attention_override(func, *args, **kwargs):
            return attn_func.__wrapped__(*args, **kwargs)

        model.model_options["transformer_options"]["optimized_attention_override"] = attention_override
        print(f"[Benchmark] Applied via optimized_attention_override: {backend}")
        return True
    except Exception as e:
        print(f"[Benchmark] apply_backend error: {e}")
        return False


def backend_to_kjnodes_mode(backend):
    """
    Convert internal backend name to kjnodes PathchSageAttentionKJ mode string.

    kjnodes modes:
    - disabled, auto
    - sageattn_qk_int8_pv_fp16_cuda, sageattn_qk_int8_pv_fp16_triton
    - sageattn_qk_int8_pv_fp8_cuda, sageattn_qk_int8_pv_fp8_cuda++
    - sageattn3, sageattn3_per_block_mean
    """
    mapping = {
        "sage_auto": "auto",
        "sage_cuda": "sageattn_qk_int8_pv_fp16_cuda",
        "sage_triton": "sageattn_qk_int8_pv_fp16_triton",
        "sage_fp8_cuda": "sageattn_qk_int8_pv_fp8_cuda",
        "sage_fp8_cuda_fast": "sageattn_qk_int8_pv_fp8_cuda++",
        "sage3": "sageattn3",
    }
    return mapping.get(backend, "disabled")


def get_impl_type(backend):
    """Get implementation type for a backend."""
    if "triton" in backend:
        return "triton"
    elif backend in ["sage_cuda", "sage_fp8_cuda", "sage_fp8_cuda_fast", "flash", "sage3"]:
        return "cuda"
    elif backend in ["xformers"]:
        return "cuda/triton"
    elif backend in ["basic", "sub_quad", "split", "pytorch"]:
        return "pytorch"
    elif backend == "sage_auto":
        return "auto"
    return "unknown"


# ============================================================================
# ComfyUI Node
# ============================================================================

class BenchmarkAndOptimize:
    """
    Benchmark all attention backends and automatically apply the fastest one,
    or force a specific backend without benchmarking.

    Inputs:
    - model: The model to optimize
    - attention_backend: "auto" runs benchmark, or select specific backend to force
    - force_refresh: Re-run benchmark even if cached
    - auto_apply: Apply the selected backend globally
    - seq_len / num_heads: Benchmark parameters

    Available backends:
    - pytorch: PyTorch SDPA (always available)
    - xformers: xFormers memory efficient attention
    - sage_auto, sage_cuda, sage_triton: SageAttention variants
    - sage_fp8_cuda, sage_fp8_cuda_fast: SageAttention FP8
    - sage3: SageAttention 3 (Blackwell GPUs only)
    - flash: Flash Attention 2
    - basic, sub_quad, split: ComfyUI built-in backends

    Outputs:
    - model: Model with attention applied globally
    - best_attention: Applied backend name
    - kjnodes_mode: Compatible mode for PathchSageAttentionKJ node
    - impl_type: Implementation type ("cuda", "triton", "pytorch")
    - speedup: Speedup vs pytorch (0.0 if forced)
    - time_ms: Time per attention call (0.0 if forced)
    - head_dim: Detected head dimension
    - report: Full text report
    """

    # All possible backends for dropdown
    ALL_BACKENDS = [
        "auto",  # Run benchmark and pick best
        "pytorch",
        "xformers",
        "sage_auto",
        "sage_cuda",
        "sage_triton",
        "sage_fp8_cuda",
        "sage_fp8_cuda_fast",
        "sage3",
        "flash",
        "basic",
        "sub_quad",
        "split",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "attention_backend": (cls.ALL_BACKENDS, {"default": "auto"}),
                "force_refresh": ("BOOLEAN", {"default": False}),
                "auto_apply": ("BOOLEAN", {"default": True}),
                "seq_len": ("INT", {"default": 8192, "min": 256, "max": 32768}),
                "num_heads": ("INT", {"default": 24, "min": 1, "max": 128}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("model", "best_attention", "kjnodes_mode", "impl_type", "speedup", "time_ms", "head_dim", "report")
    FUNCTION = "benchmark"
    CATEGORY = "model_patches/optimization"

    def benchmark(self, model, attention_backend="auto", force_refresh=False, auto_apply=True, seq_len=4096, num_heads=24):
        model_hash = cache.get_model_hash(model)
        model_dtype = cache.get_model_dtype(model)
        head_dim = cache.get_head_dim(model)

        # Clone model so we set override on the clone
        model_clone = model.clone()

        # Force specific backend (skip benchmark)
        if attention_backend != "auto":
            available = get_available_backends()
            if attention_backend not in available:
                print(f"[Benchmark] WARNING: {attention_backend} not available, falling back to pytorch")
                attention_backend = "pytorch"

            if auto_apply:
                if apply_backend(attention_backend, model_clone):
                    print(f"[Benchmark] Force applied: {attention_backend}")
                else:
                    print(f"[Benchmark] Failed to apply {attention_backend}")

            kjmode = backend_to_kjnodes_mode(attention_backend)
            impl = get_impl_type(attention_backend)
            report = f"Force selected: {attention_backend}\nNo benchmark run."

            return (
                model_clone,
                attention_backend,
                kjmode,
                impl,
                0.0,  # No speedup data
                0.0,  # No time data
                head_dim,
                report
            )

        cache_key = f"bench2_{model_hash}_{head_dim}_{seq_len}_{num_heads}"

        # Check cache
        if not force_refresh:
            cached = cache.get_cached_result(cache_key)
            if cached:
                best = cached.get("_best", "pytorch")
                if auto_apply:
                    apply_backend(best, model_clone)
                    print(f"[Benchmark] Applied cached: {best}")

                kjmode = backend_to_kjnodes_mode(best)
                impl = get_impl_type(best)
                report = self._build_report(cached, head_dim, seq_len, model_dtype, from_cache=True)
                return (
                    model_clone,
                    best,
                    kjmode,
                    impl,
                    cached.get("_best_speedup", 1.0),
                    cached.get("_best_time", 0.0),
                    head_dim,
                    report
                )

        # Run benchmark
        print(f"[Benchmark] Running... (head_dim={head_dim}, seq_len={seq_len})")
        print(f"[Benchmark] Available: {get_available_backends()}")

        results = run_benchmark(
            head_dim=head_dim,
            seq_len=seq_len,
            num_heads=num_heads
        )

        # Save to cache
        cache.set_cached_result(cache_key, results)

        # Apply best
        best = results["_best"]
        if auto_apply:
            if apply_backend(best, model_clone):
                print(f"[Benchmark] Applied: {best} ({results['_best_speedup']}x)")
            else:
                print(f"[Benchmark] Failed to apply {best}")

        kjmode = backend_to_kjnodes_mode(best)
        impl = get_impl_type(best)
        report = self._build_report(results, head_dim, seq_len, model_dtype, from_cache=False)

        return (
            model_clone,
            best,
            kjmode,
            impl,
            results["_best_speedup"],
            results["_best_time"],
            head_dim,
            report
        )

    def _build_report(self, data, head_dim, seq_len, dtype, from_cache=False):
        lines = [
            "=" * 65,
            "BENCHMARK REPORT" + (" (cached)" if from_cache else ""),
            "=" * 65,
        ]

        # System info
        info = data.get("_info", {})
        sys_info = f"dtype: {dtype} | head_dim: {head_dim} | seq_len: {seq_len}"
        if info.get("cuda_version"):
            sys_info += f" | CUDA: {info['cuda_version']}"
        if info.get("triton_available"):
            sys_info += f" | Triton: {info.get('triton_version', '?')}"
        lines.append(sys_info)

        if info.get("sage_version"):
            lines.append(f"SageAttention: v{info['sage_version']}")

        lines.append("")
        best_backend = data.get('_best', '?')
        kjmode = backend_to_kjnodes_mode(best_backend)
        impl = get_impl_type(best_backend)
        lines.append(f">>> BEST: {best_backend} ({data.get('_best_speedup', 1.0)}x speedup) <<<")
        lines.append(f"    impl: {impl} | kjnodes mode: {kjmode}")
        lines.append("")
        lines.append("Results (fastest first):")
        lines.append("-" * 65)

        # Sort by time
        backend_data = [(k, v) for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)]
        backend_data.sort(key=lambda x: x[1].get("time_ms", float('inf')))

        for backend, info in backend_data:
            impl = info.get("impl", "?")
            validated = "[v]" if info.get("validated") else "   "

            if info.get("available"):
                time_ms = info.get("time_ms", 0)
                speedup = info.get("speedup", 0)
                best_mark = " <<<" if backend == data.get("_best") else ""
                lines.append(f" {validated} {backend:20} {time_ms:>8.3f}ms  {speedup:>5.2f}x  ({impl}){best_mark}")
            else:
                error = info.get("error", "N/A")[:30]
                lines.append(f" {validated} {backend:20} ---       ({impl}) {error}")

        lines.append("-" * 65)
        lines.append("[v] = validated (tested underlying library directly)")
        lines.append("")
        lines.append("Implementation: cuda = CUDA kernels, triton = Triton kernels")

        if "timestamp" in data:
            lines.append(f"\nCached: {data['timestamp']}")

        lines.append("=" * 65)
        return "\n".join(lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AttentionOptimizer": BenchmarkAndOptimize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AttentionOptimizer": "Attention Optimizer",
}
