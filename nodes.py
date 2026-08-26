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
- ck: Comfy Kitchen int8 attention (CUDA)
"""
import json
import math
import statistics as _stats
import time
from pathlib import Path

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


def check_ck_available():
    """Check if Comfy Kitchen int8 attention is available."""
    try:
        from comfy.ldm.modules import attention as comfy_attn
        if getattr(comfy_attn, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False):
            return True
    except Exception:
        pass
    try:
        import comfy_kitchen
        return bool(comfy_kitchen.int8_attention_is_available())
    except Exception:
        return False


def get_ck_version():
    """Get installed comfy-kitchen version string."""
    try:
        from importlib.metadata import version
        return version("comfy-kitchen")
    except Exception:
        return None


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

    if check_ck_available():
        backends.append("ck")

    return backends


def get_backend_info():
    """Get detailed info about available backends."""
    info = {
        "triton_available": False,
        "triton_version": None,
        "cuda_version": None,
        "sage_version": None,
        "ck_available": check_ck_available(),
        "ck_version": get_ck_version(),
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


def validate_ck(q, k, v):
    try:
        import comfy_kitchen
        out = comfy_kitchen.int8_attention(q, k, v)
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
        elif backend == "ck":
            return validate_ck(q, k, v), None
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

    elif backend == "ck":
        return getattr(comfy_attn, "attention_comfy_kitchen_int8", None)

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
#
# Statistical methodology ported from diag_nvfp4_extended.py:
#   • per-iteration CUDA events → mean ± σ, p50, p95 (no single stopwatch)
#   • TFLOPS reporting per backend
#   • exact exception capture (root-cause, not a bare "FALLBACK")
#   • BF16(→pytorch SDPA) baseline reference in every table
#   • optional M-scaling sweep across sequence lengths + summary table
#   • JSON export (--out equivalent)
# ============================================================================

def timed_stats(fn, iters=50, warmup=15):
    """
    Time `fn` with one CUDA event pair per iteration.

    Returns (mean_ms, std_ms, p50_ms, p95_ms). Individual event pairs make
    the percentiles meaningful; a single aggregate stopwatch cannot.
    Falls back to perf_counter for CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _ in range(warmup):
        fn()

    if device == "cuda":
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)
    mean = sum(times) / n
    std = _stats.pstdev(times) if n > 1 else 0.0
    p50 = times[n // 2]
    p95 = times[min(int(n * 0.95), n - 1)]
    return mean, std, p50, p95


def attention_flops(batch_size, heads, seq_len, head_dim):
    """FLOPs in one non-causal attention forward pass.

    QK^T and PV each cost 2·B·H·S·S·D multiply-adds → 2 FLOPs per GEMM term.
    """
    return 4.0 * batch_size * heads * seq_len * seq_len * head_dim


def tflops_from(flops, mean_ms):
    """Arithmetic throughput  FLOPs / time → TFLOPS.  mean_ms=None → None."""
    if not mean_ms or mean_ms <= 0:
        return None
    return flops / (mean_ms * 1e-3) / 1e12


def benchmark_backend_stats(backend, q, k, v, heads, iters=50, warmup=15):
    """
    Benchmark one backend with statistical timing.

    Returns a dict with keys:
      mean_ms, std_ms, p50_ms, p95_ms, tflops, error (None on success).
    The exact exception type + message is captured for fallback root-causing,
    replicating the diag script's error handling instead of a bare "FALLBACK".
    """
    attn_func = get_attention_function(backend)
    if attn_func is None:
        return dict(mean_ms=None, std_ms=None, p50_ms=None, p95_ms=None,
                    tflops=None, error="backend not registered / available")

    try:
        mean, std, p50, p95 = timed_stats(
            lambda: attn_func(q, k, v, heads, skip_reshape=True),
            iters=iters, warmup=warmup,
        )
        flops = attention_flops(q.shape[0], heads, q.shape[2], q.shape[3])
        return dict(mean_ms=mean, std_ms=std, p50_ms=p50, p95_ms=p95,
                    tflops=tflops_from(flops, mean), error=None)
    except Exception as ex:
        return dict(mean_ms=None, std_ms=None, p50_ms=None, p95_ms=None,
                    tflops=None, error=f"{type(ex).__name__}: {ex}")


def benchmark_backend(backend, q, k, v, heads, num_iterations=10):
    """Benchmark a specific backend; returns mean ms (legacy wrapper)."""
    stats = benchmark_backend_stats(backend, q, k, v, heads,
                                    iters=num_iterations,
                                    warmup=max(3, num_iterations // 3))
    if stats["mean_ms"] is None:
        raise RuntimeError(stats["error"] or f"Backend {backend} failed")
    return stats["mean_ms"]


def _jsonify(obj):
    """Recursively make obj JSON-safe (handles nan/inf → null, tuples → lists)."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_benchmark_json(results, path):
    """Write a benchmark results dict to a JSON file (--out equivalent)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonify(results), indent=2))
    print(f"[Benchmark] JSON results saved -> {out.resolve()}")


def run_benchmark(head_dim=128, seq_len=4096, num_heads=24, batch_size=1,
                  iters=30, warmup=10, seq_sweep=()):
    """
    Benchmark all available backends with statistical timing.

    seq_sweep: iterable of extra sequence lengths. When non-empty, an
    M-scaling sweep is run for the strongest backends and the one-row-per-M
    totals table is stored under results["_m_scaling"].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    def _make_tensors(slen):
        return tuple(
            torch.randn(batch_size, num_heads, slen, head_dim, device=device, dtype=dtype)
            for _ in range(3)
        )

    q, k, v = _make_tensors(seq_len)

    results = {}
    available = get_available_backends()

    # All possible backends in order
    all_backends = [
        "basic", "sub_quad", "split", "pytorch", "xformers",
        "sage_auto", "sage_cuda", "sage_triton", "sage_fp8_cuda", "sage_fp8_cuda_fast",
        "sage3", "flash", "ck"
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
        elif "cuda" in backend or backend in ["flash", "sage3", "ck"]:
            impl = "cuda"
        elif backend == "sage_auto":
            impl = "auto"
        elif backend == "xformers":
            impl = "cuda/triton"

        # Validate first (tests the underlying library directly)
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

        # Statistical benchmark
        stats = benchmark_backend_stats(backend, q, k, v, num_heads,
                                        iters=iters, warmup=warmup)
        if stats["mean_ms"] is not None:
            results[backend] = {
                "time_ms": round(stats["mean_ms"], 3),
                "std_ms": round(stats["std_ms"], 3),
                "p50_ms": round(stats["p50_ms"], 3),
                "p95_ms": round(stats["p95_ms"], 3),
                "tflops": round(stats["tflops"], 2) if stats["tflops"] else None,
                "available": True,
                "impl": impl,
                "validated": True
            }
        else:
            results[backend] = {
                "time_ms": float('inf'),
                "available": False,
                "error": (stats["error"] or "Benchmark failed")[:80],
                "impl": impl,
                "validated": True
            }

    # Calculate speedups relative to pytorch (bf16/fp16 SDPA reference)
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

    # M-scaling sweep: benchmark every available backend at each extra length
    # so the winners can be compared per-M (diag --m sweep equivalent).
    sweep_seqs = sorted({int(s) for s in (seq_sweep or ()) if int(s) > 0})
    if sweep_seqs:
        sweep_backends = [b for b in available]
        scaling = {"seqs": sweep_seqs, "backends": sweep_backends, "data": {}}
        for slen in sweep_seqs:
            qs, ks, vs = _make_tensors(slen)
            entry = {}
            valid = []
            for b in sweep_backends:
                st = benchmark_backend_stats(b, qs, ks, vs, num_heads,
                                             iters=min(iters, 20), warmup=warmup)
                entry[b] = {"mean_ms": st["mean_ms"],
                            "tflops": st["tflops"],
                            "error": st["error"]}
                if st["mean_ms"] is not None:
                    valid.append((b, st["mean_ms"]))
            if valid:
                entry["_best"] = min(valid, key=lambda x: x[1])[0]
            scaling["data"][slen] = entry
            del qs, ks, vs
        results["_m_scaling"] = scaling

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

        # Prefer ComfyUI's official API; it copies the backend's
        # container_function when present (needed for comfy_kitchen_int8).
        set_attn = getattr(model, "set_model_optimized_attention", None)
        if set_attn is not None:
            set_attn(attn_func)
        else:
            def attention_override(func, *args, **kwargs):
                return attn_func.__wrapped__(*args, **kwargs)

            if hasattr(attn_func, "container_function") and attn_func.container_function is not None:
                attention_override.container_function = attn_func.container_function

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
    elif backend in ["sage_cuda", "sage_fp8_cuda", "sage_fp8_cuda_fast", "flash", "sage3", "ck"]:
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

    Timing uses the diag_nvfp4_extended methodology: per-iteration CUDA events
    reporting mean ± σ, p50 and p95, plus TFLOPS throughput for every backend.
    An optional M-scaling sweep highlights how each backend behaves at other
    sequence lengths, and results can be exported to JSON (--out equivalent).

    Inputs:
    - model: The model to optimize
    - attention_backend: "auto" runs benchmark, or select specific backend to force
    - force_refresh: Re-run benchmark even if cached
    - auto_apply: Apply the selected backend globally
    - seq_len / num_heads: Benchmark parameters
    - timing_iters / timing_warmup: Statistical timing iterations / warmup
    - seq_sweep: "off" (default), "quick" (seq/2, seq*2),
                "full" (power-of-2 grid 256..16384)
    - json_path: Optional path to export full results as JSON ("" = off)

    Available backends:
    - pytorch: PyTorch SDPA (always available)
    - xformers: xFormers memory efficient attention
    - sage_auto, sage_cuda, sage_triton: SageAttention variants
    - sage_fp8_cuda, sage_fp8_cuda_fast: SageAttention FP8
    - sage3: SageAttention 3 (Blackwell GPUs only)
    - flash: Flash Attention 2
    - ck: Comfy Kitchen int8 attention (CUDA)
    - basic, sub_quad, split: ComfyUI built-in backends

    Outputs:
    - model: Model with attention applied globally
    - best_attention: Applied backend name
    - kjnodes_mode: Compatible mode for PathchSageAttentionKJ node
    - impl_type: Implementation type ("cuda", "triton", "pytorch")
    - speedup: Speedup vs pytorch (0.0 if forced)
    - time_ms: Time per attention call (0.0 if forced)
    - head_dim: Detected head dimension
    - report: Full text report (incl. std/p50/p95/TFLOPS and M-scaling)
    """

    # All possible backends for dropdown
    ALL_BACKENDS = [
        "auto",  # Run benchmark and pick best
        "pytorch",
        "ck",
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
                "timing_iters": ("INT", {"default": 30, "min": 3, "max": 500}),
                "timing_warmup": ("INT", {"default": 10, "min": 1, "max": 200}),
                "seq_sweep": (["off", "quick", "full"], {"default": "off"}),
                "json_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("model", "best_attention", "kjnodes_mode", "impl_type", "speedup", "time_ms", "head_dim", "report")
    FUNCTION = "benchmark"
    CATEGORY = "model_patches/optimization"

    def _resolve_sweep(self, preset, seq_len):
        """Convert a seq_sweep preset into a list of extra sequence lengths."""
        if preset == "quick":
            return [max(256, seq_len // 2), seq_len * 2]
        if preset == "full":
            # Power-of-2 shape grid, mirroring diag_nvfp4_extended --full
            grid = [256, 512, 1024, 2048, 4096, 8192, 16384]
            return [s for s in grid if s != seq_len]
        return []

    def benchmark(self, model, attention_backend="auto", force_refresh=False,
                  auto_apply=True, seq_len=4096, num_heads=24,
                  timing_iters=30, timing_warmup=10, seq_sweep="off",
                  json_path=""):
        model_hash = cache.get_model_hash(model)
        model_dtype = cache.get_model_dtype(model)
        head_dim = cache.get_head_dim(model)
        sweep = tuple(self._resolve_sweep(seq_sweep, seq_len))

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

            if json_path:
                save_benchmark_json({"forced": attention_backend,
                                     "impl": impl, "head_dim": head_dim}, json_path)

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

        cache_key = (f"bench3_{model_hash}_{head_dim}_{seq_len}_{num_heads}_"
                     f"i{timing_iters}_w{timing_warmup}_s{seq_sweep}")

        # Check cache
        if not force_refresh:
            cached = cache.get_cached_result(cache_key)
            if cached:
                best = cached.get("_best", "pytorch")
                if auto_apply:
                    apply_backend(best, model_clone)
                    print(f"[Benchmark] Applied cached: {best}")

                if json_path:
                    save_benchmark_json(cached, json_path)

                kjmode = backend_to_kjnodes_mode(best)
                impl = get_impl_type(best)
                report = self._build_report(cached, head_dim, seq_len, model_dtype,
                                            from_cache=True, sweep=sweep)
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
        print(f"[Benchmark] Running... (head_dim={head_dim}, seq_len={seq_len}, "
              f"iters={timing_iters}, warmup={timing_warmup})")
        print(f"[Benchmark] Available: {get_available_backends()}")

        results = run_benchmark(
            head_dim=head_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            iters=timing_iters,
            warmup=timing_warmup,
            seq_sweep=sweep
        )

        # Save to cache
        cache.set_cached_result(cache_key, results)

        if json_path:
            save_benchmark_json(results, json_path)

        # Apply best
        best = results["_best"]
        if auto_apply:
            if apply_backend(best, model_clone):
                print(f"[Benchmark] Applied: {best} ({results['_best_speedup']}x)")
            else:
                print(f"[Benchmark] Failed to apply {best}")

        kjmode = backend_to_kjnodes_mode(best)
        impl = get_impl_type(best)
        report = self._build_report(results, head_dim, seq_len, model_dtype,
                                    from_cache=False, sweep=sweep)

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

    def _build_report(self, data, head_dim, seq_len, dtype, from_cache=False, sweep=()):
        lines = [
            "=" * 74,
            "BENCHMARK REPORT" + (" (cached)" if from_cache else ""),
            "=" * 74,
        ]

        # System info
        info = data.get("_info", {})
        sys_info = f"dtype: {dtype} | head_dim: {head_dim} | seq_len: {seq_len}"
        if info.get("cuda_version"):
            sys_info += f" | CUDA: {info['cuda_version']}"
        if info.get("triton_available"):
            sys_info += f" | Triton: {info.get('triton_version', '?')}"
        if info.get("ck_available"):
            ckv = info.get("ck_version") or "?"
            sys_info += f" | Comfy Kitchen: {ckv}"
        lines.append(sys_info)

        if info.get("sage_version"):
            lines.append(f"SageAttention: v{info['sage_version']}")

        lines.append("")
        best_backend = data.get('_best', '?')
        kjmode = backend_to_kjnodes_mode(best_backend)
        impl = get_impl_type(best_backend)

        best_tflops = data.get(best_backend, {}).get("tflops") if best_backend in data else None
        tf_suffix = f"  @ {best_tflops:.1f} TFLOPS" if best_tflops else ""
        lines.append(f">>> BEST: {best_backend} ({data.get('_best_speedup', 1.0)}x speedup){tf_suffix} <<<")
        lines.append(f"    impl: {impl} | kjnodes mode: {kjmode}")
        lines.append("")
        lines.append("Results (fastest first):")
        lines.append("-" * 74)

        # Sort by time
        backend_data = [(k, v) for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)]
        backend_data.sort(key=lambda x: x[1].get("time_ms", float('inf')))

        for backend, binfo in backend_data:
            impl_s = binfo.get("impl", "?")
            validated = "[v]" if binfo.get("validated") else "   "

            if binfo.get("available"):
                time_ms = binfo.get("time_ms", 0)
                std_ms = binfo.get("std_ms")
                p50 = binfo.get("p50_ms")
                p95 = binfo.get("p95_ms")
                speedup = binfo.get("speedup", 0)
                tf = binfo.get("tflops")
                best_mark = " <<<" if backend == data.get("_best") else ""

                jitter = ""
                if p95 and time_ms and p95 > time_ms * 1.25:
                    jitter = " [jitter]"

                stat_s = ""
                if std_ms is not None and p50 is not None and p95 is not None:
                    stat_s = f" p50={p50:>7.3f} p95={p95:>7.3f}"
                tf_s = f" {tf:>7.1f}T" if tf else f" {'--':>7}"
                lines.append(f" {validated} {backend:20} {time_ms:>8.3f}±{std_ms or 0:>5.3f}ms"
                             f"{stat_s}  {speedup:>5.2f}x{tf_s}  ({impl_s}){best_mark}{jitter}")
            else:
                error = binfo.get("error", "N/A")[:44]
                lines.append(f" {validated} {backend:20} ---       ({impl_s}) {error}")

        lines.append("-" * 74)
        lines.append("[v] = validated (tested underlying library directly)  [jitter] = p95 > 1.25x mean")
        lines.append("")

        base_tflops = data.get("pytorch", {}).get("tflops")
        if base_tflops:
            lines.append(f"pytorch SDPA baseline: {base_tflops:.1f} TFLOPS")

        # M-scaling summary (one row per sequence length)
        scaling = data.get("_m_scaling")
        if scaling and scaling.get("data"):
            lines.append("")
            lines.append("M-scaling summary (mean ms; * = best at that len):")
            lines.append("-" * 100)
            seqs = scaling.get("seqs", [])
            backends = scaling.get("backends", [])
            hdr = f"  {'seq':>7} " + "  ".join(f"{b:>10}" for b in backends) + f"  {'ratio':>7}"
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))
            for slen in seqs:
                d = scaling["data"].get(slen, {})
                best_here = d.get("_best")
                pyt = d.get("pytorch", {}).get("mean_ms")
                row = f"  {slen:>7} "
                for b in backends:
                    entry = d.get(b, {})
                    mean = entry.get("mean_ms")
                    if mean is None:
                        row += "  " + f"{'--':>10}"
                        continue
                    cell = f"{mean:>9.3f}" + ("*" if b == best_here else " ")
                    row += "  " + f"{cell:>10}"
                bst = d.get(best_here, {}).get("mean_ms") if best_here else None
                ratio = f"{(pyt / bst):>6.2f}x" if (pyt and bst) else "     --"
                row += f"  {ratio:>7}"
                lines.append(row)
            lines.append("")

        if "timestamp" in data:
            lines.append(f"Cached: {data['timestamp']}")

        lines.append("=" * 74)
        return "\n".join(lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AttentionOptimizer": BenchmarkAndOptimize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AttentionOptimizer": "Attention Optimizer",
}
