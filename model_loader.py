import logging
import os
from typing import Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

import config

logger = logging.getLogger(__name__)


def resolve_torch_dtype(dtype_setting: str):
    """Resolve the torch dtype based on user configuration and hardware capability."""
    if torch is None:
        return None
    dtype_lower = str(dtype_setting).strip().lower()

    if dtype_lower == "bfloat16":
        return torch.bfloat16
    elif dtype_lower == "float16":
        return torch.float16
    elif dtype_lower == "float32":
        return torch.float32
    elif dtype_lower == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    else:
        # Default or fallback
        return getattr(torch, dtype_lower, torch.float32)


def resolve_device_map(device_map_setting: str) -> Optional[str]:
    """Resolve the device_map parameter safely based on hardware availability."""
    if device_map_setting == "auto":
        if torch is not None and torch.cuda.is_available():
            return "auto"
        # On CPU-only environments, accelerate device_map="auto" is not needed
        return None
    return device_map_setting


def get_balanced_max_memory(headroom_ratio: float = 0.82) -> Optional[dict]:
    """Calculate balanced max_memory mapping across available CUDA devices to reserve headroom and prevent OOM."""
    if torch is None or not torch.cuda.is_available():
        return None
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        return None
    max_memory = {}
    for i in range(num_gpus):
        vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        usable_gb = max(1, int(vram_gb * headroom_ratio))
        max_memory[i] = f"{usable_gb}GiB"
    max_memory["cpu"] = "24GiB"
    return max_memory


def load_model_and_tokenizer(
    model_id_or_path: Optional[str] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[str] = None,
):
    """Load the DynaGuard model and tokenizer using Hugging Face Transformers.

    Args:
        model_id_or_path: Hugging Face model ID or path to local model weights.
        device_map: Device mapping strategy (e.g. 'auto', 'cpu', 'cuda').
        torch_dtype: Torch precision setting ('auto', 'bfloat16', 'float16',
          'float32').

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        RuntimeError: If model or tokenizer fails to load. Never silently substitutes.
    """
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError(
            "PyTorch and Hugging Face Transformers are required to load models. "
            "Please install dependencies with: pip install -r requirements.txt"
        )
    model_id = model_id_or_path or config.MODEL_ID
    dev_map = (
        resolve_device_map(device_map or config.DEVICE_MAP)
        if device_map or config.DEVICE_MAP
        else None
    )
    dtype = resolve_torch_dtype(torch_dtype or config.TORCH_DTYPE)

    logger.info("Initializing DynaGuard model from: %s", model_id)
    logger.info("Resolved device_map: %s, torch_dtype: %s", dev_map, dtype)

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    except Exception as exc:
        err_msg = (
            f"CRITICAL: Failed to load tokenizer for model '{model_id}'. "
            f"Original error: {exc}"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg) from exc

    # Ensure pad_token_id is defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load Model
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dev_map is not None:
        model_kwargs["device_map"] = dev_map
        max_mem = get_balanced_max_memory()
        if max_mem:
            model_kwargs["max_memory"] = max_mem

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
    except Exception as exc:
        err_msg = (
            f"CRITICAL: Failed to load model weights for '{model_id}'. "
            f"Check model ID or local path and ensure hardware dependencies are installed. "
            f"Original error: {exc}"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg) from exc

    logger.info("Successfully loaded DynaGuard model and tokenizer.")
    return model, tokenizer
