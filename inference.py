import logging
import os
import time
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

import config

logger = logging.getLogger(__name__)


def build_messages(
    test_case_content: str,
    system_instruction: Optional[str] = config.SYSTEM_INSTRUCTION,
) -> List[Dict[str, str]]:
    """Build the chat messages list conforming to DynaGuard's chat template conventions.

    Args:
        test_case_content: The full, unmodified string from the input .txt file.
        system_instruction: Configurable system prompt.

    Returns:
        List of message dicts with role and content.
    """
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": test_case_content})
    return messages


def build_inference_prompt(
    test_case_content: str,
    tokenizer: AutoTokenizer,
    mode: str = config.INFERENCE_MODE,
    system_instruction: str = config.SYSTEM_INSTRUCTION,
) -> str:
    """Construct the exact prompt string to be sent to DynaGuard.

    Preserves the test case content completely unmodified.
    Applies the model's tokenizer chat template where available.
    """
    messages = build_messages(test_case_content, system_instruction=system_instruction)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass

    # Fallback to documented [INST] prompt format when no chat template is available
    prefix = "<answer>\n" if mode.lower() == "fast" else "<think>\n"
    return (
        f"[INST]\n"
        f"{system_instruction}\n\n"
        f"{test_case_content}\n"
        f"[/INST]\n"
        f"{prefix}"
    )


def run_inference_single(
    test_case_content: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    mode: str = config.INFERENCE_MODE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    do_sample: bool = config.DO_SAMPLE,
) -> Tuple[str, str, float]:
    """Execute inference for a single test case using DynaGuard's prompting format.

    Uses `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, ...)`
    and decodes only newly generated tokens: `outputs[0][inputs['input_ids'].shape[-1]:]`.

    Args:
        test_case_content: Full unmodified test case text.
        model: Loaded DynaGuard model.
        tokenizer: Loaded DynaGuard tokenizer.
        mode: 'fast' or 'reasoning'.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        do_sample: Whether to sample.

    Returns:
        Tuple of (raw_generated_text, full_prompt, inference_time_seconds).
    """
    if torch is None:
        raise ImportError(
            "PyTorch ('torch') is required to run model inference. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    messages = build_messages(
        test_case_content=test_case_content,
        system_instruction=config.SYSTEM_INSTRUCTION,
    )

    # Standard DynaGuard chat template invocation matching Hugging Face documentation
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        full_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    else:
        full_prompt = build_inference_prompt(
            test_case_content=test_case_content,
            tokenizer=tokenizer,
            mode=mode,
        )
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    prompt_length = inputs["input_ids"].shape[-1]

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if do_sample and temperature > 0.0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    elapsed_time = time.perf_counter() - start_time

    # Decode only newly generated tokens: outputs[0][inputs["input_ids"].shape[-1]:]
    newly_generated_ids = outputs[0][prompt_length:]
    raw_generated_text = tokenizer.decode(
        newly_generated_ids,
        skip_special_tokens=False,
    )

    return raw_generated_text, full_prompt, elapsed_time
