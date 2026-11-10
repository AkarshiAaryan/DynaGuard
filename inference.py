import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

import config

logger = logging.getLogger(__name__)


def serialize_dialogue(messages: List[Dict[str, str]]) -> str:
    """Serialize a list of message dicts into readable multi-turn dialogue text.

    Formats turns with explicit actor roles (User: / Agent:) separated by double newlines.
    Preserves original message order, usernames, roles, course scopes, and target resources.

    Args:
        messages: List of chat message objects with 'role' and 'content'.

    Returns:
        Formatted multi-turn dialogue string.
    """
    dialogue_turns = []
    for msg in messages:
        role_raw = msg.get("role", "").lower()
        role_label = "User" if role_raw == "user" else "Agent"
        content = msg.get("content", "").strip()
        if content:
            dialogue_turns.append(f"{role_label}: {content}")
    return "\n\n".join(dialogue_turns)


def extract_policy_and_dialogue(content: Union[List[Dict[str, str]], str]) -> Tuple[str, str]:
    """Extract policy rules and serialized dialogue from either a JSON message list or raw text.

    Args:
        content: Either a list of message dicts (from .json) or a raw text string (from .txt).

    Returns:
        Tuple of (policy_text, serialized_dialogue).
    """
    if isinstance(content, list):
        if not content:
            return "", ""

        first_content = content[0].get("content", "")
        if "Policy:" in first_content:
            parts = first_content.split("\n\n", 1)
            policy = parts[0].replace("Policy:", "").strip()
            user_start = parts[1].strip() if len(parts) > 1 else ""
        else:
            policy = ""
            user_start = first_content

        dialogue_turns = []
        if user_start:
            dialogue_turns.append(f"User: {user_start}")

        for turn in content[1:]:
            role = "User" if turn.get("role", "").lower() == "user" else "Agent"
            dialogue_turns.append(f"{role}: {turn.get('content', '').strip()}")

        return policy, "\n\n".join(dialogue_turns)
    else:
        text = str(content)
        if "Dialogue:" in text:
            parts = text.split("Dialogue:", 1)
            policy = parts[0].replace("Policy:", "").replace("## Policy", "").strip()
            return policy, parts[1].strip()
        elif "Policy:" in text:
            parts = text.split("\n\n", 1)
            policy = parts[0].replace("Policy:", "").replace("## Policy", "").strip()
            return policy, (parts[1].strip() if len(parts) > 1 else "")
        return "", text


def build_evaluator_prompt(
    test_case_content: Union[List[Dict[str, str]], str],
    mode: str = config.INFERENCE_MODE,
    tokenizer: Optional[Any] = None,
) -> str:
    """Construct ONE standardized evaluator prompt following DynaGuard model-card fast-inference conventions.

    Args:
        test_case_content: Test case message list (from .json) or raw text (from .txt).
        mode: 'fast' (<answer>) or 'reasoning' (<think>).
        tokenizer: Optional tokenizer (unused, prompt is constructed directly).

    Returns:
        Full formatted evaluator prompt string.
    """
    policy_text, serialized_dialogue = extract_policy_and_dialogue(test_case_content)

    suffix = "<think>\n" if mode.lower() in ("reasoning", "cot") else "<answer>\n"

    prompt = f"""[INST]
{config.EVALUATOR_HEADER}

## POLICY

{policy_text}

## DIALOGUE

{serialized_dialogue}

## EVALUATION

{config.EVALUATION_FOOTER}
[/INST]
{suffix}"""
    return prompt


# Alias for backward compatibility
build_inference_prompt = build_evaluator_prompt


def parse_verdict(raw_output: str) -> Tuple[str, str]:
    """Strictly extract PASS or FAIL verdict from model output.

    Valid outputs:
        <answer>PASS</answer>
        <answer>
        PASS
        </answer>
        PASS\n</answer> (when <answer> is the prompt prefix)
        PASS

    Invalid outputs:
        Yes, No, Compliant, Reject, None of the above, natural language explanations,
        or ambiguous texts.

    Returns:
        Tuple of (prediction, validity_status)
        where prediction is 'PASS', 'FAIL', or 'INVALID_OUTPUT',
        and validity_status is 'VALID' or 'INVALID'.
    """
    if not raw_output or not isinstance(raw_output, str):
        return "INVALID_OUTPUT", "INVALID"

    text = raw_output.strip()

    # Case 1: text contains an explicit <answer>...</answer> block
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        answer_body = match.group(1).strip()
    else:
        # Case 2: prompt ended with <answer>, so generation emitted completion closing with </answer>
        close_match = re.search(r"^(.*?)(?:</answer>|$)", text, re.IGNORECASE | re.DOTALL)
        if close_match and "</answer>" in text:
            answer_body = close_match.group(1).strip()
        else:
            # Case 3: generated tokens only contain the verdict token
            answer_body = text.strip()

    tokens = answer_body.split()
    if len(tokens) == 1:
        token = tokens[0].upper()
        if token == "PASS":
            return "PASS", "VALID"
        elif token == "FAIL":
            return "FAIL", "VALID"

    return "INVALID_OUTPUT", "INVALID"


def run_inference_single(
    test_case_content: Union[List[Dict[str, str]], str],
    model: Any,
    tokenizer: Any,
    test_case_id: str = "TC01",
    mode: str = config.INFERENCE_MODE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    do_sample: bool = config.DO_SAMPLE,
    repetition_penalty: float = config.REPETITION_PENALTY,
) -> Dict[str, Any]:
    """Execute single-case inference returning all 9 required evaluation fields.

    Args:
        test_case_content: Test case message list (from .json) or raw text (from .txt).
        model: Loaded DynaGuard model.
        tokenizer: Loaded DynaGuard tokenizer.
        test_case_id: Identifier of the test case (e.g. 'TC01').
        mode: 'fast' or 'reasoning'.
        max_new_tokens: Maximum newly generated tokens (default: 16).
        temperature: Sampling temperature (default: 0.0).
        top_p: Nucleus sampling probability (default: 1.0).
        do_sample: Whether to sample (default: False for greedy decoding).
        repetition_penalty: Repetition penalty (default: 1.0).

    Returns:
        Dict with keys:
            - test_case_id
            - full_prompt
            - raw_generated_text
            - prediction
            - validity_status
            - inference_time
            - prompt_token_count
            - generated_token_count
            - tokens_per_second
    """
    if torch is None:
        raise ImportError(
            "PyTorch ('torch') is required to run model inference. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    full_prompt = build_evaluator_prompt(test_case_content, mode=mode)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[-1]

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }

    if do_sample and temperature > 0.0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    elapsed_time = time.perf_counter() - start_time

    newly_generated_ids = outputs[0][prompt_length:]
    generated_token_count = len(newly_generated_ids)
    tokens_per_second = (
        round(generated_token_count / elapsed_time, 2) if elapsed_time > 0 else 0.0
    )

    raw_generated_text = tokenizer.decode(
        newly_generated_ids,
        skip_special_tokens=False,
    )

    prediction, validity_status = parse_verdict(raw_generated_text)

    return {
        "test_case_id": test_case_id,
        "full_prompt": full_prompt,
        "raw_generated_text": raw_generated_text,
        "prediction": prediction,
        "validity_status": validity_status,
        "inference_time": round(elapsed_time, 4),
        "prompt_token_count": prompt_length,
        "generated_token_count": generated_token_count,
        "tokens_per_second": tokens_per_second,
    }

