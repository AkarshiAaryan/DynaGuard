import argparse
import datetime
import json
import logging
import os
import re
import sys
from typing import List

import config
from inference import run_inference_single
from model_loader import load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dynaguard_pipeline")


def natural_sort_key(s: str) -> List:
    """Sort strings containing numbers naturally so TC2 comes before TC10."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def append_run_log(
    log_path: str,
    input_filename: str,
    output_filename: str,
    status: str,
    inference_time: float,
    error_message: str = None,
) -> None:
    """Append a single run record to run_log.jsonl."""
    log_entry = {
        "input_filename": input_filename,
        "output_filename": output_filename,
        "status": status,
        "inference_time": round(inference_time, 4),
        "error_message": error_message,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal, transparent DynaGuard-8B Inference Pipeline."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=config.MODEL_ID,
        help="Hugging Face model ID or path to local weights.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=config.INPUT_DIR,
        help="Directory containing input test cases as .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory where raw model outputs are saved.",
    )
    parser.add_argument(
        "--sent-prompts-dir",
        type=str,
        default=config.SENT_PROMPTS_DIR,
        help="Directory where exact prompts sent to the model are archived.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "reasoning"],
        default=config.INFERENCE_MODE,
        help="DynaGuard inference mode: 'fast' (default) or 'reasoning'.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun test cases even if output file already exists.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.MAX_NEW_TOKENS,
        help="Maximum newly generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="Sampling temperature (0.0 for deterministic output).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=config.TOP_P,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=config.DEVICE_MAP,
        help="Device map (e.g. 'auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=config.TORCH_DTYPE,
        help="Torch dtype ('auto', 'bfloat16', 'float16', 'float32').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure necessary directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sent_prompts_dir, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # 1. Identify and naturally sort test cases
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    txt_files = [f for f in os.listdir(args.input_dir) if f.endswith(".txt")]
    txt_files.sort(key=natural_sort_key)

    total_cases = len(txt_files)
    if total_cases == 0:
        logger.warning(f"No .txt test cases found in input directory: {args.input_dir}")
        print("Done. 0 test cases processed.")
        return

    print("=" * 70)
    print("DynaGuard Inference Pipeline")
    print(f"Model ID / Path   : {args.model_id}")
    print(f"Inference Mode    : {args.mode}")
    print(f"Input Directory   : {args.input_dir}")
    print(f"Output Directory  : {args.output_dir}")
    print(f"Total Test Cases  : {total_cases}")
    print(f"Rerun Existing    : {args.rerun}")
    print("=" * 70)

    # 2. Load model and tokenizer once
    print(f"Loading DynaGuard model ({args.model_id})...")
    model, tokenizer = load_model_and_tokenizer(
        model_id_or_path=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    print("Model initialized successfully.\n")

    successful_count = 0
    failed_count = 0
    skipped_count = 0

    # 3. Process test cases sequentially
    for idx, filename in enumerate(txt_files, start=1):
        case_stem = os.path.splitext(filename)[0]
        output_filename = f"{case_stem}_output.txt"
        output_filepath = os.path.join(args.output_dir, output_filename)
        prompt_filepath = os.path.join(args.sent_prompts_dir, f"{case_stem}_prompt.txt")
        input_filepath = os.path.join(args.input_dir, filename)

        # Check if output already exists (resume behavior)
        if not args.rerun and os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
            print(f"[{idx}/{total_cases}] {filename} -> SKIPPED (output exists)")
            skipped_count += 1
            append_run_log(
                log_path=config.RUN_LOG_PATH,
                input_filename=filename,
                output_filename=output_filename,
                status="skipped",
                inference_time=0.0,
                error_message=None,
            )
            continue

        print(f"[{idx}/{total_cases}] Processing {filename} ... ", end="", flush=True)

        try:
            # Read test case file in full, preserving original contents
            with open(input_filepath, "r", encoding="utf-8") as f:
                test_case_content = f.read()

            # Execute inference
            raw_response, sent_prompt, elapsed_time = run_inference_single(
                test_case_content=test_case_content,
                model=model,
                tokenizer=tokenizer,
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=(args.temperature > 0.0),
            )

            # Save verbatim prompt sent to model
            with open(prompt_filepath, "w", encoding="utf-8") as pf:
                pf.write(sent_prompt)

            # CRITICAL: Save raw generated response without parsing or alteration
            with open(output_filepath, "w", encoding="utf-8") as of:
                of.write(raw_response)

            append_run_log(
                log_path=config.RUN_LOG_PATH,
                input_filename=filename,
                output_filename=output_filename,
                status="success",
                inference_time=elapsed_time,
                error_message=None,
            )

            print(f"DONE ({elapsed_time:.2f}s)")
            successful_count += 1

        except Exception as exc:
            print(f"FAILED: {exc}")
            logger.error(f"Error processing test case '{filename}': {exc}", exc_info=True)
            failed_count += 1
            append_run_log(
                log_path=config.RUN_LOG_PATH,
                input_filename=filename,
                output_filename=output_filename,
                status="failed",
                inference_time=0.0,
                error_message=str(exc),
            )

    # 4. Final summary report
    print("\n" + "=" * 70)
    print("Inference Run Complete")
    print(f"Total Cases     : {total_cases}")
    print(f"Successful      : {successful_count}")
    print(f"Failed          : {failed_count}")
    print(f"Skipped         : {skipped_count}")
    print(f"Outputs saved to: {args.output_dir}")
    print(f"Prompts saved to: {args.sent_prompts_dir}")
    print(f"Run log at      : {config.RUN_LOG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
