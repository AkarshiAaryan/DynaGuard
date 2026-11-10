import argparse
import csv
import datetime
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional

import config
from inference import run_inference_single, parse_verdict
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
    test_case_id: str,
    input_filename: str,
    output_filename: str,
    status: str,
    prediction: str,
    validity_status: str,
    inference_time: float,
    prompt_tokens: int = 0,
    generated_tokens: int = 0,
    tokens_per_second: float = 0.0,
    error_message: Optional[str] = None,
) -> None:
    """Append a single run record to run_log.jsonl with all 9 required fields."""
    log_entry = {
        "test_case_id": test_case_id,
        "input_filename": input_filename,
        "output_filename": output_filename,
        "status": status,
        "prediction": prediction,
        "validity_status": validity_status,
        "inference_time": round(inference_time, 4),
        "prompt_token_count": prompt_tokens,
        "generated_token_count": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "error_message": error_message,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def evaluate_against_ground_truth(
    records: List[Dict],
    ground_truth_path: str = config.GROUND_TRUTH_PATH,
) -> Dict:
    """Compare extracted model predictions against ground truth labels."""
    if not os.path.exists(ground_truth_path):
        return {}

    gt_map = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("case_id", "").strip()
            expected_raw = row.get("expected_label", "").strip()
            expected_verdict = config.LABEL_MAPPING.get(expected_raw, expected_raw.upper())
            gt_map[cid] = {
                "expected_label": expected_raw,
                "expected_verdict": expected_verdict,
            }

    total = 0
    correct = 0
    incorrect = 0
    invalid = 0

    tp = 0  # GT FAIL, Pred FAIL
    fp = 0  # GT PASS, Pred FAIL
    tn = 0  # GT PASS, Pred PASS
    fn = 0  # GT FAIL, Pred PASS

    comparison_results = []
    for r in records:
        cid = r["test_case_id"]
        pred = r["prediction"]
        gt_info = gt_map.get(cid)
        if not gt_info:
            continue

        total += 1
        expected = gt_info["expected_verdict"]

        if pred == "INVALID_OUTPUT":
            invalid += 1
            is_correct = False
        elif pred == expected:
            correct += 1
            is_correct = True
        else:
            incorrect += 1
            is_correct = False

        if expected == "FAIL" and pred == "FAIL":
            tp += 1
        elif expected == "PASS" and pred == "FAIL":
            fp += 1
        elif expected == "PASS" and pred == "PASS":
            tn += 1
        elif expected == "FAIL" and pred == "PASS":
            fn += 1

        comparison_results.append({
            "case_id": cid,
            "expected_verdict": expected,
            "predicted_verdict": pred,
            "status": "CORRECT" if is_correct else ("INVALID" if pred == "INVALID_OUTPUT" else "INCORRECT"),
            "inference_time": r.get("inference_time", 0.0),
        })

    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    valid_total = correct + incorrect
    accuracy_valid_only = (correct / valid_total * 100.0) if valid_total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "invalid": invalid,
        "accuracy": round(accuracy, 2),
        "accuracy_valid_only": round(accuracy_valid_only, 2),
        "confusion_matrix": {
            "true_fail": tp,
            "false_fail": fp,
            "true_pass": tn,
            "false_pass": fn,
            "invalid": invalid,
        },
        "details": comparison_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardized DynaGuard Policy Compliance Inference Pipeline."
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
        help="Directory containing input test cases as .json or .txt files.",
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
        help="Sampling temperature (0.0 for deterministic greedy output).",
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

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sent_prompts_dir, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    test_files = [
        f for f in os.listdir(args.input_dir)
        if (f.endswith(".json") or f.endswith(".txt")) and f != "dynaguard_messages.json"
    ]
    test_files.sort(key=natural_sort_key)

    total_cases = len(test_files)
    if total_cases == 0:
        logger.warning(f"No test cases (.json or .txt) found in input directory: {args.input_dir}")
        print("Done. 0 test cases processed.")
        return

    print("=" * 70)
    print("DynaGuard Policy Compliance Evaluation Pipeline")
    print(f"Model ID / Path   : {args.model_id}")
    print(f"Inference Mode    : {args.mode}")
    print(f"Input Directory   : {args.input_dir}")
    print(f"Output Directory  : {args.output_dir}")
    print(f"Total Test Cases  : {total_cases}")
    print(f"Rerun Existing    : {args.rerun}")
    print(f"Max New Tokens    : {args.max_new_tokens} (Greedy Decoding)")
    print("=" * 70)

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
    all_case_records = []

    for idx, filename in enumerate(test_files, start=1):
        case_stem = os.path.splitext(filename)[0]
        case_id = case_stem.split("_")[0]
        output_filename = f"{case_stem}_output.txt"
        output_filepath = os.path.join(args.output_dir, output_filename)
        prompt_filepath = os.path.join(args.sent_prompts_dir, f"{case_stem}_prompt.txt")
        input_filepath = os.path.join(args.input_dir, filename)

        if not args.rerun and os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
            print(f"[{idx}/{total_cases}] {filename} -> SKIPPED (output exists)")
            skipped_count += 1
            with open(output_filepath, "r", encoding="utf-8") as rf:
                raw_out = rf.read()
            pred, val_status = parse_verdict(raw_out)
            all_case_records.append({
                "test_case_id": case_id,
                "input_filename": filename,
                "prediction": pred,
                "validity_status": val_status,
                "raw_output": raw_out,
                "inference_time": 0.0,
            })
            continue

        print(f"[{idx}/{total_cases}] Processing {filename} ... ", end="", flush=True)

        try:
            if filename.endswith(".json"):
                with open(input_filepath, "r", encoding="utf-8") as f:
                    test_case_content = json.load(f)
            else:
                with open(input_filepath, "r", encoding="utf-8") as f:
                    test_case_content = f.read()

            res = run_inference_single(
                test_case_content=test_case_content,
                model=model,
                tokenizer=tokenizer,
                test_case_id=case_id,
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=(args.temperature > 0.0),
            )

            with open(prompt_filepath, "w", encoding="utf-8") as pf:
                pf.write(res["full_prompt"])

            with open(output_filepath, "w", encoding="utf-8") as of:
                of.write(res["raw_generated_text"])

            append_run_log(
                log_path=config.RUN_LOG_PATH,
                test_case_id=case_id,
                input_filename=filename,
                output_filename=output_filename,
                status="success",
                prediction=res["prediction"],
                validity_status=res["validity_status"],
                inference_time=res["inference_time"],
                prompt_tokens=res["prompt_token_count"],
                generated_tokens=res["generated_token_count"],
                tokens_per_second=res["tokens_per_second"],
                error_message=None,
            )

            all_case_records.append({
                "test_case_id": case_id,
                "input_filename": filename,
                "prediction": res["prediction"],
                "validity_status": res["validity_status"],
                "raw_output": res["raw_generated_text"],
                "inference_time": res["inference_time"],
                "prompt_token_count": res["prompt_token_count"],
                "generated_token_count": res["generated_token_count"],
                "tokens_per_second": res["tokens_per_second"],
                "full_prompt": res["full_prompt"],
            })

            print(f"DONE -> {res['prediction']} ({res['inference_time']:.2f}s)")
            successful_count += 1

        except Exception as exc:
            print(f"FAILED: {exc}")
            logger.error(f"Error processing test case '{filename}': {exc}", exc_info=True)
            failed_count += 1
            append_run_log(
                log_path=config.RUN_LOG_PATH,
                test_case_id=case_id,
                input_filename=filename,
                output_filename=output_filename,
                status="failed",
                prediction="INVALID_OUTPUT",
                validity_status="INVALID",
                inference_time=0.0,
                error_message=str(exc),
            )

    # Save consolidated outputs
    with open(config.CONSOLIDATED_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(all_case_records, jf, indent=2, ensure_ascii=False)

    if all_case_records:
        csv_headers = [
            "test_case_id",
            "input_filename",
            "prediction",
            "validity_status",
            "inference_time",
            "prompt_token_count",
            "generated_token_count",
            "tokens_per_second",
            "raw_output",
        ]
        with open(config.CONSOLIDATED_CSV_PATH, "w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_headers, extrasaction="ignore")
            writer.writeheader()
            for rec in all_case_records:
                writer.writerow(rec)

    print("\n" + "=" * 70)
    print("Inference Run Complete")
    print(f"Total Cases     : {total_cases}")
    print(f"Successful      : {successful_count}")
    print(f"Failed          : {failed_count}")
    print(f"Skipped         : {skipped_count}")
    print(f"Outputs saved to: {args.output_dir}")
    print(f"Prompts saved to: {args.sent_prompts_dir}")
    print(f"Log saved to    : {config.RUN_LOG_PATH}")
    print(f"CSV saved to    : {config.CONSOLIDATED_CSV_PATH}")
    print(f"JSON saved to   : {config.CONSOLIDATED_JSON_PATH}")
    print("=" * 70)

    # 5. Benchmark Evaluation
    bench = evaluate_against_ground_truth(all_case_records)
    if bench:
        print("\n" + "=" * 70)
        print("POLICY COMPLIANCE BENCHMARK EVALUATION")
        print("=" * 70)
        print(f"Total Evaluated       : {bench['total']}")
        print(f"Correct Predictions   : {bench['correct']}")
        print(f"Incorrect Predictions : {bench['incorrect']}")
        print(f"Invalid Outputs       : {bench['invalid']}")
        print(f"Overall Accuracy      : {bench['accuracy']}%")
        print(f"Valid-Only Accuracy   : {bench['accuracy_valid_only']}%")
        print("-" * 70)
        cm = bench["confusion_matrix"]
        print("Confusion Matrix:")
        print(f"  True PASS  (GT PASS -> Pred PASS): {cm['true_pass']}")
        print(f"  False FAIL (GT PASS -> Pred FAIL): {cm['false_fail']}")
        print(f"  True FAIL  (GT FAIL -> Pred FAIL): {cm['true_fail']}")
        print(f"  False PASS (GT FAIL -> Pred PASS): {cm['false_pass']}")
        print(f"  Invalid Outputs                  : {cm['invalid']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
