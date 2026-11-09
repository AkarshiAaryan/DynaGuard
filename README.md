# DynaGuard-8B Minimal Inference Pipeline

A lightweight, standalone, and transparent Python inference pipeline for evaluating the **DynaGuard-8B** guardian model on collections of test cases stored as individual `.txt` files.

This project is strictly an **inference wrapper**. It reads test cases, sends them to DynaGuard using the model's intended prompting conventions, and saves the raw generated text verbatim without parsing, extracting labels, stripping XML tags, or altering formatting.

---

## Project Structure

```text
Dynaguard/
├── config.py              # Centralized configuration (model ID, paths, parameters, system prompt)
├── model_loader.py        # Model & tokenizer loading (Hugging Face Transformers, device/dtype management)
├── inference.py           # Prompt construction, model generation, and raw token decoding
├── main.py                # Pipeline orchestration, natural sorting, resume/rerun logic, progress reporting
├── dynaguard_inference.ipynb # Interactive notebook (100% Kaggle & Google Colab compatible)
├── requirements.txt       # Dependencies (transformers, torch, accelerate)
├── README.md              # Documentation and usage guide
├── input_prompts/         # User-provided .txt test cases (TC01.txt, TC02.txt, ...)
├── outputs/               # Authoritative raw generated model responses (TC01_output.txt, ...)
├── sent_prompts/          # Archived verbatim prompts sent to the model for reproducibility
├── logs/                  # Execution log (run_log.jsonl)
└── tests/
    └── test_pipeline.py   # Unit and integration test suite
```

---

## 1. Installation

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support recommended for 8B models, or CPU)

### Setup
```bash
# Clone the repository
git clone https://github.com/AkarshiAaryan/DynaGuard.git
cd DynaGuard

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch:** For GPU acceleration with CUDA, install the PyTorch build matched to your CUDA version (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

---

## 2. Configuration

All primary settings are managed in [`config.py`](config.py) and can also be overridden via command-line arguments:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `MODEL_ID` | `"tomg-group-umd/DynaGuard-8B"` | Hugging Face model repository or local path to model weights |
| `INFERENCE_MODE` | `"fast"` | `"fast"` (direct `<answer>` prefix) or `"reasoning"` (`<think>` prefix) |
| `INPUT_DIR` | `"input_prompts"` | Directory containing input `.txt` test case files |
| `OUTPUT_DIR` | `"outputs"` | Directory where raw generated outputs are saved |
| `SENT_PROMPTS_DIR` | `"sent_prompts"` | Directory where exact prompts sent to the model are saved |
| `MAX_NEW_TOKENS` | `256` | Maximum new tokens to generate |
| `TEMPERATURE` | `0.0` | `0.0` for greedy decoding / deterministic evaluation |
| `TOP_P` | `0.95` | Nucleus sampling probability threshold |
| `DEVICE_MAP` | `"auto"` | Model device placement strategy (`"auto"`, `"cpu"`, `"cuda"`) |
| `TORCH_DTYPE` | `"auto"` | Weight precision (`"auto"`, `"bfloat16"`, `"float16"`, `"float32"`) |
| `SYSTEM_INSTRUCTION` | *(string)* | System instruction governing policy compliance evaluation |

---

## 3. Adding Test Cases

Place your `.txt` test files into the `input_prompts/` directory:

```text
input_prompts/
    TC01.txt
    TC02.txt
    TC10.txt
```

### File Format Requirements
- Each file represents one complete test case.
- The file contains both the policy/rules and the user-agent conversation.
- The pipeline reads each file **in full and completely unmodified** (no summarization, regex parsing, or splitting).
- Filenames are sorted naturally so `TC2.txt` is processed before `TC10.txt`.

---

## 4. Running the Pipeline

### Standard Run
Run inference over all `.txt` files in `input_prompts/`:

```bash
python main.py
```

### Resuming an Interrupted Run
By default, the pipeline **skips existing non-empty output files** in `outputs/`. If a run is interrupted, simply re-run:

```bash
python main.py
```
It will automatically skip already completed test cases and resume where it left off.

### Forcing a Rerun
To re-evaluate and overwrite existing test case outputs:

```bash
python main.py --rerun
```

### Command-Line Overrides
You can customize model paths, inference modes, or directories directly via CLI flags:

```bash
# Run using local model weights
python main.py --model-id /path/to/local/DynaGuard-8B

# Run in reasoning inference mode (Chain-of-Thought <think> prefix)
python main.py --mode reasoning --max-new-tokens 1024

# Custom input and output directories
python main.py --input-dir my_tests/ --output-dir my_outputs/
```

---

## 5. Kaggle & Google Colab Notebook (`dynaguard_inference.ipynb`)

For running in interactive cloud GPU environments, the repository includes [`dynaguard_inference.ipynb`](dynaguard_inference.ipynb):

- **Automatic Environment Detection**: Automatically detects Google Colab (`/content`) and Kaggle (`/kaggle/working`).
- **Adaptive Precision**: Detects GPU capabilities and selects `bfloat16` for Ampere/Hopper (A100, L4) or `float16` for Turing (T4), preventing out-of-memory errors on free-tier T4 GPUs.
- **Exact Official Prompting**: Implements DynaGuard's chat template standard:
  ```python
  inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt",
  ).to(model.device)

  outputs = model.generate(**inputs, max_new_tokens=256)
  response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
  ```
- **Self-Contained Sample Setup**: Pre-populates sample test cases if the input directory is empty, allowing immediate end-to-end execution.

---

## 6. DynaGuard Inference Modes

DynaGuard supports dual inference modes:

1. **Fast Inference (`--mode fast`, Default)**:
   - Guided by the generation prefix `<answer>`.
   - Generates the direct classification and brief explanation:
     ```text
     PASS</answer>
     <explanation>The agent followed all rules.</explanation>
     ```

2. **Reasoning Inference (`--mode reasoning`)**:
   - Guided by the generation prefix `<think>`.
   - Generates reasoning trace followed by final verdict:
     ```text
     Reasoning steps evaluating each rule...
     </think>
     <answer>PASS</answer>
     ```

---

## 7. Output Files & Logs

### Raw Model Outputs (`outputs/`)
Each output file contains **strictly the raw decoded text** produced by DynaGuard:
```text
outputs/
    TC01_output.txt
    TC02_output.txt
    TC10_output.txt
```

> **Critical Design Principle:**
> Outputs are decoded with `skip_special_tokens=False` and written directly to disk. The pipeline does **not** extract `PASS`/`FAIL`, parse XML, strip whitespace, or convert responses into JSON.

### Prompt Reproducibility Archive (`sent_prompts/`)
The exact full prompt string sent to the tokenizer for each test case is archived in `sent_prompts/`:
```text
sent_prompts/
    TC01_prompt.txt
    TC02_prompt.txt
    TC10_prompt.txt
```

### Supplementary Execution Log (`logs/run_log.jsonl`)
Every run appends execution metadata to `logs/run_log.jsonl`:
```json
{
  "input_filename": "TC01.txt",
  "output_filename": "TC01_output.txt",
  "status": "success",
  "inference_time": 1.42,
  "error_message": null,
  "timestamp": "2026-11-09T15:10:00.123456+00:00"
}
```

---

## 8. Running Unit Tests

Run the test suite to verify natural sorting, prompt generation, raw preservation, and resume logic:

```bash
python3 -m unittest tests/test_pipeline.py
```
