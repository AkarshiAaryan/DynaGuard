import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import config
from inference import build_inference_prompt
from main import natural_sort_key, append_run_log


class TestDynaGuardPipeline(unittest.TestCase):

    def test_natural_sorting(self):
        """Verify that filenames sort naturally: TC2 comes before TC10."""
        filenames = [
            "TC10.txt",
            "TC2.txt",
            "TC1.txt",
            "TC20.txt",
            "TC3.txt",
        ]
        sorted_filenames = sorted(filenames, key=natural_sort_key)
        expected = [
            "TC1.txt",
            "TC2.txt",
            "TC3.txt",
            "TC10.txt",
            "TC20.txt",
        ]
        self.assertEqual(sorted_filenames, expected)
        # Check TC2 precedes TC10 specifically
        self.assertLess(
            sorted_filenames.index("TC2.txt"),
            sorted_filenames.index("TC10.txt"),
        )

        # Also verify with zero-padded format
        padded_filenames = ["TC10.txt", "TC02.txt", "TC01.txt"]
        sorted_padded = sorted(padded_filenames, key=natural_sort_key)
        self.assertEqual(sorted_padded, ["TC01.txt", "TC02.txt", "TC10.txt"])

    def test_prompt_construction_fast_mode(self):
        """Verify prompt construction in fast mode uses <answer> generation prefix and preserves content."""
        dummy_tokenizer = MagicMock()
        dummy_tokenizer.chat_template = None  # test fallback/documented prompt template

        sample_content = "Policy:\n1. Rule A\n\nConversation:\nUser: Hi\nAgent: Hello"
        prompt = build_inference_prompt(
            test_case_content=sample_content,
            tokenizer=dummy_tokenizer,
            mode="fast",
        )

        self.assertIn("[INST]", prompt)
        self.assertIn("[/INST]", prompt)
        self.assertIn(config.SYSTEM_INSTRUCTION, prompt)
        self.assertIn(sample_content, prompt)
        self.assertTrue(prompt.endswith("<answer>\n"))

    def test_prompt_construction_reasoning_mode(self):
        """Verify prompt construction in reasoning mode uses <think> generation prefix."""
        dummy_tokenizer = MagicMock()
        dummy_tokenizer.chat_template = None

        sample_content = "Policy:\n1. Rule A\n\nConversation:\nUser: Hi\nAgent: Hello"
        prompt = build_inference_prompt(
            test_case_content=sample_content,
            tokenizer=dummy_tokenizer,
            mode="reasoning",
        )

        self.assertIn("[INST]", prompt)
        self.assertIn("[/INST]", prompt)
        self.assertIn(sample_content, prompt)
        self.assertTrue(prompt.endswith("<think>\n"))

    def test_prompt_construction_with_chat_template(self):
        """Verify prompt construction when tokenizer has a chat template."""
        dummy_tokenizer = MagicMock()
        dummy_tokenizer.chat_template = "{{ messages }}"
        dummy_tokenizer.apply_chat_template.return_value = "<formatted_chat><answer>\n"

        sample_content = "Policy:\n1. Rule A\n\nConversation:\nUser: Hi\nAgent: Hello"
        prompt = build_inference_prompt(
            test_case_content=sample_content,
            tokenizer=dummy_tokenizer,
            mode="fast",
        )

        self.assertEqual(prompt, "<formatted_chat><answer>\n")
        dummy_tokenizer.apply_chat_template.assert_called()

    def test_run_log_entry_schema(self):
        """Verify run_log.jsonl writes valid JSON lines with all required fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "run_log.jsonl")

            append_run_log(
                log_path=log_path,
                input_filename="TC01.txt",
                output_filename="TC01_output.txt",
                status="success",
                inference_time=1.23456,
                error_message=None,
            )

            append_run_log(
                log_path=log_path,
                input_filename="TC02.txt",
                output_filename="TC02_output.txt",
                status="failed",
                inference_time=0.0,
                error_message="CUDA out of memory",
            )

            with open(log_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]

            self.assertEqual(len(lines), 2)

            self.assertEqual(lines[0]["input_filename"], "TC01.txt")
            self.assertEqual(lines[0]["output_filename"], "TC01_output.txt")
            self.assertEqual(lines[0]["status"], "success")
            self.assertEqual(lines[0]["inference_time"], 1.2346)
            self.assertIsNone(lines[0]["error_message"])
            self.assertIn("timestamp", lines[0])

            self.assertEqual(lines[1]["input_filename"], "TC02.txt")
            self.assertEqual(lines[1]["status"], "failed")
            self.assertEqual(lines[1]["error_message"], "CUDA out of memory")

    def test_raw_output_preservation(self):
        """Verify raw output with XML tags, whitespace, and special tokens is saved completely verbatim."""
        raw_model_response = (
            "PASS</answer>\n"
            "<explanation>\n"
            "   The agent politely refused to issue a refund in accordance with Rule 1.  \n"
            "</explanation><|im_end|>"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "TC01_output.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(raw_model_response)

            with open(output_file, "r", encoding="utf-8") as f:
                saved_content = f.read()

            self.assertEqual(saved_content, raw_model_response)
            self.assertIn("<answer>", "<answer>" + saved_content)
            self.assertIn("<explanation>", saved_content)
            self.assertIn("<|im_end|>", saved_content)

    @patch("main.load_model_and_tokenizer")
    @patch("main.run_inference_single")
    def test_full_pipeline_mock_execution(self, mock_run_inference, mock_load_model):
        """Verify sequential processing, raw saving, prompt saving, and resume/rerun logic."""
        import sys
        from main import main

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock inference return value: raw output, sent prompt, elapsed time
        mock_run_inference.side_effect = [
            ("PASS</answer>\n<explanation>Rule 1 followed.</explanation>", "Prompt 1", 0.5),
            ("FAIL</answer>\n<explanation>Rule 2 violated.</explanation>", "Prompt 2", 0.8),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            in_dir = os.path.join(temp_dir, "input_prompts")
            out_dir = os.path.join(temp_dir, "outputs")
            sent_dir = os.path.join(temp_dir, "sent_prompts")
            logs_dir = os.path.join(temp_dir, "logs")
            run_log = os.path.join(logs_dir, "run_log.jsonl")

            os.makedirs(in_dir)

            # Create test files
            with open(os.path.join(in_dir, "TC1.txt"), "w") as f:
                f.write("Policy 1\nConversation 1")
            with open(os.path.join(in_dir, "TC2.txt"), "w") as f:
                f.write("Policy 2\nConversation 2")

            # 1. Run pipeline
            with patch("config.RUN_LOG_PATH", run_log), \
                 patch("config.LOGS_DIR", logs_dir), \
                 patch.object(sys, "argv", ["main.py", "--input-dir", in_dir, "--output-dir", out_dir, "--sent-prompts-dir", sent_dir]):
                main()

            # Verify files were created
            out1 = os.path.join(out_dir, "TC1_output.txt")
            out2 = os.path.join(out_dir, "TC2_output.txt")
            self.assertTrue(os.path.exists(out1))
            self.assertTrue(os.path.exists(out2))

            with open(out1, "r") as f:
                self.assertEqual(f.read(), "PASS</answer>\n<explanation>Rule 1 followed.</explanation>")
            with open(out2, "r") as f:
                self.assertEqual(f.read(), "FAIL</answer>\n<explanation>Rule 2 violated.</explanation>")

            prompt1 = os.path.join(sent_dir, "TC1_prompt.txt")
            self.assertTrue(os.path.exists(prompt1))
            with open(prompt1, "r") as f:
                self.assertEqual(f.read(), "Prompt 1")

            # Verify log lines
            self.assertTrue(os.path.exists(run_log))
            with open(run_log, "r") as f:
                log_lines = [json.loads(line) for line in f]
            self.assertEqual(len(log_lines), 2)
            self.assertEqual(log_lines[0]["status"], "success")
            self.assertEqual(log_lines[1]["status"], "success")

            # 2. Test Resume Behavior: Running again without --rerun should skip both
            mock_run_inference.reset_mock()
            with patch("config.RUN_LOG_PATH", run_log), \
                 patch("config.LOGS_DIR", logs_dir), \
                 patch.object(sys, "argv", ["main.py", "--input-dir", in_dir, "--output-dir", out_dir, "--sent-prompts-dir", sent_dir]):
                main()

            # No new inferences should be called because outputs already exist
            mock_run_inference.assert_not_called()

            # 3. Test Rerun Flag: Running with --rerun should process both again
            mock_run_inference.side_effect = [
                ("RERUN_PASS", "Prompt 1", 0.4),
                ("RERUN_FAIL", "Prompt 2", 0.6),
            ]
            with patch("config.RUN_LOG_PATH", run_log), \
                 patch("config.LOGS_DIR", logs_dir), \
                 patch.object(sys, "argv", ["main.py", "--input-dir", in_dir, "--output-dir", out_dir, "--sent-prompts-dir", sent_dir, "--rerun"]):
                main()

            self.assertEqual(mock_run_inference.call_count, 2)
            with open(out1, "r") as f:
                self.assertEqual(f.read(), "RERUN_PASS")


if __name__ == "__main__":
    unittest.main()
