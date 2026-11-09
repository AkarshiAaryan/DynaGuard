# DynaGuard chat-message inputs (100 test cases)

Each individual `TCxx_*.json` file contains only an array of chat message objects with `role` and `content` keys. It can be passed directly to the Transformers chat template / pipeline:

```python
import json
from transformers import pipeline

pipe = pipeline("text-generation", model="tomg-group-umd/DynaGuard-4B")
with open("TC01_student_own_grade.json", encoding="utf-8") as f:
    messages = json.load(f)
output = pipe(messages)
```

For batch evaluation, load `dynaguard_messages.json`, which is a list of objects containing `test_case_id`, `source_file`, and `messages`:

```python
import json
with open("dynaguard_messages.json", encoding="utf-8") as f:
    cases = json.load(f)

for case in cases:
    messages = case["messages"]
    output = pipe(messages)
    print(case["test_case_id"], output)
```

The policy is supplied in the first user message, beginning with `Policy:`. Conversation turns alternate `user` and `assistant`, ending with the agent's proposed action. The input contains no expected labels.

Files: `dynaguard_messages.json` (master), `TCxx_*.json` (one message array per case), `dynaguard_case_manifest.csv` (source metadata), and `dynaguard_validation_report.json` (source-match checks).
