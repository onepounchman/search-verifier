Qwen3.5-0.8B non-thinking confidence-verifier starter

This folder contains only the INITIAL starter for the six-hour 0.8B non-thinking
pilot, plus portable path wrappers, a scorer, a weak TRL LoRA example and docs.
It does not contain an optimized solution, calibration map, trained adapter,
other model sizes/thinking variants, model weights, credentials or real data.

Initial configuration
- Qwen/Qwen3.5-0.8B, revision 2fc06364715b967f1860aea9cf38778875588b17.
- Single GPU, BF16, native 262144-token context, concurrency 32.
- enable_thinking=false; no JSON schema or constrained decoding.
- Prompt requests "Think step by step" verification in ordinary completion,
  then a final line containing only probability 0..1. Template pre-closes the
  native thinking section. Actual adherence to reasoning/format may vary.
- max_new_tokens 4096; native sampling preserved in config.json.
- Full recorded trajectory and tool definitions are passed without evidence
  selection or truncation. Over-context inputs produce invalid predictions.
- No reference answer or label is passed to the model.

Files
  config.json                 Exact original prompt and sampling settings.
  judge.py                    Standalone SGLang single-pass inference harness.
  probability_scoring.py      Exact pilot final-line parser and Brier scorer.
  evaluate.py                 Local inference+scoring or saved-output scoring.
  train_lora.py               Weak TRL-only, pure-LoRA training example.
  merge_adapter.py            Pure-LoRA merger preserving untargeted tensors.
  paths.py, env.sh             Explicit offline paths and model/cache storage.
  sitecustomize.py             Per-process cache and short IPC-path helper.
  requirements-*.txt          Separate training/serving environment versions.
  task_metadata.json          Split sizes, asset checksums and starter metrics.
  examples/                   Synthetic format examples only.
  MANIFEST.sha256             Package file checksums.

Dependencies and model assets
Use separate Python 3.12 training and serving virtual environments; their
PyTorch/Transformers/CUDA requirements differ. The requirement lists reflect
the tested pilot environments. A fresh container build has not been tested.
Prepare CUDA runtime/toolchain, GCC/G++ and Ninja during image build. Download
model/tokenizer assets at build time; task training/inference must be offline.
Model source: https://huggingface.co/Qwen/Qwen3.5-0.8B
All weights, LoRA adapters, optimizer/checkpoints, merged models and model caches
must stay under /srv/disk00/wenbo/cache. No model files are shipped here.

Set up, from this folder
  source env.sh
  export PILOT_SERVE_PYTHON=/absolute/path/to/serving-env/bin/python
  export PILOT_TRAIN_PYTHON=/absolute/path/to/training-env/bin/python
  export CUDA_HOME=/absolute/path/to/cuda13-toolchain
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export CUDA_VISIBLE_DEVICES=0

For the tested pip-packaged CUDA runtime, CUDA_HOME is the serving environment's
lib/python3.12/site-packages/nvidia/cu13 directory. If needed, set
RSI_BASE_MODEL_PATH to the provided read-only base-model directory, e.g.
/models/base. The default is the pinned snapshot in /srv/disk00/wenbo/cache/hub.
PILOT_OUTPUT_ROOT defaults to
/srv/disk00/wenbo/cache/task_contribution/starter_08b/outputs.

Run the synthetic format example (not benchmark data)
  "$PILOT_SERVE_PYTHON" judge.py \
    --inputs examples/toy_inputs.jsonl \
    --outputs "$PILOT_OUTPUT_ROOT/toy/outputs.jsonl"
  python evaluate.py \
    --inputs examples/toy_inputs.jsonl --labels examples/toy_labels.jsonl \
    --predictions "$PILOT_OUTPUT_ROOT/toy/outputs.jsonl" \
    --run-dir "$PILOT_OUTPUT_ROOT/toy_score"
Each scoring --run-dir must be new. Example IDs/labels are synthetic and must
not be used to reproduce the reported benchmark scores.

Benchmark data
Source: https://huggingface.co/datasets/liuqi6777/bcp_search_agent_trajectory
Upstream code: https://github.com/liuqi6777/search_agent
The sanitized, labeled train/dev assets must be provided separately. Loading
raw upstream files directly is not equivalent to the prepared benchmark split.
Required inputs: train_inputs.jsonl, train_labels.jsonl, train_groups.jsonl,
dev_inputs.jsonl, dev_labels.jsonl and dev_groups.jsonl.
Input row: {"example_id": "...", "input": {"question": "...", "messages": [...],
            "tool_registry": {"tools": [...]}}}
Label row: {"example_id": "...", "gold_label":0}
Group row: {"example_id": "...", "question_group": "opaque question hash"}
All instances must have unique IDs. Labels remain separate from model inputs.

Train: 2825 labeled trajectories / 561 questions (2224 positive, 601 negative).
Dev: 300 labeled trajectories / 70 questions (150 positive, 150 negative).
Hidden: 500 trajectories / 198 questions (253 positive, 247 negative).
All splits are question-disjoint; audited source input length is below 200000
Qwen judge-input tokens. No hidden examples, labels or predictions are bundled.

Dev evaluation (single GPU, total 7200-second suite deadline)
  python evaluate.py \
    --inputs /provided/data/dev_inputs.jsonl \
    --labels /provided/data/dev_labels.jsonl \
    --run-dir "$PILOT_OUTPUT_ROOT/dev_run" \
    --serve-python "$PILOT_SERVE_PYTHON"
Mean Brier averages squared probability-vs-label differences over ALL assigned
instances. Missing/invalid/nonfinite/out-of-range probabilities incur loss 1.
The parser reads the last nonempty line as a bare numeric probability, without
regex repair, Markdown stripping, confidence-label removal or clamping.
Transport outputs are JSONL with example_id, confidence, raw_text and timings.
A JSONL transport record is distinct from the model's final bare-number line.

Training example
Create a JSON list of TRAIN-only IDs, then:
  "$PILOT_TRAIN_PYTHON" train_lora.py \
    --train-inputs /provided/data/train_inputs.jsonl \
    --train-labels /provided/data/train_labels.jsonl \
    --train-ids /path/to/train_ids.json \
    --output-dir "$PILOT_OUTPUT_ROOT/lora_example" --max-steps 32
This weak starting recipe uses TRL SFTTrainer and PEFT pure LoRA. Its supplied
binary labels become 0.0/1.0 completion targets; no teacher rationales or fitted
calibration are included. After training, config.json adapter_path may point to
the saved adapter. The baseline configuration keeps adapter_path=null.
Dev can be inspected and evaluated repeatedly, but cannot supply training data,
synthetic training targets or fitted calibration parameters. Any learned
calibration uses train-only question-disjoint holdouts. Original base weights,
model architecture and tokenizer are fixed. No external model or search calls.

Measured initial starter (pilot cp_20261013_072041)
  Dev:   Brier 0.3394493333, 25 invalid /300.
  Hidden:Brier 0.3573658000, 44 invalid /500.
The exported prompt, sampling configuration, message construction and parser
match the initial starter. Path wrappers do not change the judging method.
Native stochastic sampling may vary with server scheduling/batching.

Scope
This is a portable starter-code package, not a complete sealed RSI-Exam image.
The local evaluator does not isolate filesystem access or impose CPU/RAM/GPU
quotas. Final benchmark integration must mount inputs only for participant
inference, keep hidden labels outside that process/container, enforce one GPU
and the suite deadline, and supply approved offline assets at build time.
Do not put private grading files or prior solutions in the agent image.
