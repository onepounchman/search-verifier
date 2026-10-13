# Search Verifier

**Calibrated verification of long-horizon search-agent trajectories.**

A Qwen3.5-0.8B starter for an RSI-Exam task on jointly improving a verification harness and a judge model through LoRA. Given a question and a recorded search trajectory, the judge estimates the probability that the final answer is correct.

The goal is to **minimize mean Brier score** on trajectories from unseen questions. Improvements can come from better rubrics, evidence selection, verification workflows, training examples, LoRA updates, and probability calibration.

## At a glance

| Component | Starter configuration |
| :--- | :--- |
| Model | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), BF16 |
| Harness | Single-pass inference with SGLang |
| Input | Question, full recorded trajectory, and tool definitions |
| Output | Verification text followed by a bare probability between 0 and 1 |
| Generation | Non-thinking mode, up to 4,096 generated tokens |
| Serving | One GPU, concurrency 32, native 262,144-token context |
| Model updates | Pure LoRA through TRL; base architecture and tokenizer remain fixed |
| Evaluation | Mean Brier score, lower is better; two-hour suite deadline |

The prompt asks the model to **“Think step by step”** in its ordinary completion. Native thinking is disabled (`enable_thinking=false`), and no JSON schema or constrained decoding is used. The starter passes full records without evidence selection or truncation; over-context inputs yield invalid predictions.

## Measured baseline

| Split | Trajectories | Mean Brier ↓ | Invalid outputs |
| :--- | ---: | ---: | ---: |
| Visible dev | 300 | **0.339** | 25 |
| Hidden test | 500 | **0.357** | 44 |

These are the initial starter measurements from pilot `cp_20261013_072041`. The exported prompt, sampling settings, message construction, and parser match that starter. Stochastic sampling and server batching can affect reproduction.

A constant prediction of 0.5 scores **0.250** on either split, providing a useful reference.

## Setup

Use **separate Python 3.12 environments** for serving and training. Their pinned PyTorch, Transformers, and CUDA requirements differ:

- [requirements-serving.txt](requirements-serving.txt) — SGLang inference.
- [requirements-training.txt](requirements-training.txt) — TRL and PEFT training.

Install dependencies and prepare CUDA, GCC/G++, and Ninja during image build. Download model and tokenizer assets at build time; training and evaluation run offline. The requirements reflect the pilot environments; a fresh container build has not been tested.

From the repository directory:

```bash
source env.sh
export PILOT_SERVE_PYTHON=/absolute/path/to/serving-env/bin/python
export PILOT_TRAIN_PYTHON=/absolute/path/to/training-env/bin/python
export CUDA_HOME=/absolute/path/to/cuda13-toolchain
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
```

For the tested pip-packaged CUDA runtime, `CUDA_HOME` points to the serving environment's `lib/python3.12/site-packages/nvidia/cu13` directory.

| Environment variable | Default / purpose |
| :--- | :--- |
| `RSI_BASE_MODEL_PATH` | Pinned model snapshot under `/srv/disk00/wenbo/cache/hub` |
| `PILOT_OUTPUT_ROOT` | `/srv/disk00/wenbo/cache/task_contribution/starter_08b/outputs` |

All model weights, caches, adapters, optimizer states, checkpoints, and merged models must remain under **`/srv/disk00/wenbo/cache`**. Model revision: `2fc06364715b967f1860aea9cf38778875588b17`.

## Quickstart

Run inference on the included **synthetic format examples**:

```bash
"$PILOT_SERVE_PYTHON" judge.py \
  --inputs examples/toy_inputs.jsonl \
  --outputs "$PILOT_OUTPUT_ROOT/toy/outputs.jsonl"
```

Score the saved predictions:

```bash
python evaluate.py \
  --inputs examples/toy_inputs.jsonl \
  --labels examples/toy_labels.jsonl \
  --predictions "$PILOT_OUTPUT_ROOT/toy/outputs.jsonl" \
  --run-dir "$PILOT_OUTPUT_ROOT/toy_score"
```

Use a new `--run-dir` for each evaluation. The toy examples demonstrate the file format; they are not benchmark data.

## Data and evaluation

The task draws on the [six-model BrowseComp-Plus trajectory release](https://huggingface.co/datasets/liuqi6777/bcp_search_agent_trajectory) and its [upstream search-agent code](https://github.com/liuqi6777/search_agent). Prepared, labeled data are supplied separately; raw upstream files are not the prepared benchmark split.

| Split | Trajectories | Questions | Correct / incorrect | Use |
| :--- | ---: | ---: | :--- | :--- |
| Train | 2,825 | 561 | 2,224 / 601 | Training and calibration fitting |
| Visible dev | 300 | 70 | 150 / 150 | Inspection, repeated evaluation, and selection |
| Hidden test | 500 | 198 | 253 / 247 | Final grading |

All splits are **disjoint by question**. Retained records have audited source input lengths below 200,000 Qwen judge-input tokens. Correctness labels and reference answers are excluded from model inputs.

Dev must not be used for training, constructing training examples, or fitting calibration parameters. Use train-only question-disjoint holdouts for learned calibration.

<details>
<summary><strong>Prepared data files and schemas</strong></summary>

For each visible split, provide `*_inputs.jsonl`, `*_labels.jsonl`, and `*_groups.jsonl`. Every example must have a unique ID.

Input row:

```json
{
  "example_id": "example-001",
  "input": {
    "question": "Question text",
    "messages": [],
    "tool_registry": {"tools": []}
  }
}
```

Label row:

```json
{"example_id": "example-001", "gold_label": 0}
```

Question-group row:

```json
{"example_id": "example-001", "question_group": "opaque-question-hash"}
```

</details>

Run dev inference and scoring:

```bash
python evaluate.py \
  --inputs /provided/data/dev_inputs.jsonl \
  --labels /provided/data/dev_labels.jsonl \
  --run-dir "$PILOT_OUTPUT_ROOT/dev_run" \
  --serve-python "$PILOT_SERVE_PYTHON"
```

The default suite deadline is **7,200 seconds**, including initialization, adapter merging, preprocessing, inference, and output collection.

### Output and scoring

The model may write verification text, but its **last nonempty line must be a bare numeric probability**:

```text
The retrieved evidence supports the final answer and its constraints.
0.85
```

Mean Brier score averages the squared difference between the predicted probability and the binary correctness label. It measures both discrimination and calibration. All assigned examples remain in the denominator; missing, malformed, nonfinite, or out-of-range predictions receive **loss 1**.

The parser does not repair malformed output, strip Markdown, remove confidence labels, or clamp probabilities. Saved outputs use JSONL records containing `example_id`, `confidence`, `raw_text`, and timings; this storage format does not change the model's bare-number output requirement.

## LoRA training

Create a JSON list of train-only example IDs, then run the supplied TRL recipe:

```bash
"$PILOT_TRAIN_PYTHON" train_lora.py \
  --train-inputs /provided/data/train_inputs.jsonl \
  --train-labels /provided/data/train_labels.jsonl \
  --train-ids /path/to/train_ids.json \
  --output-dir "$PILOT_OUTPUT_ROOT/lora_example" \
  --max-steps 32
```

This simple starting recipe uses **TRL `SFTTrainer` and PEFT LoRA**, converting binary labels into `0.0` or `1.0` completion targets. After training, set `adapter_path` in [config.json](config.json) to the saved adapter. The untouched baseline uses `null`.

Model updates must use pure LoRA through TRL. Original base weights, model architecture, and tokenizer remain fixed. No external models or search services are used.

## Repository map

| File | Purpose |
| :--- | :--- |
| [config.json](config.json) | Original rubric, prompt, and sampling settings |
| [judge.py](judge.py) | SGLang inference harness |
| [evaluate.py](evaluate.py) | Inference and scoring, or scoring saved outputs |
| [probability_scoring.py](probability_scoring.py) | Final-line probability parser and Brier scorer |
| [train_lora.py](train_lora.py) | TRL LoRA training example |
| [merge_adapter.py](merge_adapter.py) | Pure-LoRA adapter merger |
| [env.sh](env.sh), [paths.py](paths.py), [sitecustomize.py](sitecustomize.py) | Offline cache, model storage, and runtime paths |
| [task_metadata.json](task_metadata.json) | Split counts, data hashes, and starter measurements |
| [verification.json](verification.json) | Export verification record |
| [examples/](examples/) | Synthetic inputs, labels, and ID-list examples |

<details>
<summary><strong>Package scope and benchmark integration</strong></summary>

This repository packages the initial Qwen3.5-0.8B non-thinking starter. Model weights, real trajectory data, hidden grading files, optimized solutions, trained adapters, and credentials are supplied separately or excluded.

The local evaluator is a development tool. For sealed benchmark integration, supply approved assets at build time, disable network access, enforce one GPU and the suite deadline, and keep hidden labels outside the participant process/container. The local evaluator does not enforce filesystem isolation or CPU/RAM/GPU quotas.

</details>
