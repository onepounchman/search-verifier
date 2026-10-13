#!/usr/bin/env bash
# Source before launching serving or training. Model work is offline.
export HF_HOME=/srv/disk00/wenbo/cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PILOT_OUTPUT_ROOT="${PILOT_OUTPUT_ROOT:-/srv/disk00/wenbo/cache/task_contribution/starter_08b/outputs}"
export TORCH_HOME="$PILOT_OUTPUT_ROOT/torch"
export TRITON_CACHE_DIR="$PILOT_OUTPUT_ROOT/runtime/triton"
export TORCHINDUCTOR_CACHE_DIR="$PILOT_OUTPUT_ROOT/runtime/torchinductor"
export TORCH_EXTENSIONS_DIR="$PILOT_OUTPUT_ROOT/runtime/extensions"
export CUDA_CACHE_PATH="$PILOT_OUTPUT_ROOT/runtime/cuda"
export XDG_CACHE_HOME="$PILOT_OUTPUT_ROOT/runtime"
export TMPDIR="$PILOT_OUTPUT_ROOT/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MAX_JOBS="${MAX_JOBS:-8}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export USER="${USER:-starter}" LOGNAME="${LOGNAME:-starter}"
export RSI_BASE_MODEL_PATH="${RSI_BASE_MODEL_PATH:-/srv/disk00/wenbo/cache/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"
RSI_STARTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$RSI_STARTER_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PILOT_OUTPUT_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
