"""Offline paths; all model artifacts and library caches stay under /srv cache."""
import os
from pathlib import Path

CACHE_ROOT = Path('/srv/disk00/wenbo/cache')
MODEL_PATH = Path(os.environ.get('RSI_BASE_MODEL_PATH', str(CACHE_ROOT / 'hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17')))


def validate_model_output(value):
    path = Path(value).resolve()
    if not path.is_relative_to(CACHE_ROOT.resolve()):
        raise ValueError('Model artifacts must remain under /srv/disk00/wenbo/cache')
    return path


def configure_runtime():
    out = validate_model_output(os.environ.get('PILOT_OUTPUT_ROOT', str(CACHE_ROOT / 'task_contribution/starter_08b/outputs')))
    out.mkdir(parents=True, exist_ok=True)
    os.environ['PILOT_OUTPUT_ROOT'] = str(out)
    locations = {'HF_HOME':'hf', 'HF_HUB_CACHE':'hf/hub', 'HUGGINGFACE_HUB_CACHE':'hf/hub',
                 'HF_XET_CACHE':'xet', 'HF_DATASETS_CACHE':'datasets', 'TORCH_HOME':'torch',
                 'XDG_CACHE_HOME':'runtime', 'TRITON_CACHE_DIR':'runtime/triton',
                 'TORCHINDUCTOR_CACHE_DIR':'runtime/torchinductor', 'TORCH_EXTENSIONS_DIR':'runtime/extensions',
                 'CUDA_CACHE_PATH':'runtime/cuda', 'FLASHINFER_WORKSPACE_BASE':'runtime/flashinfer',
                 'UV_CACHE_DIR':'uv', 'CUPY_CACHE_DIR':'runtime/cupy', 'NUMBA_CACHE_DIR':'runtime/numba'}
    for key, suffix in locations.items():
        path = out / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    physical_tmp = out / 'tmp'
    physical_tmp.mkdir(parents=True, exist_ok=True)
    ipc = Path('/dev/shm') / ('starter08b_' + str(os.getpid()))
    if not ipc.exists():
        ipc.symlink_to(physical_tmp, target_is_directory=True)
    os.environ['TMPDIR'] = str(ipc)
    for key in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE']:
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('USER', 'starter')
    os.environ.setdefault('LOGNAME', 'starter')
    here = str(Path(__file__).resolve().parent)
    os.environ['PYTHONPATH'] = here + os.pathsep + os.environ.get('PYTHONPATH', '')
