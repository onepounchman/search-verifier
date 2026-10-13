"""Keep library caches in isolated output; use short ephemeral IPC paths."""
import os
from pathlib import Path
root=os.environ.get('PILOT_OUTPUT_ROOT')
if root:
 base=Path(root)/'runtime_cache'/('process_'+str(os.getpid()))
 for variable,name in {'HF_HOME':'hf','HF_HUB_CACHE':'hf/hub','HUGGINGFACE_HUB_CACHE':'hf/hub','TRANSFORMERS_CACHE':'hf/transformers','HF_DATASETS_CACHE':'datasets','XDG_CACHE_HOME':'xdg','SGLANG_CACHE_DIR':'sglang','SGLANG_JIT_CACHE_DIR':'sglang-jit','FLASHINFER_WORKSPACE_BASE':'flashinfer','TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'torchinductor','TORCH_EXTENSIONS_DIR':'torch-extensions','CUDA_CACHE_PATH':'cuda-driver','CUPY_CACHE_DIR':'cupy','NUMBA_CACHE_DIR':'numba','TORCH_HOME':'torch'}.items():
  path=base/name;path.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(path)
 physical_tmp=base/'tmp';physical_tmp.mkdir(parents=True,exist_ok=True)
 ipc=Path('/dev/shm')/('pilot_ipc_'+str(os.getpid()))
 if not ipc.exists():ipc.symlink_to(physical_tmp,target_is_directory=True)
 os.environ['TMPDIR']=str(ipc)
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('USER','pilot');os.environ.setdefault('LOGNAME','pilot')
