"""Load the exact upstream FreqFusion implementation used by the models."""
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path

COMMIT = '3fb0c70637a3c194fb74294d3ce4681958b26241'
SHA256 = '5852efce406681ce6f3f7763ff8e034cab44f93cb0fdda0114130bc4c8c97e42'


@lru_cache(maxsize=1)
def upstream_module():
    default = Path(__file__).resolve().parents[1] / 'third_party/freqfusion/FreqFusion.py'
    path = Path(os.environ.get('FREQFUSION_SOURCE', default))
    if not path.is_file():
        raise FileNotFoundError('Run python scripts/fetch_freqfusion.py first')
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('FreqFusion source differs from the pinned revision')
    spec = importlib.util.spec_from_file_location('_maize_freqfusion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from mmcv.ops.carafe import CARAFEFunction
    if getattr(module.carafe, '__self__', None) is not CARAFEFunction:
        raise RuntimeError('Install MMCV with its compiled CUDA CARAFE operator')
    return module
