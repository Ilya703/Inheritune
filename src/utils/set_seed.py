import random
import numpy as np
import torch
from accelerate.state import AcceleratorState
from accelerate.utils.imports import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_torch_xla_available,
    is_xpu_available
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

'''
    Функция для фиксации seed
'''

def set_seed(seed: int, device_specific: bool = False, deterministic: bool = False):
    if device_specific:
        seed += AcceleratorState().process_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_xpu_available():
        torch.xpu.manual_seed_all(seed)
    elif is_npu_available():
        torch.npu.manual_seed_all(seed)
    elif is_mlu_available():
        torch.mlu.manual_seed_all(seed)
    elif is_musa_available():
        torch.musa.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    # safe to call this function even if cuda is not available
    if is_torch_xla_available():
        xm.set_rng_state(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)