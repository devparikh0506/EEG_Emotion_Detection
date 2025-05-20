import torch
import numpy as np
from configs.runtime import device
def moveTo(obj, device):
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def initialize_torch():
    torch.backends.cudnn.deterministic=True
    set_seed(42)
    print(f"Using device: {device}")