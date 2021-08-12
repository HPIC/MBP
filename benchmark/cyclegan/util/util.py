from pathlib import Path

import torch


def ensure_dir(dir_path: str):
    path = Path(dir_path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=False)


def ensure_file_can_create(path):
    p = Path(path)
    p.resolve()
    if p.is_file:
        ensure_dir(p.parent)
    elif p.is_dir:
        ensure_dir(p)


def prepare_device(n_gpu_use: int = 0, target: int = 0):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel or DistributedDataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device(f"cuda:{target}" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids
