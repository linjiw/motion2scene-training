"""CUDA execution and memory checks; NVML is diagnostic, not an execution gate."""

import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def cuda_preflight(minimum_free_bytes=2 * 1024**3):
    import torch

    started = time.perf_counter()
    nvml = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    result = {
        "nvml_exit": nvml.returncode,
        "nvml_output": nvml.stdout + nvml.stderr,
        "cuda_available": torch.cuda.is_available(),
        "minimum_free_bytes": minimum_free_bytes,
    }
    try:
        x = torch.ones((128, 128), device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        assert float(y[0, 0]) == 128.0
        del x, y
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        result.update(
            cuda_execution_passed=True,
            device=torch.cuda.get_device_name(0),
            free_bytes=free,
            total_bytes=total,
            launch_allowed=free >= minimum_free_bytes,
        )
    except Exception as error:
        result.update(cuda_execution_passed=False, launch_allowed=False, error=repr(error))
    result["wall_seconds"] = time.perf_counter() - started
    return result


def load_release_checkpoint(path):
    """Use the same old-TRL class compatibility mapping as the existing trainer."""
    import torch
    from trl.experimental.ppo import ppo_trainer
    from trl.trainer import utils

    utils.OnlineTrainerState = ppo_trainer.OnlineTrainerState
    utils.exact_div = ppo_trainer.exact_div
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
