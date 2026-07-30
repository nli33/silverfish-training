"""
checksum_pt.py

Computes a deterministic checksum of a PyTorch .pt file
based on tensor contents (not raw file bytes).

Usage:
    python checksum_pt.py nnue.pt
"""

import sys
import torch
import numpy as np
from pathlib import Path


def tensor_checksum(t: torch.Tensor) -> float:
    """
    Compute a stable checksum for a tensor.
    Uses float64 accumulation for numerical stability.
    """
    return float(t.double().sum().item())


def model_checksum(state_dict: dict) -> float:
    """
    Compute checksum over all tensors in the state_dict.
    Order is fixed by sorted keys.
    """
    checksum = 0.0
    for key in sorted(state_dict.keys()):
        t = state_dict[key]
        if torch.is_tensor(t):
            checksum += tensor_checksum(t)
    return checksum


def main():
    if len(sys.argv) != 2:
        print("Usage: python checksum_pt.py nnue.pt")
        sys.exit(1)

    path = Path(sys.argv[1])
    ckpt = torch.load(path, map_location="cpu")

    state_dict = ckpt.get("model_state", ckpt)

    checksum = model_checksum(state_dict)

    print(f"PyTorch model checksum: {checksum:.10f}")


if __name__ == "__main__":
    main()

