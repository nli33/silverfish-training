"""
export_nnue_float.py

Exports a trained PyTorch NNUE model to a float32 .nnue file.
No quantization. Reference implementation for correctness.

Usage:
    python export_nnue_float.py nnue.pt model.nnue
"""

import sys
import struct
from pathlib import Path
import torch
import numpy as np
from model import NUM_INPUTS, L1


def main():
    if len(sys.argv) != 3:
        print("Usage: python export_nnue_float.py nnue.pt out.nnue")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)

    # Extract weights
    W_in = state_dict["input.weight"].cpu().numpy().astype(np.float32)
    b_in = state_dict["input.bias"].cpu().numpy().astype(np.float32)
    W_out = state_dict["output.weight"].cpu().numpy().astype(np.float32)
    b_out = state_dict["output.bias"].cpu().numpy().astype(np.float32)

    assert W_in.shape == (L1, NUM_INPUTS)
    assert b_in.shape == (L1,)
    assert W_out.shape == (1, 2 * L1)
    assert b_out.shape == (1,)

    with open(out_path, "wb") as f:
        f.write(b"NNUE")
        f.write(struct.pack("<I", 1))           # version
        f.write(struct.pack("<I", NUM_INPUTS))
        f.write(struct.pack("<I", L1))

        # Weights (row-major)
        params = {
            "W_in": W_in,
            "b_in": b_in,
            "W_out": W_out,
            "b_out": b_out,
        }
        
        for name, param in params.items():
            f.write(param.tobytes(order="C"))
            print(name, param.shape)

    print(f"Wrote float NNUE to {out_path}")


if __name__ == "__main__":
    main()

