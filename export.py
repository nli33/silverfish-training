import torch
import numpy as np
from model import NNUE

SCALE = 256
OUT_FILE = "engine.nnue"


def quantize(w, scale):
    q = np.round(w * scale)
    q = np.clip(q, -32768, 32767)
    return q.astype(np.int16)


def quantize_bias(b, scale):
    q = np.round(b * scale)
    q = np.clip(q, -2**31, 2**31 - 1)
    return q.astype(np.int32)


def export_nnue(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = NNUE()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # extract weights
    in_w = model.input.weight.detach().cpu().numpy()   # [1024, 768]
    in_b = model.input.bias.detach().cpu().numpy()     # [1024]

    out_w = model.output.weight.detach().cpu().numpy() # [1, 2048]
    out_b = model.output.bias.detach().cpu().numpy()   # [1]

    # quantize
    in_w_q = quantize(in_w, SCALE)
    in_b_q = quantize_bias(in_b, SCALE)

    out_w_q = quantize(out_w, SCALE)
    out_b_q = quantize_bias(out_b, SCALE)

    # write binary
    with open(OUT_FILE, "wb") as f:
        f.write(in_w_q.astype("<i2").tobytes())
        f.write(in_b_q.astype("<i4").tobytes())
        f.write(out_w_q.astype("<i2").tobytes())
        f.write(out_b_q.astype("<i4").tobytes())

    print(f"Exported NNUE to {OUT_FILE}")
    print("Scale:", SCALE)
    print("Input layer:", in_w_q.shape)
    print("Output layer:", out_w_q.shape)


if __name__ == "__main__":
    export_nnue("nnue_checkpoint.pt")

