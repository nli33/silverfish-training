# use one (or several) models to evaluate a position

import chess
from dataset import encode_board
from model import NNUEModel
import torch
import sys


def print_params(model):
    for name, param in model.named_parameters():
        print(f"Layer name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Weights/Biases data:\n{param.data}\n")


def evaluate(board, pt_file):
    model = NNUEModel()
    checkpoint = torch.load(pt_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    w_features = encode_board(board, chess.WHITE)
    b_features = encode_board(board, chess.BLACK)

    x_white = torch.from_numpy(w_features).float()
    x_black = torch.from_numpy(b_features).float()
    stm = torch.tensor(board.turn, dtype=torch.bool)

    with torch.no_grad():
        value = model(
            x_white.unsqueeze(0),
            x_black.unsqueeze(0),
            stm.unsqueeze(0),
        )

    return value.item() * 1000

if __name__ == '__main__':
    fen = "6k1/5p1p/1q2p1p1/1PnpP3/3N4/1Pr5/P5PP/3QR1K1 w - - 3 37"
    board = chess.Board(fen)
    #fen = chess.STARTING_FEN

    pt_files = sys.argv[1:]
    for pt_file in pt_files:
        print(pt_file + ": ", end="")
        try:
            print(evaluate(board, pt_file))
        except Exception as e:
            print("Error:", e)
