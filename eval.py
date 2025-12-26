import chess
from dataset import encode_board
from model import NNUE
import torch

pt_file = "nnue.pt"

def evaluate(board):
    model = NNUE()
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
    fen = "rnbq1bnr/ppppkppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b - - 3 3"
    board = chess.Board(fen)
    print(evaluate(board))
