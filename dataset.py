import chess
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PIECE_TO_INDEX = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def feature_index_for_perspective(perspective, piece_color, piece_type, sq: int) -> int:
    friendly = (piece_color == perspective)
    piece_idx = piece_type + (0 if friendly else 6)  # 0..11
    return 64 * piece_idx + sq  # 0..767


def encode_board(board: chess.Board, perspective: bool):
    x = np.zeros(768, dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        feature = feature_index_for_perspective(perspective, piece.color, PIECE_TO_INDEX[piece.piece_type], square)
        x[feature] = 1.0
    
    return x


class NNUEDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = []
        
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for fen, eval_cp in reader:
                self.rows.append((fen, int(eval_cp)))
        
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fen, eval_cp = self.rows[idx]
        board = chess.Board(fen)
        x_white = encode_board(board, perspective=chess.WHITE)
        x_black = encode_board(board, perspective=chess.BLACK)
        white_to_move = board.turn == chess.WHITE
        target = np.clip(eval_cp, -1000, 1000) / 1000.0

        return (
            torch.from_numpy(x_white),
            torch.from_numpy(x_black),
            torch.tensor(white_to_move, dtype=torch.bool),
            torch.tensor(target, dtype=torch.float32),
        )

if __name__ == '__main__':
    dataset = NNUEDataset("train.csv")

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    xw, xb, stm, y = dataset[0]
    assert xw.shape == (768,)
    assert xb.shape == (768,)

    print(stm, y)
    print(xw.sum(), xb.sum())   # should equal number of pieces
