import chess
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

FLIP = 0b111000

PIECE_TO_INDEX = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def feature(perspective, piece_color, piece_type, sq: int) -> int:
    friendly = (piece_color == perspective)
    piece_idx = piece_type + (0 if friendly else 6)  # 0..11
    if perspective == chess.BLACK:
        sq ^= FLIP
    return 64 * piece_idx + sq  # 0..767


assert feature(chess.WHITE, chess.WHITE, chess.PAWN, 12) \
        == feature(chess.BLACK, chess.BLACK, chess.PAWN, 12 ^ FLIP)


def encode_board(board: chess.Board, perspective: bool):
    x = np.zeros(768, dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        f = feature(perspective, piece.color, PIECE_TO_INDEX[piece.piece_type], square)
        x[f] = 1.0

    return x


_FEN_PIECE_TO_TYPE = {
    "p": PIECE_TO_INDEX[chess.PAWN], "n": PIECE_TO_INDEX[chess.KNIGHT],
    "b": PIECE_TO_INDEX[chess.BISHOP], "r": PIECE_TO_INDEX[chess.ROOK],
    "q": PIECE_TO_INDEX[chess.QUEEN], "k": PIECE_TO_INDEX[chess.KING],
}


def parse_fen_features(fen: str):
    """
    Fast path used by NNUEDataset: scans the FEN's piece-placement field
    directly instead of building a chess.Board (whose constructor does a
    lot of legality-related setup this training loop never needs). Only
    active-feature indices are returned, since a position has ~32 pieces
    at most out of 768 possible features -- letting NNUEDataset build the
    dense one-hot vector only at __getitem__ time, from a small precomputed
    cache, instead of re-parsing every FEN on every epoch.
    """
    board_part, turn_part = fen.split(" ", 2)[:2]
    white_idx = []
    black_idx = []
    rank, file = 7, 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            piece_type = _FEN_PIECE_TO_TYPE[ch.lower()]
            color = chess.WHITE if ch.isupper() else chess.BLACK
            sq = rank * 8 + file
            white_idx.append(feature(chess.WHITE, color, piece_type, sq))
            black_idx.append(feature(chess.BLACK, color, piece_type, sq))
            file += 1
    white_to_move = turn_part == "w"
    return white_idx, black_idx, white_to_move


class NNUEDataset(Dataset):
    def __init__(self, csv_path):
        self.entries = []

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            # rows may carry extra WDL_Win/WDL_Draw/WDL_Loss columns
            # (unused for now); only FEN and Evaluation are needed here.
            for row in reader:
                fen, eval_cp = row[0], int(row[1])
                white_idx, black_idx, white_to_move = parse_fen_features(fen)
                target = np.clip(eval_cp, -1000, 1000) / 1000.0
                self.entries.append((
                    np.array(white_idx, dtype=np.int64),
                    np.array(black_idx, dtype=np.int64),
                    white_to_move,
                    target,
                ))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        white_idx, black_idx, white_to_move, target = self.entries[idx]

        x_white = np.zeros(768, dtype=np.float32)
        x_white[white_idx] = 1.0
        x_black = np.zeros(768, dtype=np.float32)
        x_black[black_idx] = 1.0

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
