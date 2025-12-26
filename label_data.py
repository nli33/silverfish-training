# attach evaluations to fens

import chess
import chess.engine
import csv
from tqdm import tqdm

engine = chess.engine.SimpleEngine.popen_uci("./stockfish")

MATE = 1500

source = "train.csv" # file with old evals
dest = "train_stockfish_evals.csv"


def centipawns(info, board):
    pov = info["score"].pov(board.turn)

    if pov.is_mate():
        mate = pov.mate()  # int, positive = winning
        return MATE if mate > 0 else -MATE

    cp = pov.score(mate_score=MATE)
    return max(-MATE, min(MATE, cp))

rows = [("FEN", "Evaluation")]

print("Reading")

with open(source, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in tqdm(reader):
        if len(row) < 1:
            continue
        fen = row[0].strip()
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=7))
        rows.append((fen, centipawns(info, board)))

print("Writing")

with open(dest, "w") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Done")
