# Label FENs with Stockfish score + WDL, in parallel.
#
# Depth bumped from the previous depth=7 to depth=12 -- deep enough to give
# the network a meaningfully stronger training signal without making a
# ~2M-position relabel intractable on a single machine (benchmarked at
# ~15-45ms/position at depth 12 with this repo's bundled Stockfish 17
# binary). WDL is captured alongside the centipawn score (via
# UCI_ShowWDL) so training can blend a calibrated win-probability target
# with the raw score, instead of regressing to centipawns alone.

import argparse
import csv
import multiprocessing as mp

import chess
import chess.engine
from tqdm import tqdm

MATE = 1500
DEPTH = 12
STOCKFISH_PATH = "./stockfish"

_engine = None


def _init_worker():
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _engine.configure({"UCI_ShowWDL": True, "Threads": 1})


def _label_one(fen):
    board = chess.Board(fen)
    info = _engine.analyse(board, chess.engine.Limit(depth=DEPTH))

    pov_score = info["score"].pov(board.turn)
    if pov_score.is_mate():
        mate = pov_score.mate()
        cp = MATE if mate > 0 else -MATE
    else:
        cp = max(-MATE, min(MATE, pov_score.score(mate_score=MATE)))

    pov_wdl = info["wdl"].pov(board.turn)
    win, draw, loss = pov_wdl.wins, pov_wdl.draws, pov_wdl.losses

    return fen, cp, win, draw, loss


def main():
    global DEPTH

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("dest")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--depth", type=int, default=DEPTH)
    args = parser.parse_args()

    DEPTH = args.depth

    with open(args.source, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        fens = [row[0].strip() for row in reader if row]

    print(f"Labeling {len(fens)} positions at depth {DEPTH} with {args.workers} workers")

    with open(args.dest, "w", newline="") as out, mp.Pool(args.workers, initializer=_init_worker) as pool:
        writer = csv.writer(out)
        writer.writerow(["FEN", "Evaluation", "WDL_Win", "WDL_Draw", "WDL_Loss"])
        for fen, cp, win, draw, loss in tqdm(pool.imap(_label_one, fens, chunksize=64), total=len(fens)):
            writer.writerow([fen, cp, win, draw, loss])

    print("Done ->", args.dest)


if __name__ == "__main__":
    main()
