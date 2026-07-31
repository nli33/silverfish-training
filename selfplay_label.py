"""Label self-play PGN games for NNUE fine-tuning.

Per position, the label blends two signals:
  - the engine's own search score at that ply (queried live via UCI against
    the silverfish binary that produced the games) -- this is the primary
    signal, since it reflects local static/tactical evaluation rather than
    the eventual game result.
  - the game's final outcome, from the mover's perspective, blended in with
    a weight that grows the closer the position is to the end of the game
    (TD-style distance discounting). This exists to nudge positions toward
    their actual outcome without letting one early transient advantage that
    was never converted dominate the label -- see the design discussion in
    conversation: pure outcome-only labeling systematically mislabels
    "had an edge, blundered it away" positions.

No Stockfish involvement here by design -- this is meant to be the
Stockfish-free refinement step after the initial distillation.

Parallelized across a worker pool (one silverfish process per worker),
same pattern as label_data.py, since this is otherwise a slow serial
UCI round-trip per position.

Usage:
  python selfplay_label.py games.pgn out.csv --engine ./silverfish [--depth 8] [--workers N]
"""
import argparse
import csv
import multiprocessing as mp
import subprocess

import chess
import chess.pgn

MATE = 1500
OUTCOME_CP = 1000  # magnitude used for the "won"/"lost" blend target
MAX_OUTCOME_WEIGHT = 0.5  # cap: never let outcome fully override own score
SKIP_OPENING_PLIES = 10  # book moves aren't the engine's own judgment
SKIP_ENDGAME_PLIES = 4   # last few plies are often trivial/near-mate

DEPTH = 8
_engine = None


class Engine:
    def __init__(self, path):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line or line.startswith(token):
                return

    def eval_cp(self, fen, depth):
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        last_score = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("info") and " score " in line:
                parts = line.split()
                i = parts.index("score")
                kind, val = parts[i + 1], int(parts[i + 2])
                last_score = MATE if (kind == "mate" and val > 0) else \
                    -MATE if kind == "mate" else \
                    max(-MATE, min(MATE, val))
            if line.startswith("bestmove"):
                break
        return last_score

    def close(self):
        self._send("quit")
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def outcome_from_stm(result, white_to_move):
    if result == "1/2-1/2":
        return 0
    white_won = result == "1-0"
    stm_won = white_won == white_to_move
    return OUTCOME_CP if stm_won else -OUTCOME_CP


def _init_worker(engine_path, depth):
    global _engine, DEPTH
    _engine = Engine(engine_path)
    DEPTH = depth


def _label_game_str(game_str):
    import io
    game = chess.pgn.read_game(io.StringIO(game_str))
    if game is None:
        return []
    result = game.headers.get("Result", "*")
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return []

    board = game.board()
    positions = []
    for move in game.mainline_moves():
        positions.append(board.fen())
        board.push(move)
    positions.append(board.fen())

    n = len(positions)
    lo = SKIP_OPENING_PLIES
    hi = n - SKIP_ENDGAME_PLIES
    rows = []
    for ply in range(max(lo, 0), max(hi, 0)):
        fen = positions[ply]
        white_to_move = fen.split(" ")[1] == "w"
        search_cp = _engine.eval_cp(fen, DEPTH)
        if search_cp is None:
            continue
        outcome_cp = outcome_from_stm(result, white_to_move)
        proximity = ply / max(n - 1, 1)
        weight = MAX_OUTCOME_WEIGHT * proximity
        target_cp = round((1 - weight) * search_cp + weight * outcome_cp)
        rows.append((fen, target_cp))
    return rows


def _split_games(pgn_path):
    """Split a PGN file into per-game text chunks without holding the
    whole file's parsed representation in memory -- games are separated
    by a blank line following the last movetext line."""
    games = []
    buf = []
    with open(pgn_path) as f:
        for line in f:
            if line.strip() == "" and buf and not buf[-1].startswith("["):
                games.append("".join(buf))
                buf = []
            else:
                buf.append(line)
    if buf:
        games.append("".join(buf))
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn")
    parser.add_argument("dest")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    games = _split_games(args.pgn)
    print(f"Labeling {len(games)} games at depth {args.depth} with {args.workers} workers")

    total = 0
    with open(args.dest, "w", newline="") as out, \
            mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.engine, args.depth)) as pool:
        writer = csv.writer(out)
        writer.writerow(["FEN", "Evaluation"])
        for i, rows in enumerate(pool.imap(_label_game_str, games, chunksize=1)):
            for fen, cp in rows:
                writer.writerow([fen, cp])
            total += len(rows)
            if (i + 1) % 200 == 0:
                print(f"{i + 1} games, {total} positions labeled")

    print(f"Done: {len(games)} games, {total} positions -> {args.dest}")


if __name__ == "__main__":
    main()
