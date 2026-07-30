# Project Instructions

## Stack
- Python/PyTorch NNUE trainer for the `silverfish` chess engine (sibling repo `../silverfish`). Architecture: 768→256×2→1 (`model.py`).
- `dataset.py`: `NNUEDataset` precomputes sparse active-feature indices for every row in `__init__` (fast FEN scanner, no `chess.Board`), so `__getitem__` is cheap. Because of this, DataLoader `num_workers=0` is fastest — worker processes would just re-pickle the whole precomputed entries list every epoch.
- `label_data.py`: multiprocessing Stockfish labeling CLI (`python3 label_data.py source.csv dest.csv --workers N --depth D`). Large relabels (~2M positions) should run on `orca` (96 cores), not locally — see `ssh orca`.
- `export.py`: exports a trained `.pt` checkpoint to `.nnue` for silverfish to embed via `go:embed`.

## Process
- Don't write substantial one-off Python and run it via `python3 -c "..."` on the command line for training/benchmarking — it's not reproducible and doesn't get reviewed. If the existing pipeline script (`train.py`, `dataset.py`, `label_data.py`, `export.py`) doesn't do what's needed, fix or extend the script itself and run that.
