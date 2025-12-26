import csv
import sys
from pathlib import Path
from collections import defaultdict

FEN_PREVIEW_LEN = 50
MAX_PRINT = 10


def truncate_fen(fen, n=FEN_PREVIEW_LEN):
    return fen[:n] + ("..." if len(fen) > n else "")


def check_unique(csv_files):
    fen_seen_global = {}  # fen -> filename
    duplicates_within = defaultdict(list)
    duplicates_across = []

    for csv_path in csv_files:
        csv_path = Path(csv_path)
        print(f"Checking {csv_path} ...")

        fen_seen_local = set()

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            if header != ["FEN", "Evaluation"]:
                raise ValueError(f"{csv_path}: unexpected header {header}")

            for line_num, row in enumerate(reader, start=2):
                fen = row[0]

                # duplicate inside the same file
                if fen in fen_seen_local:
                    duplicates_within[csv_path.name].append((line_num, fen))
                else:
                    fen_seen_local.add(fen)

                # duplicate across files
                if fen in fen_seen_global:
                    duplicates_across.append(
                        (fen, fen_seen_global[fen], csv_path.name)
                    )
                else:
                    fen_seen_global[fen] = csv_path.name

    if duplicates_within:
        print("\nDuplicates within files:")
        for fname, entries in duplicates_within.items():
            print(f"\n  {fname}: {len(entries)} duplicates")
            for line, fen in entries[:MAX_PRINT]:
                print(f"    line {line}: {truncate_fen(fen)}")
            if len(entries) > MAX_PRINT:
                print(f"    ... ({len(entries) - MAX_PRINT} more)")
    else:
        print("\nNo duplicates within files")

    if duplicates_across:
        print("\nDuplicates across files:")
        for fen, file_a, file_b in duplicates_across[:MAX_PRINT]:
            print(f"  {truncate_fen(fen)}")
            print(f"    in: {file_a} AND {file_b}")
        if len(duplicates_across) > MAX_PRINT:
            print(f"  ... ({len(duplicates_across) - MAX_PRINT} more)")
        print(f"\nTotal cross-file duplicates: {len(duplicates_across)}")
    else:
        print("\nNo duplicates across files")

    if not duplicates_within and not duplicates_across:
        print("\nAll FENs are globally unique")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_unique.py file1.csv file2.csv ...")
        sys.exit(1)

    check_unique(sys.argv[1:])

