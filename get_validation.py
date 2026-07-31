import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="train.csv")
parser.add_argument("--val-size", type=int, default=50000)
parser.add_argument("--train-out", default="train_reduced.csv")
parser.add_argument("--val-out", default="validation.csv")
args = parser.parse_args()

df = pd.read_csv(args.source)
val_df = df.sample(n=args.val_size, random_state=42)
train_df = df.drop(val_df.index)

val_df.to_csv(args.val_out, index=False)
train_df.to_csv(args.train_out, index=False)
print(f"{args.source}: {len(train_df)} train -> {args.train_out}, {len(val_df)} val -> {args.val_out}")
