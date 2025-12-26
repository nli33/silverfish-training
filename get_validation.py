import pandas as pd

df = pd.read_csv("train.csv")
val_df = df.sample(n=50000, random_state=42)
train_df = df.drop(val_df.index)

val_df.to_csv("validation.csv", index=False)
train_df.to_csv("train_reduced.csv", index=False)
