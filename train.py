from dataset import NNUEDataset
from model import NNUE
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader


checkpoint_path = "nnue.pt"
train_dataset = "train.csv"
test_dataset = "test.csv"
num_epochs = 10


def train():
    dataset = NNUEDataset(train_dataset)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    nnue_checkpoint_path = Path(checkpoint_path)
    loss_fn = nn.MSELoss()

    model = NNUE()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
    )

    start_epoch = 0

    if nnue_checkpoint_path.is_file():
        checkpoint = torch.load(nnue_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    model.train()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("Epoch", epoch)

        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            x_w, x_b, stm, y = batch

            pred = model(x_w, x_b, stm)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  train MSE: {avg_loss:.6f}")

        scheduler.step(avg_loss)

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, nnue_checkpoint_path)


def test():
    dataset = NNUEDataset(test_dataset)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,   # safer for testing
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = NNUE()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_w, x_b, stm, y in loader:
            pred = model(x_w, x_b, stm)
            loss = loss_fn(pred, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    print(f"Test MSE: {avg_loss:.6f}")



if __name__ == '__main__':
    # train()
    test()
