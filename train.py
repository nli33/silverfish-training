from dataset import NNUEDataset
from model import NNUEModel
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader


checkpoint_path = "nnue_256.pt"
train_path = "train_reduced.csv"
val_path = "validation.csv"
test_path = "test.csv"
num_epochs = 3


def train():
    train_dataset = NNUEDataset(train_path)
    val_dataset = NNUEDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    nnue_checkpoint_path = Path(checkpoint_path)
    loss_fn = nn.MSELoss()

    model = NNUEModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        threshold=1e-4,
        min_lr=1e-5,
    )

    start_epoch = 0

    if nnue_checkpoint_path.is_file():
        checkpoint = torch.load(nnue_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("Epoch", epoch)

        model.train()
        train_loss = 0.0
        train_batches = 0

        for x_w, x_b, stm, y in train_loader:
            pred = model(x_w, x_b, stm)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_mse = train_loss / train_batches
        print(f"  train MSE: {train_mse:.6f}")

        # validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_w, x_b, stm, y in val_loader:
                pred = model(x_w, x_b, stm)
                loss = loss_fn(pred, y)

                val_loss += loss.item()
                val_batches += 1

        val_mse = val_loss / val_batches
        print(f"  val   MSE: {val_mse:.6f}")

        # scheduler should use validation loss
        scheduler.step(val_mse)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            print("  New best model, saving")

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, nnue_checkpoint_path)

def test():
    dataset = NNUEDataset(test_path)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,   # safer for testing
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = NNUEModel()
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
    #train()
    test()
