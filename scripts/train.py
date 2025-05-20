import torch
from tqdm.notebook import tqdm as tqdm_notebook
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for x_batch, y_batch, _ in tqdm_notebook(dataloader, desc="Training...", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1) if model.is_binary else y_batch
        optimizer.zero_grad()
        outputs = model(x_batch)  # shape: [B]
        loss = loss_fn(outputs, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        if model.is_binary:
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct_preds += (preds == y_batch.long()).sum().item()
        else:
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == y_batch).sum().item()
        total_samples += x_batch.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def validate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm_notebook(dataloader, desc="Validating..", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1) if model.is_binary else y_batch

            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)

            running_loss += loss.item() * x_batch.size(0)
            if model.is_binary:
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                correct_preds += (preds == y_batch.long()).sum().item()
            else:
                preds = torch.argmax(outputs, dim=1)
                correct_preds += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def run_training(
    model, train_dataset, val_dataset, loss_fn, optimizer,
    device, batch_size=64, epochs=50, lr_schedule=None, early_stopping=None, model_destination=None
):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    epoch_bar = tqdm_notebook(range(epochs), desc="Epochs", leave=True)

    for epoch in epoch_bar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        if lr_schedule is not None:
            lr_schedule.step(val_loss)

        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        postfix_dict = {
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc:.4f}",
        }

        if early_stopping:
            improved = early_stopping(val_acc)
            if improved and model_destination:
                torch.save(model.state_dict(), f"{model_destination}_best.pth")

            postfix_dict["Best Acc"] = f"{early_stopping.best_score:.4f}"
            postfix_dict["Status"] = (
                f"[Epoch {epoch+1}] 🔼 Improved → {val_acc:.4f}" if improved
                else f"[Epoch {epoch+1}] ⚠️ No improvement. Patience {early_stopping.counter}/{early_stopping.patience}"
            )
            epoch_bar.set_postfix(postfix_dict)

            if early_stopping.early_stop:
                print(
                    f"\n⛔ Early stopping triggered at epoch {epoch+1}. Stopping training.")
                break
        else:
            epoch_bar.set_postfix(postfix_dict)

        # Save latest model regardless
        if model_destination:
            torch.save(model.state_dict(), f"{model_destination}.pth")

    print(
        f"\n🏆 Best Validation Accuracy: {early_stopping.best_score if early_stopping else max(history['val_acc']):.4f}")

    if model_destination and os.path.exists(f"{model_destination}_best.pth"):
        model.load_state_dict(torch.load(f"{model_destination}_best.pth"))
        print(f"Best model loaded from {model_destination}_best.pth")

    return model, pd.DataFrame(history)
