import torch
from tqdm.notebook import tqdm as tqdm_notebook

from torch.utils.data import DataLoader

def predict(model, test_dataset, device, batch_size=64, return_probs=False):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm_notebook(test_loader, desc="Predicting...", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)

            if model.is_binary:
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs >= 0.5).long()
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
            if return_probs:
                all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    if return_probs:
        all_probs = torch.cat(all_probs).numpy()
        return all_preds, all_probs, all_targets
    else:
        return all_preds, all_targets