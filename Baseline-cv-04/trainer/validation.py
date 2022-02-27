import torch
import numpy as np
from tqdm import tqdm

def validation(model, loader, criterion, device):
    val_losses = []
    all_preds, all_labels =[], []
    num_correct, num_samples = 0, 0

    for X, y in tqdm(loader, total=len(loader)):
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        loss = criterion(outputs, y)

        preds = torch.argmax(outputs, dim=-1)
        num_correct += (preds==y).sum()
        num_samples += preds.shape[0]

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        val_losses.append(loss.item())

    val_accuracy = "{:.6f}".format(num_correct/num_samples)
    val_loss_total = "{:.6f}".format(sum(val_losses)/len(val_losses))
    torch.cuda.empty_cache()
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0), np.float(val_accuracy), np.float(val_loss_total)
