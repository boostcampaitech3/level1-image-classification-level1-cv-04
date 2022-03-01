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


def validation_multi(model, loader, criterion0, criterion1, criterion2, device, args):
    val_losses = []
    all_preds, all_labels =[], []
    num_correct, num_samples = 0, 0

    for X, y in tqdm(loader, total=len(loader)):
        X = X.to(device)
        y_m, y_g, y_a, y = \
            y["mask"].to(device), y["gender"].to(device), y["age"].to(device), y["ans"].to(device)

        outputs_m, outputs_g, outputs_a = model(X)
        loss0 = criterion0(outputs_m, y_m)
        loss1 = criterion1(outputs_g, y_g)
        loss2 = criterion2(outputs_a, y_a)
        loss = (args["MULTIWEIGHT"][0] * loss0 + args["MULTIWEIGHT"][1] * loss1 + args["MULTIWEIGHT"][2] * loss2) / 3

        preds_m = torch.argmax(outputs_m, dim=-1)
        preds_g = torch.argmax(outputs_g, dim=-1)
        preds_a = torch.argmax(outputs_a, dim=-1)
        n = (preds_m==y_m).sum()
        n += (preds_g==y_g).sum()
        n += (preds_a==y_a).sum()
        num_correct += torch.div(n, 3)
        num_samples += preds_m.shape[0]

        if args["MULTI"][-1] == 3:
            preds = preds_m * 6 + preds_g * 3 + preds_a
        elif args["MULTI"][-1] == 6:
            preds = preds_m * 6 + preds_g * 3 + torch.div(preds_a+1, 3, rounding_mode="floor")
        else:
            pass

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        val_losses.append(loss.item())

    val_accuracy = "{:.6f}".format(num_correct/num_samples)
    val_loss_total = "{:.6f}".format(sum(val_losses)/len(val_losses))
    torch.cuda.empty_cache()
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0), np.float(val_accuracy), np.float(val_loss_total)
