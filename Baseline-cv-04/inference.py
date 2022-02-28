import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.dataset import MaskTestDataset

def inference(args, model, test_loader, info):
    preds = []
    with torch.no_grad():
        for idx, (images, id_) in enumerate(tqdm(test_loader, total=len(test_loader))):
            images = images.to("cuda")
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    if not os.path.exists("submission/" + args["MODEL"]):
        os.makedirs("submission/" + args["MODEL"])

    info["ans"] = preds
    info.to_csv(f"submission/{args['MODEL']}/sub.csv", index=False)
    print(info["ans"].value_counts().sort_index())
    print(f'Inference Done!')

def infer_logits(args, model, train_loader, train_data, valid_loader, valid_data, test_loader, info):
    if not os.path.exists("submission/" + args["MODEL"]):
        os.makedirs("submission/" + args["MODEL"])

    with torch.no_grad():
        for idx, (images, id_) in enumerate(tqdm(train_loader, total=len(train_loader))):
            images = images.to("cuda")
            logit = model(images)
            if idx == 0:
                logits = logit.cpu().numpy()
            else:
                logits = np.append(logits, logit.cpu().numpy(), axis=0)
    
    train_logits = train_data[["img_path"]].copy()
    logits_df = pd.DataFrame(logits, columns=[f"l{i:0>2}" for i in range(len(logits[0]))])
    train_logits = pd.concat([train_logits, logits_df], axis=1)
    train_logits.to_csv(f"submission/{args['MODEL']}/logits_train_{args['MODEL']}.csv", index=False)
    print(f'Train Logits Done!')
    
    logits = np.array([])
    with torch.no_grad():
        for idx, (images, id_) in enumerate(tqdm(valid_loader, total=len(valid_loader))):
            images = images.to("cuda")
            logit = model(images)
            if idx == 0:
                logits = logit.cpu().numpy()
            else:
                logits = np.append(logits, logit.cpu().numpy(), axis=0)
    
    valid_logits = valid_data[["img_path"]].copy()
    logits_df = pd.DataFrame(logits, columns=[f"l{i:0>2}" for i in range(len(logits[0]))])
    valid_logits = pd.concat([valid_logits, logits_df], axis=1)
    valid_logits.to_csv(f"submission/{args['MODEL']}/logits_val_{args['MODEL']}.csv", index=False)
    print(f'Validation Logits Done!')

    logits = np.array([])
    with torch.no_grad():
        for idx, (images, id_) in enumerate(tqdm(test_loader, total=len(test_loader))):
            images = images.to("cuda")
            logit = model(images)
            if idx == 0:
                logits = logit.cpu().numpy()
            else:
                logits = np.append(logits, logit.cpu().numpy(), axis=0)
    
    test_logits = info[["ImageID"]].copy()
    logits_df = pd.DataFrame(logits, columns=[f"l{i:0>2}" for i in range(len(logits[0]))])
    test_logits = pd.concat([test_logits, logits_df], axis=1)
    test_logits.to_csv(f"submission/{args['MODEL']}/logits_test_{args['MODEL']}.csv", index=False)
    print(f'Test Logits Done!')
