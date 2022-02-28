import os
import torch
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