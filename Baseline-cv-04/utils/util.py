import os
import time
import random
import logging
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb

# def get_train_transform(args):
#     return A.Compose([
#                     A.CenterCrop(height=int(512*0.9), width=int(384*0.9), p=1),
#                     A.ShiftScaleRotate(scale_limit=0, shift_limit=0.02, rotate_limit=0, p=0.5),
#                     A.HorizontalFlip(p=0.5),
#                     A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
#                     A.Normalize(
#                         mean=[0.56, 0.524, 0.501],
#                         std=[0.258, 0.265, 0.267],
#                         max_pixel_value=255.0),
#                     ToTensorV2()
#                     ])

# def get_valid_transform(args):
#     return A.Compose([
#                     A.CenterCrop(height=int(512*0.9), width=int(384*0.9), p=1),
#                     A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
#                     A.Normalize(
#                         mean=[0.56, 0.524, 0.501],
#                         std=[0.258, 0.265, 0.267],
#                         max_pixel_value=255.0),
#                     ToTensorV2()
#                     ])

# def pre_transform(args):
#     return A.Compose([
#                     A.CenterCrop(height=470, width=340, p=1),
#                     A.ShiftScaleRotate(scale_limit=0, shift_limit=0.02, rotate_limit=0, p=0.5),
#                     A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
#                     ]) 
    

def train_test_split_by_folder(data, test_size):

    test_size = int(data["path"].nunique() * test_size)
    train_path = random.sample(list(data["path"].unique()), test_size)

    X_train = data[~data["path"].isin(train_path)].reset_index(drop=True)
    X_valid = data[data["path"].isin(train_path)].reset_index(drop=True)

    return X_train, X_valid

def age_group(x, upper_bound):
    if x < 30:
        return 0
    elif x < upper_bound:
        return 1
    else:
        return 2

def age_bound(args, dataframe):
    dataframe["age_group"] = dataframe['age_indv'].apply(lambda x : age_group(x, args["AGE_BOUND"]))
    dataframe['all'] = dataframe['mask'] * 6 + dataframe['gender']*3 + dataframe['age_group']
    return dataframe

def load_checkpoint(checkpoint, model, optimizer, lr=None):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if lr is not None:
        for param_groups in optimizer.param_groups:
            param_groups["lr"] = lr

    print("Loaded Checkpoint.. ")

def get_log(args):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    
    model_type = args["MODEL"]

    if not os.path.exists("log"):
        os.makedirs("log")
        
    stream_handler = logging.FileHandler(f"log/{model_type}_{time.strftime('%m%d-%H-%M-%S')}.txt", mode='w', encoding='utf8')
    logger.addHandler(stream_handler)
    
    return logger

def wandb_init(args, wandb, time_stamp):
    
    wandb.init(project="test-project", entity="ai3_cv4", name = f"{args['MODEL']}_MYINITIAL")

    wandb.config.update({
    "Model": args["MODEL"],
    "Loss": args["CRITERION"],
    "Optimizer": args["OPTIMIZER"],
    "Resize": args["RESIZE"],
    "learning_rate": args["LEARNING_RATE"],
    "batch_size": args["BATCH_SIZE"],
    "Weight decay": args["WEIGHT_DECAY"],
    "Age bound": args["AGE_BOUND"],
    })

def print_metrics(best_val_preds, val_labels, fpath):

    fname = fpath.split("/")[-1]
    save_dir = "fig/" + "/".join(fpath.split("/")[:-1])
    
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    f1 = f1_score(val_labels, best_val_preds, average='macro')
    f1 = round(f1, 4)

    cm = confusion_matrix(val_labels, best_val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    fig.suptitle(f"{fname}, F-1 score: {f1}")
    plt.savefig(f"{save_dir}/{fname}.png")