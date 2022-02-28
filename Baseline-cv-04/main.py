import time
import torch
import random
import numpy as np
import pandas as pd
from args import Args
from glob import glob
from tqdm import tqdm
from utils.util import *
from inference import inference, infer_logits
from model_utils.model import load_model
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.dataset import MaskDataset, MaskTestDataset
from trainer.train import train
from sklearn.model_selection import KFold


def main(args, logger, wandb):

    device = args["DEVICE"]
    test_mode = args["INFERENCE"]
    random_seed = args["RANDOM_SEED"]
    time_stamp = "_".join(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()).split(" "))
    current_dir = os.path.dirname(os.path.realpath(__file__))



    if not test_mode:
        # init
        wandb_init(args, wandb, time_stamp)

        # Load Dataframe
        data = pd.read_csv(current_dir + "/data/final_train_df.csv")
        
        # Age label revision
        data = age_bound(args, data)

        # Class Weight
        class_dist = data["all"].value_counts().sort_index().values
        class_weights = [int(class_dist.sum()/num) for num in class_dist]
        class_weights = torch.tensor(class_weights,dtype=torch.float)
        
        # K-fold
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        unique_path =  data['path'].unique() 

        for fold_num, (train_index, valid_index) in enumerate(kfold.split(unique_path), start=1):
            
        
            train_path = unique_path[train_index]
            valid_path = unique_path[valid_index]

            X_train = data[data["path"].isin(train_path)].reset_index(drop=True)
            X_valid = data[data["path"].isin(valid_path)].reset_index(drop=True)

            # Generate train data loader
            train_dataset = MaskDataset(X_train, get_train_transform(args))
        
            # Oversampling
            if args["OVERSAMPLING"]:
                sample_weights = [0] * len(train_dataset)
                for idx, (_, y) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
                    class_weight = class_weights[y]
                    sample_weights[idx] = class_weight
                sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)

                train_loader = DataLoader(train_dataset, batch_size=args["BATCH_SIZE"], shuffle=False, sampler=sampler, num_workers=2)

            else:
                train_loader = DataLoader(train_dataset, batch_size=args["BATCH_SIZE"], shuffle=True, num_workers=2)

            valid_dataset = MaskDataset(X_valid, get_valid_transform(args))
            valid_loader = DataLoader(valid_dataset, batch_size=args["BATCH_SIZE"], shuffle=False, num_workers=2)
        
            # Load model
            model = load_model(args)
            model.to(device)

            # Train model
            logger.info("============== (" + str(fold_num) + "-th cross validation start) =================\n")
            best_val_preds, val_labels, best_f1, best_acc = train(args, model, train_loader, valid_loader, fold_num, time_stamp, class_weights, logger, wandb)
            
            # Visualize Confusion metrics
            print_metrics(best_val_preds, val_labels,  "{}/fold{}_{}_F1_{:.4f}_Acc_{:.4f}".format(time_stamp, fold_num, args['MODEL'], best_f1, best_acc))

            if not args["KFOLD"]:
                return 

        return 
    
    ### Test ###
    info = pd.read_csv(current_dir + "/data/info.csv")
    save_path = args["SAVE_PATH"]
    logger.info(f"Model Path: {save_path}")

    # Load trained model
    model = load_model(args)
    model.to(device)

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Generate test data loader
    test_dataset = MaskTestDataset(info, get_valid_transform(args))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)

    # predict & submit
    inference(args, model, test_loader, info)

    # save logits
    if args["SAVE_LOGITS"]:
        print(f"SAVE_LOGITS set to {args['SAVE_LOGITS']}. Saving logits csv file...")
        # Load Dataframe
        data = pd.read_csv(current_dir + "/data/final_train_df.csv")
        
        # Age label revision
        data = age_bound(args, data)

        # Class Weight
        class_dist = data["all"].value_counts().sort_index().values
        class_weights = [int(class_dist.sum()/num) for num in class_dist]
        class_weights = torch.tensor(class_weights,dtype=torch.float)
        
        # K-fold
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        unique_path =  data['path'].unique() 

        for fold_num, (train_index, valid_index) in enumerate(kfold.split(unique_path), start=1):
            train_path = unique_path[train_index]
            valid_path = unique_path[valid_index]

            X_train = data[data["path"].isin(train_path)].reset_index(drop=True)
            X_valid = data[data["path"].isin(valid_path)].reset_index(drop=True)

            # Generate train data loader
            train_dataset = MaskDataset(X_train, get_valid_transform(args))
            train_loader = DataLoader(train_dataset, batch_size=args["BATCH_SIZE"], shuffle=False, num_workers=2)

            valid_dataset = MaskDataset(X_valid, get_valid_transform(args))
            valid_loader = DataLoader(valid_dataset, batch_size=args["BATCH_SIZE"], shuffle=False, num_workers=2)

            break
        
        infer_logits(args, model, train_loader, X_train, valid_loader, X_valid, test_loader, info)

if __name__ == "__main__":

    args = Args().params

    logger = get_log(args)
    logger.info("\n=========Training Info=========\n"
                "Model: {}".format(args['MODEL']) + "\n" +
                "Loss: {}".format(args['CRITERION']) + "\n" +
                "Optimizer: {}".format(args['OPTIMIZER']) + "\n" +
                "Resize: {}".format(args['RESIZE']) + "\n" +
                "Batch size: {}".format(args['BATCH_SIZE']) + "\n" +
                "Learning rate: {}".format(args['LEARNING_RATE']) + "\n" +
                "Weight Decay: {}".format(args['WEIGHT_DECAY']) + "\n" +
                "Age bound(>60): {}".format(args['AGE_BOUND']) + "\n" +
                "Oversampling: {}".format(args['OVERSAMPLING']) + "\n" + 
                "Class weights: {}".format(args['CLASS_WEIGHTS']) + "\n" + 
                "===============================")

    random_seed = args['RANDOM_SEED']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args, logger, wandb)
