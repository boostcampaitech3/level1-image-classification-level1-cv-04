import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm 
from importlib import import_module
from utils.loss import create_criterion
from utils.augmentation import *
from trainer.validation import validation
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args, model, train_loader, valid_loader, fold_num, time_stamp, class_weights, logger, wandb):

    model.train()
    device = args["DEVICE"]
    
    criterion = create_criterion(args["CRITERION"])
    if args["CRITERION"]=='cross_entropy' and args["CLASS_WEIGHTS"]:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(f"Class Weight: {class_weights}")

    earlystopping_patience = args["EARLYSTOPPING_PATIENCE"]
    scheduler_patience = args["SCHEDULER_PATIENCE"]
    opt_module = getattr(import_module("torch.optim"), args["OPTIMIZER"])
    optimizer = opt_module(model.parameters(), lr=args["LEARNING_RATE"], weight_decay=args["WEIGHT_DECAY"]) 
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=scheduler_patience, verbose=True)

    # 모델 저장 경로 설정
    save_dir = f"checkpoint/{time_stamp}"
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    num_epochs = 50
    best_f1 = 0
    best_acc = 0
    earlystopping_counter = 0
    best_val_preds = None

    for epoch in tqdm(range(1, num_epochs+1), total=num_epochs):
        total_loss = []
        num_correct, num_samples = 0., 0

        for i, (X, y) in enumerate(tqdm(train_loader, total=len(train_loader))):
            X = X.to(device)
            y = y.to(device)
            if  np.random.random() > 0.5:   
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(X.size()[0]).to(args["DEVICE"])
                target_a = y
                target_b = y[rand_index]            
                bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                X[:, :, bbx1:bbx2, bby1:bby2] = X[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))

                outputs = model(X)
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            
            else:
                outputs = model(X)
                loss = criterion(outputs, y)

            loss = criterion(outputs, y)
            total_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = num_correct/num_samples
        train_loss = sum(total_loss)/len(total_loss)

        with torch.no_grad():
            model.eval()
            val_preds, val_labels, val_accuarcy, val_loss = validation(model, valid_loader, criterion, device)
            
            # F-1 Score
            f1 = f1_score(val_labels, val_preds, average='macro')

        logger.info("Epoch: {}/{}.. ".format(epoch, num_epochs) +
                    "Training Accuracy: {:.4f}.. ".format(train_accuracy) + 
                    "Training Loss: {:.4f}.. ".format(train_loss) +
                    "Valid Accuracy: {:.4f}.. ".format(val_accuarcy) + 
                    "Valid F1-Score: {:.4f}.. ".format(f1) + 
                    "Valid Loss: {:.4f}.. ".format(val_loss))
        if fold_num==1:
            wandb.log({'Valid accuracy': val_accuarcy, 'Valid F1': f1, 'Valid Loss': val_loss})
        model.train()
        
        # Save Model
        if best_f1 < f1:
            logger.info("Val F1 improved from {:.3f} -> {:.3f}".format(best_f1, f1))
            wandb.run.summary["Best F1"] = f1
            best_f1 = f1
            best_acc = val_accuarcy
            best_val_preds = val_preds

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
                }    
            
            # 기존 경로 제거
            try:
                os.remove(save_path)
            except:
                pass
            save_path = save_dir + "/Fold{}_{}_Epoch{}_{:.3f}_{}.tar".format(fold_num, args['MODEL'], epoch, best_f1, args["CRITERION"])
            
            torch.save(checkpoint, save_path)
            earlystopping_counter = 0

        else:
            earlystopping_counter += 1
            logger.info("Valid F1 did not improved from {:.3f}.. Counter {}/{}".format(best_f1, earlystopping_counter, earlystopping_patience))
            if earlystopping_counter > earlystopping_patience:
                logger.info("Early Stopped ...")
                break
        
        scheduler.step(f1)

    return best_val_preds, val_labels, best_f1, best_acc
