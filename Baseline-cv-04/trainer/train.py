import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm 
from importlib import import_module
from utils.loss import create_criterion
from utils.augmentation import cutmix
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
           
            this_epoch_cutmix = np.random.random() > 0.5
            this_epoch_mixup = np.random.random() > 0.5

            if args["FACECENTER"]:
                X_ = X["image"].to(device)
                r = X["ratio"].to(device)
            else:
                X_ = X.to(device)
                r = torch.ones(y.shape[-1]).to(device) / 2 # 0.5
            y = y.to(device)

            if args["CUTMIX"] and this_epoch_cutmix:
                X_, ratio_l, ratio_r, y_l, y_r = cutmix(X_, r, y)
                ratio_all = ratio_l + ratio_r
                y_same = ((y_l==y_r).type(torch.float) - 1) * -1 # if y_l==y_r -> 0 else 1
            elif args["MIXUP"] and this_epoch_mixup:
                pass

            outputs = model(X_)
            if args["CUTMIX"] and this_epoch_cutmix:
                _, preds = torch.topk(outputs, 2)
                p_1 = preds[:, 0] # first top value
                p_2 = preds[:, 1] / y_same # second top value, if y_l==y_r -> inf else same
                n = (p_1==y_l).sum() + (p_1==y_r).sum() + (p_2==y_l).sum() + (p_2==y_r).sum()
                num_correct += torch.div(n, 2)
            elif args["MIXUP"] and this_epoch_mixup:
                pass
            else:
                preds = torch.argmax(outputs, dim=-1)
                num_correct += (preds==y).sum()
            num_samples += preds.shape[0]
            
            if args["CUTMIX"] and this_epoch_cutmix:
                loss_l = criterion(outputs, y_l)
                loss_r = criterion(outputs, y_r)
                ratio_l = torch.sum(ratio_l / ratio_all) / ratio_all.shape[0]
                ratio_r = torch.sum(ratio_r / ratio_all) / ratio_all.shape[0]
                loss = ratio_l * loss_l + ratio_r * loss_r
            elif args["MIXUP"] and this_epoch_mixup:
                pass
            else:
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
            
            torch.save(checkpoint, "/Fold{}_{}_Epoch{}_{:.3f}_{}.tar".format(fold_num, args['MODEL'], epoch, best_f1, args["CRITERION"]))
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

def train_multi(args, model, train_loader, valid_loader, fold_num, time_stamp, class_weights, logger, wandb):

    model.train()
    device = args["DEVICE"]
    
    criterion0 = create_criterion(args["MULTICRITERION"][0])
    criterion1 = create_criterion(args["MULTICRITERION"][1])
    criterion2 = create_criterion(args["MULTICRITERION"][2])
    if args["CLASS_WEIGHTS"]:
        if args["MULTICRITERION"][0]=='cross_entropy':
            criterion0 = nn.CrossEntropyLoss(weight=class_weights.to(device))
        if args["MULTICRITERION"][1]=='cross_entropy':
            criterion1 = nn.CrossEntropyLoss(weight=class_weights.to(device))
        if args["MULTICRITERION"][2]=='cross_entropy':
            criterion2 = nn.CrossEntropyLoss(weight=class_weights.to(device))
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
           
            this_epoch_cutmix = np.random.random() > 0.5
            this_epoch_mixup = np.random.random() > 0.5

            if args["FACECENTER"]:
                X_ = X["image"].to(device)
                r = X["ratio"].to(device)
            else:
                X_ = X.to(device)
                r = torch.ones(y["ans"].shape[-1]).to(device) / 2 # 0.5
            y_m, y_g, y_a, y = \
                y["mask"].to(device), y["gender"].to(device), y["age"].to(device), y["ans"].to(device)

            if args["CUTMIX"] and this_epoch_cutmix:
                y_all = torch.cat([y_m.unsqueeze(1), y_g.unsqueeze(1), y_a.unsqueeze(1)], dim=1)
                X_, ratio_l, ratio_r, y_l, y_r = cutmix(X_, r, y_all)
                y_m_l, y_g_l, y_a_l = y_l[:, 0], y_l[:, 1], y_l[:, 2]
                y_m_r, y_g_r, y_a_r = y_r[:, 0], y_r[:, 1], y_r[:, 2]
                ratio_all = ratio_l + ratio_r
                y_m_same = ((y_m_l==y_m_r).type(torch.float) - 1) * -1 # if y_m_l==y_m_r -> 0 else 1
                y_g_same = ((y_g_l==y_g_r).type(torch.float) - 1) * -1 # if y_g_l==y_g_r -> 0 else 1
                y_a_same = ((y_a_l==y_a_r).type(torch.float) - 1) * -1 # if y_a_l==y_a_r -> 0 else 1
            elif args["MIXUP"] and this_epoch_mixup:
                pass

            outputs_m, outputs_g, outputs_a = model(X_)
            if args["CUTMIX"] and this_epoch_cutmix:
                _, preds_m = torch.topk(outputs_m, 2)
                p_m_1 = preds_m[:, 0] # first top value
                p_m_2 = preds_m[:, 1] / y_m_same # second top value, if y_m_l==y_m_r -> inf else same
                n = (p_m_1==y_m_l).sum() + (p_m_1==y_m_r).sum() + (p_m_2==y_m_l).sum() + (p_m_2==y_m_r).sum()

                _, preds_g = torch.topk(outputs_g, 2)
                p_g_1 = preds_g[:, 0] # first top value
                p_g_2 = preds_g[:, 1] / y_g_same # second top value, if y_g_l==y_g_r -> inf else same
                n += (p_g_1==y_g_l).sum() + (p_g_1==y_g_r).sum() + (p_g_2==y_g_l).sum() + (p_g_2==y_g_r).sum()

                _, preds_a = torch.topk(outputs_a, 2)
                p_a_1 = preds_a[:, 0] # first top value
                p_a_2 = preds_a[:, 1] / y_a_same # second top value, if y_a_l==y_a_r -> inf else same
                n += (p_a_1==y_a_l).sum() + (p_a_1==y_a_r).sum() + (p_a_2==y_a_l).sum() + (p_a_2==y_a_r).sum()

                num_correct += torch.div(n, 6)
            elif args["MIXUP"] and this_epoch_mixup:
                pass
            else:
                preds_m = torch.argmax(outputs_m, dim=-1)
                preds_g = torch.argmax(outputs_g, dim=-1)
                preds_a = torch.argmax(outputs_a, dim=-1)
                n = (preds_m==y_m).sum()
                n += (preds_g==y_g).sum()
                n += (preds_a==y_a).sum()
                num_correct += torch.div(n, 3)
            num_samples += preds_m.shape[0]
            
            if args["CUTMIX"] and this_epoch_cutmix:
                loss_l = criterion0(outputs_m, y_m_l)
                loss_r = criterion0(outputs_m, y_m_r)
                ratio_l = torch.sum(ratio_l / ratio_all) / ratio_all.shape[0]
                ratio_r = torch.sum(ratio_r / ratio_all) / ratio_all.shape[0]
                loss0 = ratio_l * loss_l + ratio_r * loss_r
                
                loss_l = criterion1(outputs_m, y_g_l)
                loss_r = criterion1(outputs_m, y_g_r)
                ratio_l = torch.sum(ratio_l / ratio_all) / ratio_all.shape[0]
                ratio_r = torch.sum(ratio_r / ratio_all) / ratio_all.shape[0]
                loss1 = ratio_l * loss_l + ratio_r * loss_r

                loss_l = criterion2(outputs_m, y_a_l)
                loss_r = criterion2(outputs_m, y_a_r)
                ratio_l = torch.sum(ratio_l / ratio_all) / ratio_all.shape[0]
                ratio_r = torch.sum(ratio_r / ratio_all) / ratio_all.shape[0]
                loss2 = ratio_l * loss_l + ratio_r * loss_r
            elif args["MIXUP"] and this_epoch_mixup:
                pass
            else:
                loss0 = criterion0(outputs_m, y_m)
                loss1 = criterion1(outputs_g, y_g)
                loss2 = criterion2(outputs_a, y_a)
            loss = (args["MULTIWEIGHT"][0] * loss0 + args["MULTIWEIGHT"][1] * loss1 + args["MULTIWEIGHT"][2] * loss2) / 3
            total_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = num_correct/num_samples
        train_loss = sum(total_loss)/len(total_loss)

        with torch.no_grad():
            model.eval()
            val_preds, val_labels, val_accuarcy, val_loss = validation_multi(model, valid_loader, criterion0, criterion1, criterion2, device, args)
            
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
            best_f1 = f1
            best_acc = val_accuarcy
            best_val_preds = val_preds

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
                }    
            
            torch.save(checkpoint, "/Fold{}_{}_Epoch{}_{:.3f}_{}.tar".format(fold_num, args['MODEL'], epoch, best_f1, args["CRITERION"]))
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
