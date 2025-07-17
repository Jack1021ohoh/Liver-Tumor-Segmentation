import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(preds, targets, n_classes = 3):
    dice_scores = []
    iou_scores = []

    for cls in range(1, n_classes):  # ignore background
        dice_sum = 0.0
        iou_sum = 0.0
        valid_count = 0  # only count valid samples

        for pred, target in zip(preds, targets):
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            intersection = torch.logical_and(pred_cls, target_cls).sum().item()
            union = torch.logical_or(pred_cls, target_cls).sum().item()
            pred_sum = pred_cls.sum().item()
            target_sum = target_cls.sum().item()

            if target_sum + pred_sum == 0:
                # If both are empty, treat as perfect match
                dice = 1.0
                iou = 1.0
            else:
                dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-6)
                iou = intersection / (union + 1e-6)

            dice_sum += dice
            iou_sum += iou
            valid_count += 1

        dice_scores.append(dice_sum / valid_count)
        iou_scores.append(iou_sum / valid_count)

    return dice_scores, iou_scores

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, num_epochs, patience = 20, save_dir = './model_storage/transunet_tcia', class_weights = [1., 1., 1.]):
    best_dice = 0.0 
    early_stop_count = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        running_dice_1 = 0.0
        running_dice_2 = 0.0

        with tqdm(train_loader, desc = f'Epoch {epoch + 1}/{num_epochs}', unit = 'batch') as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pred = torch.argmax(outputs, dim = 1)
                running_dices, _ = calculate_metrics(pred, labels)
                running_dice_1 += running_dices[0]
                running_dice_2 += running_dices[1]
                running_loss += loss.item()
                t.set_postfix(loss = running_loss / (t.n + 1), dice_1 = running_dice_1 / (t.n + 1), dice_2 = running_dice_2 / (t.n + 1))

        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_dice_1 = running_dice_1 / len(train_loader)
        epoch_dice_2 = running_dice_2 / len(train_loader)

        # Validation
        model.eval()  
        val_loss = 0.0
        val_dice_1 = 0.0
        val_dice_2 = 0.0
        with torch.no_grad(): 
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = torch.argmax(outputs, dim = 1)
                val_dices,_ = calculate_metrics(pred, labels)
                val_dice_1 += val_dices[0]
                val_dice_2 += val_dices[1]
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_dice_1 = val_dice_1 / len(val_loader)
        val_dice_2 = val_dice_2 / len(val_loader)

        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Dice1: {epoch_dice_1:.4f}, Train Dice2: {epoch_dice_2:.4f}')
        tqdm.write(f'Validation Loss: {val_loss:.4f}, Validation Dice1: {val_dice_1:.4f}, Validation Dice2: {val_dice_2:.4f}')

        writer.add_scalar('Loss/train', epoch_loss, epoch+1)
        writer.add_scalar('Dice_1/train', epoch_dice_1, epoch + 1)
        writer.add_scalar('Dice_2/train', epoch_dice_2, epoch + 1)

        writer.add_scalar('Loss/valid', val_loss, epoch+1)
        writer.add_scalar('Dice_1/valid', val_dice_1, epoch + 1)
        writer.add_scalar('Dice_2/valid', val_dice_2, epoch + 1)

        combined_val_dice = np.dot([val_dice_1, val_dice_2], class_weights[1:])

        if combined_val_dice > best_dice:
            best_dice = combined_val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            tqdm.write("Best model saved!")
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        if early_stop_count >= patience:
            tqdm.write(f"Early Stop at {epoch + 1}!")
            break