import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
from model_training_function import calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_mapping = {
    0: 0,     
    1: 128,   
    2: 255    
}

def test_model(model, test_loader, criterion):
    model.eval()  
    test_dice_1 = []
    test_dice_2 = []
    test_iou_1 = []
    test_iou_2 = []
    test_loss = 0.0

    with torch.no_grad():  
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            preds = torch.argmax(outputs, dim = 1)
            test_dices, test_ious = calculate_metrics(preds, labels)
            test_dice_1.append(test_dices[0])
            test_dice_2.append(test_dices[1])
            test_iou_1.append(test_ious[0])
            test_iou_2.append(test_ious[1])

            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    dice1 = sum(test_dice_1) / len(test_dice_1)
    dice2 = sum(test_dice_2) / len(test_dice_2)
    iou1 = sum(test_iou_1) / len(test_iou_1)
    iou2 = sum(test_iou_2) / len(test_iou_2)

    tqdm.write(f'Loss: {test_loss:.4f}, Dice1: {dice1:.4f}, Dice2: {dice2:.4f}, IoU1: {iou1:.4f}, IoU2: {iou2:.4f}')

    return test_dice_1, test_dice_2, test_iou_1, test_iou_2

def infernece(model, image):
    model.eval()  
    if image.ndim == 3:
        image = image.unsqueeze(dim = 0)
    
    with torch.no_grad():          
        image = image.to(device)
        output = model(image)
        pred = torch.argmax(output, dim = 1)

    return pred.cpu()

def display_result(image, pred, label):
    image = image * 255
    image = image.to(torch.uint8)
    image = image.permute(1, 2, 0)
    label = label.apply_(label_mapping.get)
    pred = pred.apply_(label_mapping.get)
    pred = pred.permute(1, 2, 0)

    plt.figure(figsize = (6, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap = 'gray')
    plt.title('Prediction')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(label, cmap = 'gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.tight_layout()
    plt.show()