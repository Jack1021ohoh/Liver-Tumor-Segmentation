import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def __init__(self, smooth = 1e-5, ignore_index = 0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index  # classes to skip

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim = 1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        dice = []
        for c in range(num_classes):
            if c == self.ignore_index:
                continue  # skip background
            pred_c = pred_softmax[:, c]
            target_c = target_onehot[:, c]
            intersection = (pred_c * target_c).sum(dim = (1, 2))
            union = pred_c.sum(dim = (1, 2)) + target_c.sum(dim = (1, 2))
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            dice.append(dice_c)

        return 1. - torch.mean(torch.stack(dice))

class CombinedLoss(nn.Module):
    def __init__(self, alpha = 0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + (1 - self.alpha) * self.dice(pred, target)
    
class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth = 1e-5, ignore_index = 0, class_weights = None):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index  # classes to skip
        self.class_weights = class_weights

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim = 1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        dice_losses = []
        weights = []

        for c in range(num_classes):
            if c == self.ignore_index:
                continue  # skip background
            pred_c = pred_softmax[:, c]
            target_c = target_onehot[:, c]
            intersection = (pred_c * target_c).sum(dim = (1, 2))
            union = pred_c.sum(dim = (1, 2)) + target_c.sum(dim = (1, 2))
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss_c = 1. - dice_c
            dice_losses.append(dice_loss_c.mean())

            # Apply class weight
            if self.class_weights is not None:
                weights.append(self.class_weights[c])
            else:
                weights.append(1.0)

        weighted_dice_losses = [w * loss for w, loss in zip(weights, dice_losses)]
        return torch.stack(weighted_dice_losses).mean()

class WeightedCombinedLoss(nn.Module):
    def __init__(self, alpha = 0.5, class_weights = None):
        super(WeightedCombinedLoss, self).__init__()
        self.alpha = alpha
        self.class_weights = torch.FloatTensor(class_weights).to(device)
        self.ce = nn.CrossEntropyLoss(weight = self.class_weights)
        self.dice = WeightedDiceLoss(class_weights = self.class_weights)

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + (1 - self.alpha) * self.dice(pred, target)