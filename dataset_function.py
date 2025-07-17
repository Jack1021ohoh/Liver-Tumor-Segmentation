import numpy as np
import os
import random
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

def split_data(cases, size = [0.7, 0.1, 0.2]):
    np.random.shuffle(cases)

    total_size = len(cases)
    train_size = int(total_size * size[0])
    val_size = int(total_size * size[1])

    train_cases = cases[:train_size]
    val_cases = cases[train_size:train_size + val_size]
    test_cases= cases[train_size + val_size:]

    return train_cases, val_cases, test_cases

def split_data_tcia(cases, cases_idx, size = [0.7, 0.1, 0.2]):
    np.random.shuffle(cases_idx)

    total_size = len(cases_idx)
    train_size = int(total_size * size[0])
    val_size = int(total_size * size[1])

    train_cases_idx = cases_idx[:train_size]
    val_cases_idx = cases_idx[train_size:train_size + val_size]
    test_cases_idx = cases_idx[train_size + val_size:]

    train_cases = [cases[i] for i in train_cases_idx]
    val_cases = [cases[i] for i in val_cases_idx]
    test_cases = [cases[i] for i in test_cases_idx]

    return train_cases, val_cases, test_cases

def get_image_training_transforms():
    """Training transforms for images only"""
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),        
        # Normalization
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

def get_image_testing_transforms():
    """Testing/validation transforms for images only"""
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

def get_label_transforms():
    """Transforms for labels - only resize with nearest neighbor"""
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST),
    ])

class JointTransform:
    """Apply the same geometric transforms to both image and label"""
    
    def __init__(self, size = (256, 256), training = True):
        self.size = size
        self.training = training
    
    def __call__(self, image, label):
        # Resize both with appropriate interpolation
        image = TF.resize(image, self.size, interpolation = transforms.InterpolationMode.NEAREST)
        label = TF.resize(label, self.size, interpolation = transforms.InterpolationMode.NEAREST)
        
        if self.training:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            
            # Random vertical flip  
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
        
        return image, label

class ImageOnlyTransform:
    """Apply transforms only to images (after joint transforms)"""    
    def __init__(self):
        self.transform = transforms.Normalize(mean = [0.5], std = [0.5])
    
    def __call__(self, image):
        return self.transform(image)
    
class TCIADataset(Dataset):
    def __init__(self, cases, training = True, use_joint_transforms = True):
        self.img_dir = './tcia_dataset_preprocessed/'
        self.cases = cases
        self.training = training
        self.use_joint_transforms = use_joint_transforms
        self.data_pairs = []

        for case in cases:
            image_dir = os.path.join(self.img_dir, case, 'Image')
            label_dir = os.path.join(self.img_dir, case, 'GT')

            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                continue

            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
            
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file)
                
                if os.path.exists(label_path):
                    self.data_pairs.append((img_path, label_path))
        
        # Setup transforms
        if use_joint_transforms:
            self.joint_transform = JointTransform(size = (256, 256), training = training)
            self.image_transform = ImageOnlyTransform()
            self.label_transform = None
        else:
            self.joint_transform = None
            self.image_transform = (get_image_training_transforms() if training 
                                  else get_image_testing_transforms())
            self.label_transform = get_label_transforms()

    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.data_pairs[idx]

        image = np.load(img_path)
        label = np.load(label_path)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        if image.ndim == 2:
            image = image.unsqueeze(0)
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # Shape: [3, H, W]
        
        label = label.unsqueeze(dim = 0)

        # Apply transforms
        if self.use_joint_transforms and self.joint_transform:
            # Apply geometric transforms to both
            image, label = self.joint_transform(image, label)
            # Apply image-only transforms
            if self.image_transform:
                image = self.image_transform(image)
        else:
            # Apply separate transforms
            if self.image_transform:
                image = self.image_transform(image)
            if self.label_transform:
                label = self.label_transform(label)
        
        return image, label.squeeze(dim = 0)
    
class LitsDataset(Dataset):
    def __init__(self, cases, training = True, use_joint_transforms = True):
        self.img_dir = './lits_dataset_preprocessed/'
        self.cases = cases
        self.training = training
        self.use_joint_transforms = use_joint_transforms
        self.data_pairs = []

        for case in cases:
            image_dir = os.path.join(self.img_dir, str(case), 'Image')
            label_dir = os.path.join(self.img_dir, str(case), 'GT')

            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                continue

            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
            
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file)
                
                if os.path.exists(label_path):
                    self.data_pairs.append((img_path, label_path))
        
        # Setup transforms
        if use_joint_transforms:
            self.joint_transform = JointTransform(size = (256, 256), training = training)
            self.image_transform = ImageOnlyTransform()
            self.label_transform = None
        else:
            self.joint_transform = None
            self.image_transform = (get_image_training_transforms() if training 
                                  else get_image_testing_transforms())
            self.label_transform = get_label_transforms()

    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.data_pairs[idx]

        image = np.load(img_path)
        label = np.load(label_path)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        if image.ndim == 2:
            image = image.unsqueeze(0)
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # Shape: [3, H, W]
        
        label = label.unsqueeze(dim = 0)

        # Apply transforms
        if self.use_joint_transforms and self.joint_transform:
            # Apply geometric transforms to both
            image, label = self.joint_transform(image, label)
            # Apply image-only transforms
            if self.image_transform:
                image = self.image_transform(image)
        else:
            # Apply separate transforms
            if self.image_transform:
                image = self.image_transform(image)
            if self.label_transform:
                label = self.label_transform(label)
        
        return image, label.squeeze(dim = 0)