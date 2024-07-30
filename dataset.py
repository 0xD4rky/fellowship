import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import scipy.io
import numpy as np
from tqdm import tqdm

"""
enter data paths
"""

data_dir = r'C:\Users\DELL\OneDrive\Documents\fellowship\data\jpg'
labels_file = r'C:\Users\DELL\OneDrive\Documents\fellowship\data\imagelabels.mat'
splits_file = r'C:\Users\DELL\OneDrive\Documents\fellowship\data\setid.mat'

labels = scipy.io.loadmat(labels_file)['labels'][0]
splits = scipy.io.loadmat(splits_file)
train_ids = splits['trnid'][0]-1
val_ids = splits['valid'][0]-1
test_ids = splits['tstid'][0]-1

class FlowerDataset(Dataset):
    def __init__(self,data_dir,file_ids,labels,transform=None):
        self.data_dir = data_dir
        self.file_ids = file_ids
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self,idx):
        img_name = f'image_{self.file_ids[idx]:05d}.jpg'
        img_path = os.path.join(self.data_dir,img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[self.file_ids[idx]]-1
        
        if self.transform:
            image = self.transform(image)
            
        return image,label
    
# Data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


