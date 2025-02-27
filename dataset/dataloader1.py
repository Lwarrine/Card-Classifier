import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def classes(self):
        return self.data.classes
    
    def get_data_loaders(base_path='/projects/dsci410_510/Luke_Card_Classifier', batch_size=32):

        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),])
    
        train_data = PlayingCardDataset(data_dir= os.path.join(base_path, 'train'), transform = transform)
        test_data = PlayingCardDataset(data_dir= os.path.join(base_path, 'test'), transform = transform)
        val_data = PlayingCardDataset(data_dir= os.path.join(base_path, 'valid'), transform = transform)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, val_loader