import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None, random_seed=None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.set_seed()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def classes(self):
        return self.data.classes

    def set_seed(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def get_data_loaders(base_path='/projects/dsci410_510/Luke_Card_Classifier', batch_size=32, random_seed = None):

        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),])
    
        train_data = PlayingCardDataset(data_dir= os.path.join(base_path, 'train'), transform = transform, random_seed = random_seed)
        test_data = PlayingCardDataset(data_dir= os.path.join(base_path, 'test'), transform = transform, random_seed = random_seed)
        val_data = PlayingCardDataset(data_dir= os.path.join(base_path, 'valid'), transform = transform, random_seed = random_seed)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, val_loader