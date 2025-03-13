import torch
import torch.nn as nn
from torchvision import models

class CardClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        size = 64
        self.conv1 = nn.Conv2d(input_size, 16, 3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(size, size, 3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 32 * 32, size)
        self.fc2 = nn.Linear(size, output_size)
        self.act = torch.nn.LeakyReLU() 
        self.act2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv2_1(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return self.fc2(x)
