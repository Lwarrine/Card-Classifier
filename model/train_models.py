import torch
import torch.nn as nn
from torchvision import models
import sys
import os
sys.path.append(os.path.abspath('..'))
from model.model1 import CardClassifier
from dataset.dataloader1 import PlayingCardDataset

class Trainer():
    def __init__(self, input_size = 3, output_size = 6, random_seed= None, epochs=15, loss_fn=None, reg_param=0.001,
                 lr=1e-3,base_path='/projects/dsci410_510/Luke_Card_Classifier', batch_size = 128):
        self.input_size = 3
        self.output_size = 6
        self.random_seed = random_seed
        self.epochs = epochs
        self.reg_param = reg_param
        self.lr = lr
        self.base_path = base_path
        self.batch_size = batch_size
        if loss_fn == None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
        self.train_loader, self.test_loader, self.val_loader = PlayingCardDataset.get_data_loaders(self.base_path,self.batch_size,random_seed=self.random_seed)
        self.set_seed()
                     
    def train_step(self, model, train_loader, loss_fn, optimizer, reg_param, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss  = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        return avg_loss, train_acc

    def evaluation_step(self, model, data_loader, loss_fn, reg_param, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss  = loss_fn(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = running_loss / total
        acc = correct / total
        return avg_loss, acc

    def set_seed(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
    def train_conv_model(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        else:
            if torch.backends.mps.is_available():
                device = "mps"


        model = CardClassifier(self.input_size,self.output_size)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, weight_decay= self.reg_param)
        #loss_fn = loss_fn

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        val_losses = []
        val_accs = []
        best_model = "best_model.pth"
        best_acc = 0
        best_losss = 0

        for i in range(self.epochs):
            train_loss, train_acc = self.train_step(model, self.train_loader, self.loss_fn, optimizer, self.reg_param, device)
            val_loss, val_acc = self.evaluation_step(model, self.val_loader, self.loss_fn, self.reg_param, device)
            test_loss, test_acc = self.evaluation_step(model, self.test_loader, self.loss_fn, self.reg_param, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f" {i} epoch done")
            if test_acc > best_acc:
                best_acc = test_acc
                bestlosss = test_loss
                best_epoch = i
                torch.save(model.state_dict(), best_model)
        #print(f"train loss: {train_losses[-1]}")
        #print(f"val loss: {val_losses[-1]}")
        #print(f"test loss: {test_losses[-1]}")
        #print(f"train acc:{train_accs[-1]}")
        #print(f"val acc: {val_accs[-1]}")
        #print(f"test acc: {test_accs[-1]}")
        print(f"Best Accuracy: {best_acc}, Epoch: {best_epoch}, Loss: {test_loss}")
        return model, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs