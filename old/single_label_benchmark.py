import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import json

class ClassificationModel:
    def __init__(self, devicestr=None, num_classes=4, lr=0.001):
        # Set up device
        self.num_classes = num_classes
        if devicestr is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(devicestr)
        self.transform = transforms.Compose([
            transforms.Resize(256),            # Resize the smaller edge to 256 while keeping the aspect ratio.
            transforms.CenterCrop(224),        # Crop the center 224x224 region.
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)
        self.classes = None
    
    def train_model(self, train_dir, num_epochs=10, batch_size=400,learing_rate=0.01, early_stopping=0.05):
        # ...existing code to train...
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learing_rate)    
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            total_iterations = len(self.train_loader)
            iteration = 0
            total_batches = len(self.train_loader)
            current_batch = 0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                print(f"Iteration {iteration}/{total_iterations}")
                iteration += 1
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                print(f"batch {current_batch}/{total_batches}, loss: {loss.item():.4f}")
                current_batch += 1
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / total_iterations}")
            if running_loss / total_iterations < early_stopping:
                print(f"Early stopping at epoch {epoch} with loss {running_loss / total_iterations:.4f}")
                break
            else:
                print(f"no early stopping at epoch {epoch} with loss {running_loss / total_iterations:.4f}")

    
    def evaluate(self, test_dir, batch_size):
        # ...existing code to evaluate...
        self.test_dataset = datasets.ImageFolder(test_dir, transform=self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(f"Accuracy: {100 * correct / total}")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model:ClassificationModel = ClassificationModel(num_classes=9)
model.train_model("D:/datasets/multilabel/missingclass/train/", num_epochs=2, batch_size=400,learing_rate=0.001)
model.evaluate("D:/datasets/multilabel/missingclass/test/", batch_size=400)
model.evaluate("D:/datasets/multilabel/missingclass/reduced_test/", batch_size=400)
model.evaluate("D:/datasets/multilabel/missingclass/evil_test/", batch_size=400)