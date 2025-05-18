import random
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from circle_dataset import CircleSet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CircleModel(torch.nn.Module):
    def __init__(self, numCircles,neurons_per_circle=4):
        super(CircleModel, self).__init__()
        self.numCircles = numCircles
        self.net=torch.nn.Sequential(
            torch.nn.Linear(2, neurons_per_circle*numCircles),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons_per_circle*numCircles, neurons_per_circle*numCircles),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons_per_circle*numCircles, numCircles)
        ) 

    def forward(self, x):
        return self.net(x)
    
    def train_model(self, dataset, num_epochs=1000, batch_size=32, learning_rate=0.001, early_stopping=0.05):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            avg_loss = 0.0
            self.train()
            for x_batch, y_batch in dataloader:
                logits = self(x_batch)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
            if avg_loss < early_stopping:
                print(f"Early stopping at epoch {epoch} with loss {avg_loss:.4f}")
                break
    def evaluate(self, dataset,batch_size=1024,probabilitie_threshold=0.5):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            correct_predictions = 0
            total_predictions = 0
            dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for x_batch, y_batch in dataloader:
                logits = self(x_batch)
                predictions = torch.sigmoid(logits) > probabilitie_threshold
                real_labels = y_batch > probabilitie_threshold
                results = predictions == real_labels
                correct += results.sum().item()
                total += results.numel()
                total_predictions += len(x_batch)
                # only count the predictions where all lables are correct
                correct_predictions += (predictions.sum(dim=1) == real_labels.sum(dim=1)).sum().item()
                
            accuracy = correct / total
            print(f'class count: {self.numCircles} total points: {len(dataset)} max correct labels: {self.numCircles*len(dataset)}')
            print(f"total: {total}, correct: {correct}, accuracy: {accuracy:.4f}")
            print(f"total predictions: {total_predictions}, correct predictions: {correct_predictions}, accuracy: {correct_predictions/total_predictions:.4f}")
    def train_and_evaluate(self, dataset,split, num_epochs=1000, batch_size=32, learning_rate=0.001, early_stopping=0.05):
        train_size = int(len(dataset) * split)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_model(train_dataset, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, early_stopping=early_stopping)
        self.evaluate(test_dataset,batch_size=batch_size)

circle_count=8
circle_model = CircleModel(numCircles=circle_count,neurons_per_circle=4)
circle_model=circle_model.to(device)
dataset = CircleSet(minX=0, minY=0, maxX=100, maxY=100, minRadius=10, maxRadius=30, numCircles=circle_count, numPoints=500)
# dataset.display()
print(f'device: {device}')
# dataset.print()
# circle_model.train_and_evaluate(dataset,split=0.8, num_epochs=5000, batch_size=1024, learning_rate=0.001, early_stopping=0.005)






