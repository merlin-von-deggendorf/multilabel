import random
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from circle_dataset import CircleSet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CircleModel(torch.nn.Module):
    def __init__(self, circle_index,neurons_per_circle=4):
        super(CircleModel, self).__init__()
        self.circle_index = circle_index
        self.net=torch.nn.Sequential(
            torch.nn.Linear(2, neurons_per_circle),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons_per_circle, neurons_per_circle),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons_per_circle, 1)
            #Sigmoid is applied in the loss function (BCEWithLogitsLoss)
        ) 

    def forward(self, x):
        return self.net(x)
    
    def train_model(self, dataset, num_epochs=1000, batch_size=32, learning_rate=0.001, early_stopping=0.05):
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logits Loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        self.losses = []
        for epoch in range(num_epochs):
            
            self.train()
            avg_loss = 0.0
            for x_batch, y_batch in dataloader:
                logits = self(x_batch)
                y_batch = y_batch[:, self.circle_index].unsqueeze(1)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            self.losses.append(avg_loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
            if avg_loss < early_stopping:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
                print(f"Early stopping at epoch {epoch} with loss {avg_loss:.4f}")
                break
        

    def train_and_evaluate(self, dataset,split=0.8, num_epochs=1000, batch_size=32, learning_rate=0.001, early_stopping=0.1):
        # Split the dataset into training and validation sets
        train_size = int(split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Train the model
        self.train_model(train_dataset, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, early_stopping=early_stopping)
        
        # Evaluate the model on the validation set
        self.evaluate(val_dataset,batch_size=batch_size)


    def plot_losses(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.losses, label="train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

class CircleModelContainer:
    def __init__(self, circle_dataset: CircleSet, neurons_per_circle=4):
        self.circle_dataset = circle_dataset
        self.models : list[CircleSet]= []
        for i in range(circle_dataset.numCircles):
            model = CircleModel(i,neurons_per_circle=neurons_per_circle).to(device)
            self.models.append(model)
    def train_all(self,dataset, num_epochs=1000, batch_size=32, learning_rate=0.001, early_stopping=0.05):
        for i, model in enumerate(self.models):
            print(f"Training model for circle {i}")
            model.train_model(dataset, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, early_stopping=early_stopping)
    def evaluate_all(self, dataset,batch_size=1024,probabilitie_threshold=0.5):
        for model in self.models:
            model.eval()
        with torch.no_grad():
            dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=False)
            correct = 0
            total = 0

            for x_batch, y_batch in dataloader:
                full_predicts=[]
                # generate seperate y_batch for each model
                for model in self.models:
                    # get logits
                    logits = model(x_batch)
                    # get probabilities, then binary predictions
                    probs = torch.sigmoid(logits)
                    preds = (probs >= probabilitie_threshold).float()
                    full_predicts.append(preds)
                stacked_preds = torch.cat(full_predicts, dim=1)
                correct += (stacked_preds == y_batch).sum().item()
                total += y_batch.numel()
            accuracy = correct / total
            print(f"Accuracy: {accuracy:.4f} correct: {correct} total: {total}")
                    
                    
    def train_and_evaluate_all(self,split=0.8, num_epochs=1000, batch_size=32, learning_rate=0.001, early_stopping=0.1):
        dataset = self.circle_dataset
        # Split the dataset into training and validation sets
        train_size = int(split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Train the model
        self.train_all(train_dataset,num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, early_stopping=early_stopping)
        
        # Evaluate the model on the validation set
        self.evaluate_all(val_dataset,batch_size=batch_size)
    def plot_losses(self):
        for i, model in enumerate(self.models):
            plt.figure()
            plt.plot(model.losses, label=f"train loss circle {i}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training Loss Curve for Circle {i}")
            plt.legend()
            plt.show()
    
   
circle_count = 5
dataset = CircleSet(minX=0, minY=0, maxX=100, maxY=100, minRadius=20, maxRadius=40, numCircles=circle_count, numPoints=500)
# dataset.display()
# for i in range(circle_count):
#     dataset.display(i)


# create model for each circle

circle_model_container = CircleModelContainer(dataset,neurons_per_circle=12)
circle_model_container.train_and_evaluate_all(split=0.8, num_epochs=50000, batch_size=1024, learning_rate=0.001, early_stopping=0.01)
circle_model_container.plot_losses()

