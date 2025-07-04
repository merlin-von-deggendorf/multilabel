import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from imagedataset import CustomImageDataset, ImageLabel
import labeler
import StringIDMapper


class ResNet18MultiLabel(nn.Module):
    def __init__(self, num_labels, hidden_dim=512, dropout=0.5):
        super().__init__()
        # 1) Load pretrained backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = self.backbone.fc.in_features
        self.num_classes = num_labels
        # 2) Replace its head with a two‐layer MLP
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),                 # optional
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        logits = self.backbone(x)                # raw scores
        probs  = torch.sigmoid(logits)           # if you need probabilities
        return probs, logits
    
    def train_model(self,dataset,num_epochs=1000, batch_size=32, learning_rate=0.001, early_stopping=0.05):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            avg_loss = 0.0
            total_batches = len(dataloader)
            current_batch = 0
            for x_batch, y_batch in dataloader:
                logits = self(x_batch)[1]
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                print(f"batch {current_batch}/{total_batches}, loss: {loss.item():.4f}")
                current_batch += 1
            avg_loss /= len(dataloader)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
            if avg_loss < early_stopping:
                print(f"Early stopping at epoch {epoch} with loss {avg_loss:.4f}")
                break
            else:
                print(f"no early stopping at epoch {epoch} with loss {avg_loss:.4f}")
    def evaluate(self, dataset,batch_size=1024,probabilitie_threshold=0.5):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            correct_predictions = 0
            total_predictions = 0
            dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=False)
            total_batches = len(dataloader)
            current_batch = 0
            for x_batch, y_batch in dataloader:
                logits = self(x_batch)[1]
                predictions = torch.sigmoid(logits) > probabilitie_threshold
                real_labels = y_batch > probabilitie_threshold
                results = predictions == real_labels
                correct += results.sum().item()
                total += results.numel()
                total_predictions += len(x_batch)
                # only count the predictions where all lables are correct
                correct_predictions += (predictions.sum(dim=1) == real_labels.sum(dim=1)).sum().item()
                print(f"batch {current_batch}/{total_batches}, correct: {correct}, total: {total}")
                current_batch += 1
                
            accuracy = correct / total
            print(f'class count: {self.num_classes} total points: {len(dataset)} max correct labels: {self.num_classes*len(dataset)}')
            print(f"total: {total}, correct: {correct}, accuracy: {accuracy:.4f}")
            print(f"total predictions: {total_predictions}, correct predictions: {correct_predictions}, accuracy: {correct_predictions/total_predictions:.4f}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set = CustomImageDataset(root_dir="D:/datasets/multilabel/missingclass/train/",device=device)
string_id_mapper = train_set.string_id_mapper
test_set = CustomImageDataset(root_dir="D:/datasets/multilabel/missingclass/test/",device=device,string_id_mapper=string_id_mapper)
reduced_test_set = CustomImageDataset(root_dir="D:/datasets/multilabel/missingclass/reduced_test/",device=device,string_id_mapper=string_id_mapper)
evil_test_set = CustomImageDataset(root_dir="D:/datasets/multilabel/missingclass/evil_test/",device=device,string_id_mapper=string_id_mapper)
model = ResNet18MultiLabel(num_labels=test_set.num_classes).to(device)
model.train_model(train_set, num_epochs=10, batch_size=400, learning_rate=0.001, early_stopping=0.05)
print("Training finished")
print("Evaluating model")
print("test set")
model.evaluate(test_set,batch_size=400,probabilitie_threshold=0.5)
print("reduced test set")
model.evaluate(reduced_test_set,batch_size=400,probabilitie_threshold=0.5)
print("evil test set")
model.evaluate(evil_test_set,batch_size=400,probabilitie_threshold=0.5)

