import torch
from torch.utils.data import TensorDataset, DataLoader

# -- Generate 100 samples of 10-dim features
num_samples = 4
num_features = 10
num_classes = 3

# Random inputs
X = torch.randn(num_samples, num_features)

# Random integer labels in {0,1,2}
y = torch.randint(low=0, high=num_classes, size=(num_samples,))

dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

for i, (x, y) in enumerate(loader):
    print(f"Batch {i+1}:")
    print(f"X: {x}")
    print(f"y: {y}")
    print()