import torch
import torch.nn as nn
import torch.optim as optim

tensor = torch.tensor([0.0, 1.0])

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2,1)
    # nn.Softmax(dim=1) Softmax is applied in the loss function
)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss applies softmax internally because it simplifies the computation





