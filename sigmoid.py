import torch
import torch.nn as nn
import torch.optim as optim

tensor = torch.tensor([0.0, 1.0])

model2 = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2,1),
    nn.Sigmoid()
)

# loss with binary cross entropy loss
# this is the same as applying sigmoid and then applying BCELoss
# but this is not numerically stable
loss2 = nn.BCELoss()(model2(tensor), torch.tensor([1.0]))

# this is the same as applying sigmoid and then applying BCELoss
model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2,1),
    # nn.Sigmoid() we skip this because we are using BCEWithLogitsLoss which combines
    # Sigmoid and BCELoss in one single class
)

criterion = nn.BCEWithLogitsLoss() # This combines Sigmoid and BCELoss in one single class
loss= criterion(model(tensor), torch.tensor([1.0])) 







