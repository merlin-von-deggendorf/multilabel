import torch

num_classes   = 5
sample_labels = [0, 2, 4]               # this example belongs to classes 0, 2 and 4
y = torch.zeros(num_classes)            # start with all zeros
y[sample_labels] = 1.                   # set the positive classes to 1

print(y)  # tensor([1., 0., 1., 0., 1.])
