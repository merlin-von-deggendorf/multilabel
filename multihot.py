import torch
# Suppose K=6 labels, and 3 samples:
labels = [
    [0, 2],        # sample 0 has classes 0 & 2
    [1, 3, 5],     # sample 1 has classes 1, 3 & 5
    [2, 4]         # sample 2 has classes 2 & 4
]
num_classes = 6
batch_size = len(labels)
multihot = torch.zeros(batch_size, num_classes, dtype=torch.float)
for i in range(batch_size):
    multihot[i, labels[i]] = 1.
if torch.cuda.is_available():
    multihot = multihot.to(device='cuda')
print(multihot)
   
