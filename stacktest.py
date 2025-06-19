import torch

# three column‐vectors, each N×1
t1 = torch.tensor([[1], [2], [3]])   # shape (3,1)
t2 = torch.tensor([[4], [5], [6]])   # shape (3,1)
t3 = torch.tensor([[7], [8], [9]])   # shape (3,1)

# stack them into a 3×3 matrix
X = torch.cat([t1, t2, t3], dim=1)
print(X)
# tensor([[1, 4, 7],
#         [2, 5, 8],
#         [3, 6, 9]])
print(X.shape)  # torch.Size([3, 3])
