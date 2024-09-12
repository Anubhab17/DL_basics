import torch
import numpy as np

#  create a vector of evenly spaced values, starting at 0 (included) and ending at n (not included)
x = torch.arange(12, dtype=torch.float32)
print(x)

# total number of elements in a tensor
print(x.numel())

# tensor’s shape
print(x.shape)

#  change the shape of a tensor without altering its size or values, by invoking reshape
X = x.reshape(3, 4)
print(X)

# tensor with all elements set to 0
print(torch.zeros((2, 3, 4)))

# tensor with all elements set to 1
print(torch.ones((2, 3, 4)))

# tensor with elements drawn from a standard Gaussian (normal) distribution
print(torch.randn(3, 4))

# tensors by supplying the exact values for each element by supplying (possibly nested) Python list(s) containing numerical literals
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# Indexing and Slicing

print(X[-1], X[1:3])
X[1, 2] = 17
print(X)

X[:2, :] = 12
print(X)

