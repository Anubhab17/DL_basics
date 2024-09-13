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

# Operations

print(f'exp(x) is: {torch.exp(x)}')

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

print(X == Y)

print(X.sum())

# Broadcasting

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)
print(a - b)
print(a * b)

# Saving memory

before = id(Y)
Y = Y + X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)

before = id(X)
X += Y
print(id(X) > before)


# Conversion to Other Python Objects

A = X.numpy()
B = torch.from_numpy(A)
print(type(A), type(B))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))