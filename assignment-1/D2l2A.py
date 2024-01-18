# D2l2A.py D2l 2.1 2023 cheng CS5173/6073
# Usage: python D2l2A.py

# D2l 2.1 Data Manipulation
# D2l 2.1.1 Getting Started
import torch
x = torch.arange(12, dtype=torch.float32)
print(x)
print(x.numel())
print(x.shape)
X = x.reshape(3, 4)
print(X)
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.randn(3, 4))
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# D2l 2.1.2 Indexing and Slicing
print(X[-1], X[1:3])
X[1, 2] = 17
print(X)
X[:2, :] = 12
print(X)

# D2l 2.1.3 Operations
print(torch.exp(x))
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())

# D2l 2.1.4 Broadcasting
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

# D2l 2.1.5 Saving Memory
print(id(Y))
Y = Y + X
print(id(Y))
Z = torch.zeros_like(Y)
print(id(Z))
Z[:] = X + Y
print(id(Z))
print(id(X))
X += Y
print(id(X))

# D2l 2.1.6 Conversion to Other Python Objects
A = X.numpy()
B = torch.from_numpy(A)
print(type(A), type(B))
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
