# D2l2C.py D2l 2.3 2023 cheng CS5173/6073
# Usage: python D2l2C.py

# D2l 2.3 Linear Algebra
# D2l 2.3.1 Scalars
import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x**y)

# D2l 2.3.2 Vectors
x = torch.arange(3)
print(x)
print(x[2])
print(len(x))
print(x.shape)

# D2l 2.3.3 Matrices
A = torch.arange(6).reshape(3, 2)
print(A)
print(A.T)
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(A == A.T)

# D2l 2.3.4 Tensors
print(torch.arange(24).reshape(2, 3, 4))

# D2l 2.3.5 Basic Properties of Tensor Arithmetic
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone() # Assign a copy of `A` to `B` by allocating new memory
print(A)
print(A + B)
print(A * B)
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)

# D2l 2.3.6 Reduction
x = torch.arange(3, dtype=torch.float32)
print(x, x.sum())
print(A.shape)
print(A.sum())
print(A.sum(axis=0).shape)
print(A.sum(axis=1).shape)
print(A.sum(axis=[0, 1]))
print(A.mean(), A.sum() / A.numel())
print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

# D2l 2.3.7 Non-Reduction Sum
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A, sum_A.shape)
print(A / sum_A)
print(A.cumsum(axis=0))

# D2l 2.3.8 Dot Products
y = torch.ones(3, dtype = torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))

# D2l 2.3.9 Matrix-Vector Products
print(A.shape, x.shape, torch.mv(A, x), A@x)

# D2l 2.3.10 Matrix-Matrix Multiplication
B = torch.ones(3, 4)
print(torch.mm(A, B))
print(A@B)

# D2l 2.3.11 Norms
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
print(torch.abs(u).sum())
print(torch.norm(torch.ones((4, 9))))