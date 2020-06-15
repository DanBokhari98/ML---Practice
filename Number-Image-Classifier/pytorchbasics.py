import torch

x = torch.tensor([5,3])
y = torch.tensor([2,1])

print(x * y)

x = torch.zeros([2,5])
print(x)

x.shape
torch.Size([2,5])

y = torch.rand([2,5])
print(y)

y = y.view([1,10])
print(y)
