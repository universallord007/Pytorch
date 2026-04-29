import torch

x = torch.tensor([1.0,2.0,3.0])
y = torch.tensor([4.0,5.0,6.0])

# print(x)
# print(y)
# print(x.shape)
# print(x+y)

a = torch.tensor(2.0, requires_grad=True)
b = a**3
b.backward()
print(a.grad)