import torch

x = torch.tensor(2.0)
m = torch.tensor(3.0 , requires_grad=True)
b = torch.tensor(1.0,requires_grad=True)
y = m*x + b
y_true = torch.tensor(5.0)
loss = (y_true - y)**2
loss.backward()
lr = 0.01 
with torch.no_grad():
    m = m - lr*m.grad
    b = b - lr*b.grad
if m.grad is not None:
    m.grad.zero_()
if b.grad is not None :
    b.grad.zero_()
print(m)
print(b)