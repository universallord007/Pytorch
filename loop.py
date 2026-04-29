import torch

x = torch.tensor(2.0)
m = torch.tensor(3.0 , requires_grad=True)
b = torch.tensor(1.0 , requires_grad=True)

lr = 0.1
epochs = 500

for i in range(epochs):
    #Forward Pass
    y = m*x + b
    y_true = 5
    #Loss
    error = (y_true - y)**2
    error.backward()

    with torch.no_grad():
        m -= lr*m.grad
        b -= lr*b.grad

    if m.grad is not None:
        m.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    # if i%10 == 0:
    #     print(f"The epoch {i}: error = {error.item()}")

# .item converts the tensor to normal python
print(m.item())
print(b.item())