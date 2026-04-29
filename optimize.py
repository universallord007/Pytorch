import torch

x = torch.tensor(2.0)
m = torch.tensor(3.0 , requires_grad=True)
b = torch.tensor(1.0 , requires_grad=True)

lr = 0.1
epochs = 500

optimizer = torch.optim.SGD([m,b],lr = lr)


for i in range(epochs):
    #Forward Pass
    y = m*x + b
    y_true = 5
    #Loss
    error = (y_true - y)**2
    error.backward()

    optimizer.step()

    optimizer.zero_grad()
    # if i%10 == 0:
    #     print(f"The epoch {i}: error = {error.item()}")

# .item converts the tensor to normal python
print(m.item())
print(b.item())