import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)
    
#Here we are creating a Linear model which will do our task instead of typing it manually like we were doing previously . 

model = LinearModel()

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0]])
y_true = torch.tensor([[2.9],[5.2],[6.8],[9.1],[10.9],[13.2],[14.8]])

optimizer = torch.optim.SGD(model.parameters(),lr=0.03)

for i in range(1000):
    y = model(x)

    loss = ((y_true - y)**2).mean()

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
    if i % 100 == 0:
        print(loss.item())

print(model.linear.weight.item())
print(model.linear.bias.item())

plt.scatter(x.numpy(),y_true.numpy())

y_pred = model(x).detach()

plt.plot(x.numpy(),y_pred.numpy())
print(model(torch.tensor([[10.0]])).item())
plt.show()

torch.save(model.state_dict(), "linear_model.pth")
saved = torch.load("linear_model.pth")
print(saved)