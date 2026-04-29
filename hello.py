import torch
import torch.nn as nn 

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(3.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self,x):
        return self.m*x + self.b
    

model = LinearModel()

x = torch.tensor([1.0,2.0,3.0,4.0])
y_true = torch.tensor([3.0,5.0,7.0,9.0])

optimizer = torch.optim.SGD(model.parameters(),lr=0.03)

for i in range(1000):
    y = model(x)

    loss = ((y_true - y)**2).mean()

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

print(model.m.item())
print(model.b.item())