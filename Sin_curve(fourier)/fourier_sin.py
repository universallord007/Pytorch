import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Model

x = torch.linspace(-50,50,100).view(-1,1)
y = torch.sin(x)

x_features = torch.cat([torch.sin(x),torch.cos(x)],dim=1)

model = Model()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for i in range(1000):
    y_true = model(x_features)
    loss = ((y_true - y)**2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
with torch.no_grad():
    pred = model(x_features)

plt.scatter(x.numpy(),y.numpy(),color='blue')
plt.plot(x.numpy(),pred.numpy(),color='red')
plt.show()