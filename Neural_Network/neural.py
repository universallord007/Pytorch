import torch 
import numpy as np
import matplotlib.pyplot as plt
from model import Model

x = torch.linspace(-5,5,100).view(-1,1)
y = torch.sin(x)
model = Model()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.009)

for i in range(6000):
    y_pred = model(x)
    loss = ((y_pred - y)**2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # if i % 100 == 0 :
    #     print(f"The loss is : {loss.item()}")


pred = model(x).detach()
#This is the true curve
plt.scatter(x.numpy(), y.numpy(), color='red')
#This is the curve the model has predicted
plt.plot(x.numpy(), pred.numpy(), color='blue')


tests = torch.tensor([[1.0], [2.0], [3.0]])
pred = model(tests).detach()
real = torch.sin(tests)
print(pred)
print(real)

torch.save(model.state_dict(),"sin_curve.pth")