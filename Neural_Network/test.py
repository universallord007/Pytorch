import torch
from model import Model

model = Model()
model.load_state_dict(torch.load("sin_curve.pth"))
model.eval()

tests = torch.tensor([[4.0],[5.0],[6.0],[7.0]])

with torch.no_grad():
    pred = model(tests)

real = torch.sin(tests)

print("The real value is:")
print(real)
print("The model's predicted value is:")
print(pred)