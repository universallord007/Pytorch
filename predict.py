import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self,x):
        return self.linear(x)
    

model = LinearModel()
model.load_state_dict(torch.load("linear_model.pth"))
model.eval()

x = torch.tensor([15.0])
y_true = model(x)

print(y_true.item())