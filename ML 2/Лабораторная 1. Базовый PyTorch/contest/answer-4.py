import torch
import torch.nn as nn


'''
hidden_dim=100 just works
'''
class SimpleModel(nn.Module):
    def __init__(self, hidden_dim: int = 100):
        super().__init__()
        self.linear1 = torch.nn.Sequential(torch.nn.Linear(in_features=784, out_features=1024*10),
         torch.nn.ReLU())
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(in_features=1024*10, out_features=100),
         torch.nn.ReLU())
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(in_features=100, out_features=10),
         torch.nn.Sigmoid(),torch.nn.Softmax())

    def forward(self, images):
        x = self.linear1(images)
        x = self.linear2(x)
        return self.linear3(x)