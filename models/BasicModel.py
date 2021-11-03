import torch
import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            
            nn.Linear(200,100),
        )

        
    def forward(self, x):
        x = self.layer(x)
        return x