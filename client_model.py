import torch
import torch.nn as nn

class ClientModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            
            # Flatten and FC
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
