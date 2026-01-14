import torch
import torch.nn as nn

class ServerModel(nn.Module):
    """
    Server 端的模型 (後半部分)
    輸入: 256 維的 activation
    輸出: 10 個類別的 logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  
        )

    def forward(self, A_k):
        return self.net(A_k)
