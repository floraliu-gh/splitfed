import torch
import torch.nn as nn
import torch.optim as optim

class MainServer:
    def __init__(self, model, device, lr=0.01):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def ServerUpdate(self, A_k, y):
        """
        Server 的 forward + backward
        回傳: dA_k (傳給 client 的梯度)
        """
        # 確保 A_k 在正確的 device 並且需要梯度
        A_k = A_k.to(self.device)
        y = y.to(self.device)
        A_k.requires_grad_(True)

        # Forward pass
        y_hat = self.model(A_k)
        loss = self.criterion(y_hat, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # 取得 A_k 的梯度 (傳回給 client)
        dA_k = A_k.grad.detach()
        return dA_k, loss.item()

    def step(self):
        """
        更新 server model 的參數
        """
        self.optimizer.step()

