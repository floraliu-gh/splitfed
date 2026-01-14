import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, model, data_loader, device, lr=1e-3):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        # optimizer 一定要是 class 成員
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.last_A_k = None
        self.data_iter = iter(self.data_loader) 

    def ClientUpdate(self):
        """
        Forward only
        return smashed data A_k,t and label Y_k
        """
        try:
            x, y = next(self.data_iter)
        except StopIteration:
            # 如果 epoch 結束,重新開始
            self.data_iter = iter(self.data_loader)
            x, y = next(self.data_iter)

        A_k = self.model(x)
        self.last_A_k = A_k      # 存起來給 backward 用

        return A_k.detach(), y

    def ClientBackprop(self, dA_k):
        """
        Backprop using received gradient
        """
        self.optimizer.zero_grad()

        # 核心：對 smashed data backward
        self.last_A_k.backward(dA_k)
        self.optimizer.step()

