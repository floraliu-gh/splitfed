import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, model, device, lr=1e-3, channel=None, pixel_noise_injector=None):
        self.model                = model.to(device)
        self.device               = device
        self.optimizer            = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.last_A_k             = None
        self.channel              = channel
        self.pixel_noise_injector = pixel_noise_injector
        
    # Forward
    def ClientUpdate(self, x, add_pixel_noise=False):
        self.model.train()
        self.optimizer.zero_grad()

        x = x.to(self.device)

        if add_pixel_noise and self.pixel_noise_injector is not None:
            x = self.pixel_noise_injector.add_noise(x)

        A_k = self.model(x)
        self.last_A_k = A_k  # 保留計算圖，供 backward 使用

        return A_k  # 回傳乾淨特徵（未量化、未加噪）

    # Backward
    def ClientBackprop(self, dA_k):
        if dA_k is None or self.last_A_k is None:
            return
        dA_k = dA_k.to(self.device)
        self.last_A_k.backward(dA_k)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.last_A_k = None
