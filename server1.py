import torch
import torch.nn as nn
import torch.optim as optim


class MainServer:
    
    def __init__(self, model, device, lr=0.01):
        self.model     = model.to(device)
        self.device    = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=6)

    # Forward + Backward
    def ServerUpdate(self, A_k, y, clear_grad=True):
        if clear_grad:
            self.optimizer.zero_grad()

        y = y.to(self.device)

        # 防呆：確保 input_tensor 是 leaf tensor 且有 requires_grad
        if not A_k.is_leaf or not A_k.requires_grad:
            input_tensor = A_k.detach().requires_grad_(True).to(self.device)
        else:
            input_tensor = A_k.to(self.device)

        # Forward（ServerModel 內部依 use_denoiser 決定是否去噪）
        y_hat = self.model(input_tensor)
        loss  = self.criterion(y_hat, y)

        # Backward
        loss.backward()

        # 取梯度
        dA_k = (input_tensor.grad.detach()
                if input_tensor.grad is not None
                else torch.zeros_like(input_tensor))

        return dA_k, loss.item()

    # Optimizer step（含 gradient clipping）
    def step(self):
        """clip_grad_norm_ 在此處執行，永遠早於 optimizer.step()"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
