import torch
import torch.nn as nn
import torch.optim as optim


class MainServer:
    """
    管理 ServerModel 的訓練流程。

    修正說明（對比原版）：
      - 移除 denoiser 參數：去噪器現在是 ServerModel 的一部分（use_denoiser 旗標），
        由 main.py 統一控制，不再由 MainServer 重複管理。
      - clip_grad_norm_ 移入 step()，確保永遠在 optimizer.step() 之前執行。
      - 移除重複的 self.denoiser = denoiser 賦值。
    """

    def __init__(self, model, device, lr=0.01):
        self.model     = model.to(device)
        self.device    = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=6)

    # ------------------------------------------------------------------
    # Forward + Backward（一次 mini-batch）
    # ------------------------------------------------------------------
    def ServerUpdate(self, A_k, y, clear_grad=True):
        """
        Parameters
        ----------
        A_k       : Tensor，來自 main.py 的 leaf tensor（detach + requires_grad=True）
                    ServerModel.forward() 內部會依 use_denoiser 旗標決定是否去噪。
        y         : 標籤
        clear_grad: 是否清空 server optimizer 的梯度

        Returns
        -------
        dA_k     : Tensor，對 A_k 的梯度（供 Client 反向傳播用）
        loss_val : float
        """
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

    # ------------------------------------------------------------------
    # Optimizer step（含 gradient clipping）
    # ------------------------------------------------------------------
    def step(self):
        """clip_grad_norm_ 在此處執行，永遠早於 optimizer.step()"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()