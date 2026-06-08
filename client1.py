import torch
import torch.nn as nn
import torch.optim as optim


class Client:
    """
    負責職責：
      1. ClientUpdate  : 只做 model forward，回傳乾淨特徵（保留計算圖）
      2. ClientBackprop: 接收已處理過的梯度，執行 backward → clip → step

    通道傳輸、量化、SNR 縮放、梯度 chain rule 全部由 main.py 統一管理，
    避免重複加噪 / 重複量化的問題。
    """

    def __init__(self, model, device, lr=1e-3, channel=None, pixel_noise_injector=None):
        self.model                = model.to(device)
        self.device               = device
        self.optimizer            = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.last_A_k             = None
        self.channel              = channel
        self.pixel_noise_injector = pixel_noise_injector

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def ClientUpdate(self, x, add_pixel_noise=False):
        """
        只做 model forward，回傳乾淨特徵。

        修正說明（對比原版）：
          - 移除內部 fp16 量化（不再固定雙重量化）
          - 移除內部 channel.transmit（通道邏輯由 main.py 統一執行）
          - 回傳的 A_k 保有計算圖，供 ClientBackprop 使用

        Returns
        -------
        A_k : Tensor（有計算圖，requires_grad 源自模型參數）
        """
        self.model.train()
        self.optimizer.zero_grad()

        x = x.to(self.device)

        if add_pixel_noise and self.pixel_noise_injector is not None:
            x = self.pixel_noise_injector.add_noise(x)

        A_k = self.model(x)
        self.last_A_k = A_k   # 保留計算圖，供 backward 使用

        return A_k             # 回傳乾淨特徵（未量化、未加噪）

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def ClientBackprop(self, dA_k):
        """
        接收從 main.py 傳入、已完成下列處理的梯度：
          (a) chain rule（SNR 縮放）
          (b) 反向通道模擬（若需要）
          (c) EMA 平滑

        執行 backward → clip(1.0) → optimizer.step()。

        修正說明（對比原版）：
          - 移除內部 channel.transmit（main.py 已做，不再雙重加噪）
          - clip_grad_norm_ 在 optimizer.step() 之前執行（原版在 step 之後無效）
        """
        if dA_k is None or self.last_A_k is None:
            return

        dA_k = dA_k.to(self.device)

        self.last_A_k.backward(dA_k)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.last_A_k = None