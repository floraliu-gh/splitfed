import torch
import torch.nn as nn


# ============================================================
#  AutoencoderDenoiser（獨立模組，不再內嵌於 ServerModel）
# ============================================================
class AutoencoderDenoiser(nn.Module):
    """
    殘差式 Autoencoder 去噪器。
    在瓶頸層強迫模型丟棄隨機雜訊，只保留核心語意；
    輸出為殘差補償值，加回原始特徵。

    修正說明（對比原版）：
      - 移除最後一層 BatchNorm2d：
        BN 會把殘差強制標準化（均值 0、方差 1），限制補償幅度。
        殘差連接的輸出層一般不加 BN，讓模型自由決定補償量級。
    """
    def __init__(self, in_channels=64, bottleneck_channels=16):
        super().__init__()

        # Encoder：壓縮到瓶頸維度
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )

        # Decoder：從瓶頸還原，輸出殘差補償值
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            # 最後一層：輸出殘差，不加啟動函數，也不加 BN（避免壓縮補償幅度）
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        residual = self.decoder(self.encoder(x))
        return x + residual   # 殘差連接：原始特徵 + 補償值


# ============================================================
#  ServerModel
# ============================================================
class ServerModel(nn.Module):
    """
    Server 端 Vision Transformer 分割模型。

    修正說明（對比原版）：
      - AutoencoderDenoiser 改由 use_denoiser 旗標控制，可在建立時或執行時關閉。
      - 傳統通訊 baseline 評估時，呼叫端將 use_denoiser 設為 False，
        確保比較公平（去噪器是語意通訊系統的一部分，不應套用到傳統通訊）。
      - 去噪器從「永遠執行」改為「可控制」，配合 main.py 的 ENABLE_LEARNED_DENOISER 旗標。

    Parameters
    ----------
    use_denoiser : bool
        True  → 前向傳播時先過 AutoencoderDenoiser（訓練 + 語意通訊評估）
        False → 直接進 ViT（傳統通訊 baseline 評估時使用）
    """
    def __init__(self, embed_dim=64, num_heads=4, depth=2,
                 num_classes=7, use_denoiser=True):
        super().__init__()
        self.use_denoiser = use_denoiser

        # 去噪器（條件建立，避免不需要時佔用參數）
        if use_denoiser:
            self.denoiser = AutoencoderDenoiser(
                in_channels=embed_dim, bottleneck_channels=16)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分割頭：16×16 → 64×64，輸出 num_classes 通道
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),  # 16 → 32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),         # 32 → 64
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1),                   # → num_classes
        )

    def forward(self, A_k):
        # ── 去噪（可控制）────────────────────────────────────
        if self.use_denoiser and hasattr(self, 'denoiser'):
            A_k = self.denoiser(A_k)

        # ── ViT：spatial token → Transformer → reshape ──────
        B, C, H, W = A_k.shape
        x = A_k.view(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
        x = self.blocks(x)
        spatial = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # ── 分割頭 ────────────────────────────────────────────
        return self.segmentation_head(spatial)