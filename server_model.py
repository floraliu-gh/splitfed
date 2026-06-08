import torch
import torch.nn as nn

class AutoencoderDenoiser(nn.Module):
    def __init__(self, in_channels=64, bottleneck_channels=16):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )

        # Decoder
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
        
class ServerModel(nn.Module):
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
        # 去噪
        if self.use_denoiser and hasattr(self, 'denoiser'):
            A_k = self.denoiser(A_k)

        # ViT：spatial token → Transformer → reshape 
        B, C, H, W = A_k.shape
        x = A_k.view(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
        x = self.blocks(x)
        spatial = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return self.segmentation_head(spatial)
