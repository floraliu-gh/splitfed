import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
from pathlib import Path
import pandas as pd
import os
import time

from client_model import ClientModel
from server_model import ServerModel
from client1 import Client
from server1 import MainServer
from robust_aggregation import fedserver
from channel_simulation import CommunicationChannel, PixelNoiseInjector, FeatureQuantizer

SIMULATE_IDEAL_CHANNEL   = False
ENABLE_FORWARD_SCALING   = True
ENABLE_BACKWARD_EMA      = True
ENABLE_LEARNED_DENOISER  = True

CHANNEL_TYPE    = 'awgn'
RICIAN_K_FACTOR = 2.0

ENABLE_QUANTIZATION = False
QUANTIZATION_BITS   = 8

EXPERIMENT_NAME = (
    f"SNR_Sweep_{CHANNEL_TYPE.upper()}_"
    + ("WithDenoising" if ENABLE_FORWARD_SCALING else "NoDenoising")
    + ("_LearnedDenoise" if ENABLE_LEARNED_DENOISER else "")
)
SNR_LIST = [15]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K            = 5
rounds       = 1
local_epochs = 3
batch_size   = 128
lr           = 0.002
server_lr    = 0.0005

CHANNEL_GAIN   = 1.0
BIT_ERROR_RATE = 0.001

# 存在 main.py 旁邊，不管從哪裡執行都找得到
RESULT_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results.csv')
ROUND_LOG_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'round_log.csv')
SUMMARY_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary.csv')


# ============================================================
#  CSV 寫入工具
# ============================================================
def write_row(filepath, row_dict):
    """
    將一筆資料 append 進 CSV。
    - 檔案不存在時自動建立（含表頭）
    - 已存在時直接 append，不重寫整個檔案（速度快、不鎖檔）
    """
    file_exists = Path(filepath).exists()
    df = pd.DataFrame([row_dict])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)


def overwrite_round_row(filepath, row_dict):
    """
    每輪結束時「更新」同一個 SNR 的那一列進度。
    若該 (experiment, snr_db, round) 已存在則先刪除再寫入，
    確保同一輪不會重複累積。
    """
    if Path(filepath).exists():
        df = pd.read_csv(filepath)
        mask = (
            (df['experiment'] == row_dict['experiment']) &
            (df['snr_db']     == row_dict['snr_db'])     &
            (df['round']      == row_dict['round'])
        )
        df = df[~mask]          # 刪掉舊的那列
    else:
        df = pd.DataFrame()

    df_new = pd.DataFrame([row_dict])
    df_out = pd.concat([df, df_new], ignore_index=True)
    df_out.to_csv(filepath, index=False)


# ============================================================
#  結果摘要：終端機表格 + 精簡 CSV
# ============================================================
def print_and_save_summary(all_results, snr_vals, comm_metrics_by_snr,
                           experiment_name, channel_type):
    """
    跑完所有 SNR 後，在終端機印出清楚的摘要表，
    並另存 summary.csv（只保留最重要的欄位）。
    """
    SEP  = "=" * 100
    sep2 = "-" * 100

    print(f"\n{SEP}")
    print(f"  實驗結果摘要  |  {experiment_name}  |  Channel: {channel_type.upper()}")
    print(SEP)

    # ── 第一表：分割效果 ──────────────────────────────────────
    print(f"\n{'SNR (dB)':>10} │ {'Sem Acc (%)':>12} │ {'Sem mIoU':>10} │ {'Train Loss':>11} │ {'Trad Acc (%)':>13} │ {'Trad mIoU':>10}")
    print(sep2)
    for s in snr_vals:
        r = all_results[s]
        print(f"{s:>10} │ {r['acc']*100:>12.2f} │ {r['miou']:>10.4f} │ {r['loss']:>11.4f} │ {r['trad_acc']*100:>13.2f} │ {r['trad_miou']:>10.4f}")
    print(sep2)
    best_snr = max(snr_vals, key=lambda s: all_results[s]['miou'])
    print(f"  最佳 mIoU: SNR={best_snr}dB  →  {all_results[best_snr]['miou']:.4f}")

    # ── 第二表：通訊效率 ──────────────────────────────────────
    print(f"\n{'SNR (dB)':>10} │ {'通道容量(Mbps)':>15} │ {'傳統延遲(ms)':>14} │ {'語意延遲(ms)':>14} │ {'延遲縮短':>10} │ {'壓縮比':>8}")
    print(sep2)
    for s in snr_vals:
        c = comm_metrics_by_snr[s]
        print(f"{s:>10} │ {c['channel_capacity_mbps']:>15.3f} │ {c['trad_tx_latency_ms']:>14.2f} │ {c['sem_int8_tx_latency_ms']:>14.2f} │ {c['latency_reduction_x']:>9.2f}× │ {c['compression_ratio']:>7.2f}×")
    print(sep2)
    print(f"  資料量：傳統(fp32) = {comm_metrics_by_snr[snr_vals[0]]['img_payload_kbits']} Kbits  │  語意(int8) = {comm_metrics_by_snr[snr_vals[0]]['feat_int8_kbits']} Kbits  │  通道頻寬假設 = {comm_metrics_by_snr[snr_vals[0]]['channel_bw_mhz']} MHz")
    print(SEP)

    # ── 精簡 CSV ──────────────────────────────────────────────
    rows = []
    for s in snr_vals:
        r = all_results[s]
        c = comm_metrics_by_snr[s]
        rows.append({
            'SNR_dB':                   s,
            'Sem_Pixel_Acc_%':          round(r['acc'] * 100, 2),
            'Sem_mIoU':                 round(r['miou'], 4),
            'Train_Loss':               round(r['loss'], 4),
            'Trad_Pixel_Acc_%':         round(r['trad_acc'] * 100, 2),
            'Trad_mIoU':                round(r['trad_miou'], 4),
            'Channel_Capacity_Mbps':    c['channel_capacity_mbps'],
            'Trad_Tx_Latency_ms':       c['trad_tx_latency_ms'],
            'Sem_Int8_Tx_Latency_ms':   c['sem_int8_tx_latency_ms'],
            'Latency_Reduction_x':      c['latency_reduction_x'],
            'Compression_Ratio_x':      c['compression_ratio'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_FILE, index=False)
    print(f"\n  精簡摘要已存至: {SUMMARY_FILE}")
    print(f"  完整資料已存至: {RESULT_FILE}")


# ============================================================
#  評估函式
# ============================================================
def evaluate(c_model, s_model, loader):
    c_model.eval(); s_model.eval()
    total_pixels, correct_pixels = 0, 0
    intersection = torch.zeros(7, device=device)
    union        = torch.zeros(7, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, predicted = torch.max(s_model(c_model(x)), 1)

            valid_mask      = y != 6
            total_pixels   += valid_mask.sum().item()
            correct_pixels += ((predicted == y) & valid_mask).sum().item()

            for cls in range(6):
                p = predicted == cls; t = y == cls
                intersection[cls] += (p & t & valid_mask).sum()
                union[cls]        += ((p | t) & valid_mask).sum()

    miou      = (intersection[:6] / (union[:6] + 1e-6)).mean().item()
    pixel_acc = correct_pixels / max(total_pixels, 1)
    return pixel_acc, miou


def evaluate_traditional(c_model, s_model, loader, channel, device):
    c_model.eval(); s_model.eval()
    total_pixels, correct_pixels = 0, 0
    intersection = torch.zeros(7, device=device)
    union        = torch.zeros(7, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_rx = torch.clamp(torch.nan_to_num(channel.transmit(x), nan=0.0), -3.0, 3.0)
            _, predicted = torch.max(s_model(c_model(x_rx)), 1)

            valid_mask      = y != 6
            total_pixels   += valid_mask.sum().item()
            correct_pixels += ((predicted == y) & valid_mask).sum().item()

            for cls in range(6):
                p = predicted == cls; t = y == cls
                intersection[cls] += (p & t & valid_mask).sum()
                union[cls]        += ((p | t) & valid_mask).sum()

    miou      = (intersection[:6] / (union[:6] + 1e-6)).mean().item()
    pixel_acc = correct_pixels / max(total_pixels, 1)
    return pixel_acc, miou


# ============================================================
#  硬體分析
# ============================================================
def run_hardware_analysis(client_model, server_model):
    print("\n=== Hardware & Performance Analysis Report ===")
    dummy_c = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        dummy_s = client_model(dummy_c)

    print("\n[1] Client Model:"); summary(client_model, input_data=(dummy_c,), device=device)
    print("\n[2] Server Model:"); summary(server_model, input_data=(dummy_s,), device=device)

    fa = FlopCountAnalysis(client_model, dummy_c)
    fb = FlopCountAnalysis(server_model, dummy_s)
    fa.unsupported_ops_warnings(False); fb.unsupported_ops_warnings(False)
    fc, fs = fa.total(), fb.total()
    print(f"\n[3] FLOPs: Client={fc/1e6:.2f}M  Server={fs/1e6:.2f}M  Total={(fc+fs)/1e6:.2f}M")

    def latency(model, inp, n=100):
        model.eval()
        with torch.no_grad():
            for _ in range(n): model(inp)
            if device.type == 'cuda': torch.cuda.synchronize()
            t = time.time()
            for _ in range(n): model(inp)
            if device.type == 'cuda': torch.cuda.synchronize()
        return (time.time() - t) / n * 1000

    with torch.no_grad():
        lc = latency(client_model, dummy_c)
        ls = latency(server_model, dummy_s)
    print(f"[4] Latency: Client={lc:.3f}ms  Server={ls:.3f}ms  Total={lc+ls:.3f}ms\n")

    return {
        'client_flops_M':    round(fc / 1e6, 2),
        'server_flops_M':    round(fs / 1e6, 2),
        'total_flops_M':     round((fc + fs) / 1e6, 2),
        'client_latency_ms': round(lc, 3),
        'server_latency_ms': round(ls, 3),
        'total_latency_ms':  round(lc + ls, 3),
    }

# ============================================================
#  頻寬與傳輸延遲計算
# ============================================================
def compute_comm_metrics(snr_db, channel_bw_hz=1e6):
    """
    計算語意通訊 vs 傳統通訊的頻寬使用量與傳輸延遲。

    資料量：
      - 傳統通訊：原始影像 fp32  = 3×64×64×32 bits = 393,216 bits
      - 語意通訊：特徵向量 fp32  = 64×16×16×32 bits = 524,288 bits
      - 語意通訊：特徵向量 int8  = 64×16×16×8  bits = 131,072 bits
        （int8 量化後比原始影像小 3×，這才是語意通訊的壓縮優勢）

    通道速率（Shannon 公式）：
      C = B × log2(1 + SNR_linear)   [bits/s]

    傳輸延遲 = 資料量 / C   [ms]

    Parameters
    ----------
    snr_db        : float   訊雜比（dB）
    channel_bw_hz : float   通道頻寬（Hz），預設 1 MHz

    Returns
    -------
    dict  包含資料量、通道容量、延遲等指標
    """
    # 資料量（bits）
    img_bits      = 3 * 64 * 64 * 32       # 原始影像 fp32
    feat_fp32_bits = 64 * 16 * 16 * 32     # 特徵 fp32
    feat_int8_bits = 64 * 16 * 16 * 8      # 特徵 int8（量化後）

    # Shannon 通道容量
    snr_linear    = 10 ** (snr_db / 10.0)
    capacity_bps  = channel_bw_hz * np.log2(1 + snr_linear)
    capacity_mbps = capacity_bps / 1e6

    # 傳輸延遲（ms）
    trad_latency_ms    = (img_bits       / capacity_bps) * 1000
    sem_fp32_latency_ms = (feat_fp32_bits / capacity_bps) * 1000
    sem_int8_latency_ms = (feat_int8_bits / capacity_bps) * 1000

    # 壓縮比（vs 原始影像 fp32）
    compression_ratio  = img_bits / feat_int8_bits  # ≈ 3.0×

    print(f"\n[通訊指標] SNR={snr_db}dB  通道容量={capacity_mbps:.3f} Mbps")
    print(f"  資料量：傳統(fp32)={img_bits/1000:.1f}Kbits  "
          f"語意(fp32)={feat_fp32_bits/1000:.1f}Kbits  "
          f"語意(int8)={feat_int8_bits/1000:.1f}Kbits")
    print(f"  傳輸延遲：傳統={trad_latency_ms:.3f}ms  "
          f"語意(fp32)={sem_fp32_latency_ms:.3f}ms  "
          f"語意(int8)={sem_int8_latency_ms:.3f}ms")
    print(f"  壓縮比（int8 vs 傳統）: {compression_ratio:.2f}×")

    return {
        'channel_bw_mhz':         channel_bw_hz / 1e6,
        'channel_capacity_mbps':  round(capacity_mbps, 4),
        'img_payload_kbits':      round(img_bits / 1000, 1),
        'feat_fp32_kbits':        round(feat_fp32_bits / 1000, 1),
        'feat_int8_kbits':        round(feat_int8_bits / 1000, 1),
        'compression_ratio':      round(compression_ratio, 2),
        'trad_tx_latency_ms':     round(trad_latency_ms, 4),
        'sem_fp32_tx_latency_ms': round(sem_fp32_latency_ms, 4),
        'sem_int8_tx_latency_ms': round(sem_int8_latency_ms, 4),
        'latency_reduction_x':    round(trad_latency_ms / sem_int8_latency_ms, 2),
    }


# ============================================================
#  ActivationDecoder（特徵重建視覺化用）
# ============================================================
class ActivationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


# ============================================================
#  特徵重建（收集結果，最後統一畫圖）
# ============================================================
def visualize_reconstruction(client, x_single, device, snr_db_label, decoder):
    """
    用已訓練好的 decoder，對單一 SNR 輸出一張 1×4 比較圖：
      Original | Reconstructed: Clean | Reconstructed: Noisy | Reconstructed: SNR Scaled
    所有 SNR 傳入同一個 x_single 和同一個 decoder，確保可以直接比較。
    """
    mean_np = np.array([0.344, 0.380, 0.407])
    std_np  = np.array([0.203, 0.136, 0.114])

    def to_display(t):
        img = t.squeeze(0).cpu().permute(1, 2, 0).numpy()
        return np.clip(img * std_np + mean_np, 0, 1)

    def psnr(a, b):
        mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        return float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

    client.model.eval()
    decoder.eval()
    x_single = x_single.to(device)

    snr_lin = 10 ** (snr_db_label / 10.0)
    snr_w   = snr_lin / (snr_lin + 1.0)

    with torch.no_grad():
        clean_act    = client.model(x_single)
        noisy_act    = client.channel.transmit(clean_act)
        denoised_act = noisy_act * snr_w

        vis_clean    = to_display(decoder(clean_act))
        vis_noisy    = to_display(decoder(noisy_act))
        vis_denoised = to_display(decoder(denoised_act))

    orig_vis = to_display(x_single)
    psnr_c   = psnr(vis_clean,    orig_vis)
    psnr_n   = psnr(vis_noisy,    orig_vis)
    psnr_d   = psnr(vis_denoised, orig_vis)
    delta    = psnr_d - psnr_n

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f'Activation Reconstruction Comparison  (SNR={snr_db_label:.1f}dB, {CHANNEL_TYPE.upper()})',
        fontsize=13, fontweight='bold')

    panels = [
        (orig_vis,    'Original Input',            '',                                    'black'),
        (vis_clean,   'Reconstructed: Clean',      f'PSNR={psnr_c:.1f}dB',               'green'),
        (vis_noisy,   'Reconstructed: Noisy',      f'PSNR={psnr_n:.1f}dB',               'red'),
        (vis_denoised,'Reconstructed: SNR Scaled', f'PSNR={psnr_d:.1f}dB  (Δ{delta:+.1f}dB)', 'royalblue'),
    ]
    for ax, (img, title_top, title_bot, color) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(f'{title_top}\n{title_bot}' if title_bot else title_top,
                     fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5); sp.set_visible(True)

    plt.tight_layout()
    plt.show(block=False)
    print(f'[視覺化] SNR={snr_db_label}dB  ' 
          f'Clean={psnr_c:.1f}dB  Noisy={psnr_n:.1f}dB  ' 
          f'SNR-Scaled={psnr_d:.1f}dB (Δ{delta:+.1f}dB)')


# ============================================================
#  Main
# ============================================================
if __name__ == '__main__':
    print(f"[{EXPERIMENT_NAME}] device={device}")
    print(f"Channel={CHANNEL_TYPE.upper()}  "
          f"Quantize={'ON('+str(QUANTIZATION_BITS)+'bit)' if ENABLE_QUANTIZATION else 'OFF'}  "
          f"LearnedDenoiser={'ON' if ENABLE_LEARNED_DENOISER else 'OFF'}")
    print(f"\n{'='*60}")
    print(f"[存檔位置]")
    print(f"  每輪進度 → {ROUND_LOG_FILE}")
    print(f"  最終結果 → {RESULT_FILE}")
    print(f"{'='*60}\n")

    print("Preparing DeepGlobe dataset...")
    from deepglobe import DeepGlobeDataset

    full_dataset = DeepGlobeDataset(
        root_dir='C:/Users/winlab/Desktop/Flora/deepglobe',
        split=None, img_size=(64, 64))
    train_size = int(0.8 * len(full_dataset))
    train_full, test_dataset = random_split(
        full_dataset, [train_size, len(full_dataset) - train_size])

    nk_list       = [len(train_full) // K] * K
    nk_list[-1]  += len(train_full) - sum(nk_list)
    client_subsets = random_split(train_full, nk_list)

    dataloaders = [DataLoader(sub, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True) for sub in client_subsets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    print("\n[System] Hardware analysis...")
    _dc = ClientModel().to(device)
    _ds = ServerModel(num_classes=7, use_denoiser=ENABLE_LEARNED_DENOISER).to(device)
    hw_stats = run_hardware_analysis(_dc, _ds)
    del _dc, _ds

    exp_label = "Semantic (Denoised)" if ENABLE_FORWARD_SCALING else "Semantic (No Denoised)"
    if ENABLE_LEARNED_DENOISER:
        exp_label += " + AutoAE"

    all_results        = {}
    comm_metrics_by_snr = {}   # 每個 SNR 的通訊指標，最後統一輸出摘要
    quantizer   = FeatureQuantizer(num_bits=QUANTIZATION_BITS) if ENABLE_QUANTIZATION else None
    fixed_vis_batch, _ = next(iter(test_loader))
    fixed_vis_batch    = fixed_vis_batch[0:1].to(device)   # 固定同一張圖，所有 SNR 共用

    # 每個 SNR 各自訓練 decoder：
    # 每個 SNR 訓練出來的 client model 特徵空間不同，
    # 共用 decoder 會導致後續 SNR 的 Clean 重建也是亂碼。

    for SNR_DB in SNR_LIST:
        print(f"\n{'='*60}")
        print(f"  Training  SNR={SNR_DB}dB  Channel={CHANNEL_TYPE.upper()}")
        print(f"{'='*60}")

        channels = [
            CommunicationChannel(snr_db=SNR_DB, channel_gain=CHANNEL_GAIN,
                                 bit_error_rate=BIT_ERROR_RATE, channel_type=CHANNEL_TYPE,
                                 rician_k=RICIAN_K_FACTOR, block_fading=True)
            for _ in range(K)
        ]

        global_server_model = ServerModel(
            num_classes=7, use_denoiser=ENABLE_LEARNED_DENOISER).to(device)
        main_server = MainServer(global_server_model, device, lr=server_lr)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
                          main_server.optimizer, T_max=rounds, eta_min=1e-5)

        clients = []
        for i in range(K):
            clients.append(Client(model=ClientModel().to(device), device=device,
                                  lr=lr, channel=channels[i]))

        init_w = clients[0].model.state_dict()
        for c in clients:
            c.model.load_state_dict(init_w)

        client_grad_ema = {i: None for i in range(K)}
        train_losses, test_accs, test_mious = [], [], []

        # ── Training Loop ──────────────────────────────────────
        for r in range(rounds):
            round_snrs = []
            for i in range(K):
                ns = 100.0 if SIMULATE_IDEAL_CHANNEL else SNR_DB + np.random.randn() * 2
                clients[i].channel.snr_db = ns
                round_snrs.append(ns)

            print(f'\n--- Round {r+1}/{rounds}  SNR={SNR_DB}dB  Avg={np.mean(round_snrs):.2f}dB ---')
            round_loss, total_steps = 0.0, 0

            for i in range(K):
                for epoch in range(local_epochs):
                    for batch_idx, (x, y) in enumerate(dataloaders[i]):
                        x, y = x.to(device), y.to(device)

                        A_k_raw    = clients[i].ClientUpdate(x, add_pixel_noise=False)
                        A_k_for_tx = A_k_raw.detach()

                        if ENABLE_QUANTIZATION:
                            A_k_for_tx = quantizer.quantize_dequantize(A_k_for_tx)

                        A_k_received = clients[i].channel.transmit(A_k_for_tx)
                        A_k_received = torch.nan_to_num(A_k_received, nan=0.0)
                        A_k_received = torch.clamp(A_k_received, -10.0, 10.0)

                        cur_snr = clients[i].channel.snr_db
                        snr_lin = 10 ** (cur_snr / 10.0)
                        base_w  = snr_lin / (snr_lin + 1.0)
                        fwd_w   = base_w if ENABLE_FORWARD_SCALING else 1.0
                        A_k_server_input = (A_k_received * fwd_w).clone().requires_grad_(True)

                        dA_k, loss_val = main_server.ServerUpdate(
                            A_k_server_input, y, clear_grad=True)
                        dA_k = torch.clamp(dA_k, -0.5, 0.5)

                        dA_k_chain = dA_k * fwd_w
                        dA_k_from_server = (clients[i].channel.transmit(dA_k_chain)
                                            if not SIMULATE_IDEAL_CHANNEL else dA_k_chain)

                        ema_alpha = (1.0 - base_w) if ENABLE_BACKWARD_EMA else 0.0
                        if (client_grad_ema[i] is None or
                                client_grad_ema[i].shape != dA_k_from_server.shape):
                            client_grad_ema[i] = dA_k_from_server.clone()
                            dA_k_smoothed      = dA_k_from_server.clone()
                        else:
                            sm = (ema_alpha * client_grad_ema[i]
                                  + (1.0 - ema_alpha) * dA_k_from_server)
                            client_grad_ema[i] = sm.clone()
                            dA_k_smoothed      = sm.clone()

                        clients[i].ClientBackprop(dA_k_smoothed)
                        main_server.step()

                        if i == 0 and batch_idx == 0:
                            gnorm = dA_k_smoothed.abs().mean().item()
                            wgrad = next((p.grad.abs().mean().item()
                                          for p in clients[i].model.parameters()
                                          if p.grad is not None), 0)
                            print(f"  C0 SNR={cur_snr:.2f}dB  fwd_w={fwd_w:.4f}  "
                                  f"ema_α={ema_alpha:.4f}  |dA|={gnorm:.2e}  |∇w|={wgrad:.6f}")

                        round_loss  += loss_val
                        total_steps += 1

            train_losses.append(round_loss / max(total_steps, 1))

            agg = fedserver([c.model for c in clients], nk_list, sum(nk_list),
                            method='fedavg', trim_ratio=0.2, f=1)
            gw  = agg.state_dict()
            for c in clients:
                c.model.load_state_dict(gw)

            acc, miou = evaluate(clients[0].model, main_server.model, test_loader)
            test_accs.append(acc); test_mious.append(miou)
            scheduler.step()

            # ── 每輪結束立刻寫入進度（可隨時開 CSV 查看）────────
            overwrite_round_row(ROUND_LOG_FILE, {
                'experiment':    EXPERIMENT_NAME,
                'channel':       CHANNEL_TYPE,
                'snr_db':        SNR_DB,
                'round':         r + 1,
                'total_rounds':  rounds,
                'pixel_acc_%':   round(acc * 100, 2),
                'miou':          round(miou, 4),
                'train_loss':    round(train_losses[-1], 4),
                'status':        'done' if r + 1 == rounds else 'running',
            })
            print(f"  [進度已更新] round_log.csv  Round {r+1}/{rounds}  "
                  f"Acc={acc*100:.2f}%  mIoU={miou:.4f}")

        # ── 最終成績（最後 5 輪平均）──────────────────────────
        final_acc  = float(np.mean(test_accs[-5:]))
        final_miou = float(np.mean(test_mious[-5:]))
        final_loss = float(np.mean(train_losses[-5:]))

        # ── 傳統通訊 Baseline ───────────────────────────────────
        print(f"\n[Traditional Baseline] SNR={SNR_DB}dB ...")
        trad_ch = CommunicationChannel(
            snr_db=SNR_DB, channel_gain=CHANNEL_GAIN, bit_error_rate=BIT_ERROR_RATE,
            channel_type=CHANNEL_TYPE, rician_k=RICIAN_K_FACTOR, block_fading=False)

        main_server.model.use_denoiser = False
        trad_acc, trad_miou = evaluate_traditional(
            clients[0].model, main_server.model, test_loader, trad_ch, device)
        main_server.model.use_denoiser = ENABLE_LEARNED_DENOISER

        all_results[SNR_DB] = dict(
            acc=final_acc, miou=final_miou, loss=final_loss,
            trad_acc=trad_acc, trad_miou=trad_miou)

        print(f"[Semantic   ] Acc={final_acc*100:.2f}%  mIoU={final_miou:.4f}  Loss={final_loss:.4f}")
        print(f"[Traditional] Acc={trad_acc*100:.2f}%  mIoU={trad_miou:.4f}")

        # ── 通訊指標 ───────────────────────────────────────────
        comm = compute_comm_metrics(SNR_DB, channel_bw_hz=1e6)
        comm_metrics_by_snr[SNR_DB] = comm

        # ── SNR 結束後寫入最終結果 ─────────────────────────────
        per_round_acc  = ','.join(f'{v*100:.2f}' for v in test_accs)
        per_round_miou = ','.join(f'{v:.4f}'     for v in test_mious)
        per_round_loss = ','.join(f'{v:.4f}'     for v in train_losses)

        write_row(RESULT_FILE, {
            'experiment':        EXPERIMENT_NAME,
            # 通訊指標
            'channel_bw_mhz':            comm['channel_bw_mhz'],
            'channel_capacity_mbps':     comm['channel_capacity_mbps'],
            'img_payload_kbits':         comm['img_payload_kbits'],
            'feat_int8_kbits':           comm['feat_int8_kbits'],
            'compression_ratio':         comm['compression_ratio'],
            'trad_tx_latency_ms':        comm['trad_tx_latency_ms'],
            'sem_int8_tx_latency_ms':    comm['sem_int8_tx_latency_ms'],
            'latency_reduction_x':       round(comm['trad_tx_latency_ms'] / comm['sem_int8_tx_latency_ms'], 2),
            'channel':           CHANNEL_TYPE,
            'snr_db':            SNR_DB,
            'K':                 K,
            'rounds':            rounds,
            'local_epochs':      local_epochs,
            'batch_size':        batch_size,
            'lr':                lr,
            'server_lr':         server_lr,
            'forward_scaling':   ENABLE_FORWARD_SCALING,
            'backward_ema':      ENABLE_BACKWARD_EMA,
            'learned_denoiser':  ENABLE_LEARNED_DENOISER,
            'quantization':      ENABLE_QUANTIZATION,
            'rician_k':          RICIAN_K_FACTOR,
            'channel_gain':      CHANNEL_GAIN,
            'bit_error_rate':    BIT_ERROR_RATE,
            'sem_pixel_acc_%':   round(final_acc  * 100, 2),
            'sem_miou':          round(final_miou, 4),
            'train_loss':        round(final_loss, 4),
            'trad_pixel_acc_%':  round(trad_acc   * 100, 2),
            'trad_miou':         round(trad_miou, 4),
            'client_flops_M':    hw_stats['client_flops_M'],
            'server_flops_M':    hw_stats['server_flops_M'],
            'total_flops_M':     hw_stats['total_flops_M'],
            'client_latency_ms': hw_stats['client_latency_ms'],
            'server_latency_ms': hw_stats['server_latency_ms'],
            'total_latency_ms':  hw_stats['total_latency_ms'],
            'per_round_acc_%':   per_round_acc,
            'per_round_miou':    per_round_miou,
            'per_round_loss':    per_round_loss,
        })
        print(f"[最終結果寫入] {RESULT_FILE}")

        # ── 視覺化：每個 SNR 各自訓練 decoder，確保重建品質正確 ──
        # 每個 SNR 的 client model 特徵空間不同，不能共用 decoder
        clients[0].channel.snr_db = SNR_DB   # 重置為標準值
        print(f"\n[Decoder Training] SNR={SNR_DB}dB  1000 steps...")
        _decoder = ActivationDecoder().to(device)
        _opt     = torch.optim.Adam(_decoder.parameters(), lr=1e-3)
        _crit    = nn.MSELoss()
        clients[0].model.eval()
        for _ in range(1000):
            _decoder.train()
            with torch.no_grad():
                _act = clients[0].model(fixed_vis_batch)
            _loss = _crit(_decoder(_act), fixed_vis_batch)
            _opt.zero_grad(); _loss.backward(); _opt.step()
        visualize_reconstruction(
            clients[0], fixed_vis_batch, device, SNR_DB, _decoder)

    # ============================================================
    #  最終報表（格式化表格 + 精簡 CSV）
    # ============================================================
    snr_vals = sorted(SNR_LIST)
    print_and_save_summary(all_results, snr_vals, comm_metrics_by_snr,
                           EXPERIMENT_NAME, CHANNEL_TYPE)

    # ── 三指標對比圖 ──────────────────────────────────────────
    sem_miou = [all_results[s]['miou']          for s in snr_vals]
    sem_acc  = [all_results[s]['acc'] * 100      for s in snr_vals]
    sem_loss = [all_results[s]['loss']           for s in snr_vals]
    t_miou   = [all_results[s]['trad_miou']      for s in snr_vals]
    t_acc    = [all_results[s]['trad_acc'] * 100  for s in snr_vals]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Semantic vs Traditional  ({CHANNEL_TYPE.upper()})\n"
        f"Learned Denoiser: {'ON' if ENABLE_LEARNED_DENOISER else 'OFF'}",
        fontsize=13, fontweight='bold')

    cfgs = [
        dict(ax=axes[0], title='mIoU vs SNR',           ylabel='mIoU',
             sd=sem_miou, td=t_miou, sc='royalblue', tc='tomato',     sm='o', tm='s'),
        dict(ax=axes[1], title='Pixel Accuracy vs SNR', ylabel='Pixel Acc (%)',
             sd=sem_acc,  td=t_acc,  sc='seagreen',  tc='darkorange', sm='^', tm='D'),
        dict(ax=axes[2], title='Training Loss vs SNR\n(Semantic Only)', ylabel='Loss',
             sd=sem_loss, td=None,   sc='purple',    tc=None,         sm='v', tm=None),
    ]
    for cfg in cfgs:
        ax = cfg['ax']
        ax.plot(snr_vals, cfg['sd'], marker=cfg['sm'], color=cfg['sc'],
                linewidth=2.5, markersize=8, label=exp_label)
        if cfg['td']:
            ax.plot(snr_vals, cfg['td'], marker=cfg['tm'], color=cfg['tc'],
                    linewidth=2, linestyle='--', markersize=7, alpha=0.85,
                    label='Traditional (Pixel Tx)')
        ax.set_title(cfg['title'], fontsize=12); ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel(cfg['ylabel'], fontsize=11); ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6); ax.set_xticks(snr_vals)

    plt.tight_layout()
    plt.show(block=False)



    print(f"\n[{EXPERIMENT_NAME}] Finished！")
    plt.show()