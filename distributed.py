# Distributed Learning (Federated Split Learning)
# Pipeline:
#   Client -> ClientModel -> [feature activations -> noisy channel] -> ServerModel
#          -> ServerModel trains -> [gradients -> noisy channel] -> Client
#          -> ClientModel backward -> FedAvg aggregation
#
# 與 Centralized 的差別：raw data 永遠不離開 client；
# 傳輸 feature activations，不是 raw pixels；Server 看不到原始影像。

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

from client_model import ClientModel
from server_model import ServerModel
from client1 import Client
from server1 import MainServer
from robust_aggregation import fedserver
from channel_simulation import CommunicationChannel
from deepglobe import DeepGlobeDataset

CHANNEL_TYPE    = 'awgn'
RICIAN_K_FACTOR = 2.0
SNR_LIST        = [7,9,11,13]
CHANNEL_GAIN    = 1.0
BIT_ERROR_RATE  = 0.001

ENABLE_FORWARD_SCALING = True   # SNR scaling on received activations
ENABLE_BACKWARD_EMA    = True   # EMA smoothing on received gradients

NUM_CLIENTS        = 5
ROUNDS             = 40
LOCAL_EPOCHS       = 3
BATCH_SIZE         = 128
LR                 = 2e-3
SERVER_LR          = 1e-3
AGGREGATION_METHOD = 'fedavg'

DATASET_ROOT = 'C:/Users/winlab/Desktop/Flora/deepglobe'
IMG_SIZE     = (64, 64)
NUM_CLASSES  = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_hardware_analysis():
    print("\n=== Hardware & Performance Analysis (Split Learning) ===")

    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    c_part = ClientModel().to(device)
    s_part = ServerModel(num_classes=NUM_CLASSES).to(device)

    with torch.no_grad():
        mid = c_part(dummy_input)

    print("\n[1] Client Part (Feature Extractor):")
    summary(c_part, input_data=dummy_input, device=device)

    print("\n[2] Server Part (Decoder + Head):")
    summary(s_part, input_data=mid, device=device)

    # FLOPs
    def count_flops(model, inp):
        a = FlopCountAnalysis(model, inp)
        a.unsupported_ops_warnings(False)
        return a.total()

    flops_c = count_flops(c_part, dummy_input)
    flops_s = count_flops(s_part, mid)
    print("\n[3] Computational Complexity (per image):")
    print(f"  Client Part FLOPs : {flops_c / 1e6:.2f} MFLOPs")
    print(f"  Server Part FLOPs : {flops_s / 1e6:.2f} MFLOPs")
    print(f"  Total System FLOPs: {(flops_c + flops_s) / 1e6:.2f} MFLOPs")

    # Latency
    def measure_latency(model, inp, n=100):
        model.eval()
        with torch.no_grad():
            for _ in range(50):
                model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000

    print("\n[4] Latency (50 warmup + 100 test runs):")
    lat_c = measure_latency(c_part, dummy_input)
    lat_s = measure_latency(s_part, mid)
    print(f"  Client Part Latency : {lat_c:.3f} ms/image")
    print(f"  Server Part Latency : {lat_s:.3f} ms/image")
    print(f"  Total System Latency: {lat_c + lat_s:.3f} ms/image\n")

    del c_part, s_part


# ─────────────────────────────────────────
#  Activation Reconstruction Decoder (for visualization only)
# ─────────────────────────────────────────
class ActivationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(x)


def train_decoder(c_model, dataloader, steps=1000):
    decoder   = ActivationDecoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    c_model.eval()
    step = 0
    
    # 加上這行提示，準備印出進度
    print("    [Decoder Training Progress]:", end=" ", flush=True) 
    
    while step < steps:
        for x, _ in dataloader:
            x = x.to(device)
            with torch.no_grad():
                act = c_model(x)
            recon = decoder(act)
            loss  = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            
            # 每 100 步印出一次數字，讓你知道它還活著
            if step % 100 == 0:
                print(f"{step}..", end="", flush=True)
                
            if step >= steps:
                break
                
    print(" Done!") # 1000 步跑完印出 Done
    decoder.eval()
    return decoder


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))


def build_reconstruction_figure(c_model, decoder, channel, snr_db, x_single):
    c_model.eval()
    decoder.eval()

    mean_np = np.array([0.344, 0.380, 0.407])
    std_np  = np.array([0.203, 0.136, 0.114])

    def to_display(t):
        img = t.squeeze(0).cpu().permute(1, 2, 0).numpy()
        return np.clip(img * std_np + mean_np, 0, 1)

    with torch.no_grad():
        clean_act    = c_model(x_single)
        noisy_act    = channel.transmit(clean_act.clone())
        noisy_act    = torch.nan_to_num(noisy_act, nan=0.0)
        noisy_act    = torch.clamp(noisy_act, -10.0, 10.0)

        snr_linear   = 10 ** (snr_db / 10.0)
        snr_weight   = snr_linear / (snr_linear + 1.0)
        scaled_act   = noisy_act * snr_weight

        recon_clean  = decoder(clean_act)
        recon_noisy  = decoder(noisy_act)
        recon_scaled = decoder(scaled_act)

    orig_vis    = to_display(x_single)
    vis_clean   = to_display(recon_clean)
    vis_noisy   = to_display(recon_noisy)
    vis_scaled  = to_display(recon_scaled)

    p_clean  = psnr(vis_clean,  orig_vis)
    p_noisy  = psnr(vis_noisy,  orig_vis)
    p_scaled = psnr(vis_scaled, orig_vis)
    delta    = p_scaled - p_noisy

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Activation Reconstruction Comparison (SNR={snr_db:.1f}dB)",
        fontsize=13
    )
    panels = [
        (orig_vis,   "Original Input",                        "black"),
        (vis_clean,  f"Reconstructed: Clean\nPSNR={p_clean:.1f}dB",   "green"),
        (vis_noisy,  f"Reconstructed: Noisy\nPSNR={p_noisy:.1f}dB",   "red"),
        (vis_scaled, f"Reconstructed: SNR Scaled\n"
                     f"PSNR={p_scaled:.1f}dB  ({delta:+.1f}dB)",      "royalblue"),
    ]
    for ax, (img, title, color) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
#  Evaluate: clean inference (no channel noise)
# ─────────────────────────────────────────
def evaluate(c_model, s_model, loader):
    c_model.eval()
    s_model.eval()
    correct = total = 0
    intersection = torch.zeros(6, device=device)
    union        = torch.zeros(6, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            preds  = s_model(c_model(x)).argmax(dim=1)
            valid  = y != 6
            total   += valid.sum().item()
            correct += ((preds == y) & valid).sum().item()
            for cls in range(6):
                p = preds == cls
                t = y     == cls
                intersection[cls] += (p & t & valid).sum()
                union[cls]        += ((p | t) & valid).sum()
    miou    = (intersection / (union + 1e-6)).mean().item()
    pix_acc = correct / max(total, 1)
    return pix_acc, miou


# ─────────────────────────────────────────
#  Transmit model weights through noisy channel
# ─────────────────────────────────────────
def transmit_weights_through_channel(model, channel):
    params       = [p.data for p in model.parameters()]
    shapes       = [p.shape for p in params]
    flat         = torch.cat([p.reshape(-1) for p in params])

    flat_4d      = flat.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    flat_rx      = channel.transmit(flat_4d)
    flat_rx      = torch.nan_to_num(flat_rx, nan=0.0)
    flat_rx      = flat_rx.squeeze().cpu()

    received_sd  = {}
    offset       = 0
    for (name, _), shape in zip(model.named_parameters(), shapes):
        numel              = 1
        for s in shape:
            numel *= s
        received_sd[name]  = flat_rx[offset:offset + numel].reshape(shape)
        offset            += numel

    import copy
    rx_model = copy.deepcopy(model)
    rx_model.load_state_dict(received_sd, strict=False)
    return rx_model


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
if __name__ == '__main__':
    print(f"[Distributed / Federated Split Learning] Device: {device}")
    print(f"Channel: {CHANNEL_TYPE.upper()}  |  SNR sweep: {SNR_LIST}")
    print(f"Forward Scaling: {'ON' if ENABLE_FORWARD_SCALING else 'OFF'}  |  "
          f"Backward EMA: {'ON' if ENABLE_BACKWARD_EMA else 'OFF'}\n")

    # 執行硬體分析
    run_hardware_analysis()

    # Dataset
    full_dataset = DeepGlobeDataset(root_dir=DATASET_ROOT, split=None, img_size=IMG_SIZE)
    train_size   = int(0.8 * len(full_dataset))
    test_size    = len(full_dataset) - train_size
    train_full, test_dataset = random_split(full_dataset, [train_size, test_size])

    nk_list = [len(train_full) // NUM_CLIENTS] * NUM_CLIENTS
    nk_list[-1] += len(train_full) - sum(nk_list)
    client_subsets = random_split(train_full, nk_list)

    dataloaders = [
        DataLoader(sub, batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=0, pin_memory=True) 
        for sub in client_subsets
    ]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    _vis_x, _ = next(iter(test_loader))
    x_fixed   = _vis_x[0:1].to(device)

    all_results = {}

    for SNR_DB in SNR_LIST:
        print(f"\n{'='*60}")
        print(f"  Distributed  |  SNR = {SNR_DB} dB  |  {CHANNEL_TYPE.upper()}")
        print(f"{'='*60}")

        global_s_model = ServerModel(num_classes=NUM_CLASSES).to(device)
        main_server    = MainServer(global_s_model, device, lr=SERVER_LR, denoiser=None)
        scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(
            main_server.optimizer, T_max=ROUNDS, eta_min=1e-5)

        clients = []
        for i in range(NUM_CLIENTS):
            channel    = CommunicationChannel(
                snr_db=SNR_DB, channel_gain=CHANNEL_GAIN,
                bit_error_rate=BIT_ERROR_RATE,
                channel_type=CHANNEL_TYPE, rician_k=RICIAN_K_FACTOR,
                block_fading=True
            )
            c_model    = ClientModel().to(device)
            c_instance = Client(model=c_model, device=device,
                                lr=LR, channel=channel,
                                pixel_noise_injector=None)
            clients.append(c_instance)

        init_weights = clients[0].model.state_dict()
        for c in clients:
            c.model.load_state_dict(init_weights)

        client_grad_ema = {i: None for i in range(NUM_CLIENTS)}
        train_losses, test_accs, test_mious = [], [], []

        for r in range(ROUNDS):
            round_snrs = []
            for i in range(NUM_CLIENTS):
                new_snr = SNR_DB + np.random.randn() * 2
                clients[i].channel.snr_db = new_snr
                round_snrs.append(new_snr)

            print(f'\n--- Round {r+1}/{ROUNDS}  '
                  f'(SNR={SNR_DB}dB, Avg={np.mean(round_snrs):.2f}dB) ---')
            round_loss  = 0.0
            total_steps = 0

            for i in range(NUM_CLIENTS):
                for epoch in range(LOCAL_EPOCHS):
                    for batch_idx, (x, y) in enumerate(dataloaders[i]):
                        x, y = x.to(device), y.to(device)

                        A_k_clean = clients[i].ClientUpdate(x, add_pixel_noise=False)

                        A_k_received = clients[i].channel.transmit(A_k_clean)
                        A_k_received = torch.nan_to_num(A_k_received, nan=0.0)
                        A_k_received = torch.clamp(A_k_received, -10.0, 10.0)

                        current_snr = clients[i].channel.snr_db
                        snr_linear  = 10 ** (current_snr / 10.0)
                        snr_weight  = snr_linear / (snr_linear + 1.0)
                        fwd_weight  = snr_weight if ENABLE_FORWARD_SCALING else 1.0

                        A_k_denoised     = A_k_received * fwd_weight
                        A_k_server_input = A_k_denoised.detach().clone().requires_grad_(True)

                        dA_k, loss_val = main_server.ServerUpdate(
                            A_k_server_input, y, clear_grad=True)
                        dA_k = torch.clamp(dA_k, -0.5, 0.5)

                        dA_k_chain = dA_k * fwd_weight

                        dA_k_received = clients[i].channel.transmit(dA_k_chain)

                        ema_alpha = (1.0 - snr_weight) if ENABLE_BACKWARD_EMA else 0.0
                        B = dA_k_received.shape[0]

                        if client_grad_ema[i] is None:
                            client_grad_ema[i] = dA_k_received.clone()
                            dA_k_smoothed = client_grad_ema[i].clone()
                        else:
                            hist = client_grad_ema[i]
                            if hist.shape[0] != B:
                                hist = dA_k_received.clone()
                            smoothed = ema_alpha * hist + (1.0 - ema_alpha) * dA_k_received
                            client_grad_ema[i] = smoothed.clone()
                            dA_k_smoothed = smoothed.clone()

                        clients[i].ClientBackprop(dA_k_smoothed)
                        nn.utils.clip_grad_norm_(clients[i].model.parameters(), 5.0)
                        nn.utils.clip_grad_norm_(main_server.model.parameters(), 5.0)
                        main_server.step()

                        if i == 0 and batch_idx == 0:
                            g_norm = dA_k_smoothed.abs().mean().item()
                            w_grad = next(
                                (p.grad.abs().mean().item()
                                 for p in clients[i].model.parameters()
                                 if p.grad is not None), 0.0)
                            print(f"SNR={current_snr:.2f}dB | "
                                  f"fwd_W={fwd_weight:.4f} | EMA_a={ema_alpha:.4f} | "
                                  f"grad={g_norm:.2e} | w_grad={w_grad:.6f}")

                        round_loss  += loss_val
                        total_steps += 1

            avg_loss = round_loss / max(total_steps, 1)
            train_losses.append(avg_loss)

            upload_channel = CommunicationChannel(
                snr_db=SNR_DB, channel_gain=CHANNEL_GAIN,
                bit_error_rate=BIT_ERROR_RATE,
                channel_type=CHANNEL_TYPE, rician_k=RICIAN_K_FACTOR,
                block_fading=False
            )
            received_models = [
                transmit_weights_through_channel(c.model, upload_channel)
                for c in clients
            ]

            aggregated     = fedserver(received_models, nk_list, sum(nk_list),
                                       method=AGGREGATION_METHOD,
                                       trim_ratio=0.2, f=1)

            global_weights = aggregated.state_dict()
            for c in clients:
                agg_copy = type(c.model)().to(device)
                agg_copy.load_state_dict(global_weights)
                rx_global = transmit_weights_through_channel(agg_copy, upload_channel)
                c.model.load_state_dict(rx_global.state_dict())

            test_acc, miou = evaluate(clients[0].model, main_server.model, test_loader)
            test_accs.append(test_acc)
            test_mious.append(miou)
            scheduler.step()

            print(f"  loss={avg_loss:.4f}  test_acc={test_acc*100:.2f}%  mIoU={miou:.4f}")

        final_acc  = float(np.mean(test_accs[-5:]))
        final_miou = float(np.mean(test_mious[-5:]))
        final_loss = float(np.mean(train_losses[-5:]))
        all_results[SNR_DB] = {
            "acc": final_acc, "miou": final_miou, "loss": final_loss,
            "accs": test_accs, "mious": test_mious, "losses": train_losses,
        }
        print(f"\n[Result] SNR={SNR_DB}dB -> "
              f"Pixel Acc={final_acc*100:.2f}%  mIoU={final_miou:.4f}  Loss={final_loss:.4f}")

        import copy
        all_results[SNR_DB]["client_state"] = copy.deepcopy(clients[0].model.state_dict())

    # ── Summary table
    print(f"\n{'='*55}")
    print(f"{'SNR (dB)':>10} | {'Pixel Acc (%)':>14} | {'mIoU':>8} | {'Loss':>8}")
    print(f"{'-'*55}")
    for s in sorted(SNR_LIST):
        r = all_results[s]
        print(f"{s:>10} | {r['acc']*100:>14.2f} | {r['miou']:>8.4f} | {r['loss']:>8.4f}")
    print(f"{'='*55}")

    # ── Training curve plot
    colors = ['royalblue', 'seagreen', 'tomato', 'darkorange']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Distributed (Federated Split) Learning | {CHANNEL_TYPE.upper()} Channel\n"
        f"[Fwd Scaling={'ON' if ENABLE_FORWARD_SCALING else 'OFF'} | "
        f"Bwd EMA={'ON' if ENABLE_BACKWARD_EMA else 'OFF'}]",
        fontsize=12, fontweight='bold'
    )
    for ax, (title, ylabel, key, scale) in zip(axes, [
        ("mIoU vs Round",       "mIoU",         "mious",  1),
        ("Pixel Acc vs Round",  "Pixel Acc (%)", "accs",   100),
        ("Train Loss vs Round", "Loss",          "losses", 1),
    ]):
        for j, snr in enumerate(sorted(SNR_LIST)):
            data = [v * scale for v in all_results[snr][key]]
            ax.plot(range(1, len(data)+1), data, label=f"SNR={snr}dB",
                    color=colors[j], linewidth=1.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # ── Activation reconstruction
    print("\n[Visualization] Training decoders and collecting reconstruction data...")
    x_single = x_fixed[0:1]

    mean_np = np.array([0.344, 0.380, 0.407])
    std_np  = np.array([0.203, 0.136, 0.114])

    def to_display(t):
        img = t.squeeze(0).cpu().permute(1, 2, 0).numpy()
        return np.clip(img * std_np + mean_np, 0, 1)

    snr_vals = sorted(SNR_LIST)
    n_rows   = len(snr_vals)
    fig_all, axes_all = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes_all = axes_all[np.newaxis, :]

    fig_all.suptitle(
        f"Activation Reconstruction Comparison ({CHANNEL_TYPE.upper()} Channel)",
        fontsize=14, fontweight='bold'
    )

    col_labels = ["Original Input",
                  "Reconstructed: Clean",
                  "Reconstructed: Noisy",
                  "Reconstructed: SNR Scaled"]
    col_colors = ["black", "green", "red", "royalblue"]
    for col, (lbl, col_c) in enumerate(zip(col_labels, col_colors)):
        axes_all[0, col].set_title(lbl, fontsize=11,
                                   color=col_c, fontweight='bold')

    for row_idx, snr in enumerate(snr_vals):
        # 注意這裡：步驟被我們縮減到了 200 步
        decode_steps = 200
        print(f"  SNR={snr}dB: training decoder ({decode_steps} steps)...")

        c_model_vis = ClientModel().to(device)
        c_model_vis.load_state_dict(all_results[snr]["client_state"])
        c_model_vis.eval()

        decoder = train_decoder(c_model_vis, dataloaders[0], steps=decode_steps)

        vis_ch = CommunicationChannel(
            snr_db=snr, channel_gain=CHANNEL_GAIN,
            bit_error_rate=BIT_ERROR_RATE,
            channel_type=CHANNEL_TYPE, rician_k=RICIAN_K_FACTOR,
            block_fading=False
        )

        with torch.no_grad():
            clean_act   = c_model_vis(x_single)
            noisy_act   = vis_ch.transmit(clean_act.clone())
            noisy_act   = torch.nan_to_num(noisy_act, nan=0.0)
            noisy_act   = torch.clamp(noisy_act, -10.0, 10.0)
            snr_linear  = 10 ** (snr / 10.0)
            snr_weight  = snr_linear / (snr_linear + 1.0)
            scaled_act  = noisy_act * snr_weight

            recon_clean  = decoder(clean_act)
            recon_noisy  = decoder(noisy_act)
            recon_scaled = decoder(scaled_act)

        orig_vis    = to_display(x_single)
        vis_c       = to_display(recon_clean)
        vis_n       = to_display(recon_noisy)
        vis_s       = to_display(recon_scaled)

        p_clean  = psnr(vis_c, orig_vis)
        p_noisy  = psnr(vis_n, orig_vis)
        p_scaled = psnr(vis_s, orig_vis)
        delta    = p_scaled - p_noisy

        panels = [
            (orig_vis, f"SNR={snr}dB",                             "black"),
            (vis_c,    f"PSNR={p_clean:.1f}dB",                    "green"),
            (vis_n,    f"PSNR={p_noisy:.1f}dB",                    "red"),
            (vis_s,    f"PSNR={p_scaled:.1f}dB  ({delta:+.1f}dB)", "royalblue"),
        ]
        for col, (img, subtitle, color) in enumerate(panels):
            ax = axes_all[row_idx, col]
            ax.imshow(img)
            ax.set_ylabel(subtitle if col == 0 else "",
                          fontsize=10, color=color, fontweight='bold',
                          rotation=0, labelpad=60, va='center')
            if col != 0:
                ax.set_title(subtitle, fontsize=9, color=color)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)

        del decoder, c_model_vis

    plt.tight_layout()
    plt.show()
    print("\nDistributed Learning done.")