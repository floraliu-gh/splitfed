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
from channel_simulation import CommunicationChannel
from deepglobe import DeepGlobeDataset

CHANNEL_TYPE    = 'awgn'
RICIAN_K_FACTOR = 2.0
SNR_LIST        = [0,5,10,15]
CHANNEL_GAIN    = 1.0
BIT_ERROR_RATE  = 0.001

NUM_CLIENTS  = 5
ROUNDS       = 70
LOCAL_EPOCHS = 3
BATCH_SIZE   = 128
LR           = 2e-3

DATASET_ROOT = 'C:/Users/winlab/Desktop/Flora/deepglobe'
IMG_SIZE     = (64, 64)
NUM_CLASSES  = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FullModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.client_part = ClientModel()
        self.server_part = ServerModel(num_classes=num_classes)

    def forward(self, x):
        return self.server_part(self.client_part(x))

class CentralizedServer:
    def __init__(self, lr=LR):
        self.model     = FullModel(num_classes=NUM_CLASSES).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=6)

    def train_on_received(self, x_noisy, y):
        # Server trains on the noisy pixels received from the channel
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(x_noisy), y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return loss.item()

    def get_weights(self):
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def evaluate(self, loader, channel=None):
        # channel=None -> no noise added during evaluation
        self.model.eval()
        correct = total = 0
        intersection = torch.zeros(6, device=device)
        union        = torch.zeros(6, device=device)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if channel is not None:
                    x = channel.transmit(x)
                    x = torch.nan_to_num(x, nan=0.0)
                    x = torch.clamp(x, -3.0, 3.0)
                preds = self.model(x).argmax(dim=1)
                valid = y != 6
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

class CentralizedClient:
    def __init__(self, client_id, dataset, channel):
        self.client_id = client_id
        self.loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=4, pin_memory=True)
        self.channel   = channel

    def transmit_pixels(self, x):
        # Raw image pixels (3 x 64 x 64) -> noisy channel -> server
        # This is the key difference from distributed: raw data leaves the client
        x_rx = self.channel.transmit(x)
        x_rx = torch.nan_to_num(x_rx, nan=0.0)
        x_rx = torch.clamp(x_rx, -3.0, 3.0)
        return x_rx

DEEPGLOBE_COLORS = np.array([
    [0,   255, 255],   # 0 urban
    [255, 255,   0],   # 1 agriculture
    [255,   0, 255],   # 2 rangeland
    [  0, 255,   0],   # 3 forest
    [  0,   0, 255],   # 4 water
    [255, 255, 255],   # 5 barren
    [  0,   0,   0],   # 6 unknown
], dtype=np.uint8)

CLASS_NAMES = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']

MEAN = np.array([0.344, 0.380, 0.407])
STD  = np.array([0.203, 0.136, 0.114])


def mask_to_rgb(mask_np):
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(7):
        rgb[mask_np == cls] = DEEPGLOBE_COLORS[cls]
    return rgb


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))


def collect_pixel_recon(model, x_single, channel):
    # 只收集原始圖片與加上通道雜訊後的圖片
    model.eval()
    with torch.no_grad():
        # 模擬原始像素經過雜訊通道
        x_noisy = channel.transmit(x_single)
        x_noisy = torch.nan_to_num(x_noisy, nan=0.0)
        x_noisy = torch.clamp(x_noisy, -3.0, 3.0)

    def to_img(t):
        img = t.cpu().permute(1, 2, 0).numpy()
        return np.clip(img * STD + MEAN, 0, 1)

    orig_vis  = to_img(x_single[0])
    noisy_vis = to_img(x_noisy[0])
    p_noisy   = psnr(noisy_vis, orig_vis)

    return orig_vis, noisy_vis, p_noisy


def plot_all_pixel_recon(vis_data, snr_list, channel_type):
    snr_vals = sorted(snr_list)
    n_rows   = len(snr_vals)

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Centralized Learning Pixel Transmission ({channel_type.upper()} Channel)\n"
        f"Columns: Original Input | Received Noisy Pixels",
        fontsize=13, fontweight='bold'
    )

    col_labels = ["Original Input", "Received Noisy Pixels"]
    col_colors = ["black",          "red"]

    for col, (lbl, col_c) in enumerate(zip(col_labels, col_colors)):
        axes[0, col].set_title(lbl, fontsize=11, color=col_c, fontweight='bold')

    for row_idx, snr in enumerate(snr_vals):
        orig_vis, noisy_vis, p_noisy = vis_data[snr]

        panels = [
            (orig_vis,  f"SNR={snr}dB",           "black"),
            (noisy_vis, f"PSNR={p_noisy:.1f}dB", "red"),
        ]
        for col, (img, subtitle, color) in enumerate(panels):
            ax = axes[row_idx, col]
            ax.imshow(img)
            if col == 0:
                ax.set_ylabel(f"SNR={snr}dB", fontsize=11,
                              color="black", fontweight='bold',
                              rotation=0, labelpad=55, va='center')
            if subtitle:
                ax.set_title(subtitle, fontsize=9, color=color)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  
    return fig

def run_hardware_analysis():
    print("\n=== Hardware & Performance Analysis (Centralized FullModel) ===")

    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    c_part = ClientModel().to(device)
    s_part = ServerModel(num_classes=NUM_CLASSES).to(device)
    full   = FullModel(num_classes=NUM_CLASSES).to(device)

    with torch.no_grad():
        mid = c_part(dummy_input)

    print("\n[1] Client Part (feature extractor):")
    summary(c_part, input_data=dummy_input, device=device)

    print("\n[2] Server Part (decoder + head):")
    summary(s_part, input_data=mid, device=device)

    print("\n[3] Full Model (combined):")
    summary(full, input_data=dummy_input, device=device)

    # FLOPs
    def count_flops(model, inp):
        a = FlopCountAnalysis(model, inp)
        a.unsupported_ops_warnings(False)
        return a.total()

    flops_c = count_flops(c_part, dummy_input)
    flops_s = count_flops(s_part, mid)
    print("\n[4] Computational Complexity (per image):")
    print(f"  Client Part FLOPs : {flops_c / 1e6:.2f} MFLOPs")
    print(f"  Server Part FLOPs : {flops_s / 1e6:.2f} MFLOPs")
    print(f"  Total System FLOPs: {(flops_c + flops_s) / 1e6:.2f} MFLOPs")

    # Latency
    def measure_latency(model, inp, n=100):
        model.eval()
        with torch.no_grad():
            for _ in range(100):
                model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000

    print("\n[5] Latency (100 warmup + 100 test runs):")
    lat_c    = measure_latency(c_part, dummy_input)
    lat_s    = measure_latency(s_part, mid)
    lat_full = measure_latency(full, dummy_input)
    print(f"  Client Part Latency : {lat_c:.3f} ms/image")
    print(f"  Server Part Latency : {lat_s:.3f} ms/image")
    print(f"  Full Model Latency  : {lat_full:.3f} ms/image")
    print(f"  (C+S split sum)     : {lat_c + lat_s:.3f} ms/image\n")

    del c_part, s_part, full

if __name__ == '__main__':
    print(f"[Centralized Learning] Device: {device}")
    print(f"Channel: {CHANNEL_TYPE.upper()}  |  SNR sweep: {SNR_LIST}\n")

    run_hardware_analysis()

    # Dataset
    full_dataset = DeepGlobeDataset(root_dir=DATASET_ROOT, split=None, img_size=IMG_SIZE)
    train_size   = int(0.8 * len(full_dataset))
    test_size    = len(full_dataset) - train_size
    train_full, test_dataset = random_split(full_dataset, [train_size, test_size])

    nk_list = [len(train_full) // NUM_CLIENTS] * NUM_CLIENTS
    nk_list[-1] += len(train_full) - sum(nk_list)
    client_subsets = random_split(train_full, nk_list)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Fixed visualization samples — grabbed once, reused across all SNR runs
    VIS_N = 4
    _vis_x, _vis_y = next(iter(test_loader))
    x_fixed = _vis_x[:VIS_N].to(device)   # always same 4 images
    y_fixed = _vis_y[:VIS_N]               # always same 4 GT masks

    all_results = {}
    vis_data    = {}   # stores per-SNR pixel reconstruction data for final plot

    for SNR_DB in SNR_LIST:
        print(f"\n{'='*60}")
        print(f"  Centralized  |  SNR = {SNR_DB} dB  |  {CHANNEL_TYPE.upper()}")
        print(f"{'='*60}")

        server    = CentralizedServer(lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            server.optimizer, T_max=ROUNDS, eta_min=1e-5)

        clients = [
            CentralizedClient(
                client_id=i,
                dataset=client_subsets[i],
                channel=CommunicationChannel(
                    snr_db=SNR_DB, channel_gain=CHANNEL_GAIN,
                    bit_error_rate=BIT_ERROR_RATE,
                    channel_type=CHANNEL_TYPE, rician_k=RICIAN_K_FACTOR,
                    block_fading=True
                )
            )
            for i in range(NUM_CLIENTS)
        ]

        train_losses, test_accs, test_mious = [], [], []

        for r in range(ROUNDS):
            print(f'\n--- Round {r+1}/{ROUNDS} ---')
            round_loss  = 0.0
            total_steps = 0

            for i, client in enumerate(clients):
                # dynamic SNR perturbation (same as original experiment)
                client.channel.snr_db = SNR_DB + np.random.randn() * 2

                for _ in range(LOCAL_EPOCHS):
                    for x, y in client.loader:
                        x, y = x.to(device), y.to(device)

                        # Step 1: client transmits raw pixels through noisy channel
                        x_received = client.transmit_pixels(x)

                        # Step 2: server trains full model on received noisy image
                        loss_val = server.train_on_received(x_received, y)

                        if i == 0 and total_steps == 0:
                            print(f"  [Client 0] SNR={client.channel.snr_db:.2f}dB  "
                                  f"loss={loss_val:.4f}")

                        round_loss  += loss_val
                        total_steps += 1

            avg_loss = round_loss / max(total_steps, 1)
            train_losses.append(avg_loss)

            # Evaluate with channel noise on test images
            eval_ch = CommunicationChannel(
                snr_db=SNR_DB, channel_gain=CHANNEL_GAIN,
                bit_error_rate=BIT_ERROR_RATE,
                channel_type=CHANNEL_TYPE, rician_k=RICIAN_K_FACTOR,
                block_fading=False  # fixed channel for stable eval
            )
            test_acc, miou = server.evaluate(test_loader, channel=eval_ch)
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

        # Collect visualization data (no plotting yet)
        vis_ch = CommunicationChannel(
            snr_db=SNR_DB, channel_gain=CHANNEL_GAIN,
            bit_error_rate=BIT_ERROR_RATE,
            channel_type=CHANNEL_TYPE, rician_k=RICIAN_K_FACTOR,
            block_fading=False
        )
        vis_data[SNR_DB] = collect_pixel_recon(server.model, x_fixed, vis_ch)

    # Summary table
    print(f"\n{'='*55}")
    print(f"{'SNR (dB)':>10} | {'Pixel Acc (%)':>14} | {'mIoU':>8} | {'Loss':>8}")
    print(f"{'-'*55}")
    for s in sorted(SNR_LIST):
        r = all_results[s]
        print(f"{s:>10} | {r['acc']*100:>14.2f} | {r['miou']:>8.4f} | {r['loss']:>8.4f}")
    print(f"{'='*55}")

    # Plot: training curves
    colors = ['royalblue', 'seagreen', 'tomato', 'darkorange']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Centralized Learning | {CHANNEL_TYPE.upper()} Channel",
                 fontsize=13, fontweight='bold')
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

    # Plot: pixel reconstruction comparison (all SNRs in one figure)
    print("\n[Visualization] Building pixel reconstruction figure...")
    plot_all_pixel_recon(vis_data, SNR_LIST, CHANNEL_TYPE)

    plt.show()   # show all pending figures at once; close windows to exit
    print("\nCentralized Learning done.")
