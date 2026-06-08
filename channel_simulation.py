import torch
import numpy as np

class CommunicationChannel:
    """
    模擬通訊通道，支援多種 channel type：
    - 'awgn'     : 純高斯白雜訊
    - 'rayleigh' : Rayleigh fading（無直射路徑，如都市環境）
    - 'rician'   : Rician fading（有直射路徑，如 LEO 衛星通訊）
    - 'phase'    : 純相位旋轉雜訊（模擬相位偏移）
    - 'ideal'    : 無雜訊
    """
    def __init__(self, snr_db=20, channel_gain=1.0, bit_error_rate=0.0,
                 channel_type='awgn', rician_k=2.0, block_fading=True):
        self.snr_db       = snr_db
        self.h            = channel_gain
        self.ber          = bit_error_rate
        self.channel_type = channel_type
        self.rician_k     = rician_k
        self.block_fading = block_fading # 啟用 Block Fading

    def _rayleigh_gain(self, shape, device):
        # 模擬 Block Fading，同一張圖片(Batch 內的一個樣本)共用同一個衰落係數
        if self.block_fading and len(shape) == 4:
            fading_shape = (shape[0], 1, 1, 1) 
        else:
            fading_shape = shape
            
        real = torch.randn(fading_shape, device=device)
        imag = torch.randn(fading_shape, device=device)
        magnitude = torch.sqrt(real**2 + imag**2) / np.sqrt(2)
        return magnitude 

    def _rician_gain(self, shape, device):
        if self.block_fading and len(shape) == 4:
            fading_shape = (shape[0], 1, 1, 1)
        else:
            fading_shape = shape
            
        nu = np.sqrt(self.rician_k / (self.rician_k + 1))
        sigma = np.sqrt(1 / (2 * (self.rician_k + 1)))

        real = torch.randn(fading_shape, device=device) * sigma + nu
        imag = torch.randn(fading_shape, device=device) * sigma
        magnitude = torch.sqrt(real**2 + imag**2)
        return magnitude

    def _phase_noise(self, x):
        snr_linear = 10 ** (self.snr_db / 10.0)
        phase_std = 1.0 / np.sqrt(snr_linear)
        theta = torch.randn_like(x) * phase_std
        return x * torch.cos(theta)

    def add_awgn_noise(self, x):
        if self.snr_db > 100:
            return x
        signal_power = torch.mean(x ** 2)
        if signal_power == 0:
            return x
        snr_linear  = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std   = torch.sqrt(noise_power)
        noise       = torch.randn_like(x) * noise_std
        return x + noise

    def add_bit_errors(self, x, num_bits=8):
        if self.ber <= 0.0:
            return x
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return x
        scale           = (2 ** num_bits - 1) / (x_max - x_min)
        x_int           = ((x - x_min) * scale).long()
        final_flip_mask = torch.zeros_like(x_int)
        for b in range(num_bits):
            prob_matrix     = torch.rand_like(x, dtype=torch.float32)
            flip_decision   = (prob_matrix < self.ber).long()
            final_flip_mask = final_flip_mask | (flip_decision << b)
        x_int_corrupted = x_int ^ final_flip_mask
        return x_int_corrupted.float() / scale + x_min

    def transmit(self, x, add_awgn=True, add_bit_error=False):
        device = x.device

        if self.channel_type == 'ideal':
            return x
        elif self.channel_type == 'awgn':
            y = x * self.h
            if add_awgn: y = self.add_awgn_noise(y)
        elif self.channel_type == 'rayleigh':
            h_fading = self._rayleigh_gain(x.shape, device)
            y = x * h_fading
            if add_awgn: y = self.add_awgn_noise(y)
        elif self.channel_type == 'rician':
            h_fading = self._rician_gain(x.shape, device)
            y = x * h_fading
            if add_awgn: y = self.add_awgn_noise(y)
        elif self.channel_type == 'phase':
            y = self._phase_noise(x)
            if add_awgn: y = self.add_awgn_noise(y)
        else:
            raise ValueError(f"Unknown channel_type: '{self.channel_type}'")

        if add_bit_error and self.ber > 0:
            y = self.add_bit_errors(y)
        return y

class PixelNoiseInjector:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def add_noise(self, images):
        if self.noise_std <= 0:
            return images
        noise = torch.randn_like(images) * self.noise_std
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, -3.0, 3.0)
        return noisy_images


class Denoiser:
    def __init__(self, alpha=0.3, method='ema'):
        self.base_alpha  = alpha
        self.alpha       = alpha
        self.method      = method
        self.prev_signal = None
        self.prev_shape  = None

    def update_dynamic_alpha(self, current_snr, current_round, total_rounds):
        if self.method != 'dynamic':
            return self.alpha
        snr_factor      = max(0.0, min(1.0, current_snr / 40.0))
        progress_factor = current_round / max(1, total_rounds)
        dynamic_a       = self.base_alpha + (snr_factor * 0.4) + (progress_factor * 0.3)
        self.alpha      = max(0.1, min(1.0, dynamic_a))
        return self.alpha

    def denoise(self, noisy_signal):
        if self.method == 'none':
            return noisy_signal
        current_shape = noisy_signal.shape
        if self.prev_signal is None or current_shape != self.prev_shape:
            self.prev_signal = noisy_signal.detach().clone()
            self.prev_shape  = current_shape
            return noisy_signal
        denoised         = self.alpha * noisy_signal + (1 - self.alpha) * self.prev_signal
        self.prev_signal = denoised.detach().clone()
        return denoised

    def reset_client(self):
        self.prev_signal = None
        self.prev_shape  = None

    def reset(self):
        self.prev_signal = None
        self.prev_shape  = None


class MMSEDenoiser:
    def __init__(self, snr_db=15.0):
        self.snr_db     = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

    def update_snr(self, snr_db):
        self.snr_db     = snr_db
        self.snr_linear = 10 ** (snr_db / 10.0)

    def denoise(self, noisy_signal):
        received_power = torch.mean(noisy_signal ** 2)
        if received_power == 0:
            return noisy_signal
        wiener_coeff = self.snr_linear / (self.snr_linear + 1.0)
        return wiener_coeff * noisy_signal

    def reset_client(self): pass
    def reset(self): pass

class FeatureQuantizer:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmax = 2 ** num_bits - 1

    def quantize_dequantize(self, x):
        if self.num_bits >= 32: # float32 不量化
            return x
        x_min, x_max = x.min(), x.max()
        scale = self.qmax / (x_max - x_min + 1e-6)
        x_q = torch.round((x - x_min) * scale)
        x_deq = (x_q / scale) + x_min
        return x_deq
