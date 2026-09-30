import torch
from torch.utils.data import Dataset
import numpy as np

class EEWRobustDataset(Dataset):
    def __init__(self, waveforms, labels, sample_rate=100, causal_truncation=True, time_shift=True):
        self.waveforms = waveforms
        self.labels = labels
        self.sample_rate = sample_rate
        self.causal_truncation = causal_truncation
        self.time_shift = time_shift

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        wave = self.waveforms[idx].copy()
        label = self.labels[idx]

        # 1. 時域平移防禦
        if self.time_shift and np.random.rand() > 0.5:
            shift = np.random.randint(-100, 100)
            wave = np.roll(wave, shift)

        # 2. 因果截斷防禦（模擬初動 0.5~2 秒極限情境）
        if self.causal_truncation and label == 1 and np.random.rand() > 0.3:
            cutoff = np.random.randint(int(0.5 * self.sample_rate), int(2.0 * self.sample_rate))
            p_arrival_approx = 500
            if p_arrival_approx + cutoff < len(wave):
                wave[p_arrival_approx + cutoff:] = 0.0

        # Standardize
        std = np.std(wave)
        wave = (wave - np.mean(wave)) / std if std > 1e-6 else np.zeros_like(wave)

        return torch.tensor(wave, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32).unsqueeze(0)
