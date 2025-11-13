import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class SequenceBuilder:
    # (2) Consider a windowed Dataset that **indexes slices on-the-fly** instead of materializing np.array(xs); this reduces peak RAM and speeds up startup on large tick data.
    def __init__(self, seq_len: int = 120, horizon: int = 1, step: int = 1, zscore: bool = True):
        self.seq_len = seq_len
        self.horizon = horizon
        self.step = step
        self.zscore = zscore
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.zscore:
            self.mean_ = X.mean(axis=0, keepdims=True)
            self.std_ = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - self.mean_) / self.std_
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.zscore and self.mean_ is not None:
            X = (X - self.mean_) / self.std_
        return X

    def build(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(X)
        xs, ys = [], []
        last_start = T - (self.seq_len + self.horizon)
        for start in range(0, last_start + 1, self.step):
            end = start + self.seq_len
            target_idx = end + self.horizon - 1
            xs.append(X[start:end])
            ys.append(y[target_idx])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

class LOBSTERSequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        assert X_seq.ndim == 3, "X_seq must be (N, seq_len, num_features)"
        self.X = X_seq
        self.y = y_seq

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
