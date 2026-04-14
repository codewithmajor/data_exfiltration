import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    def __init__(self, df, feature_cols):
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]  # normalize col names

        if "label" not in df.columns:
            raise ValueError(f"'label' column not found. Columns: {list(df.columns)}")

        df[feature_cols] = df[feature_cols].fillna(0)

        self.features = df[feature_cols].values.astype(np.float32)
        self.labels   = df["label"].fillna(0).values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class FlowSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=20, attack_threshold=0.3):
        self.seq_len = seq_len
        self.attack_threshold = attack_threshold

        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]  # normalize col names

        # Validate required columns
        required_cols = ["src_ip", "timestamp", "label"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Found: {list(df.columns)}"
            )

        df[feature_cols] = df[feature_cols].fillna(0)
        df["label"]      = df["label"].fillna(0)

        df = df.sort_values(by=["src_ip", "timestamp"])
        self.samples = []

        for src_ip, group in df.groupby("src_ip"):
            feats  = group[feature_cols].values.astype(np.float32)
            labels = group["label"].values.astype(np.int64)

            # Handle short sequences — add as single padded sample
            if len(feats) < seq_len:
                pad_size = seq_len - len(feats)
                feats  = np.pad(feats,  ((0, pad_size), (0, 0)), mode='constant')
                labels = np.pad(labels, (0, pad_size),            mode='constant')
                attack_ratio = labels.mean()
                seq_y = 1 if attack_ratio > self.attack_threshold else 0
                self.samples.append((feats, seq_y))
                continue  # skip sliding window

            # Sliding window over full sequences
            for i in range(0, len(feats) - seq_len + 1):
                seq_x        = feats[i:i + seq_len]
                attack_ratio = labels[i:i + seq_len].mean()
                seq_y        = 1 if attack_ratio > self.attack_threshold else 0
                self.samples.append((seq_x, seq_y))

        print(f"Total sequences created: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_x, seq_y = self.samples[idx]
        return (
            torch.from_numpy(seq_x),
            torch.tensor(seq_y, dtype=torch.float32)
        )