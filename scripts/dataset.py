import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    """Single-flow dataset for baseline models."""
    def __init__(self, df, feature_cols):
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df["label"].values.astype(np.int64)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx]).float()
        return x, y

class FlowSequenceDataset(Dataset):
    """Each sample is a sequence of flows for one src_ip."""
    def __init__(self, df, feature_cols, seq_len=20):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        df = df.sort_values(by=["src_ip", "timestamp"])
        self.samples = []
        
        for src_ip, group in df.groupby("src_ip"):
            feats = group[feature_cols].values.astype(np.float32)
            labels = group["label"].values.astype(np.int64)
            
            if len(feats) < seq_len:
                continue
            
            for i in range(0, len(feats) - seq_len + 1):
                seq_x = feats[i:i+seq_len]
                seq_y = 1 if labels[i:i+seq_len].max() == 1 else 0
                self.samples.append((seq_x, seq_y))
        
        print(f"Total sequences: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_x, seq_y = self.samples[idx]
        x = torch.tensor(seq_x)  # (seq_len, feat_dim)
        y = torch.tensor(seq_y).float()
        return x, y
