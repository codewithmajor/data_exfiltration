import torch
import torch.nn as nn


class HybridExfiltrationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4):
        super().__init__()
        assert hidden_dim % nhead == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})"

        # MLP Branch
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Transformer Branch
        self.token_dim = 16  # features per token
        self.embedding = nn.Linear(self.token_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion + Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # MLP branch
        mlp_out = self.mlp(x)

        # Transformer branch - split features into tokens
        B, F = x.shape
        # pad if not divisible by token_dim
        pad = (self.token_dim - F % self.token_dim) % self.token_dim
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        x_seq = x.view(B, -1, self.token_dim)  # (batch, seq_len, token_dim)

        trans_out = self.embedding(x_seq)       # (batch, seq_len, hidden_dim)
        trans_out = self.transformer(trans_out)
        trans_out = trans_out.mean(dim=1)       # (batch, hidden_dim)

        # Fuse & classify
        combined = torch.cat((mlp_out, trans_out), dim=1)
        return self.fc(combined).squeeze(-1)