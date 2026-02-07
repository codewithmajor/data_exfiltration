import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = self.embedding(x)  # (batch_size, 1, hidden_dim)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer_encoder(x)  # (batch_size, 1, hidden_dim)
        return x

class ExfiltrationTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=3, num_classes=2, dropout=0.1):
        super(ExfiltrationTransformer, self).__init__()
        
        self.transformer = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout_fc = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        transformer_out = self.transformer(x)  # (batch_size, 1, hidden_dim)
        x = transformer_out[:, 0, :]  # (batch_size, hidden_dim)
        
        # Classification head
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.sigmoid(self.fc3(x)[:, 0])  # Binary classification
        
        return x

class SimpleDNSDetector(nn.Module):
    """Simplified model for quick testing"""
    def __init__(self, input_dim, num_classes=2):
        super(SimpleDNSDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.sigmoid(x[:, 0])
