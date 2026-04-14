import os
import torch
import numpy as np
import pandas as pd

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models.exfil_transformer import HybridExfiltrationModel

# ==============================
# DEVICE
# ==============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================
# LOAD DATA
# ==============================
def load_and_prepare_data(data_dir='data2'):
    csv_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for file in csv_files:
        print(f"Loading: {file}")
        df = pd.read_csv(file)

        # 🔥 Label creation
        if "Attack" in file or "heavy" in file or "light" in file:
            df['Label'] = 1
        else:
            df['Label'] = 0

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    print("Total samples:", len(data))
    print("Label distribution:\n", data['Label'].value_counts())

    X = data.drop(columns=['Label'])
    y = data['Label']

    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)

    return X, y

# ==============================
# TRAIN
# ==============================
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ==============================
# EVALUATE
# ==============================
def evaluate_model(model, loader):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            predictions = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()

            preds.extend(predictions)
            actuals.extend(y_batch.numpy())

    return accuracy_score(actuals, preds)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    X, y = load_and_prepare_data('data2')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    # Model
    input_dim = X_train.shape[1]
    model = HybridExfiltrationModel(input_dim).to(device)

    # 🔥 FIXED class imbalance handling
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training
    epochs = 20

    for epoch in range(epochs):
        loss = train_model(model, train_loader, optimizer, criterion)
        acc = evaluate_model(model, test_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/exfil_model.pth")

    print("✅ Training Complete & Model Saved!")