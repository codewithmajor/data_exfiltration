import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from models.exfil_transformer import ExfiltrationTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =========================
# LOAD DATA
# =========================
def load_and_prepare_data(data_dir='data/'):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            dfs.append(df)
            print(f"Loaded {file}: {df.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    data = pd.concat(dfs, ignore_index=True)
    print(f"Total data shape: {data.shape}")
    return data


# =========================
# FEATURE ENGINEERING
# =========================
def prepare_features_labels(data):
    # Handle label
    if 'Label' not in data.columns:
        data['Label'] = 0

    y = data['Label'].fillna(0)

    if isinstance(y.iloc[0], str):
        y = (y == 'Exfiltration').astype(int)
    else:
        y = y.astype(int)

    # Encode Protocol if exists
    if 'Protocol' in data.columns:
        data['Protocol'] = data['Protocol'].astype('category').cat.codes

    # Keep only numeric features
    X = data.select_dtypes(include=[np.number]).drop(columns=['Label'], errors='ignore')

    # Fill missing values
    X = X.fillna(X.mean())

    print(f"Features shape: {X.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")

    # Scale features ONCE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values


# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for X, y in tqdm(train_loader, desc="Training"):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X).squeeze()

        loss = criterion(outputs, y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X).squeeze()
            loss = criterion(outputs, y.float())
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    return total_loss / len(val_loader), accuracy, all_preds, all_labels


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    data = load_and_prepare_data()
    X, y = prepare_features_labels(data)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance dataset
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {X_train.shape}, Labels: {np.bincount(y_train)}")

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    # Model
    input_dim = X_train.shape[1]
    model = ExfiltrationTransformer(input_dim=input_dim, num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # =========================
    # TRAIN LOOP
    # =========================
    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, accuracy, preds, labels = evaluate_model(model, test_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # =========================
    # FINAL EVALUATION
    # =========================
    print("\nFinal Evaluation:")
    _, _, all_preds, all_labels = evaluate_model(model, test_loader, criterion)

    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds):.4f}")

    print("Model saved at checkpoints/best_model.pth")