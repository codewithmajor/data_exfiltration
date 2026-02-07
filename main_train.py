import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from models.exfil_transformer import ExfiltrationTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_prepare_data(data_dir='data/'):
    """Load all CSV files and prepare dataset"""
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
    print(f"\nTotal data shape: {data.shape}")
    return data

def prepare_features_labels(data):
    """Prepare features and labels for training"""
    # Separate features and labels
    X = data.drop(columns=['Label'] if 'Label' in data.columns else [])
    y = data['Label'] if 'Label' in data.columns else data.iloc[:, -1]
    
    # Encode labels if string
    if isinstance(y.iloc[0], str):
        y = (y == 'Exfiltration').astype(int)
    else:
        y = y.astype(int)
    
    print(f"Features shape: {X.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, X.columns

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in tqdm(train_loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), accuracy, all_preds, all_labels

if __name__ == "__main__":
    # Load data
    data = load_and_prepare_data()
    X, y, feature_names = prepare_features_labels(data)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = ExfiltrationTransformer(input_dim=input_dim, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    epochs = 50
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, preds, labels = evaluate_model(model, test_loader, criterion, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    _, _, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds):.4f}")
    print(f"F1-Score: {f1_score(all_labels, all_preds):.4f}")
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/exfil_model.pth')
    print("Model saved to checkpoints/exfil_model.pth")
