import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from main_train import load_and_prepare_data
from models.exfil_transformer import HybridExfiltrationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model_detailed():
    print("Loading data...")
    X, y = load_and_prepare_data('data2')
    feature_names = X.columns.tolist()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply same scaler used during training
    scaler = joblib.load('checkpoints/scaler.pkl')
    X_test_scaled = scaler.transform(X_test)

    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)

    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=32, shuffle=False
    )

    print("Loading trained model...")
    input_dim = X_test_tensor.shape[1]
    model = HybridExfiltrationModel(input_dim=input_dim).to(device)
    model.load_state_dict(
        torch.load('checkpoints/exfil_model.pth', map_location=device)
    )
    model.eval()

    all_preds_raw, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze(-1)

            preds_prob = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds_prob > 0.5).astype(int)

            all_preds_raw.extend(preds_prob)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds_raw = np.array(all_preds_raw)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc   = roc_auc_score(all_labels, all_preds_raw)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
          target_names=['Benign', 'Exfiltration']))

    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Exfiltration'],
                yticklabels=['Benign', 'Exfiltration'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_preds_raw)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Exfiltration Detection')
    plt.legend()
    plt.savefig('results/roc_curve.png')
    plt.close()

    print("\nPlots saved to results/")
    print("="*60)


if __name__ == "__main__":
    evaluate_model_detailed()