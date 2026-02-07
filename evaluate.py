import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from main_train import load_and_prepare_data, prepare_features_labels
from models.exfil_transformer import ExfiltrationTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model_detailed():
    """Detailed evaluation of the trained model"""
    
    # Load data
    print("Loading data...")
    data = load_and_prepare_data()
    X, y, feature_names = prepare_features_labels(data)
    
    # Train-test split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create dataloader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load trained model
    print("Loading trained model...")
    input_dim = X_test.shape[1]
    model = ExfiltrationTransformer(input_dim=input_dim, num_classes=1).to(device)
    model.load_state_dict(torch.load('checkpoints/exfil_model.pth', map_location=device))
    model.eval()
    
    # Get predictions
    all_preds_raw = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            
            # Raw probabilities
            preds_prob = torch.sigmoid(outputs).cpu().numpy()
            all_preds_raw.extend(preds_prob)
            
            # Binary predictions (threshold 0.5)
            preds = (preds_prob > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    
    all_preds_raw = np.array(all_preds_raw)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_preds_raw)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
          target_names=['Benign', 'Exfiltration']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved to: confusion_matrix.png")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds_raw)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    print("ROC curve saved to: roc_curve.png")
    
    print("="*60)

if __name__ == "__main__":
    evaluate_model_detailed()
