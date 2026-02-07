# DNS Exfiltration Detection with Transformer Model

A deep learning project for detecting DNS data exfiltration attacks using Transformer-based neural networks.

## Dataset
- **Source**: CIC-Bell-DNS-EXF-2021 (or similar DNS traffic datasets)
- **Files**: 12 CSV files with network traffic features
- **Features**: DNS query patterns, response times, entropy, TTL values, and more
- **Labels**: Binary classification (Normal / Exfiltration)

## Project Structure
```
data_exfiltration/
├── README.md
├── requirements.txt
├── main_train.py              # Main training script
├── .gitignore
├── data/
│   └── *.csv                  # Your dataset files
├── scripts/
│   └── preprocess.py          # Data preprocessing
├── models/
│   └── exfil_transformer.py   # Transformer model
└── checkpoints/               # Saved models
```

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Prepare dataset**:
- Place all 12 CSV files in the `data/` folder
- Run preprocessing:
```bash
python scripts/preprocess.py
```

3. **Train model**:
```bash
python main_train.py
```

## Model Architecture
- **Transformer Encoder** with multi-head self-attention
- **Feature Embedding** layer
- **Classification Head** for binary output
- **Dropout & Layer Normalization** for regularization

## Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve
- Confusion matrix

## Results
(Will be updated after training)

## Author
CodeWithMajor
