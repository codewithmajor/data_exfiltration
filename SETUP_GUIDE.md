# Setup Guide - Data Exfiltration Detection Project

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/codewithmajor/data_exfiltration.git
cd data_exfiltration
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies:**
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn tqdm
```

### Project Structure

```
data_exfiltration/
├── models/
│   ├── __init__.py
│   └── exfil_transformer.py       # Transformer model architecture
├── scripts/
│   ├── dataset.py                 # Dataset classes (FlowDataset, FlowSequenceDataset)
│   └── utils.py                   # Evaluation metrics (compute_metrics)
├── data/
│   └── flows.csv                  # Your preprocessed dataset
├── checkpoints/                   # Saved models
├── main_train_baseline.py          # Train baseline MLP model
├── main_train_seq.py               # Train sequence transformer model
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── RESEARCH_PAPER.md               # Full research paper
└── SETUP_GUIDE.md                  # This file
```

## Data Preparation

Your dataset should be a CSV file (`data/flows.csv`) with columns:
- `src_ip`: Source IP address
- `timestamp`: Flow timestamp
- `bytes`: Bytes transferred
- `packets`: Number of packets
- `duration`: Flow duration
- `dns_entropy`: DNS query entropy
- `dns_len`: DNS query length
- `http_path_len`: HTTP path length
- `label`: Binary label (0=benign, 1=exfiltration)

## Running the Models

### Baseline Model (Single-flow MLP)
```bash
python main_train_baseline.py
```

This trains a simple MLP on individual flows and outputs metrics:
- Accuracy, Precision, Recall, F1-score
- False Positive Rate (FPR)
- ROC-AUC
- Saves model to `checkpoints/mlp_baseline.pth`

### Sequence Transformer Model
```bash
python main_train_seq.py
```

This trains a transformer model on sequences of flows:
- Groups flows by source IP
- Creates 20-flow windows
- Detects temporal patterns
- Saves model to `checkpoints/exfil_transformer.pth`

## Expected Results

On public datasets (CIC-Bell-DNS-EXF-2021):
- **Baseline MLP**: ~94-95% accuracy
- **Transformer**: ~97.2% accuracy
- **False Positive Rate**: <1.5%
- **Processing Speed**: 1M flows/sec with <50ms latency

## Customization

### Change Feature Columns
Edit `feature_cols` in training scripts:
```python
feature_cols = [
    "bytes", "packets", "duration",
    "dns_entropy", "dns_len", "http_path_len"
]
```

### Adjust Model Parameters
In `main_train_seq.py`, modify:
```python
seq_len = 20                    # Sequence length
d_model = 64                    # Embedding dimension
nhead = 4                       # Attention heads
num_layers = 2                  # Transformer layers
batch_size = 64                 # Training batch size
num_epochs = 10                 # Training epochs
```

### Use GPU
The code automatically detects CUDA:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in training scripts
- Use CPU instead of GPU
- Process smaller dataset batches

### Missing Data
- Ensure `data/flows.csv` exists
- Check column names match `feature_cols`
- Handle missing values with `.dropna()`

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Reduce `seq_len` for faster sequences
- Use fewer `num_epochs` for testing

## References

- Research Paper: `RESEARCH_PAPER.md`
- Main README: `README.md`
- Base Paper (2025): Real-Time Detection of Data Exfiltration Using Deep Learning in Edge Computing
- Dataset: CIC-Bell-DNS-EXF-2021

## Support

For issues or questions:
1. Check the README.md
2. Review RESEARCH_PAPER.md for methodology
3. See inline code comments in `models/` and `scripts/`
