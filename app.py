from flask import Flask, request, jsonify
import torch
import numpy as np
import os
from models.exfil_transformer import ExfiltrationTransformer

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model at startup
print(f"Loading model on device: {device}...")
model = ExfiltrationTransformer(input_dim=32, num_classes=1).to(device)
model.load_state_dict(torch.load('checkpoints/exfil_model.pth', map_location=device))
model.eval()
print("✓ Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'DNS Exfiltration Detection API is running', 'device': str(device)})

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction for DNS exfiltration"""
    try:
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features field'}), 400
        
        features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
        
        if features.shape[1] != 32:
            return jsonify({'error': f'Expected 32 features, got {features.shape[1]}'}), 400
        
        X = torch.FloatTensor(features).to(device)
        
        with torch.no_grad():
            output = model(X).squeeze()
            confidence = torch.sigmoid(output).item()
            prediction = 1 if confidence > 0.5 else 0
        
        return jsonify({
            'prediction': 'Exfiltration' if prediction == 1 else 'Benign',
            'confidence': float(confidence),
            'is_exfiltration': prediction == 1,
            'risk_level': 'HIGH' if confidence > 0.7 else 'MEDIUM' if confidence > 0.5 else 'LOW'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make predictions for multiple DNS queries"""
    try:
        data = request.json
        
        if 'features_list' not in data:
            return jsonify({'error': 'Missing features_list field'}), 400
        
        features_list = data['features_list']
        if not isinstance(features_list, list) or len(features_list) == 0:
            return jsonify({'error': 'features_list must be a non-empty list'}), 400
        
        features = np.array(features_list, dtype=np.float32)
        
        if features.shape[1] != 42:
            return jsonify({'error': f'Expected 42 features, got {features.shape[1]}'}), 400
        
        X = torch.FloatTensor(features).to(device)
        
        with torch.no_grad():
            outputs = model(X).squeeze()
            confidences = torch.sigmoid(outputs).cpu().numpy()
            predictions = (confidences > 0.5).astype(int)
        
        results = []
        for conf, pred in zip(confidences, predictions):
            results.append({
                'prediction': 'Exfiltration' if pred == 1 else 'Benign',
                'confidence': float(conf),
                'is_exfiltration': pred == 1,
                'risk_level': 'HIGH' if conf > 0.7 else 'MEDIUM' if conf > 0.5 else 'LOW'
            })
        
        return jsonify({'predictions': results, 'total': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("DNS Exfiltration Detection API")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Starting server on http://0.0.0.0:5000")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
