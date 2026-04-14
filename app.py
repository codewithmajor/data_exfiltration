import os
import joblib
import numpy as np
import torch
from flask import Flask, request, jsonify
from models.exfil_transformer import HybridExfiltrationModel

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_BATCH = 1000

# ==============================
# MODEL LOADING
# ==============================
def load_model():
    checkpoint = 'checkpoints/exfil_model.pth'
    meta_path  = 'checkpoints/model_meta.pkl'
    scaler_path = 'checkpoints/scaler.pkl'

    for path in [checkpoint, meta_path, scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}. Train the model first."
            )

    meta      = joblib.load(meta_path)
    input_dim = meta['input_dim']

    model = HybridExfiltrationModel(input_dim=input_dim).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )
    model.eval()

    scaler = joblib.load(scaler_path)
    print(f"✓ Model loaded | input_dim={input_dim} | device={device}")
    return model, scaler, input_dim


try:
    model, scaler, input_dim = load_model()
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    model, scaler, input_dim = None, None, None


# ==============================
# HELPERS
# ==============================
def get_risk(confidence, is_exfiltration):
    if not is_exfiltration:
        return 'LOW'
    return 'HIGH' if confidence > 0.7 else 'MEDIUM'


def model_ready():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 503
    return None


# ==============================
# ROUTES
# ==============================
@app.route('/health', methods=['GET'])
def health():
    status = 'ready' if model is not None else 'model_not_loaded'
    return jsonify({
        'status': status,
        'device': str(device),
        'input_dim': input_dim
    })


@app.route('/predict', methods=['POST'])
def predict():
    err = model_ready()
    if err:
        return err

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400
    if 'features' not in data:
        return jsonify({'error': 'Missing features field'}), 400

    try:
        features = np.array(data['features'], dtype=np.float32).reshape(1, -1)

        if features.shape[1] != input_dim:
            return jsonify({
                'error': f'Expected {input_dim} features, got {features.shape[1]}'
            }), 400

        features = scaler.transform(features)
        X = torch.FloatTensor(features).to(device)

        with torch.no_grad():
            output     = model(X).squeeze(-1)
            confidence = torch.sigmoid(output).item()
            prediction = 1 if confidence > 0.5 else 0

        is_exfil = prediction == 1
        return jsonify({
            'prediction':     'Exfiltration' if is_exfil else 'Benign',
            'confidence':     round(float(confidence), 4),
            'is_exfiltration': is_exfil,
            'risk_level':     get_risk(confidence, is_exfil),
            'alert':          is_exfil
        })

    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    err = model_ready()
    if err:
        return err

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400
    if 'features_list' not in data:
        return jsonify({'error': 'Missing features_list field'}), 400

    try:
        features_list = data['features_list']

        if not isinstance(features_list, list) or len(features_list) == 0:
            return jsonify({'error': 'features_list must be a non-empty list'}), 400
        if len(features_list) > MAX_BATCH:
            return jsonify({
                'error': f'Batch size {len(features_list)} exceeds limit of {MAX_BATCH}'
            }), 400

        features = np.array(features_list, dtype=np.float32)

        if features.shape[1] != input_dim:
            return jsonify({
                'error': f'Expected {input_dim} features, got {features.shape[1]}'
            }), 400

        features = scaler.transform(features)
        X = torch.FloatTensor(features).to(device)

        with torch.no_grad():
            outputs     = model(X).squeeze(-1)
            confidences = torch.sigmoid(outputs).cpu().numpy()
            predictions = (confidences > 0.5).astype(int)

        results = []
        for conf, pred in zip(confidences, predictions):
            is_exfil = bool(pred == 1)
            results.append({
                'prediction':     'Exfiltration' if is_exfil else 'Benign',
                'confidence':     round(float(conf), 4),
                'is_exfiltration': is_exfil,
                'risk_level':     get_risk(float(conf), is_exfil),
                'alert':          is_exfil
            })

        return jsonify({
            'predictions': results,
            'total':       len(results),
            'alerts':      sum(r['alert'] for r in results)
        })

    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400


if __name__ == '__main__':
    # For production use: gunicorn -w 4 app:app
    print(f"\n{'='*60}")
    print("DNS Exfiltration Detection API")
    print(f"Device:  {device}")
    print(f"Server:  http://0.0.0.0:5000")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)