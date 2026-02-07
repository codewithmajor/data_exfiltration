import requests
import numpy as np

# Generate 32 random features (matching the trained model)
dummy_features = np.random.randn(32).tolist()

response = requests.post(
    'http://localhost:5000/predict',
    json={'features': dummy_features}
)

print("Single Prediction Response:")
print(response.json())
