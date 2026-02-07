import requests
import json

class DNSExfiltrationClient:
    def __init__(self, api_url='http://localhost:5000'):
        self.api_url = api_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check if API is running"""
        try:
            response = self.session.get(f'{self.api_url}/health', timeout=2)
            return response.json()
        except Exception as e:
            return {'error': str(e), 'status': 'offline'}
    
    def predict(self, features):
        """Single prediction"""
        try:
            if len(features) != 32:
                return {'error': f'Expected 32 features, got {len(features)}'}
            
            response = self.session.post(
                f'{self.api_url}/predict',
                json={'features': features},
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, features_list):
        """Batch predictions"""
        try:
            response = self.session.post(
                f'{self.api_url}/predict_batch',
                json={'features_list': features_list},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}

# Usage
if __name__ == '__main__':
    client = DNSExfiltrationClient()
    
    # Check API
    print("API Status:", client.health_check())
    
    # Single prediction
    test_features = [0.5] * 32
    result = client.predict(test_features)
    print("\nSingle Prediction:")
    print(json.dumps(result, indent=2))
    
    # Batch prediction
    batch_features = [[0.3] * 32, [0.7] * 32, [0.5] * 32]
    results = client.predict_batch(batch_features)
    print("\nBatch Predictions:")
    print(json.dumps(results, indent=2))
