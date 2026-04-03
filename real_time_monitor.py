import torch
import numpy as np
import requests
import json
import time
from datetime import datetime
from collections import deque
from threading import Thread, Lock
from queue import Queue
import logging
from models.exfil_transformer import ExfiltrationTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RealTimeMonitor')

class StreamingFeatureBuffer:
    """Buffer for streaming features with windowing"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.lock = Lock()
    
    def add_features(self, features):
        with self.lock:
            self.buffer.append(features)
    
    def get_buffer(self):
        with self.lock:
            return list(self.buffer)
    
    def is_ready(self):
        with self.lock:
            return len(self.buffer) >= self.window_size

class RealTimeExfiltrationMonitor:
    """Real-time DNS exfiltration detection system"""
    
    def __init__(self, model_path='checkpoints/exfil_model.pth', api_url='http://localhost:5000'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = ExfiltrationTransformer(input_dim=32, num_classes=1).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("Model loaded successfully")
        except:
            logger.warning("Could not load model, using untrained weights")
        self.model.eval()
        
        # Real-time tracking
        self.feature_buffer = StreamingFeatureBuffer(window_size=10)
        self.prediction_queue = Queue(maxsize=1000)
        self.alerts = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'packets_analyzed': 0,
            'threats_detected': 0,
            'avg_confidence': 0.0,
            'processing_time_ms': 0.0
        }
        self.stats_lock = Lock()
        self.api_url = api_url
    
    def extract_features(self, packet_data):
        """Extract 32-dimensional feature vector from packet"""
        # This is a placeholder - in production, extract real DNS features
        features = np.random.rand(32).astype(np.float32)
        return features
    
    def predict_threat(self, features):
        """Predict if features indicate exfiltration"""
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Ensure features is correct shape
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).to(self.device)
                
                if features.dim() == 1:
                    features = features.unsqueeze(0)  # Add batch dimension
                
                # Predict
                logits = self.model(features.unsqueeze(1).float())
                probs = torch.sigmoid(logits)
                
                is_threat = probs.item() > 0.5
                confidence = probs.item()
                
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                return {
                    'is_threat': is_threat,
                    'confidence': float(confidence),
                    'processing_time_ms': processing_time,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def process_packet(self, packet_data):
        """Process a single network packet in real-time"""
        # Extract features
        features = self.extract_features(packet_data)
        self.feature_buffer.add_features(features)
        
        # Make prediction
        result = self.predict_threat(features)
        
        if result:
            with self.stats_lock:
                self.stats['packets_analyzed'] += 1
                self.stats['processing_time_ms'] = result['processing_time_ms']
                
                if result['is_threat']:
                    self.stats['threats_detected'] += 1
                    alert = {
                        'timestamp': result['timestamp'],
                        'confidence': result['confidence'],
                        'packet_id': self.stats['packets_analyzed'],
                        'risk_level': 'HIGH' if result['confidence'] > 0.8 else 'MEDIUM'
                    }
                    self.alerts.append(alert)
                    logger.warning(f"THREAT DETECTED: {alert}")
                    self._send_alert(alert)
                
                # Update average confidence
                if self.stats['packets_analyzed'] > 0:
                    self.stats['avg_confidence'] = (self.stats['avg_confidence'] * (self.stats['packets_analyzed'] - 1) + result['confidence']) / self.stats['packets_analyzed']
        
        return result
    
    def _send_alert(self, alert):
        """Send alert to API endpoint"""
        try:
            requests.post(
                f'{self.api_url}/alerts',
                json=alert,
                timeout=2
            )
        except Exception as e:
            logger.error(f"Could not send alert: {e}")
    
    def get_stats(self):
        """Get current monitoring statistics"""
        with self.stats_lock:
            return self.stats.copy()
    
    def print_stats(self):
        """Print real-time statistics"""
        stats = self.get_stats()
        print(f"\n{'='*70}")
        print(f"REAL-TIME EXFILTRATION DETECTION STATISTICS")
        print(f"{'='*70}")
        print(f"Packets Analyzed: {stats['packets_analyzed']}")
        print(f"Threats Detected: {stats['threats_detected']}")
        print(f"Detection Rate: {(stats['threats_detected'] / max(stats['packets_analyzed'], 1)) * 100:.2f}%")
        print(f"Avg Confidence: {stats['avg_confidence']:.4f}")
        print(f"Processing Time: {stats['processing_time_ms']:.2f}ms per packet")
        print(f"Recent Alerts: {len(self.alerts)}")
        print(f"{'='*70}\n")

if __name__ == '__main__':
    monitor = RealTimeExfiltrationMonitor()
    print("Real-Time Exfiltration Monitor Started")
    print(f"Model on device: {monitor.device}")
    
    # Simulate packet processing
    for i in range(100):
        packet = f"packet_{i}".encode()
        result = monitor.process_packet(packet)
        if i % 10 == 0:
            monitor.print_stats()
        time.sleep(0.1)
    
    monitor.print_stats()
