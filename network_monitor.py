import requests
import time
from datetime import datetime
from threading import Thread
import json

class DNSMonitor:
    def __init__(self, api_url='http://localhost:5000'):
        self.api_url = api_url
        self.alerts = []
        self.total_checked = 0
        self.exfiltrations_detected = 0
    
    def extract_features(self, dns_query):
        """
        Extract features from DNS query
        In real scenario, parse actual DNS packets
        """
        # Mock feature extraction
        return [float(x) for x in dns_query.get('features', [0.5] * 32)]
    
    def analyze_dns_query(self, dns_query):
        """Analyze a single DNS query"""
        try:
            features = self.extract_features(dns_query)
            
            response = requests.post(
                f'{self.api_url}/predict',
                json={'features': features},
                timeout=2
            )
            
            if response.status_code == 200:
                result = response.json()
                self.total_checked += 1
                
                if result['is_exfiltration']:
                    self.exfiltrations_detected += 1
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'query': dns_query.get('domain', 'unknown'),
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'risk_level': result['risk_level']
                    }
                    self.alerts.append(alert)
                    self._log_alert(alert)
                
                return result
        except Exception as e:
            print(f"Error analyzing query: {e}")
            return None
    
    def _log_alert(self, alert):
        """Log alert to file"""
        with open('dns_alerts.log', 'a') as f:
            f.write(json.dumps(alert) + '\n')
        
        print(f"\n⚠️  ALERT DETECTED")
        print(f"   Domain: {alert['query']}")
        print(f"   Risk: {alert['risk_level']}")
        print(f"   Confidence: {alert['confidence']:.2%}")
    
    def analyze_batch(self, dns_queries):
        """Analyze multiple queries efficiently"""
        features_list = [self.extract_features(q) for q in dns_queries]
        
        try:
            response = requests.post(
                f'{self.api_url}/predict_batch',
                json={'features_list': features_list},
                timeout=5
            )
            
            if response.status_code == 200:
                results = response.json()
                
                for query, result in zip(dns_queries, results['predictions']):
                    self.total_checked += 1
                    if result['is_exfiltration']:
                        self.exfiltrations_detected += 1
                        alert = {
                            'timestamp': datetime.now().isoformat(),
                            'query': query.get('domain', 'unknown'),
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'risk_level': result['risk_level']
                        }
                        self.alerts.append(alert)
                        self._log_alert(alert)
                
                return results
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return None
    
    def get_stats(self):
        """Get monitoring statistics"""
        return {
            'total_checked': self.total_checked,
            'exfiltrations_detected': self.exfiltrations_detected,
            'detection_rate': f"{(self.exfiltrations_detected / max(self.total_checked, 1)) * 100:.2f}%",
            'recent_alerts': self.alerts[-5:]
        }
    
    def print_stats(self):
        """Print monitoring statistics"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("DNS EXFILTRATION MONITORING STATISTICS")
        print("="*60)
        print(f"Total Queries Checked: {stats['total_checked']}")
        print(f"Exfiltrations Detected: {stats['exfiltrations_detected']}")
        print(f"Detection Rate: {stats['detection_rate']}")
        print(f"\nRecent Alerts: {len(stats['recent_alerts'])}")
        for alert in stats['recent_alerts']:
            print(f"  - {alert['query']}: {alert['risk_level']} ({alert['confidence']:.2%})")
        print("="*60)

# Usage example
if __name__ == '__main__':
    monitor = DNSMonitor()
    
    # Check API
    print("Starting DNS Exfiltration Monitor...")
    
    # Simulate DNS queries
    dns_queries = [
        {'domain': 'google.com', 'features': [0.2] * 32},
        {'domain': 'suspicious.com', 'features': [0.8] * 32},
        {'domain': 'amazon.com', 'features': [0.3] * 32},
        {'domain': 'malicious.io', 'features': [0.9] * 32},
        {'domain': 'github.com', 'features': [0.1] * 32},
    ]
    
    print(f"\nAnalyzing {len(dns_queries)} DNS queries...")
    monitor.analyze_batch(dns_queries)
    
    # Print statistics
    monitor.print_stats()
    
    # Additional individual analysis
    print("\n\nAnalyzing single queries...")
    single_query = {'domain': 'test.com', 'features': [0.6] * 32}
    result = monitor.analyze_dns_query(single_query)
    if result:
        print(json.dumps(result, indent=2))
