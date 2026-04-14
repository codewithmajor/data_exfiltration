import json
import logging
import requests
from collections import deque
from datetime import datetime
from threading import Lock


logging.basicConfig(
    filename='dns_alerts.log',
    level=logging.INFO,
    format='%(message)s'
)


class DNSMonitor:
    def __init__(self, api_url='http://localhost:5000', input_dim=32, chunk_size=500):
        self.api_url    = api_url
        self.input_dim  = input_dim
        self.chunk_size = chunk_size

        self.total_checked          = 0
        self.exfiltrations_detected = 0
        self.alerts = deque(maxlen=1000)  # bounded — won't grow forever

        self._lock  = Lock()
        self.logger = logging.getLogger('dns_monitor')

    # ── API health ──────────────────────────────────────────────
    def check_api_health(self):
        try:
            r = requests.get(f'{self.api_url}/health', timeout=3)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    # ── Feature extraction ──────────────────────────────────────
    def extract_features(self, dns_query):
        """Extract features from DNS query dict."""
        return [float(x) for x in
                dns_query.get('features', [0.0] * self.input_dim)]

    # ── Single prediction ───────────────────────────────────────
    def analyze_dns_query(self, dns_query):
        try:
            features = self.extract_features(dns_query)
            response = requests.post(
                f'{self.api_url}/predict',
                json={'features': features},
                timeout=(3, 10)
            )
            if response.status_code == 200:
                result = response.json()
                self._record(dns_query, result)
                return result
            else:
                print(f"[Warning] API returned {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[Error] analyze_dns_query: {e}")
        return None

    # ── Batch prediction ────────────────────────────────────────
    def analyze_batch(self, dns_queries):
        """Analyze queries in safe-sized chunks."""
        all_predictions = []

        for i in range(0, len(dns_queries), self.chunk_size):
            chunk        = dns_queries[i:i + self.chunk_size]
            features_list = [self.extract_features(q) for q in chunk]

            try:
                response = requests.post(
                    f'{self.api_url}/predict_batch',
                    json={'features_list': features_list},
                    timeout=(3, 30)
                )
                if response.status_code == 200:
                    results = response.json()
                    for query, result in zip(chunk, results['predictions']):
                        self._record(query, result)
                    all_predictions.extend(results['predictions'])
                else:
                    print(f"[Warning] Batch API returned {response.status_code}")
            except Exception as e:
                print(f"[Error] analyze_batch chunk {i}: {e}")

        return all_predictions

    # ── Internal helpers ────────────────────────────────────────
    def _record(self, dns_query, result):
        """Thread-safe update of counters and alerts."""
        with self._lock:
            self.total_checked += 1
            if result.get('is_exfiltration'):
                self.exfiltrations_detected += 1
                alert = {
                    'timestamp':  datetime.now().isoformat(),
                    'query':      dns_query.get('domain', 'unknown'),
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'risk_level': result['risk_level']
                }
                self.alerts.append(alert)
                self._log_alert(alert)

    def _log_alert(self, alert):
        self.logger.info(json.dumps(alert))
        print(f"\n⚠️  ALERT: {alert['query']} | "
              f"{alert['risk_level']} | {alert['confidence']:.2%}")

    # ── Stats ───────────────────────────────────────────────────
    def get_stats(self):
        with self._lock:
            return {
                'total_checked':          self.total_checked,
                'exfiltrations_detected': self.exfiltrations_detected,
                'detection_rate':         f"{(self.exfiltrations_detected / max(self.total_checked, 1)) * 100:.2f}%",
                'recent_alerts':          list(self.alerts)[-5:]
            }

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("DNS EXFILTRATION MONITORING STATISTICS")
        print("="*60)
        print(f"Total Queries Checked:  {stats['total_checked']}")
        print(f"Exfiltrations Detected: {stats['exfiltrations_detected']}")
        print(f"Detection Rate:         {stats['detection_rate']}")
        for alert in stats['recent_alerts']:
            print(f"  ⚠ {alert['query']}: {alert['risk_level']} "
                  f"({alert['confidence']:.2%})")
        print("="*60)


# ──