import requests
import json
from datetime import datetime

class ZeekDNSAnalyzer:
    """Integrate with Zeek IDS for DNS analysis"""
    
    def __init__(self, api_url='http://localhost:5000'):
        self.api_url = api_url
    
    def parse_zeek_dns_log(self, zeek_dns_file):
        """Parse Zeek DNS log file"""
        queries = []
        
        with open(zeek_dns_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) >= 4:
                    query = {
                        'timestamp': fields[0],
                        'query': fields[3],
                        'response_code': fields[4] if len(fields) > 4 else 'NODATA'
                    }
                    queries.append(query)
        
        return queries
    
    def extract_features_from_zeek(self, zeek_record):
        """Extract features from Zeek DNS record"""
        # Extract relevant features from Zeek data
        # This is a mock - in reality you'd parse actual DNS metrics
        features = [0.5] * 32
        return features
    
    def analyze_zeek_dns_log(self, zeek_dns_file, output_file='zeek_threats.json'):
        """Analyze all DNS queries in Zeek log"""
        queries = self.parse_zeek_dns_log(zeek_dns_file)
        threats = []
        
        print(f"Analyzing {len(queries)} DNS queries from Zeek...")
        
        for i, query in enumerate(queries):
            features = self.extract_features_from_zeek(query)
            
            try:
                response = requests.post(
                    f'{self.api_url}/predict',
                    json={'features': features},
                    timeout=1
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['is_exfiltration']:
                        threat = {
                            'query_time': query['timestamp'],
                            'domain': query['query'],
                            'risk_level': result['risk_level'],
                            'confidence': result['confidence'],
                            'response_code': query['response_code']
                        }
                        threats.append(threat)
                        print(f"🚨 Threat #{len(threats)}: {query['query']}")
            except:
                pass
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(queries)} queries...")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(threats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total Queries: {len(queries)}")
        print(f"Threats Found: {len(threats)}")
        print(f"Threat Rate: {(len(threats) / max
