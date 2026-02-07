import socket
import struct
import textwrap
import requests
import json

class DNSPacketAnalyzer:
    def __init__(self, api_url='http://localhost:5000'):
        self.api_url = api_url
        self.packet_count = 0
        self.threats = 0
    
    def extract_dns_features_from_packet(self, packet_data):
        """
        Extract DNS features from raw packet data
        Returns 32-dimensional feature vector
        """
        # In production, you would:
        # 1. Parse DNS packet headers
        # 2. Extract query names, response times, TTL values
        # 3. Calculate entropy, packet sizes, etc.
        
        # Mock implementation
        features = [
            len(packet_data) / 1000,  # Packet size
            hash(str(packet_data)) % 100 / 100,  # Hash-based feature
        ]
        features.extend([0.5] * 30)  # Placeholder features
        return features[:32]
    
    def analyze_packet(self, packet_data):
        """Analyze a single packet"""
        self.packet_count += 1
        
        try:
            features = self.extract_dns_features_from_packet(packet_data)
            
            response = requests.post(
                f'{self.api_url}/predict',
                json={'features': features},
                timeout=1
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result['is_exfiltration']:
                    self.threats += 1
                    print(f"\n🚨 THREAT DETECTED (Packet {self.packet_count})")
                    print(f"   Risk Level: {result['risk_level']}")
                    print(f"   Confidence: {result['confidence']:.2%}")
                
                return result
        except:
            pass  # Timeout or API unavailable
        
        return None
    
    def sniff_packets(self, interface='eth0', packet_count=100):
        """Sniff network packets"""
        print(f"Starting packet sniffer on {interface}...")
        print(f"Analyzing up to {packet_count} packets...\n")
        
        # Create raw socket (requires admin/root)
        try:
            if hasattr(socket, 'AF_PACKET'):
                sniffer = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))
            else:
                print("Packet sniffing requires Linux/Unix")
                return
            
            packets_analyzed = 0
            while packets_analyzed < packet_count:
                raw_buffer, addr = sniffer.recvfrom(65535)
                self.analyze_packet(raw_buffer)
                packets_analyzed += 1
            
            print(f"\n{'='*60}")
            print("PACKET ANALYSIS SUMMARY")
            print(f"{'='*60}")
            print(f"Total Packets Analyzed: {self.packet_count}")
            print(f"Threats Detected: {self.threats}")
            print(f"Threat Rate: {(self.threats / max(self.packet_count, 1)) * 100:.2f}%")
            print(f"{'='*60}")
        
        except Exception as e:
            print(f"Error: {e}")
            print("Note: Packet sniffing requires administrator privileges")

# Usage
if __name__ == '__main__':
    analyzer = DNSPacketAnalyzer()
    
    # Simulate packet sniffing
    print("DNS Packet Analyzer")
    print("Note: Run with 'sudo' for real packet capture")
    
    # In real scenario:
    # analyzer.sniff_packets('eth0', packet_count=1000)
