import json
import logging
from datetime import datetime
from collections import deque
from threading import Lock
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AlertSystem')

class AlertManager:
    """Manages alerts and notifications for detected threats"""
    
    def __init__(self, max_alerts=1000):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_log_file = 'threat_alerts.log'
        self.lock = Lock()
        self.severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
    
    def create_alert(self, detection_result, packet_id=None):
        """Create an alert from a detection result"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'packet_id': packet_id,
            'confidence': detection_result.get('confidence', 0),
            'processing_time_ms': detection_result.get('processing_time_ms', 0),
            'is_threat': detection_result.get('is_threat', False),
            'risk_level': self._calculate_risk_level(detection_result.get('confidence', 0))
        }
        
        with self.lock:
            self.alerts.append(alert)
            self._log_alert_to_file(alert)
        
        return alert
    
    def _calculate_risk_level(self, confidence):
        """Calculate risk level based on confidence score"""
        if confidence >= 0.9:
            return 'CRITICAL'
        elif confidence >= 0.75:
            return 'HIGH'
        elif confidence >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _log_alert_to_file(self, alert):
        """Log alert to file"""
        try:
            with open(self.alert_log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def get_recent_alerts(self, count=10):
        """Get recent alerts"""
        with self.lock:
            return list(self.alerts)[-count:]
    
    def get_high_priority_alerts(self):
        """Get high priority alerts"""
        with self.lock:
            return [a for a in self.alerts if self.severity_levels.get(a.get('risk_level', 'LOW'), 0) >= 3]
    
    def print_alert_summary(self):
        """Print alert summary"""
        with self.lock:
            total = len(self.alerts)
            critical = sum(1 for a in self.alerts if a.get('risk_level') == 'CRITICAL')
            high = sum(1 for a in self.alerts if a.get('risk_level') == 'HIGH')
            medium = sum(1 for a in self.alerts if a.get('risk_level') == 'MEDIUM')
        
        print(f"\n{'='*60}")
        print(f"THREAT ALERT SUMMARY")
        print(f"{'='*60}")
        print(f"Total Alerts: {total}")
        print(f"Critical: {critical} | High: {high} | Medium: {medium}")
        print(f"{'='*60}\n")

class NotificationHandler:
    """Sends notifications for critical threats"""
    
    def __init__(self, email_config=None):
        self.email_config = email_config or {}
        self.notification_log = 'notifications.log'
    
    def send_email_alert(self, alert, recipients):
        """Send email notification for threat alert"""
        if not self.email_config.get('enabled', False):
            logger.warning("Email notifications disabled")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_address')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"🚨 THREAT ALERT - {alert.get('risk_level')} SEVERITY"
            
            body = f"""
            Threat Detection Alert
            
            Time: {alert.get('timestamp')}
            Risk Level: {alert.get('risk_level')}
            Confidence: {alert.get('confidence'):.2%}
            Packet ID: {alert.get('packet_id')}
            Processing Time: {alert.get('processing_time_ms'):.2f}ms
            
            ACTION: Review logs and take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # In production, connect to actual SMTP server
            # For now, just log
            logger.info(f"Email notification would be sent to {recipients}")
            self._log_notification(alert, recipients)
            
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _log_notification(self, alert, recipients):
        """Log notification attempt"""
        try:
            with open(self.notification_log, 'a') as f:
                f.write(json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'alert': alert,
                    'recipients': recipients
                }) + '\n')
        except Exception as e:
            logger.error(f"Error logging notification: {e}")

if __name__ == '__main__':
    alert_mgr = AlertManager()
    notification = NotificationHandler()
    
    # Test alert creation
    test_result = {'confidence': 0.85, 'is_threat': True, 'processing_time_ms': 1.5}
    alert = alert_mgr.create_alert(test_result, packet_id=1)
    
    print(f"Alert created: {alert}")
    alert_mgr.print_alert_summary()
