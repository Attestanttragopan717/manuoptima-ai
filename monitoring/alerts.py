from typing import Dict, List, Any
import smtplib
from email.mime.text import MimeText
import json

class AlertSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = []
        
    def check_alerts(self, predictions: List[Dict], quality_results: Dict[str, Any]) -> List[Dict]:
        alerts = []
        
        for pred in predictions:
            if pred['alert_level'] == 'HIGH':
                alert = {
                    'type': 'EQUIPMENT_FAILURE_RISK',
                    'level': 'HIGH',
                    'message': f"High failure risk detected: {pred['failure_probability']:.2%}",
                    'timestamp': pred['timestamp'],
                    'data': pred
                }
                alerts.append(alert)
        
        if quality_results['defect_rate'] > 0.1:
            alert = {
                'type': 'QUALITY_ISSUE',
                'level': 'MEDIUM',
                'message': f"High defect rate: {quality_results['defect_rate']:.2%}",
                'timestamp': pd.Timestamp.now(),
                'data': quality_results
            }
            alerts.append(alert)
        
        if quality_results['defect_rate'] > 0.2:
            alert['level'] = 'HIGH'
            alerts.append(alert)
        
        self.alert_history.extend(alerts)
        return alerts
    
    def send_email_alert(self, alert: Dict[str, Any]):
        if not self.config.get('email_enabled', False):
            return
        
        smtp_config = self.config.get('smtp', {})
        
        msg = MimeText(alert['message'])
        msg['Subject'] = f"ManuOptima Alert: {alert['type']} - {alert['level']}"
        msg['From'] = smtp_config.get('from_email', 'alerts@manuoptima.com')
        msg['To'] = ', '.join(smtp_config.get('recipients', []))
        
        try:
            server = smtplib.SMTP(smtp_config.get('smtp_server', 'localhost'), smtp_config.get('smtp_port', 587))
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Failed to send email alert: {e}")