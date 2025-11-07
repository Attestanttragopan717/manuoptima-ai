import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import pandas as pd
from core.predictive_maintenance import PredictiveMaintenance
from core.quality_control import QualityControl
from core.production_optimizer import ProductionOptimizer
from data.sensor_processor import SensorDataProcessor
from data.image_processor import ImageDataProcessor
from monitoring.dashboard import MonitoringDashboard
from monitoring.alerts import AlertSystem
from utils.config import Config
import torch

def main():
    config = Config()
    
    print("Initializing ManuOptima AI Monitoring System...")
    
    pm_model = PredictiveMaintenance(config.get('predictive_maintenance', {}))
    qc_model = QualityControl(config.get('quality_control', {}))
    optimizer = ProductionOptimizer(config.get('production_optimization', {}))
    
    sensor_processor = SensorDataProcessor(config.get('data', {}))
    image_processor = ImageDataProcessor(config.get('data', {}))
    
    dashboard = MonitoringDashboard()
    alert_system = AlertSystem(config.get('alerts', {}))
    
    try:
        pm_model.model.load_state_dict(torch.load('models/predictive_maintenance.pth'))
        qc_model.model.load_state_dict(torch.load('models/quality_control.pth'))
        print("Models loaded successfully")
    except FileNotFoundError:
        print("Trained models not found. Please run train_models.py first.")
        return
    
    print("Starting real-time monitoring...")
    
    iteration = 0
    while True:
        print(f"\n--- Monitoring Iteration {iteration} ---")
        
        current_sensor_data = sensor_processor.simulate_sensor_data(100)
        processed_data = sensor_processor.preprocess_sensor_data(current_sensor_data)
        
        latest_sensor_readings = processed_data.iloc[-50:]
        prediction = pm_model.predict_failure(latest_sensor_readings)
        
        product_images = image_processor.simulate_product_images(50)
        quality_results = qc_model.analyze_production_batch(product_images)
        
        current_conditions = {
            'machine_speed': latest_sensor_readings['rotation_speed'].iloc[-1],
            'temperature': latest_sensor_readings['temperature'].iloc[-1],
            'pressure': latest_sensor_readings['pressure'].iloc[-1],
            'material_flow': latest_sensor_readings['motor_current'].iloc[-1],
            'vibration': latest_sensor_readings['vibration'].iloc[-1],
            'current_production': 95.0
        }
        
        constraints = {
            'machine_speed_min': 1500, 'machine_speed_max': 2000,
            'temperature_min': 60, 'temperature_max': 90,
            'pressure_min': 80, 'pressure_max': 120,
            'material_flow_min': 10, 'material_flow_max': 20,
            'vibration_min': 1, 'vibration_max': 4
        }
        
        optimization_result = optimizer.optimize_production(current_conditions, constraints)
        
        alerts = alert_system.check_alerts([prediction], quality_results)
        
        for alert in alerts:
            print(f"ALERT: {alert['level']} - {alert['message']}")
            if alert['level'] == 'HIGH':
                alert_system.send_email_alert(alert)
        
        print(f"Failure Risk: {prediction['failure_probability']:.2%}")
        print(f"Quality Score: {quality_results['batch_quality_score']:.2%}")
        print(f"Expected Improvement: {optimization_result['expected_improvement']:.2%}")
        
        if iteration % 10 == 0:
            health_dashboard = dashboard.create_equipment_health_dashboard(
                processed_data, [prediction] * 10
            )
            quality_dashboard = dashboard.create_quality_control_dashboard(quality_results)
            
            health_dashboard.write_html(f"dashboards/health_monitor_{iteration}.html")
            quality_dashboard.write_html(f"dashboards/quality_monitor_{iteration}.html")
            print("Dashboards updated")
        
        iteration += 1
        time.sleep(300)

if __name__ == "__main__":
    main()