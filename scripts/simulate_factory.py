import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from data.sensor_processor import SensorDataProcessor
from data.image_processor import ImageDataProcessor
from utils.config import Config

def simulate_factory_operations(days: int = 7):
    config = Config()
    
    sensor_processor = SensorDataProcessor(config.get('data', {}))
    image_processor = ImageDataProcessor(config.get('data', {}))
    
    print(f"Simulating {days} days of factory operations...")
    
    operations_data = []
    
    for day in range(days):
        daily_sensor_data = sensor_processor.simulate_sensor_data(24)
        daily_images = image_processor.simulate_product_images(500)
        
        daily_quality = len([img for img in daily_images if np.random.random() < 0.15])
        daily_production = len(daily_images)
        
        operation_summary = {
            'day': day + 1,
            'total_production': daily_production,
            'defective_products': daily_quality,
            'quality_rate': (daily_production - daily_quality) / daily_production,
            'avg_temperature': daily_sensor_data['temperature'].mean(),
            'avg_vibration': daily_sensor_data['vibration'].mean(),
            'equipment_uptime': np.random.uniform(0.85, 0.98),
            'energy_consumption': np.random.normal(5000, 500)
        }
        
        operations_data.append(operation_summary)
        
        print(f"Day {day + 1}: {daily_production} products, "
              f"Quality: {operation_summary['quality_rate']:.2%}, "
              f"Uptime: {operation_summary['equipment_uptime']:.2%}")
    
    df_operations = pd.DataFrame(operations_data)
    
    overall_quality = df_operations['quality_rate'].mean()
    total_production = df_operations['total_production'].sum()
    total_defects = df_operations['defective_products'].sum()
    
    print(f"\n--- Factory Simulation Summary ---")
    print(f"Total Production: {total_production} units")
    print(f"Overall Quality Rate: {overall_quality:.2%}")
    print(f"Total Defects: {total_defects}")
    print(f"Average Equipment Uptime: {df_operations['equipment_uptime'].mean():.2%}")
    print(f"Average Daily Production: {df_operations['total_production'].mean():.0f} units")
    
    return df_operations

if __name__ == "__main__":
    simulate_factory_operations(7)