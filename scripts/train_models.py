import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.predictive_maintenance import PredictiveMaintenance
from core.quality_control import QualityControl
from core.production_optimizer import ProductionOptimizer
from data.sensor_processor import SensorDataProcessor
from data.image_processor import ImageDataProcessor
from utils.config import Config
import torch

def main():
    config = Config()
    
    print("Generating training data...")
    sensor_processor = SensorDataProcessor(config.get('data', {}))
    image_processor = ImageDataProcessor(config.get('data', {}))
    
    sensor_data = sensor_processor.simulate_sensor_data(5000)
    processed_sensor_data = sensor_processor.preprocess_sensor_data(sensor_data)
    
    print("Training predictive maintenance model...")
    pm_model = PredictiveMaintenance(config.get('predictive_maintenance', {}))
    pm_model.train(processed_sensor_data, 'failure_risk', epochs=100)
    
    torch.save(pm_model.model.state_dict(), 'models/predictive_maintenance.pth')
    
    print("Training quality control model...")
    qc_model = QualityControl(config.get('quality_control', {}))
    training_images = image_processor.simulate_product_images(1000)
    
    torch.save(qc_model.model.state_dict(), 'models/quality_control.pth')
    
    print("Training production optimizer...")
    historical_data = sensor_data.copy()
    historical_data['production_rate'] = np.random.normal(100, 10, len(historical_data))
    historical_data['quality_score'] = np.random.uniform(0.8, 1.0, len(historical_data))
    
    optimizer = ProductionOptimizer(config.get('production_optimization', {}))
    optimizer.train_models(historical_data)
    
    print("All models trained successfully!")

if __name__ == "__main__":
    main()