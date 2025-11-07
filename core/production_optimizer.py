import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

class ProductionOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train_models(self, historical_data: pd.DataFrame):
        features = ['machine_speed', 'temperature', 'pressure', 'material_flow', 'vibration']
        target_performance = 'production_rate'
        target_quality = 'quality_score'
        
        X = historical_data[features]
        y_performance = historical_data[target_performance]
        y_quality = historical_data[target_quality]
        
        self.performance_model.fit(X, y_performance)
        self.quality_model.fit(X, y_quality)
        self.is_trained = True
        
    def optimize_production(self, current_conditions: Dict[str, float], 
                          constraints: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        def objective(x):
            params = {
                'machine_speed': x[0],
                'temperature': x[1],
                'pressure': x[2],
                'material_flow': x[3],
                'vibration': x[4]
            }
            
            production_rate = self.performance_model.predict([list(params.values())])[0]
            quality_score = self.quality_model.predict([list(params.values())])[0]
            
            combined_score = production_rate * quality_score
            return -combined_score
        
        bounds = [
            (constraints['machine_speed_min'], constraints['machine_speed_max']),
            (constraints['temperature_min'], constraints['temperature_max']),
            (constraints['pressure_min'], constraints['pressure_max']),
            (constraints['material_flow_min'], constraints['material_flow_max']),
            (constraints['vibration_min'], constraints['vibration_max'])
        ]
        
        x0 = [
            current_conditions['machine_speed'],
            current_conditions['temperature'],
            current_conditions['pressure'],
            current_conditions['material_flow'],
            current_conditions['vibration']
        ]
        
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        optimized_params = {
            'machine_speed': result.x[0],
            'temperature': result.x[1],
            'pressure': result.x[2],
            'material_flow': result.x[3],
            'vibration': result.x[4]
        }
        
        predicted_production = -result.fun
        predicted_quality = self.quality_model.predict([result.x])[0]
        
        improvement = (predicted_production - 
                      current_conditions.get('current_production', predicted_production)) / current_conditions.get('current_production', 1)
        
        return {
            'optimized_parameters': optimized_params,
            'predicted_production_rate': float(predicted_production),
            'predicted_quality_score': float(predicted_quality),
            'expected_improvement': float(improvement),
            'success': result.success
        }