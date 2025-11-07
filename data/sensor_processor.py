import pandas as pd
import numpy as np
from typing import Dict, List, Any

class SensorDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def simulate_sensor_data(self, num_samples: int = 1000) -> pd.DataFrame:
        np.random.seed(42)
        
        timestamps = pd.date_range(start='2024-01-01', periods=num_samples, freq='H')
        
        data = {
            'timestamp': timestamps,
            'temperature': np.random.normal(75, 5, num_samples),
            'pressure': np.random.normal(100, 10, num_samples),
            'vibration': np.random.normal(2, 0.5, num_samples),
            'motor_current': np.random.normal(15, 2, num_samples),
            'rotation_speed': np.random.normal(1800, 50, num_samples),
            'bearing_temperature': np.random.normal(80, 8, num_samples)
        }
        
        df = pd.DataFrame(data)
        
        failure_indicator = self.simulate_failure_pattern(df)
        df['failure_risk'] = failure_indicator
        
        return df
    
    def simulate_failure_pattern(self, df: pd.DataFrame) -> np.ndarray:
        risk_scores = np.zeros(len(df))
        
        for i in range(len(df)):
            risk = 0
            risk += max(0, (df.loc[i, 'temperature'] - 85) / 10)
            risk += max(0, (df.loc[i, 'vibration'] - 3) / 1)
            risk += max(0, (df.loc[i, 'bearing_temperature'] - 90) / 10)
            
            if i > 10:
                recent_vibration = df.loc[i-10:i, 'vibration'].mean()
                risk += max(0, (recent_vibration - 2.5) / 1)
            
            risk_scores[i] = risk
        
        return risk_scores / np.max(risk_scores) if np.max(risk_scores) > 0 else risk_scores
    
    def preprocess_sensor_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        df = raw_data.copy()
        
        for column in ['temperature', 'pressure', 'vibration', 'motor_current', 'rotation_speed', 'bearing_temperature']:
            df[column] = df[column].interpolate(method='linear')
        
        df['vibration_rolling_mean'] = df['vibration'].rolling(window=5).mean()
        df['temperature_trend'] = df['temperature'].diff().rolling(window=10).mean()
        
        return df.dropna()