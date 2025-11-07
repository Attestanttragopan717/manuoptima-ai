import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class PredictiveMaintenance:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.sequence_length = config.get('sequence_length', 50)
        
    def prepare_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        features = data.drop(columns=[target_col])
        targets = data[target_col].values
        
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(targets[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_size: int, output_size: int = 1) -> nn.Module:
        hidden_size = self.config.get('hidden_size', 64)
        num_layers = self.config.get('num_layers', 2)
        dropout = self.config.get('dropout', 0.2)
        
        model = LSTMForecaster(input_size, hidden_size, num_layers, output_size, dropout)
        return model.to(self.device)
    
    def train(self, train_data: pd.DataFrame, target_col: str, epochs: int = 100):
        X_train, y_train = self.prepare_sequences(train_data, target_col)
        
        input_size = X_train.shape[2]
        self.model = self.build_model(input_size)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    def predict_failure(self, sensor_data: pd.DataFrame, threshold: float = 0.8) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        with torch.no_grad():
            features = sensor_data.drop(columns=['timestamp'], errors='ignore')
            scaled_features = self.scaler.transform(features)
            
            if len(scaled_features) < self.sequence_length:
                padded = np.zeros((self.sequence_length, scaled_features.shape[1]))
                padded[-len(scaled_features):] = scaled_features
                scaled_features = padded
            
            sequence = torch.FloatTensor(scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)).to(self.device)
            prediction = self.model(sequence).cpu().numpy()[0][0]
            
            failure_probability = self.sigmoid(prediction)
            alert_level = 'HIGH' if failure_probability > threshold else 'MEDIUM' if failure_probability > 0.5 else 'LOW'
            
            return {
                'failure_probability': float(failure_probability),
                'alert_level': alert_level,
                'predicted_value': float(prediction),
                'timestamp': pd.Timestamp.now()
            }
    
    def sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))