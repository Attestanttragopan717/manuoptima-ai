import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetector:
    def __init__(self, input_dim: int):
        self.model = Autoencoder(input_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.threshold = None
        
    def compute_reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(data).to(self.device)
            reconstructed = self.model(tensor_data)
            error = torch.mean((tensor_data - reconstructed) ** 2, dim=1)
            return error.cpu().numpy()
    
    def detect_anomalies(self, sensor_data: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        errors = self.compute_reconstruction_error(sensor_data)
        
        if self.threshold is None:
            self.threshold = np.percentile(errors, 100 * (1 - contamination))
        
        anomalies = errors > self.threshold
        return anomalies