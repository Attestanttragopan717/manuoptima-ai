import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from typing import Dict, List, Any

class DefectDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class QualityControl:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DefectDetector()
        self.defect_threshold = config.get('defect_threshold', 0.7)
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.FloatTensor(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def detect_defects(self, image: np.ndarray) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            tensor = self.preprocess_image(image)
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            defect_prob = probabilities[0][1].item()
            
            has_defect = defect_prob > self.defect_threshold
            confidence = defect_prob if has_defect else 1 - defect_prob
            
            return {
                'has_defect': has_defect,
                'defect_confidence': float(defect_prob),
                'overall_confidence': float(confidence),
                'defect_type': self.classify_defect_type(image) if has_defect else 'none'
            }
    
    def classify_defect_type(self, image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contour_area_ratio = np.sum(edges) / (image.shape[0] * image.shape[1])
        
        if contour_area_ratio > 0.1:
            return 'crack'
        else:
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            var = np.var(blur)
            if var < 100:
                return 'discoloration'
            else:
                return 'surface_imperfection'
    
    def analyze_production_batch(self, images: List[np.ndarray]) -> Dict[str, Any]:
        results = [self.detect_defects(img) for img in images]
        defect_count = sum(1 for r in results if r['has_defect'])
        total_count = len(images)
        
        quality_score = 1 - (defect_count / total_count)
        
        return {
            'batch_quality_score': float(quality_score),
            'defect_count': defect_count,
            'total_products': total_count,
            'defect_rate': defect_count / total_count,
            'individual_results': results
        }