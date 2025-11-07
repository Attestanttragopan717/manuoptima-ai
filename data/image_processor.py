import cv2
import numpy as np
from typing import List

class ImageDataProcessor:
    def __init__(self, config: dict):
        self.config = config
        
    def simulate_product_images(self, num_images: int = 100) -> List[np.ndarray]:
        images = []
        image_size = (256, 256)
        
        for i in range(num_images):
            if np.random.random() < 0.2:
                img = self.create_defective_image(image_size)
            else:
                img = self.create_normal_image(image_size)
            images.append(img)
        
        return images
    
    def create_normal_image(self, size: tuple) -> np.ndarray:
        img = np.random.normal(128, 10, (*size, 3)).astype(np.uint8)
        return cv2.GaussianBlur(img, (5, 5), 0)
    
    def create_defective_image(self, size: tuple) -> np.ndarray:
        img = self.create_normal_image(size)
        
        defect_type = np.random.choice(['crack', 'discoloration', 'scratch'])
        
        if defect_type == 'crack':
            points = np.random.randint(0, size[0], (10, 2))
            for j in range(len(points)-1):
                cv2.line(img, tuple(points[j]), tuple(points[j+1]), (0, 0, 0), 2)
        elif defect_type == 'discoloration':
            x, y = np.random.randint(0, size[0]-50, 2)
            img[y:y+50, x:x+50] = np.random.normal(200, 5, (50, 50, 3)).astype(np.uint8)
        else:
            x1, y1 = np.random.randint(0, size[0], 2)
            x2, y2 = np.random.randint(0, size[0], 2)
            cv2.line(img, (x1, y1), (x2, y2), (50, 50, 50), 3)
        
        return img