import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

class DataVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
        
    def plot_sensor_trends(self, sensor_data: pd.DataFrame, save_path: str = None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        sensors = ['temperature', 'pressure', 'vibration', 'motor_current', 'rotation_speed', 'bearing_temperature']
        
        for i, sensor in enumerate(sensors):
            ax = axes[i//3, i%3]
            ax.plot(sensor_data['timestamp'], sensor_data[sensor])
            ax.set_title(f'{sensor.replace("_", " ").title()} Trend')
            ax.set_xlabel('Time')
            ax.set_ylabel(sensor)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_quality_analysis(self, quality_results: Dict[str, Any], save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].pie([quality_results['defect_count'], 
                      quality_results['total_products'] - quality_results['defect_count']],
                     labels=['Defective', 'Good'], autopct='%1.1f%%')
        axes[0,0].set_title('Product Quality Distribution')
        
        axes[0,1].bar(['Quality Score'], [quality_results['batch_quality_score']])
        axes[0,1].set_ylim(0, 1)
        axes[0,1].set_title('Batch Quality Score')
        
        if quality_results['individual_results']:
            confidences = [r['defect_confidence'] for r in quality_results['individual_results']]
            axes[1,0].hist(confidences, bins=20, alpha=0.7)
            axes[1,0].set_title('Defect Confidence Distribution')
            axes[1,0].set_xlabel('Confidence')
            axes[1,0].set_ylabel('Count')
        
        axes[1,1].text(0.1, 0.5, f"Defect Rate: {quality_results['defect_rate']:.2%}\n"
                     f"Total Products: {quality_results['total_products']}\n"
                     f"Defects Found: {quality_results['defect_count']}",
                     fontsize=12, va='center')
        axes[1,1].set_title('Quality Summary')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()