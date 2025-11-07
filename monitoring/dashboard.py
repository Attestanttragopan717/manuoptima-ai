import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any

class MonitoringDashboard:
    def __init__(self):
        self.figures = {}
        
    def create_equipment_health_dashboard(self, sensor_data: pd.DataFrame, 
                                        predictions: List[Dict]) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Temperature Trend', 'Vibration Levels', 
                          'Pressure Monitoring', 'Motor Current',
                          'Failure Risk Prediction', 'Equipment Health Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=sensor_data['timestamp'], y=sensor_data['temperature'],
                      name='Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sensor_data['timestamp'], y=sensor_data['vibration'],
                      name='Vibration', line=dict(color='blue')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=sensor_data['timestamp'], y=sensor_data['pressure'],
                      name='Pressure', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sensor_data['timestamp'], y=sensor_data['motor_current'],
                      name='Motor Current', line=dict(color='orange')),
            row=2, col=2
        )
        
        risk_timestamps = [p['timestamp'] for p in predictions]
        risk_values = [p['failure_probability'] for p in predictions]
        
        fig.add_trace(
            go.Scatter(x=risk_timestamps, y=risk_values,
                      name='Failure Risk', line=dict(color='purple')),
            row=3, col=1, secondary_y=False
        )
        
        health_scores = [1 - risk for risk in risk_values]
        fig.add_trace(
            go.Scatter(x=risk_timestamps, y=health_scores,
                      name='Health Score', line=dict(color='green', dash='dash')),
            row=3, col=1, secondary_y=True
        )
        
        alert_counts = [1 if p['alert_level'] == 'HIGH' else 0 for p in predictions]
        fig.add_trace(
            go.Bar(x=risk_timestamps, y=alert_counts, name='High Alerts',
                  marker_color='red'),
            row=3, col=2
        )
        
        fig.update_layout(height=800, title_text="Real-time Equipment Monitoring Dashboard")
        return fig
    
    def create_quality_control_dashboard(self, quality_results: Dict[str, Any]) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Batch Quality Score', 'Defect Distribution',
                          'Defect Types', 'Quality Trend'),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_results['batch_quality_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                delta={'reference': 0.9},
                gauge={'axis': {'range': [0, 1]},
                      'steps': [{'range': [0, 0.8], 'color': "lightgray"},
                               {'range': [0.8, 0.9], 'color': "yellow"},
                               {'range': [0.9, 1], 'color': "green"}]}),
            row=1, col=1
        )
        
        defect_count = quality_results['defect_count']
        good_count = quality_results['total_products'] - defect_count
        
        fig.add_trace(
            go.Pie(labels=['Good Products', 'Defective Products'],
                  values=[good_count, defect_count],
                  name="Product Quality"),
            row=1, col=2
        )
        
        if quality_results['individual_results']:
            defect_types = [r['defect_type'] for r in quality_results['individual_results'] if r['has_defect']]
            type_counts = pd.Series(defect_types).value_counts()
            
            fig.add_trace(
                go.Bar(x=type_counts.index, y=type_counts.values,
                      name="Defect Types"),
                row=2, col=1
            )
        
        fig.update_layout(height=600, title_text="Quality Control Dashboard")
        return fig