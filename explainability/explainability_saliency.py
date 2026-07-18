import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os

class ExplainabilityEngine:
    """Gradient-based explanations for insider threat predictions"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def compute_saliency(self, x_input: torch.Tensor, y_target: float,
                        feature_names: List[str]) -> Dict:
        """
        Compute gradient-based saliency: d_y_hat / d_x
        """
        self.model.eval()
        x_input = x_input.clone().detach().to(self.device)
        x_input.requires_grad = True
        
        # Forward pass
        logit = self.model(x_input)
        
        # Backward pass
        logit.backward()
        
        # Get gradients
        saliency = torch.abs(x_input.grad).squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        saliency_sum = saliency.sum()
        saliency_normalized = saliency / (saliency_sum + 1e-8)
        
        # Get top-5 features
        top_indices = np.argsort(saliency_normalized)[-5:][::-1]
        top_features = [(feature_names[i], float(saliency_normalized[i])) 
                       for i in top_indices if i < len(feature_names)]
        
        return {
            'saliency': saliency_normalized.tolist(),
            'top_5_features': top_features
        }
    
    def generate_explanations(self, users: pd.DataFrame, predictions: np.ndarray,
                             feature_cols: List[str], risk_threshold: float = 0.001) -> List[Dict]:
        """
        Generate explanations for High/Critical alerts
        """
        explanations = []
        
        for idx, (row_idx, row) in enumerate(users.iterrows()):
            pred = predictions[idx]
            
            if pred >= risk_threshold:
                # Create input tensor
                feat_vals = row[feature_cols].values.astype(np.float32)
                features = torch.tensor(feat_vals, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Compute saliency
                saliency_info = self.compute_saliency(
                    features, float(pred), feature_names=feature_cols
                )
                
                explanations.append({
                    'user_id': row.get('user_id', f'user_{idx}'),
                    'risk_score': float(pred),
                    'top_1_feature': saliency_info['top_5_features'][0][0] if saliency_info['top_5_features'] else 'None',
                    'top_5_features': [f[0] for f in saliency_info['top_5_features']],
                    'feature_weights': [f[1] for f in saliency_info['top_5_features']]
                })
        
        return explanations


class PsychologyFeatureValidator:
    """Validate that psychological features contribute meaningfully"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_saliency_distribution(self, explanations: List[Dict],
                                     n_alerts: int = 200) -> pd.DataFrame:
        """
        Analyze which features appear in top-5 explanations
        """
        feature_occurrence = {}
        
        for expl in explanations[:n_alerts]:
            for feature in expl['top_5_features']:
                feature_occurrence[feature] = feature_occurrence.get(feature, 0) + 1
        
        # Convert to percentages
        total_occurrences = sum(feature_occurrence.values())
        if total_occurrences > 0:
            feature_pct = {k: (v / total_occurrences) * 100 for k, v in feature_occurrence.items()}
        else:
            feature_pct = {}
            
        # Rank by occurrence
        ranked = sorted(feature_pct.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(ranked, columns=['Feature', 'Occurrence (%)'])
        
        # Add actionability rating (analyst-determined mapping)
        actionability_map = {
            'file_sensitivity': 5,
            'behavioral_deviat': 4,
            'login_time_deviat': 3,
            'communication_entropy': 3,
            'psychology_score': 1
        }
        
        df['Actionability'] = df['Feature'].map(actionability_map).fillna(3).astype(int)
        
        print("\n" + "="*80)
        print("SALIENCY ANALYSIS: Feature Actionability")
        print("="*80)
        print(df.to_string(index=False))
        
        return df

if __name__ == "__main__":
    # Self-test explainability engine
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 1)
            # Give psychology and file sensitivity high weights
            self.linear.weight.data = torch.tensor([[0.1, 1.2, 0.4, 0.3, 2.5]])
        def forward(self, x):
            return self.linear(x)
            
    model = ToyModel()
    engine = ExplainabilityEngine(model, device='cpu')
    
    # 5 test features
    feat_cols = ['behavioral_deviat', 'file_sensitivity', 'login_time_deviat', 'communication_entropy', 'psychology_score']
    toy_users = pd.DataFrame([
        [0.1, 0.8, 0.2, 0.1, 0.9, 'usr001'],
        [0.2, 0.9, 0.1, 0.3, 0.8, 'usr002']
    ], columns=feat_cols + ['user_id'])
    
    preds = np.array([0.05, 0.08])
    
    explanations = engine.generate_explanations(toy_users, preds, feat_cols, risk_threshold=0.01)
    print(f"Generated {len(explanations)} explanations:")
    print(explanations[0])
    
    validator = PsychologyFeatureValidator()
    validator.analyze_saliency_distribution(explanations)
