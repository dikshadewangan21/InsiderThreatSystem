import torch
import torch.nn as nn
from typing import Dict, List, Callable, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, auc, precision_recall_curve
import numpy as np
from scipy import stats
import os

class AblationStudy:
    """Framework for systematic ablation studies"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.results = None
    
    def register_model(self, name: str, model_fn: Callable, description: str):
        """Register a model variant"""
        self.models[name] = {
            'fn': model_fn,
            'description': description
        }
    
    def train_all_variants(self, X_train: Dict, y_train: np.ndarray,
                          X_val: Dict, y_val: np.ndarray,
                          X_test: Dict, y_test: np.ndarray,
                          n_folds: int = 5) -> pd.DataFrame:
        """
        Train all registered model variants
        """
        results = []
        
        for model_name, model_info in self.models.items():
            print(f"\n[Ablation] Training variant: {model_name}")
            print(f"   Description: {model_info['description']}")
            
            fold_results = []
            
            for seed in range(42, 42 + n_folds):
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Retrieve and initialize model
                model = model_info['fn'](device=self.device)
                
                # Mock training loop with simple simulated optimizer step for NN or standard fit for sklearn
                if isinstance(model, LogisticRegression):
                    model.fit(X_train['behavioral'], y_train)
                    test_preds = model.predict_proba(X_test['behavioral'])[:, 1]
                else:
                    # Neural Network variant training
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    criterion = nn.BCEWithLogitsLoss()
                    
                    # Convert inputs to tensors
                    x_train_t = torch.tensor(X_train['behavioral'], dtype=torch.float32, device=self.device)
                    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
                    x_test_t = torch.tensor(X_test['behavioral'], dtype=torch.float32, device=self.device)
                    
                    model.train()
                    for epoch in range(15):
                        optimizer.zero_grad()
                        outputs = model(x_train_t).squeeze()
                        loss = criterion(outputs, y_train_t)
                        loss.backward()
                        optimizer.step()
                        
                    model.eval()
                    with torch.no_grad():
                        test_preds = torch.sigmoid(model(x_test_t)).squeeze().cpu().numpy()
                
                metrics = self._compute_metrics(y_test, test_preds)
                metrics['seed'] = seed
                metrics['model'] = model_name
                
                fold_results.append(metrics)
            
            fold_df = pd.DataFrame(fold_results)
            for metric in ['AUPRC', 'F1', 'Accuracy', 'ROC-AUC']:
                mean = fold_df[metric].mean()
                std = fold_df[metric].std()
                print(f"   {metric:12s}: {mean:.4f} +/- {std:.4f}")
            
            results.extend(fold_results)
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        threshold = 0.5
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        accuracy = (y_pred_binary == y_true).mean()
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_pred)
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
            auprc = auc(recall_vals, precision_vals)
        else:
            roc_auc = 0.5
            auprc = 0.5
        
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        tn = ((y_pred_binary == 0) & (y_true == 0)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
        
        mcc_denom = np.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn - fp*fn) / mcc_denom if mcc_denom > 0 else 0
        
        return {
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'ROC-AUC': float(roc_auc),
            'AUPRC': float(auprc),
            'MCC': float(mcc)
        }
    
    def print_ablation_results(self):
        print("\n" + "="*100)
        print("ABLATION STUDY RESULTS")
        print("="*100)
        
        for model_name in self.results['model'].unique():
            model_results = self.results[self.results['model'] == model_name]
            
            print(f"\n[Model] {model_name}")
            print("-" * 100)
            
            for metric in ['AUPRC', 'F1', 'Accuracy', 'ROC-AUC', 'MCC']:
                mean = model_results[metric].mean()
                std = model_results[metric].std()
                ci_low = mean - 1.96 * std / np.sqrt(len(model_results))
                ci_high = mean + 1.96 * std / np.sqrt(len(model_results))
                
                print(f"  {metric:12s}: {mean:.4f} +/- {std:.4f}  "
                      f"[95% CI: {ci_low:.4f}, {ci_high:.4f}]")
        
        print("\n" + "="*100)
        print("COMPARISON TABLE")
        print("="*100)
        comparison = self.results.groupby('model').agg({
            'AUPRC': ['mean', 'std'],
            'F1': ['mean', 'std'],
            'Accuracy': ['mean', 'std'],
            'ROC-AUC': ['mean', 'std']
        })
        print(comparison.to_string())
    
    def statistical_significance_test(self):
        print("\n" + "="*100)
        print("STATISTICAL SIGNIFICANCE TESTS (t-test, alpha=0.05)")
        print("="*100)
        
        models = self.results['model'].unique()
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                auprc_a = self.results[self.results['model'] == model_a]['AUPRC'].values
                auprc_b = self.results[self.results['model'] == model_b]['AUPRC'].values
                
                t_stat, p_value = stats.ttest_ind(auprc_a, auprc_b)
                mean_diff = auprc_a.mean() - auprc_b.mean()
                pooled_std = np.sqrt(
                    ((len(auprc_a)-1)*auprc_a.std()**2 + 
                     (len(auprc_b)-1)*auprc_b.std()**2) / 
                    (len(auprc_a) + len(auprc_b) - 2)
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                significant = "YES" if p_value < 0.05 else "NO"
                
                print(f"\n{model_a} vs {model_b}:")
                print(f"  Delta AUPRC: {mean_diff:+.4f}")
                print(f"  t-stat: {t_stat:.4f}, p-value: {p_value:.4f} (Significant: {significant})")
                print(f"  Cohen's d: {cohens_d:.4f} (effect size)")

if __name__ == "__main__":
    # Self-test using mock inputs
    X_tr = {'behavioral': np.random.randn(100, 20)}
    y_tr = np.random.randint(0, 2, size=100)
    X_te = {'behavioral': np.random.randn(50, 20)}
    y_te = np.random.randint(0, 2, size=50)
    
    ablation = AblationStudy(device='cpu')
    
    # Register baseline variants
    ablation.register_model('A_LogReg', lambda device: LogisticRegression(max_iter=1000, random_state=42), 
                           'Logistic Regression on hand-crafted features')
    
    class SimpleNN(nn.Module):
        def __init__(self, device='cpu'):
            super().__init__()
            self.fc1 = nn.Linear(20, 1)
        def forward(self, x):
            return self.fc1(x)
        def predict_proba(self, x):
            self.eval()
            with torch.no_grad():
                return torch.sigmoid(self(torch.tensor(x, dtype=torch.float32))).numpy()
                
    ablation.register_model('B_SimpleNN', lambda device: SimpleNN(device), 'Simple NN baseline')
    
    results = ablation.train_all_variants(X_tr, y_tr, X_te, y_tr, X_te, y_te, n_folds=5)
    ablation.print_ablation_results()
    ablation.statistical_significance_test()
