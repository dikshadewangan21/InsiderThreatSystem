import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Tuple, List, Dict
import pickle
import json
import os

class StratifiedSplitter:
    """
    Creates stratified train/val/test splits that preserve class distribution
    across all folds
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.15, 
                 val_size: float = 0.15, random_state: int = 42):
        """
        Args:
            n_splits: Number of folds (default 5)
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation (after test split)
            random_state: For reproducibility
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.splits = []
        self.split_stats = {}
    
    def create_splits(self, user_ids: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """
        Creates stratified k-fold splits
        
        Args:
            user_ids: Array of user identifiers (shape: N,)
            labels: Binary labels 0=benign, 1=malicious (shape: N,)
        
        Returns:
            List of dicts with keys: fold, train_idx, val_idx, test_idx, stats
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                             random_state=self.random_state)
        
        user_ids = np.array(user_ids)
        labels = np.array(labels)
        
        all_splits = []
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(
            skf.split(user_ids, labels)
        ):
            # Split train+val into train and val (maintaining stratification)
            train_val_labels = labels[train_val_idx]
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.val_size / (1 - self.test_size),  # Adjust for already-removed test
                stratify=train_val_labels,
                random_state=self.random_state + fold_idx
            )
            
            split_dict = {
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
            }
            
            # Compute statistics
            stats = self._compute_split_stats(
                user_ids, labels, train_idx, val_idx, test_idx
            )
            split_dict['stats'] = stats
            
            all_splits.append(split_dict)
        
        self.splits = all_splits
        return all_splits
    
    def _compute_split_stats(self, user_ids: np.ndarray, labels: np.ndarray,
                            train_idx: np.ndarray, val_idx: np.ndarray, 
                            test_idx: np.ndarray) -> Dict:
        """Compute statistics for each fold"""
        
        def stats_for_indices(idx):
            n_total = len(idx)
            n_pos = labels[idx].sum()
            pos_ratio = n_pos / n_total if n_total > 0 else 0
            return {
                'n_samples': n_total,
                'n_positive': int(n_pos),
                'n_negative': int(n_total - n_pos),
                'positive_ratio': float(pos_ratio)
            }
        
        return {
            'train': stats_for_indices(train_idx),
            'val': stats_for_indices(val_idx),
            'test': stats_for_indices(test_idx)
        }
    
    def print_split_report(self):
        """Pretty-print split statistics"""
        print("\n" + "="*80)
        print("STRATIFIED K-FOLD CROSS-VALIDATION REPORT")
        print("="*80)
        
        for split in self.splits:
            fold = split['fold']
            stats = split['stats']
            
            print(f"\n[Fold] FOLD {fold}")
            print("-" * 80)
            
            for split_type in ['train', 'val', 'test']:
                s = stats[split_type]
                print(f"  {split_type.upper():6s}: {s['n_samples']:5d} users | "
                      f"{s['n_positive']:3d} malicious "
                      f"({s['positive_ratio']*100:4.1f}%) | "
                      f"{s['n_negative']:5d} benign")
        
        # Overall statistics
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("-" * 80)
        
        all_train_stats = [s['stats']['train'] for s in self.splits]
        train_pos_ratios = [s['positive_ratio'] for s in all_train_stats]
        
        print(f"All folds preserve positive ratio: "
              f"{np.mean(train_pos_ratios)*100:.1f}% +/- {np.std(train_pos_ratios)*100:.1f}%")
        
        # Minimum positives per fold
        min_val_pos = min([s['stats']['val']['n_positive'] for s in self.splits])
        min_test_pos = min([s['stats']['test']['n_positive'] for s in self.splits])
        
        print(f"Minimum positives in validation: {min_val_pos}")
        print(f"Minimum positives in test: {min_test_pos}")
        
        if min_val_pos >= 1 and min_test_pos >= 1:
            print("[Status] All folds have sufficient positives for evaluation")
        else:
            print("[Warning] Some folds have <1 positive example")
    
    def save_splits(self, output_dir: str):
        """Save splits to disk for reproducibility"""
        os.makedirs(output_dir, exist_ok=True)
        
        for split in self.splits:
            fold = split['fold']
            fold_dict = {
                'train_idx': split['train_idx'].tolist(),
                'val_idx': split['val_idx'].tolist(),
                'test_idx': split['test_idx'].tolist()
            }
            
            with open(os.path.join(output_dir, f"fold_{fold}.pkl"), 'wb') as f:
                pickle.dump(fold_dict, f)
        
        # Also save JSON report
        report = {
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'splits': []
        }
        
        for split in self.splits:
            report['splits'].append({
                'fold': split['fold'],
                'stats': split['stats']
            })
        
        with open(os.path.join(output_dir, "split_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[Save] Splits saved to {output_dir}")

if __name__ == "__main__":
    import sys
    labels_csv = "data/labels/cert_r4.2_labels.csv"
    if os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        user_ids = df['user_id'].values
        labels = df['label'].values
        splitter = StratifiedSplitter(n_splits=5)
        splits = splitter.create_splits(user_ids, labels)
        splitter.print_split_report()
        splitter.save_splits("data/splits")
    else:
        print(f"[Error] Labels file {labels_csv} not found. Run label extractor first.")
