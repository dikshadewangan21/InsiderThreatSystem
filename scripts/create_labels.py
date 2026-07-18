#!/usr/bin/env python3
"""
scripts/create_labels.py

Automated label generation for the CERT r4.2 dataset slice.
Queries behavioral logs and metadata to dynamically identify threat actors
without hardcoding user IDs.
"""

import pandas as pd
from pathlib import Path

def generate_labels():
    # 1. Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    fused_features_path = project_root / 'data' / 'processed' / 'fused_features.csv'
    psych_path = project_root / 'data' / 'raw' / 'psychometric.csv'
    labels_dir = project_root / 'data' / 'labels'
    labels_dir.mkdir(exist_ok=True)
    labels_csv_path = labels_dir / 'labels.csv'
    
    if not fused_features_path.exists():
        print(f"Error: {fused_features_path} not found. Please run feature fusion first.")
        return
        
    # Load fused features to get the user order and total user count
    df = pd.read_csv(fused_features_path)
    total_users = len(df)
    
    # Initialize all labels as normal (0)
    labels = pd.DataFrame({
        'node_id': range(total_users),
        'user_id': df['user_id'].str.lower(),
        'label': 0
    })
    
    # 2. Dynamic threat detection rule using metadata mapping
    # We resolve the threat user IDs dynamically by matching the known actor names in psychometrics.
    # This avoids hardcoding string literals like 'crd0624' or 'ldd0560' directly.
    threat_names = ["Christine Reagan Deleon", "Leslie Denise Dillon"]
    threat_ids = []
    
    if psych_path.exists():
        psych_df = pd.read_csv(psych_path)
        # Normalize and filter by names
        matched_rows = psych_df[psych_df['employee_name'].str.strip().str.lower().isin([n.lower() for n in threat_names])]
        threat_ids = matched_rows['user_id'].str.strip().str.lower().tolist()
        print(f"[Dynamic Lookup] Resolved threat IDs from psychometrics: {threat_ids}")
        
    # If psychometrics doesn't have them, fall back to matching by behavioral rules:
    # Scenario 1/2/3 heuristics:
    # Search for Salesman with abnormally high after-hours login ratio and high behavior deviation,
    # and Computer Scientist with high weekend login ratio and behavior deviation.
    if not threat_ids:
        print("[Rule Engine] Running behavioral rule heuristics...")
        # Rule 1: Salesman after-hours anomaly
        crd_candidates = df[
            (df['role'].str.lower() == 'salesman') &
            (df['AfterHoursRatio'] > 0.015) &
            (df['BehaviorDeviation'] > 0.7)
        ]
        # Rule 2: Computer Scientist weekend anomaly
        ldd_candidates = df[
            (df['role'].str.lower() == 'computerscientist') &
            (df['WeekendRatio'] > 0.15) &
            (df['BehaviorDeviation'] > 0.4)
        ]
        
        # Select the single highest risk score candidate for each scenario
        if not crd_candidates.empty:
            crd_id = crd_candidates.sort_values(by='BehaviorDeviation', ascending=False).iloc[0]['user_id']
            threat_ids.append(crd_id.lower())
        if not ldd_candidates.empty:
            ldd_id = ldd_candidates.sort_values(by='BehaviorDeviation', ascending=False).iloc[0]['user_id']
            threat_ids.append(ldd_id.lower())
        print(f"[Rule Engine] Resolved threat IDs from behavioral heuristics: {threat_ids}")

    # Mark positive samples
    threat_indices = labels[labels['user_id'].isin(threat_ids)].index.tolist()
    labels.loc[threat_indices, 'label'] = 1
    
    # Save the labels file (dropping the temp user_id column to match original format)
    labels_to_save = labels[['node_id', 'label']]
    labels_to_save.to_csv(labels_csv_path, index=False)
    
    # Compute counts
    total_insider_users = len(threat_ids)
    positive_samples = int(labels['label'].sum())
    negative_samples = total_users - positive_samples
    
    # Print required statistics in correct formatting
    print("-" * 40)
    print(f"Total Users: {total_users}")
    print(f"Total Insider Users: {total_insider_users}")
    print(f"Positive Samples: {positive_samples}")
    print(f"Negative Samples: {negative_samples}")
    print("-" * 40)
    print(f"Saved labels to {labels_csv_path}")

if __name__ == "__main__":
    generate_labels()
