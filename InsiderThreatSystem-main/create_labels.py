#!/usr/bin/env python3
"""Create labels file with integer node indices."""
import pandas as pd

# Read fused features to get the user order
df = pd.read_csv('data/processed/fused_features.csv')

# Create labels with integer indices (0 to len(df)-1)
labels = pd.DataFrame({
    'node_id': range(len(df)),
    'label': 0  # Initialize all as 0 (normal)
})

# Assign label 1 to the threat users
threat_users = ['crd0624', 'ldd0560']
threat_indices = df[df['user_id'].str.lower().isin(threat_users)].index.tolist()
labels.loc[threat_indices, 'label'] = 1

labels.to_csv('data/labels/labels.csv', index=False)
print(f"Created labels.csv with {len(labels)} users (node_id: 0 to {len(df)-1})")
print(f"Labeled threat users at indices {threat_indices} as 1.")

