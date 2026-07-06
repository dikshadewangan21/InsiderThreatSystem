#!/usr/bin/env python3
"""Verify all backward compatibility imports work correctly."""

import sys

def verify_imports():
    """Test that all backward compatibility aliases work."""
    errors = []
    
    # Test 1: TGN from tgn_model
    try:
        from models.tgn_model import TGN
        print("✓ from models.tgn_model import TGN")
    except Exception as e:
        errors.append(f"Failed to import TGN: {e}")
        print(f"✗ from models.tgn_model import TGN - ERROR: {e}")
    
    # Test 2: GAT from gat_model
    try:
        from models.gat_model import GAT
        print("✓ from models.gat_model import GAT")
    except Exception as e:
        errors.append(f"Failed to import GAT: {e}")
        print(f"✗ from models.gat_model import GAT - ERROR: {e}")
    
    # Test 3: MLPClassifier from mlp_classifier
    try:
        from models.mlp_classifier import MLPClassifier
        print("✓ from models.mlp_classifier import MLPClassifier")
    except Exception as e:
        errors.append(f"Failed to import MLPClassifier: {e}")
        print(f"✗ from models.mlp_classifier import MLPClassifier - ERROR: {e}")
    
    # Test 4: Verify they're the right classes
    if 'TGN' in dir():
        from models.tgn_model import TemporalGraphNetwork
        if TGN is TemporalGraphNetwork:
            print("✓ TGN is TemporalGraphNetwork")
        else:
            errors.append("TGN is not TemporalGraphNetwork")
            print("✗ TGN is not TemporalGraphNetwork")
    
    if 'GAT' in dir():
        from models.gat_model import ProductionGAT
        if GAT is ProductionGAT:
            print("✓ GAT is ProductionGAT")
        else:
            errors.append("GAT is not ProductionGAT")
            print("✗ GAT is not ProductionGAT")
    
    if 'MLPClassifier' in dir():
        from models.mlp_classifier import MLPRiskClassifier
        if MLPClassifier is MLPRiskClassifier:
            print("✓ MLPClassifier is MLPRiskClassifier")
        else:
            errors.append("MLPClassifier is not MLPRiskClassifier")
            print("✗ MLPClassifier is not MLPRiskClassifier")
    
    # Test 5: Verify training/train.py can import them
    try:
        import training.train
        print("✓ training.train imports successfully")
    except Exception as e:
        errors.append(f"training.train import failed: {e}")
        print(f"✗ training.train import failed: {e}")
    
    print()
    if errors:
        print(f"ERRORS: {len(errors)}")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("SUCCESS: All imports verified!")
        sys.exit(0)

if __name__ == "__main__":
    verify_imports()
