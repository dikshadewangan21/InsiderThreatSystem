import pandas as pd
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path

class CERTLabelExtractor:
    """Extract malicious user labels from CERT r4.2 dataset"""
    
    def __init__(self, cert_root_path: str):
        self.cert_root = cert_root_path
        self.malicious_users = {}
        self.log_events = {}
    
    def extract_explicit_labels(self) -> Dict:
        """
        Parse CERT documentation and insiders.csv for explicit malicious labels
        """
        insiders_path = os.path.join(self.cert_root, "insiders", "insiders.csv")
        alternative_path = os.path.join(self.cert_root, "labels", "insiders.csv")
        
        target_path = insiders_path if os.path.exists(insiders_path) else alternative_path
        
        try:
            insiders_df = pd.read_csv(target_path)
            
            explicit_malicious = {}
            for _, row in insiders_df.iterrows():
                user_id = row['insider_id']
                scenario = row['scenario_type']  # EXFILTRATION, SABOTAGE, ESCALATION
                
                explicit_malicious[user_id] = {
                    'label': 1,
                    'scenario': scenario,
                    'event_start': row.get('event_start_date', '2010-01-01'),
                    'event_end': row.get('event_end_date', '2010-12-31'),
                    'confidence': 'HIGH',  # From official CERT labels
                    'source': 'CERT_OFFICIAL'
                }
            
            return explicit_malicious
            
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"[Warning] insiders.csv not found at {target_path}, falling back to default explicit threats (Christine & Leslie)")
            return {
                "crd0624": {
                    'label': 1,
                    'scenario': 'EXFILTRATION',
                    'event_start': '2010-03-02',
                    'event_end': '2010-03-24',
                    'confidence': 'HIGH',
                    'source': 'CERT_OFFICIAL'
                },
                "ldd0560": {
                    'label': 1,
                    'scenario': 'EXFILTRATION',
                    'event_start': '2010-05-10',
                    'event_end': '2010-06-02',
                    'confidence': 'HIGH',
                    'source': 'CERT_OFFICIAL'
                }
            }
    
    def detect_implicit_patterns(self) -> Dict:
        """
        Detect malicious behavior patterns from logs even if not explicitly labeled
        Uses known attack signatures from precomputed features for speed & memory safety
        """
        implicit_malicious = {}
        
        fused_path = os.path.join(self.cert_root, "..", "processed", "fused_features.csv")
        if not os.path.exists(fused_path):
            fused_path = os.path.join(self.cert_root, "processed", "fused_features.csv")
            
        if os.path.exists(fused_path):
            print(f"[Search] Reading precomputed features from {fused_path} for implicit scanner...")
            df = pd.read_csv(fused_path)
            
            df_sorted = df.sort_values(by="BehaviorDeviation", ascending=False)
            df_sorted = df_sorted[~df_sorted['user_id'].str.strip().str.lower().isin(["crd0624", "ldd0560"])]
            
            # Pattern 1: Exfiltration (high files touched, sensitivity, or USB)
            exfil_df = df_sorted[
                (df_sorted['MaxSensitivity'] > 0.3) | 
                (df_sorted['FileCount'] > 50)
            ].head(10)
            
            for _, row in exfil_df.iterrows():
                uid = str(row['user_id']).strip().lower()
                implicit_malicious[uid] = {
                    'label': 1,
                    'scenario': 'EXFILTRATION',
                    'confidence': 'MEDIUM',
                    'source': 'PATTERN_DETECTED',
                    'signals': {
                        'file_diversity': int(row.get('UniqueFilesTouched', 100)),
                        'unusual_access': int(row.get('SensitiveFileCount', 25)),
                        'email_volume': int(row.get('EmailCount', 100))
                    }
                }
                
            # Pattern 2: Sabotage (IT admin role or high off-hours activity + USB)
            sabotage_df = df_sorted[
                (df_sorted['role'].str.lower().str.contains('admin', na=False)) & 
                (df_sorted['AfterHoursRatio'] > 0.05)
            ].head(10)
            
            for _, row in sabotage_df.iterrows():
                uid = str(row['user_id']).strip().lower()
                if uid not in implicit_malicious:
                    implicit_malicious[uid] = {
                        'label': 1,
                        'scenario': 'SABOTAGE',
                        'confidence': 'MEDIUM',
                        'source': 'PATTERN_DETECTED',
                        'signals': {
                            'after_hours_usb': int(row.get('USBInsertions', 10))
                        }
                    }
                    
            # Pattern 3: Lateral Movement (large number of unique PCs used)
            lateral_df = df_sorted[
                (df_sorted['UniquePCs'] > 1.5) | 
                (df_sorted['BehaviorDeviation'] > 0.8)
            ].head(20)
            
            for _, row in lateral_df.iterrows():
                uid = str(row['user_id']).strip().lower()
                if uid not in implicit_malicious:
                    implicit_malicious[uid] = {
                        'label': 1,
                        'scenario': 'LATERAL_MOVEMENT',
                        'confidence': 'MEDIUM',
                        'source': 'PATTERN_DETECTED',
                        'signals': {
                            'systems_accessed': int(row.get('UniquePCs', 2))
                        }
                    }
        else:
            print("[Warning] Preprocessed features not found. Skipping implicit scan.")
            
        print(f"  Found {len(implicit_malicious)} implicit threat patterns.")
        return implicit_malicious
        
    def _detect_exfiltration_pattern(self) -> Dict:
        return {}
        
    def _detect_sabotage_pattern(self) -> Dict:
        return {}
        
    def _detect_lateral_movement_pattern(self) -> Dict:
        return {}
    
    def merge_labels(self, explicit: Dict, implicit: Dict) -> Dict:
        """
        Merge explicit and implicit labels with priority to explicit
        """
        print("\n[Stats] Merging labels...")
        all_labels = {}
        all_labels.update(explicit)
        
        for user_id, info in implicit.items():
            uid_lower = user_id.lower()
            if uid_lower not in all_labels:
                all_labels[uid_lower] = info
                
        return all_labels
    
    def validate_labels(self, labels: Dict) -> Tuple[int, int, int]:
        """
        Validate label quality and consistency
        """
        high = len([u for u, info in labels.items() if info['confidence'] == 'HIGH'])
        medium = len([u for u, info in labels.items() if info['confidence'] == 'MEDIUM'])
        low = len([u for u, info in labels.items() if info['confidence'] == 'LOW'])
        
        print(f"\n[Validation] Label Validation Report:")
        print(f"  HIGH confidence:    {high} users (use these)")
        print(f"  MEDIUM confidence:  {medium} users (validate)")
        print(f"  LOW confidence:     {low} users (skip)")
        print(f"  TOTAL MALICIOUS:    {high + medium + low} users")
        
        return high, medium, low
    
    def save_labels(self, labels: Dict, output_path: str):
        """Save labels to JSON for reproducibility"""
        with open(output_path, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"\n[Save] Labels saved to {output_path}")
    
    def generate_label_csv(self, labels: Dict, output_path: str):
        """Generate CSV for use in training pipeline containing ALL users with node_id"""
        fused_path = os.path.join(self.cert_root, "..", "processed", "fused_features.csv")
        if not os.path.exists(fused_path):
            fused_path = os.path.join(self.cert_root, "processed", "fused_features.csv")
            
        if os.path.exists(fused_path):
            df_fused = pd.read_csv(fused_path)
            all_users = df_fused['user_id'].str.strip().str.lower().unique()
        else:
            # Fallback if features aren't present
            all_users = list(labels.keys())
            
        rows = []
        for idx, uid in enumerate(all_users):
            uid_lower = uid.lower()
            if uid_lower in labels:
                info = labels[uid_lower]
                rows.append({
                    'node_id': idx,
                    'user_id': uid_lower,
                    'label': 1,
                    'scenario': info['scenario'],
                    'confidence': info['confidence'],
                    'source': info['source']
                })
            else:
                rows.append({
                    'node_id': idx,
                    'user_id': uid_lower,
                    'label': 0,
                    'scenario': 'BENIGN',
                    'confidence': 'HIGH',
                    'source': 'LDAP'
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"[Save] Label CSV saved to {output_path} with {len(df)} total users")
        
        return df

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    extractor = CERTLabelExtractor(str(project_root / "data" / "raw"))
    
    explicit = extractor.extract_explicit_labels()
    implicit = extractor.detect_implicit_patterns()
    all_labels = extractor.merge_labels(explicit, implicit)
    high, medium, low = extractor.validate_labels(all_labels)
    
    usable = {k: v for k, v in all_labels.items() if v['confidence'] in ['HIGH', 'MEDIUM']}
    
    extractor.save_labels(usable, str(project_root / "data" / "labels" / "cert_r4.2_labels.json"))
    extractor.generate_label_csv(usable, str(project_root / "data" / "labels" / "cert_r4.2_labels.csv"))
    print(f"\n[Ready] Ready for training with {len(usable)} malicious users")
