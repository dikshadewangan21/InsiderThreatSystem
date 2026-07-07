"""
alerts/risk_engine.py

Real-Time Neural Risk Classifier. Loads the complete pre-trained pipeline 
(TGN -> GAT -> MLP) and calculates risk probabilities for streaming CERT events.
"""

import logging
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alerts.config import AlertConfig
from graph.edge_features import (
    EdgeFeatureConfig,
    load_static_lookups,
    UserTemporalState,
    sinusoidal_encoding,
    _LOG_DENOM,
    _FI
)
from models.tgn_model import TGN
from models.gat_model import GAT, GATConfig
from models.mlp_classifier import MLPClassifier, MLPClassifierConfig
from graph.edge_weighting import DynamicEdgeWeighting
from training.train import apply_edge_weighting

logger = logging.getLogger("insider_threat.alerts.risk_engine")

EVENT_TYPE_TO_RELATION = {
    "logon": ("User", "PC"),
    "logoff": ("User", "PC"),
    "device": ("User", "PC"),
    "http": ("User", "WebsiteDomain"),
    "file": ("User", "FileExtension"),
}

TEMPORAL_RELATION_NAME = {
    ("User", "PC"): "accesses",
    ("User", "WebsiteDomain"): "visits",
    ("User", "FileExtension"): "touches",
}

class RiskEngine:
    """Class to load pre-trained checkpoints and perform real-time risk scoring."""
    
    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info("Initializing Risk Engine...")
        
        # 1. Load Preprocessing Artifacts
        if not os.path.exists(config.artifacts_path):
            raise FileNotFoundError(f"Preprocessing artifacts not found at {config.artifacts_path}")
            
        with open(config.artifacts_path, "rb") as f:
            self.artifacts = pickle.load(f)
            
        self.uid_list = self.artifacts["uid_list"]
        self.uid_to_index = self.artifacts["user_index"]
        self.pc_registry = self.artifacts["pc_registry"]
        self.domain_registry = self.artifacts["domain_registry"]
        self.ext_registry = self.artifacts["ext_registry"]
        self.event_type_registry = self.artifacts["event_type_registry"]
        
        # 2. Load Node Skeleton and Compute Offsets
        if not os.path.exists(config.graph_path):
            raise FileNotFoundError(f"Node graph skeleton not found at {config.graph_path}")
            
        self.skeleton = torch.load(config.graph_path, map_location="cpu", weights_only=False)
        self.num_nodes = sum(self.skeleton[nt].num_nodes for nt in self.skeleton.node_types)
        
        self.offsets = {}
        offset = 0
        for node_type in self.skeleton.node_types:
            self.offsets[node_type] = offset
            offset += self.skeleton[node_type].num_nodes
            
        # 3. Initialize Model Architecture
        from models.tgn_model import TGNConfig
        tgn_config = TGNConfig(
            memory_dim=128,
            time_dim=32,
            embedding_dim=128,
            message_hidden_dim=128,
            aggregator="attention",
            device=config.device,
        )
        self.tgn = TGN(
            num_nodes=self.num_nodes,
            edge_feature_dim=56,  # 40 features + 16 temporal encoding
            config=tgn_config,
        )
        
        gat_config = GATConfig(
            in_dim=128,
            hidden_dim=128,
            out_dim=128,
            heads=4,
            dropout=0.2,
        )
        self.gat = GAT(gat_config)
        
        mlp_config = MLPClassifierConfig()
        self.mlp = MLPClassifier(in_dim=128, config=mlp_config)
        
        self.edge_weighter = DynamicEdgeWeighting(in_features=40).to(self.device)
        
        # 4. Load Checkpoint
        checkpoint_path = config.best_model_path if config.best_model_path.exists() else config.last_model_path
        if checkpoint_path.exists():
            logger.info(f"Restoring checkpoints from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.tgn.load_state_dict(state["tgn"])
            self.gat.load_state_dict(state["gat"])
            self.mlp.load_state_dict(state["mlp"])
            if "edge_weighter" in state and self.edge_weighter is not None:
                self.edge_weighter.load_state_dict(state["edge_weighter"])
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}! Model is using random weights.")
            
        self.tgn.eval()
        self.gat.eval()
        self.mlp.eval()
        if self.edge_weighter is not None:
            self.edge_weighter.eval()
            
        # 5. Load Static Feature Lookups
        feat_cfg = EdgeFeatureConfig()
        feat_cfg.graph_output_dir = Path(config.graph_path.parent)
        feat_cfg.psychology_csv = Path(config.psychology_csv)
        feat_cfg.fused_csv = Path(config.fused_csv)
        feat_cfg.__post_init__()
        
        self.lookups = load_static_lookups(feat_cfg)
        
        # 6. Initialize Running Temporal States
        self.temporal_state = UserTemporalState()
        self.max_rc = 1.0
        self.max_ic = 1.0
        
        # 7. Probability shift parameter
        self.probability_logit_shift = float(np.log(config.pos_weight)) if config.calibrate_weighted_logits else 0.0
        logger.info(f"Calibration logit shift parameter set to: {self.probability_logit_shift:.4f}")
        
    def predict_event(self, event: Dict[str, Any]) -> float:
        """Run real-time inference on a single CERT event and return risk probability."""
        try:
            user_id = str(event.get("user_id", "")).strip().upper()
            event_type = str(event.get("event_type", "")).strip().lower()
            resource = str(event.get("resource", "")).strip()
            target_user = str(event.get("target_user", "none")).strip().upper()
            ts = float(event.get("timestamp", 0.0))
            
            # Match event types to relation
            relation = EVENT_TYPE_TO_RELATION.get(event_type)
            if relation is None:
                return 0.0  # Unsupported relation
                
            src_node_type, dst_node_type = relation
            relation_name = TEMPORAL_RELATION_NAME[relation]
            
            # Map user ID to index
            user_idx = self.uid_to_index.get(user_id)
            if user_idx is None:
                logger.warning(f"User {user_id} not found in user index registry. Skipping event.")
                return 0.0
                
            # Map destination resource to index
            if dst_node_type == "PC":
                dst_idx = self.pc_registry.get(resource, 0)
            elif dst_node_type == "WebsiteDomain":
                dst_idx = self.domain_registry.get(resource, 0)
            elif dst_node_type == "FileExtension":
                # Strip leading dot
                resource_clean = resource.lower().lstrip(".")
                dst_idx = self.ext_registry.get(resource_clean, 0)
            else:
                dst_idx = 0
                
            # Map target user to index
            target_idx = self.uid_to_index.get(target_user, -1) if target_user != "NONE" else -1
            
            # Compute global homogeneous graph indices using skeleton offsets
            src_global = user_idx + self.offsets["User"]
            dst_global = dst_idx + self.offsets[dst_node_type]
            
            # Construct PyTorch edge index
            edge_index = torch.tensor([[src_global], [dst_global]], dtype=torch.long, device=self.device)
            edge_time = torch.tensor([ts], dtype=torch.float32, device=self.device)
            
            # --- FEATURE ENGINEERING ---
            feat = np.zeros((1, 40), dtype=np.float32)
            
            # 1. Calendar features
            dti = pd.to_datetime([ts], unit="s", utc=True)[0]
            feat[0, _FI["hour_of_day"]] = float(dti.hour) / 23.0
            feat[0, _FI["minute_of_hour"]] = float(dti.minute) / 59.0
            feat[0, _FI["weekday"]] = float(dti.dayofweek) / 6.0
            feat[0, _FI["weekend_flag"]] = 1.0 if dti.dayofweek >= 5 else 0.0
            feat[0, _FI["month"]] = float(dti.month) / 12.0
            feat[0, _FI["quarter"]] = float(dti.quarter) / 4.0
            
            after_hours = 1.0 if (dti.hour < 7 or dti.hour >= 19) else 0.0
            working_hours = 1.0 if (after_hours == 0.0 and dti.dayofweek < 5) else 0.0
            feat[0, _FI["is_after_hours"]] = after_hours
            feat[0, _FI["is_working_hours"]] = working_hours
            
            # 2. Temporal deltas (Update streaming state machine)
            last_g = self.temporal_state._last_global.get(user_idx)
            delta_global = float(ts - last_g) if last_g is not None and ts >= last_g else 0.0
            
            rkey = (user_idx, relation_name)
            last_r = self.temporal_state._last_rel.get(rkey)
            delta_relation = float(ts - last_r) if last_r is not None and ts >= last_r else 0.0
            
            ss = self.temporal_state._sess_start.get(user_idx)
            if ss is None:
                self.temporal_state._sess_start[user_idx] = int(ts)
                session_dur = 0.0
            else:
                gap = (ts - last_g) if last_g is not None else 0
                if gap > 8 * 3600:  # 8 hours gap
                    self.temporal_state._sess_start[user_idx] = int(ts)
                    session_dur = 0.0
                else:
                    session_dur = float(max(ts - self.temporal_state._sess_start[user_idx], 0))
                    
            self.temporal_state._roll_count[user_idx] += 1
            rolling_count = float(self.temporal_state._roll_count[user_idx])
            
            ik = (user_idx, dst_idx)
            interaction_ct = float(self.temporal_state._inter_count[ik])
            self.temporal_state._inter_count[ik] += 1
            
            self.temporal_state._last_global[user_idx] = int(ts)
            self.temporal_state._last_rel[rkey] = int(ts)
            
            # Log norm temporal features
            def _lognorm_val(val):
                return float((np.log1p(max(val, 0.0)) / _LOG_DENOM))
                
            feat[0, _FI["time_since_last_event"]] = np.clip(_lognorm_val(delta_global), 0, 1)
            feat[0, _FI["session_duration"]] = np.clip(_lognorm_val(session_dur), 0, 1)
            feat[0, _FI["time_gap_prev_action"]] = np.clip(_lognorm_val(delta_relation), 0, 1)
            
            # 3. Behavior features
            feat[0, _FI["user_behavior_score"]] = float(np.clip(self.lookups.user_behavior_score[user_idx], 0, 1))
            feat[0, _FI["anomaly_score"]] = float(np.clip(self.lookups.user_anomaly_score[user_idx], 0, 1))
            feat[0, _FI["activity_frequency"]] = float(np.clip(self.lookups.user_activity_freq[user_idx], 0, 1))
            
            self.max_rc = max(self.max_rc, rolling_count)
            feat[0, _FI["rolling_action_count"]] = float(np.clip(np.log1p(rolling_count) / np.log1p(self.max_rc), 0, 1))
            
            feat[0, _FI["login_frequency"]] = float(np.clip(self.lookups.user_login_freq[user_idx], 0, 1))
            feat[0, _FI["device_frequency"]] = float(np.clip(self.lookups.user_device_freq[user_idx], 0, 1))
            feat[0, _FI["website_frequency"]] = float(np.clip(self.lookups.user_website_freq[user_idx], 0, 1))
            
            # 4. Psychological features
            feat[0, _FI["psychology_score"]] = float(np.clip(self.lookups.user_psychology[user_idx], 0, 1))
            feat[0, _FI["behavior_deviation"]] = float(np.clip(self.lookups.user_behav_dev[user_idx], 0, 1))
            feat[0, _FI["after_hours_score"]] = float(np.clip(self.lookups.user_aft_hrs[user_idx], 0, 1))
            
            # 5. Organizational features
            def _encode_org(attr_dict, registry, max_id):
                attr = attr_dict.get(user_id, "unknown")
                enc = registry.get(attr, 0)
                return float(enc) / float(max(max_id, 1))
                
            feat[0, _FI["dept_encoded"]] = _encode_org(self.lookups.uid_to_dept, self.lookups.dept_registry, self.lookups.max_dept_id)
            feat[0, _FI["role_encoded"]] = _encode_org(self.lookups.uid_to_role, self.lookups.role_registry, self.lookups.max_role_id)
            feat[0, _FI["team_encoded"]] = _encode_org(self.lookups.uid_to_team, self.lookups.team_registry, self.lookups.max_team_id)
            feat[0, _FI["bu_encoded"]] = _encode_org(self.lookups.uid_to_bu, self.lookups.bu_registry, self.lookups.max_bu_id)
            
            has_mgr = 1.0 if (user_id in self.lookups.uid_to_supervisor and self.lookups.uid_to_supervisor[user_id] not in ("", "unknown", "nan")) else 0.0
            feat[0, _FI["has_manager"]] = has_mgr
            
            # 6. Resource features
            if dst_node_type == "FileExtension":
                ext_idx_clamped = np.clip(dst_idx, 0, self.lookups.max_ext_idx)
                sens = float(self.lookups.ext_sensitivity[ext_idx_clamped])
                feat[0, _FI["file_sensitivity"]] = sens
                feat[0, _FI["extension_risk"]] = sens
            elif dst_node_type == "WebsiteDomain":
                dom_idx_clamped = np.clip(dst_idx, 0, self.lookups.max_domain_idx)
                feat[0, _FI["domain_popularity"]] = 1.0 - float(dom_idx_clamped) / float(self.lookups.max_domain_idx)
            elif dst_node_type == "PC":
                pc_idx_clamped = np.clip(dst_idx, 0, self.lookups.max_pc_idx)
                feat[0, _FI["pc_popularity"]] = 1.0 - float(pc_idx_clamped) / float(self.lookups.max_pc_idx)
                
            # 7. Graph features
            feat[0, _FI["source_degree"]] = float(np.clip(self.lookups.user_out_degree[user_idx], 0, 1))
            
            max_dst = self.lookups.max_pc_idx if dst_node_type == "PC" else (self.lookups.max_domain_idx if dst_node_type == "WebsiteDomain" else self.lookups.max_ext_idx)
            feat[0, _FI["destination_degree"]] = 1.0 - float(np.clip(dst_idx, 0, max_dst)) / float(max(max_dst, 1))
            
            self.max_ic = max(self.max_ic, interaction_ct)
            feat[0, _FI["historical_interaction_count"]] = float(np.clip(np.log1p(interaction_ct) / np.log1p(self.max_ic), 0, 1))
            
            # 8. Event type flags
            logon_ids = {self.lookups.event_type_registry.get(k, -9) for k in ("logon", "logoff")}
            device_ids = {self.lookups.event_type_registry.get("device", -9)}
            http_ids = {self.lookups.event_type_registry.get("http", -9)}
            
            ev_id = self.lookups.event_type_registry.get(event_type, -1)
            feat[0, _FI["is_logon_event"]] = 1.0 if ev_id in logon_ids else 0.0
            feat[0, _FI["is_device_event"]] = 1.0 if ev_id in device_ids else 0.0
            feat[0, _FI["is_http_event"]] = 1.0 if ev_id in http_ids else 0.0
            
            # 9. Relation type flags
            feat[0, _FI["rel_accesses_pc"]] = 1.0 if dst_node_type == "PC" else 0.0
            feat[0, _FI["rel_visits_web"]] = 1.0 if dst_node_type == "WebsiteDomain" else 0.0
            feat[0, _FI["rel_touches_file"]] = 1.0 if dst_node_type == "FileExtension" else 0.0
            
            # 10. Target-user flag
            feat[0, _FI["has_target_user"]] = 1.0 if target_idx != -1 else 0.0
            
            # Convert engineered features to tensor
            features = torch.tensor(feat, dtype=torch.float32, device=self.device)
            
            # Sinusoidal temporal encoding
            time_tensor = torch.tensor([ts], dtype=torch.long, device=self.device)
            temporal_encoding = sinusoidal_encoding(time_tensor, 16, 1e6)
            
            # Construct shard dict
            tgn_shard = {
                "edge_index": edge_index,
                "features": features,
                "temporal_encoding": temporal_encoding,
                "edge_time": edge_time,
                "relation": relation_name,
                "src_type": src_node_type,
                "dst_type": dst_node_type,
                "shard_index": 0,
            }
            
            # Apply dynamic edge weighting if weighter exists
            if self.edge_weighter is not None:
                tgn_shard = apply_edge_weighting(tgn_shard, self.edge_weighter)
                
            # --- MODEL INFERENCE ---
            with torch.no_grad():
                # 1. TGN
                tgn_result = self.tgn.process_shard(tgn_shard)
                node_embeddings = tgn_result["embeddings"]
                touched_ids = tgn_result["updated_node_ids"]
                
                # 2. local GAT edge index
                local_edge_index = torch.stack([
                    torch.searchsorted(touched_ids, edge_index[0]),
                    torch.searchsorted(touched_ids, edge_index[1])
                ], dim=0)
                
                # 3. GAT
                edge_weight = tgn_shard.get("edge_weight")
                if edge_weight is None:
                    edge_weight = torch.ones(local_edge_index.size(1), dtype=torch.float32, device=self.device)
                    
                contextual_embeddings, _ = self.gat(node_embeddings, local_edge_index, edge_weight)
                
                # 4. Extract user index embedding
                src_pos = (touched_ids == src_global).nonzero(as_tuple=True)[0]
                if len(src_pos) == 0:
                    return 0.0
                src_pos = src_pos[0].item()
                user_embedding = contextual_embeddings[src_pos].unsqueeze(0)
                
                # 5. MLP Logits
                logits = self.mlp(user_embedding)
                logits = logits.squeeze(-1) if logits.dim() > 1 else logits
                logit = float(logits.item())
                
                # --- CALIBRATION SHIFT ---
                calibrated_logit = logit - self.probability_logit_shift
                prob = 1.0 / (1.0 + np.exp(-calibrated_logit))
                
                return float(np.clip(prob, 1e-6, 1.0 - 1e-6))
                
        except Exception as e:
            logger.error(f"Error during risk engine prediction: {e}")
            return 0.0
