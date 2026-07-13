"""
explainability/feature_importance.py

Computes feature importance for GNN-TGN predictions using backpropagation saliency
gradients (primary) and value-deviation heuristics (fallback).
"""

import logging
import torch
import numpy as np
from typing import Any, Dict, List, Tuple
from alerts.risk_engine import RiskEngine, EVENT_TYPE_TO_RELATION, TEMPORAL_RELATION_NAME
from graph.edge_features import sinusoidal_encoding, _FI, FEATURE_NAMES, _LOG_DENOM
from training.train import apply_edge_weighting

logger = logging.getLogger("insider_threat.explainability.feature_importance")

def get_saliency_importance(
    risk_engine: RiskEngine, event: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute feature importance using backpropagation saliency maps.
    Returns:
        importance_dict: Dict mapping feature name to normalized saliency score.
        raw_features_dict: Dict mapping feature name to its raw value.
    """
    try:
        user_id = str(event.get("user_id", "")).strip().upper()
        event_type = str(event.get("event_type", "")).strip().lower()
        resource = str(event.get("resource", "")).strip()
        target_user = str(event.get("target_user", "none")).strip().upper()
        ts = float(event.get("timestamp", 0.0))
        
        relation = EVENT_TYPE_TO_RELATION.get(event_type)
        if relation is None:
            return {}, {}
            
        src_node_type, dst_node_type = relation
        relation_name = TEMPORAL_RELATION_NAME[relation]
        
        user_idx = risk_engine.uid_to_index.get(user_id)
        if user_idx is None:
            return {}, {}
            
        if dst_node_type == "PC":
            dst_idx = risk_engine.pc_registry.get(resource, 0)
        elif dst_node_type == "WebsiteDomain":
            dst_idx = risk_engine.domain_registry.get(resource, 0)
        elif dst_node_type == "FileExtension":
            dst_idx = risk_engine.ext_registry.get(resource.lower().lstrip("."), 0)
        else:
            dst_idx = 0
            
        target_idx = risk_engine.uid_to_index.get(target_user, -1) if target_user != "NONE" else -1
        
        src_global = user_idx + risk_engine.offsets["User"]
        dst_global = dst_idx + risk_engine.offsets[dst_node_type]
        
        edge_index = torch.tensor([[src_global], [dst_global]], dtype=torch.long, device=risk_engine.device)
        edge_time = torch.tensor([ts], dtype=torch.float32, device=risk_engine.device)
        
        # Build 40-dim feature vector
        feat = np.zeros((1, 40), dtype=np.float32)
        
        # Retrieve lookup profiles
        # Re-use engineered variables matching edge_features process_shard
        import pandas as pd
        dti = pd.to_datetime([ts], unit="s", utc=True)[0]
        feat[0, _FI["hour_of_day"]] = float(dti.hour) / 23.0
        feat[0, _FI["minute_of_hour"]] = float(dti.minute) / 59.0
        feat[0, _FI["weekday"]] = float(dti.dayofweek) / 6.0
        feat[0, _FI["weekend_flag"]] = 1.0 if dti.dayofweek >= 5 else 0.0
        feat[0, _FI["month"]] = float(dti.month) / 12.0
        feat[0, _FI["quarter"]] = float(dti.quarter) / 4.0
        
        after_hours = 1.0 if (dti.hour < 7 or dti.hour >= 19) else 0.0
        feat[0, _FI["is_after_hours"]] = after_hours
        feat[0, _FI["is_working_hours"]] = 1.0 if (after_hours == 0.0 and dti.dayofweek < 5) else 0.0
        
        # Temporal state deltas
        last_g = risk_engine.temporal_state._last_global.get(user_idx)
        delta_global = float(ts - last_g) if last_g is not None and ts >= last_g else 0.0
        
        rkey = (user_idx, relation_name)
        last_r = risk_engine.temporal_state._last_rel.get(rkey)
        delta_relation = float(ts - last_r) if last_r is not None and ts >= last_r else 0.0
        
        ss = risk_engine.temporal_state._sess_start.get(user_idx)
        session_dur = float(max(ts - ss, 0)) if ss is not None else 0.0
        if last_g is not None and (ts - last_g) > 8 * 3600:
            session_dur = 0.0
            
        rolling_count = float(risk_engine.temporal_state._roll_count.get(user_idx, 0))
        interaction_ct = float(risk_engine.temporal_state._inter_count.get((user_idx, dst_idx), 0))
        
        def _lognorm_val(val):
            return float((np.log1p(max(val, 0.0)) / _LOG_DENOM))
            
        feat[0, _FI["time_since_last_event"]] = np.clip(_lognorm_val(delta_global), 0, 1)
        feat[0, _FI["session_duration"]] = np.clip(_lognorm_val(session_dur), 0, 1)
        feat[0, _FI["time_gap_prev_action"]] = np.clip(_lognorm_val(delta_relation), 0, 1)
        
        feat[0, _FI["user_behavior_score"]] = float(np.clip(risk_engine.lookups.user_behavior_score[user_idx], 0, 1))
        feat[0, _FI["anomaly_score"]] = float(np.clip(risk_engine.lookups.user_anomaly_score[user_idx], 0, 1))
        feat[0, _FI["activity_frequency"]] = float(np.clip(risk_engine.lookups.user_activity_freq[user_idx], 0, 1))
        feat[0, _FI["rolling_action_count"]] = float(np.clip(np.log1p(rolling_count) / np.log1p(risk_engine.max_rc), 0, 1))
        feat[0, _FI["login_frequency"]] = float(np.clip(risk_engine.lookups.user_login_freq[user_idx], 0, 1))
        feat[0, _FI["device_frequency"]] = float(np.clip(risk_engine.lookups.user_device_freq[user_idx], 0, 1))
        feat[0, _FI["website_frequency"]] = float(np.clip(risk_engine.lookups.user_website_freq[user_idx], 0, 1))
        
        feat[0, _FI["psychology_score"]] = float(np.clip(risk_engine.lookups.user_psychology[user_idx], 0, 1))
        feat[0, _FI["behavior_deviation"]] = float(np.clip(risk_engine.lookups.user_behav_dev[user_idx], 0, 1))
        feat[0, _FI["after_hours_score"]] = float(np.clip(risk_engine.lookups.user_aft_hrs[user_idx], 0, 1))
        
        def _encode_org(attr_dict, registry, max_id):
            attr = attr_dict.get(user_id, "unknown")
            enc = registry.get(attr, 0)
            return float(enc) / float(max(max_id, 1))
            
        feat[0, _FI["dept_encoded"]] = _encode_org(risk_engine.lookups.uid_to_dept, risk_engine.lookups.dept_registry, risk_engine.lookups.max_dept_id)
        feat[0, _FI["role_encoded"]] = _encode_org(risk_engine.lookups.uid_to_role, risk_engine.lookups.role_registry, risk_engine.lookups.max_role_id)
        feat[0, _FI["team_encoded"]] = _encode_org(risk_engine.lookups.uid_to_team, risk_engine.lookups.team_registry, risk_engine.lookups.max_team_id)
        feat[0, _FI["bu_encoded"]] = _encode_org(risk_engine.lookups.uid_to_bu, risk_engine.lookups.bu_registry, risk_engine.lookups.max_bu_id)
        feat[0, _FI["has_manager"]] = 1.0 if (user_id in risk_engine.lookups.uid_to_supervisor and risk_engine.lookups.uid_to_supervisor[user_id] not in ("", "unknown", "nan")) else 0.0
        
        if dst_node_type == "FileExtension":
            sens = float(risk_engine.lookups.ext_sensitivity[np.clip(dst_idx, 0, risk_engine.lookups.max_ext_idx)])
            feat[0, _FI["file_sensitivity"]] = sens
            feat[0, _FI["extension_risk"]] = sens
        elif dst_node_type == "WebsiteDomain":
            feat[0, _FI["domain_popularity"]] = 1.0 - float(np.clip(dst_idx, 0, risk_engine.lookups.max_domain_idx)) / float(risk_engine.lookups.max_domain_idx)
        elif dst_node_type == "PC":
            feat[0, _FI["pc_popularity"]] = 1.0 - float(np.clip(dst_idx, 0, risk_engine.lookups.max_pc_idx)) / float(risk_engine.lookups.max_pc_idx)
            
        feat[0, _FI["source_degree"]] = float(np.clip(risk_engine.lookups.user_out_degree[user_idx], 0, 1))
        max_dst = risk_engine.lookups.max_pc_idx if dst_node_type == "PC" else (risk_engine.lookups.max_domain_idx if dst_node_type == "WebsiteDomain" else risk_engine.lookups.max_ext_idx)
        feat[0, _FI["destination_degree"]] = 1.0 - float(np.clip(dst_idx, 0, max_dst)) / float(max(max_dst, 1))
        feat[0, _FI["historical_interaction_count"]] = float(np.clip(np.log1p(interaction_ct) / np.log1p(risk_engine.max_ic), 0, 1))
        
        logon_ids = {risk_engine.lookups.event_type_registry.get(k, -9) for k in ("logon", "logoff")}
        device_ids = {risk_engine.lookups.event_type_registry.get("device", -9)}
        http_ids = {risk_engine.lookups.event_type_registry.get("http", -9)}
        ev_id = risk_engine.lookups.event_type_registry.get(event_type, -1)
        feat[0, _FI["is_logon_event"]] = 1.0 if ev_id in logon_ids else 0.0
        feat[0, _FI["is_device_event"]] = 1.0 if ev_id in device_ids else 0.0
        feat[0, _FI["is_http_event"]] = 1.0 if ev_id in http_ids else 0.0
        feat[0, _FI["rel_accesses_pc"]] = 1.0 if dst_node_type == "PC" else 0.0
        feat[0, _FI["rel_visits_web"]] = 1.0 if dst_node_type == "WebsiteDomain" else 0.0
        feat[0, _FI["rel_touches_file"]] = 1.0 if dst_node_type == "FileExtension" else 0.0
        feat[0, _FI["has_target_user"]] = 1.0 if target_idx != -1 else 0.0
        
        raw_features_dict = {name: float(feat[0, idx]) for idx, name in enumerate(FEATURE_NAMES)}
        
        # Prepare inputs with autograd tracking
        features_tensor = torch.tensor(feat, dtype=torch.float32, device=risk_engine.device, requires_grad=True)
        time_tensor = torch.tensor([ts], dtype=torch.long, device=risk_engine.device)
        temporal_enc = sinusoidal_encoding(time_tensor, 16, 1e6)
        
        tgn_shard = {
            "edge_index": edge_index,
            "features": features_tensor,
            "temporal_encoding": temporal_enc,
            "edge_time": edge_time,
            "relation": relation_name,
            "src_type": src_node_type,
            "dst_type": dst_node_type,
            "shard_index": 0,
        }
        
        if risk_engine.edge_weighter is not None:
            tgn_shard = apply_edge_weighting(tgn_shard, risk_engine.edge_weighter)
            
        # Run forward pass tracking gradients
        with torch.set_grad_enabled(True):
            tgn_result = risk_engine.tgn.process_shard(tgn_shard)
            node_embeddings = tgn_result["embeddings"]
            touched_ids = tgn_result["updated_node_ids"]
            
            local_edge_index = torch.stack([
                torch.searchsorted(touched_ids, edge_index[0]),
                torch.searchsorted(touched_ids, edge_index[1])
            ], dim=0)
            
            edge_weight = tgn_shard.get("edge_weight")
            if edge_weight is None:
                edge_weight = torch.ones(local_edge_index.size(1), dtype=torch.float32, device=risk_engine.device)
                
            contextual_embeddings, _ = risk_engine.gat(node_embeddings, local_edge_index, edge_weight)
            
            src_pos = (touched_ids == src_global).nonzero(as_tuple=True)[0]
            if len(src_pos) == 0:
                return _fallback_importance(raw_features_dict), raw_features_dict
            src_pos = src_pos[0].item()
            user_embedding = contextual_embeddings[src_pos].unsqueeze(0)
            
            logit = risk_engine.mlp(user_embedding)
            logit = logit.squeeze(-1) if logit.dim() > 1 else logit
            
            # Autograd backward to compute saliency map
            logit.backward()
            
        # Extract gradients
        saliency = features_tensor.grad.abs().squeeze(0).cpu().numpy()
        
        # Normalize saliency to [0, 1]
        sum_sal = saliency.sum()
        if sum_sal > 0:
            saliency = saliency / sum_sal
            
        importance_dict = {name: float(saliency[idx]) for idx, name in enumerate(FEATURE_NAMES)}
        return importance_dict, raw_features_dict
        
    except Exception as e:
        logger.warning(f"Saliency gradient computation failed ({e}). Falling back to value-deviation heuristic.")
        # Fallback to value-based importance
        raw_feat_fallback = {name: 0.0 for name in FEATURE_NAMES}
        if 'user_id' in event:
            # We try to extract whatever raw values we can
            try:
                user_id = event["user_id"].upper()
                user_idx = risk_engine.uid_to_index.get(user_id)
                if user_idx is not None:
                    raw_feat_fallback["psychology_score"] = float(risk_engine.lookups.user_psychology[user_idx])
                    raw_feat_fallback["behavior_deviation"] = float(risk_engine.lookups.user_behav_dev[user_idx])
                    raw_feat_fallback["login_frequency"] = float(risk_engine.lookups.user_login_freq[user_idx])
            except Exception:
                pass
        return _fallback_importance(raw_feat_fallback), raw_feat_fallback

def _fallback_importance(raw_features: Dict[str, float]) -> Dict[str, float]:
    """Fallback heuristic importance based on raw feature deviation."""
    importance = {name: 0.05 for name in FEATURE_NAMES}
    # Boost features that have high values
    for name, val in raw_features.items():
        if val > 0.5:
            importance[name] = val * 0.5
            
    # Normalize
    sum_imp = sum(importance.values())
    if sum_imp > 0:
        importance = {k: v / sum_imp for k, v in importance.items()}
    return importance
