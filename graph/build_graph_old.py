"""
graph/build_graph.py
Heterogeneous graph construction for Insider Threat Detection.
CERT Insider Threat Dataset -> PyTorch Geometric HeteroData
"""

import os
import glob
import pickle
import logging
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LDAP_DIR = DATA_DIR / "raw" / "LDAP"
GRAPH_DIR = BASE_DIR / "graph"

FUSED_FEATURES_PATH = PROCESSED_DIR / "fused_features.csv"
UNIFIED_EVENTS_PATH = PROCESSED_DIR / "unified_events.csv"
OUTPUT_GRAPH_PATH = GRAPH_DIR / "heterogeneous_graph.pt"
OUTPUT_ARTIFACTS_PATH = GRAPH_DIR / "preprocessing_artifacts.pkl"

EVENT_CHUNK_SIZE = 500_000
NUM_BEHAVIOURAL_FEATURES = 47


# =========================================================
# UTILITY
# =========================================================

def normalize_uid(uid):
    """Strip whitespace and lowercase a user id string."""
    if pd.isna(uid):
        return None
    return str(uid).strip().lower()


def detect_latest_ldap_file(ldap_dir: Path) -> Path:
    """Return the most recently modified CSV in the LDAP directory."""
    csv_files = list(ldap_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ldap_dir}")
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    log.info(f"Using LDAP file: {latest.name}")
    return latest


def validate_columns(df: pd.DataFrame, required: list, source: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{source}] Missing required columns: {missing}")


# =========================================================
# STEP 1 — LOAD USER FEATURES
# =========================================================

def load_user_features(path: Path):
    """
    Load fused_features.csv.
    Returns:
        user_feature_matrix : np.ndarray  [N_users, 47]
        user_index          : dict  uid -> int
        uid_list            : list of uids in index order
    """
    log.info("Loading fused_features.csv ...")
    df = pd.read_csv(path)
    validate_columns(df, ["user_id"], "fused_features.csv")

    df["user_id"] = df["user_id"].apply(normalize_uid)
    df = df.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"])

    # Select exactly 47 numerical columns (exclude user_id and non-numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude any accidental index columns
    feature_cols = [c for c in numeric_cols if c != "user_id"][:NUM_BEHAVIOURAL_FEATURES]

    if len(feature_cols) < NUM_BEHAVIOURAL_FEATURES:
        log.warning(
            f"Only {len(feature_cols)} numeric feature columns found; padding to {NUM_BEHAVIOURAL_FEATURES}."
        )
        feat_matrix = df[feature_cols].fillna(0.0).values
        pad = np.zeros((len(df), NUM_BEHAVIOURAL_FEATURES - len(feature_cols)), dtype=np.float32)
        feat_matrix = np.concatenate([feat_matrix, pad], axis=1).astype(np.float32)
    else:
        feat_matrix = df[feature_cols].fillna(0.0).values.astype(np.float32)

    uid_list = df["user_id"].tolist()
    user_index = {uid: i for i, uid in enumerate(uid_list)}

    log.info(f"  {len(uid_list)} users, feature shape {feat_matrix.shape}")
    return feat_matrix, user_index, uid_list


# =========================================================
# STEP 2 — LOAD LDAP
# =========================================================

def load_ldap(ldap_dir: Path, user_index: dict):
    """
    Load LDAP data.
    Returns dicts mapping uid -> department/role/team/business_unit/supervisor (str).
    Missing entries fall back to 'Unknown'.
    """
    ldap_path = detect_latest_ldap_file(ldap_dir)
    log.info("Loading LDAP ...")
    df = pd.read_csv(ldap_path)

    # Flexible column detection
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower().replace(" ", "_").replace("-", "_")
        col_map[lc] = col

    def get_col(candidates):
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    uid_col = get_col(["user_id", "userid", "id"])
    dept_col = get_col(["department", "dept"])
    role_col = get_col(["role", "job_role", "jobrole"])
    team_col = get_col(["team"])
    bu_col = get_col(["business_unit", "businessunit", "bu"])
    sup_col = get_col(["supervisor", "manager", "supervisor_id"])

    required_raw = [c for c in [uid_col] if c is not None]
    if uid_col is None:
        raise ValueError("LDAP file has no recognizable user_id column.")

    df[uid_col] = df[uid_col].apply(normalize_uid)
    df = df.dropna(subset=[uid_col]).drop_duplicates(subset=[uid_col])

    def safe_str(val):
        if pd.isna(val) or str(val).strip() == "":
            return "Unknown"
        return str(val).strip()

    uid_to_dept = {}
    uid_to_role = {}
    uid_to_team = {}
    uid_to_bu = {}
    uid_to_supervisor = {}

    for _, row in df.iterrows():
        uid = row[uid_col]
        uid_to_dept[uid] = safe_str(row[dept_col]) if dept_col else "Unknown"
        uid_to_role[uid] = safe_str(row[role_col]) if role_col else "Unknown"
        uid_to_team[uid] = safe_str(row[team_col]) if team_col else "Unknown"
        uid_to_bu[uid] = safe_str(row[bu_col]) if bu_col else "Unknown"
        uid_to_supervisor[uid] = safe_str(row[sup_col]) if sup_col else "Unknown"

    log.info(f"  LDAP rows loaded: {len(df)}")
    return uid_to_dept, uid_to_role, uid_to_team, uid_to_bu, uid_to_supervisor


# =========================================================
# STEP 3 — STREAM EVENTS (single pass)
# =========================================================

def stream_event_logs(path: Path):
    """
    Stream unified_events.csv exactly ONCE in chunks.
    Returns collected statistics as plain dicts/Counters.

    Collected:
        user_pc_logins      : dict (uid, pc)  -> int  (login count)
        user_domain_visits  : dict (uid, dom) -> int  (visit count)
        user_ext_touches    : dict (uid, ext) -> int  (access count)
        pc_stats            : dict pc  -> {'login_count': int, 'user_set': set}
        domain_stats        : dict dom -> {'visit_count': int, 'user_set': set}
        ext_stats           : dict ext -> {'access_count': int, 'user_set': set}
    """
    log.info("Streaming unified_events.csv (single pass) ...")

    user_pc_logins = Counter()
    user_domain_visits = Counter()
    user_ext_touches = Counter()

    pc_login_count = Counter()
    pc_user_sets = defaultdict(set)

    domain_visit_count = Counter()
    domain_user_sets = defaultdict(set)

    ext_access_count = Counter()
    ext_user_sets = defaultdict(set)

    required_cols = ["user_id", "event_type"]
    optional_cols = {"resource", "target_user", "timestamp"}

    chunk_num = 0
    for chunk in pd.read_csv(path, chunksize=EVENT_CHUNK_SIZE, low_memory=False):
        chunk_num += 1
        if chunk_num == 1:
            validate_columns(chunk, required_cols, "unified_events.csv")
            log.info(f"  Event columns: {list(chunk.columns)}")

        chunk["user_id"] = chunk["user_id"].apply(normalize_uid)
        chunk = chunk.dropna(subset=["user_id"])

        has_resource = "resource" in chunk.columns

        for _, row in chunk.iterrows():
            uid = row["user_id"]
            etype = str(row.get("event_type", "")).strip().lower()
            resource = str(row["resource"]).strip() if has_resource and not pd.isna(row.get("resource")) else ""

            # --- PC events (logon / logoff / device) ---
            if etype in ("logon", "logoff", "device") and resource:
                pc = resource
                user_pc_logins[(uid, pc)] += 1
                pc_login_count[pc] += 1
                pc_user_sets[pc].add(uid)

            # --- HTTP / website events ---
            elif etype in ("http", "www", "web") and resource:
                # Extract domain from URL
                domain = resource.split("/")[0].split("?")[0].lower()
                if not domain:
                    domain = resource[:64]
                user_domain_visits[(uid, domain)] += 1
                domain_visit_count[domain] += 1
                domain_user_sets[domain].add(uid)

            # --- File events ---
            elif etype in ("file", "email") and resource:
                # Extract file extension
                ext = os.path.splitext(resource)[-1].lower()
                if not ext:
                    ext = ".unknown"
                user_ext_touches[(uid, ext)] += 1
                ext_access_count[ext] += 1
                ext_user_sets[ext].add(uid)

        if chunk_num % 10 == 0:
            log.info(f"  Processed {chunk_num * EVENT_CHUNK_SIZE:,} events ...")

    log.info(f"  Streaming complete. {chunk_num} chunks processed.")

    # Consolidate node-level stats
    pc_stats = {
        pc: {
            "login_count": pc_login_count[pc],
            "unique_users": len(pc_user_sets[pc]),
        }
        for pc in pc_login_count
    }
    domain_stats = {
        dom: {
            "visit_count": domain_visit_count[dom],
            "unique_users": len(domain_user_sets[dom]),
        }
        for dom in domain_visit_count
    }
    ext_stats = {
        ext: {
            "access_count": ext_access_count[ext],
            "unique_users": len(ext_user_sets[ext]),
        }
        for ext in ext_access_count
    }

    log.info(
        f"  PCs: {len(pc_stats)}, Domains: {len(domain_stats)}, Extensions: {len(ext_stats)}"
    )
    log.info(
        f"  user-PC pairs: {len(user_pc_logins)}, "
        f"user-domain pairs: {len(user_domain_visits)}, "
        f"user-ext pairs: {len(user_ext_touches)}"
    )

    return user_pc_logins, user_domain_visits, user_ext_touches, pc_stats, domain_stats, ext_stats


# =========================================================
# STEP 4 — BUILD REGISTRIES
# =========================================================

def build_registries(
    uid_list,
    uid_to_dept, uid_to_role, uid_to_team, uid_to_bu,
    pc_stats, domain_stats, ext_stats,
):
    """
    Build index maps for each non-user node type.
    Always includes an 'Unknown' entry for structural attributes.
    """
    def make_registry(values):
        unique = sorted(set(values))
        if "Unknown" not in unique:
            unique = ["Unknown"] + unique
        return {v: i for i, v in enumerate(unique)}

    dept_registry = make_registry(
        [uid_to_dept.get(uid, "Unknown") for uid in uid_list]
    )
    role_registry = make_registry(
        [uid_to_role.get(uid, "Unknown") for uid in uid_list]
    )
    team_registry = make_registry(
        [uid_to_team.get(uid, "Unknown") for uid in uid_list]
    )
    bu_registry = make_registry(
        [uid_to_bu.get(uid, "Unknown") for uid in uid_list]
    )
    pc_registry = {pc: i for i, pc in enumerate(sorted(pc_stats.keys()))}
    domain_registry = {d: i for i, d in enumerate(sorted(domain_stats.keys()))}
    ext_registry = {e: i for i, e in enumerate(sorted(ext_stats.keys()))}

    log.info(
        f"Registries: dept={len(dept_registry)}, role={len(role_registry)}, "
        f"team={len(team_registry)}, bu={len(bu_registry)}, "
        f"pc={len(pc_registry)}, domain={len(domain_registry)}, ext={len(ext_registry)}"
    )

    return dept_registry, role_registry, team_registry, bu_registry, \
           pc_registry, domain_registry, ext_registry


# =========================================================
# STEP 5 — BUILD NODE FEATURES
# =========================================================

def build_node_features(
    user_feat_matrix,
    uid_list, user_index,
    uid_to_dept, uid_to_role, uid_to_team, uid_to_bu,
    dept_registry, role_registry, team_registry, bu_registry,
    pc_stats, pc_registry,
    domain_stats, domain_registry,
    ext_stats, ext_registry,
):
    """
    Compute feature tensors for every node type.
    Structural nodes (dept/role/team/bu) use mean-pooled user vectors.
    Resource nodes (pc/domain/ext) use their statistics.
    """
    D = NUM_BEHAVIOURAL_FEATURES

    # -- user --
    user_x = torch.tensor(user_feat_matrix, dtype=torch.float)

    # -- structural: mean-pool user features --
    def mean_pool_features(registry, uid_to_attr):
        n = len(registry)
        accum = np.zeros((n, D), dtype=np.float64)
        count = np.zeros(n, dtype=np.int64)
        for uid in uid_list:
            attr = uid_to_attr.get(uid, "Unknown")
            idx = registry.get(attr, registry["Unknown"])
            uidx = user_index[uid]
            accum[idx] += user_feat_matrix[uidx]
            count[idx] += 1
        count = np.where(count == 0, 1, count)
        return torch.tensor((accum / count[:, None]).astype(np.float32), dtype=torch.float)

    dept_x = mean_pool_features(dept_registry, uid_to_dept)
    role_x = mean_pool_features(role_registry, uid_to_role)
    team_x = mean_pool_features(team_registry, uid_to_team)
    bu_x = mean_pool_features(bu_registry, uid_to_bu)

    # -- pc: [login_count, unique_users] --
    n_pc = len(pc_registry)
    pc_feat = np.zeros((n_pc, 2), dtype=np.float32)
    for pc, idx in pc_registry.items():
        stats = pc_stats[pc]
        pc_feat[idx, 0] = stats["login_count"]
        pc_feat[idx, 1] = stats["unique_users"]
    pc_x = torch.tensor(pc_feat, dtype=torch.float)

    # -- domain: [visit_count, unique_users] --
    n_dom = len(domain_registry)
    dom_feat = np.zeros((n_dom, 2), dtype=np.float32)
    for dom, idx in domain_registry.items():
        stats = domain_stats[dom]
        dom_feat[idx, 0] = stats["visit_count"]
        dom_feat[idx, 1] = stats["unique_users"]
    domain_x = torch.tensor(dom_feat, dtype=torch.float)

    # -- extension: [access_count, unique_users] --
    n_ext = len(ext_registry)
    ext_feat = np.zeros((n_ext, 2), dtype=np.float32)
    for ext, idx in ext_registry.items():
        stats = ext_stats[ext]
        ext_feat[idx, 0] = stats["access_count"]
        ext_feat[idx, 1] = stats["unique_users"]
    ext_x = torch.tensor(ext_feat, dtype=torch.float)

    log.info(
        f"Node features: user={user_x.shape}, dept={dept_x.shape}, "
        f"role={role_x.shape}, team={team_x.shape}, bu={bu_x.shape}, "
        f"pc={pc_x.shape}, domain={domain_x.shape}, ext={ext_x.shape}"
    )

    return user_x, dept_x, role_x, team_x, bu_x, pc_x, domain_x, ext_x


# =========================================================
# STEP 6 — BUILD EDGES
# =========================================================

def build_structural_edges(uid_list, user_index, uid_to_attr, registry):
    """
    One edge per user -> structural node (dept/role/team/bu).
    Missing mappings go to 'Unknown'.
    """
    src, dst = [], []
    for uid in uid_list:
        attr = uid_to_attr.get(uid, "Unknown")
        node_idx = registry.get(attr, registry["Unknown"])
        src.append(user_index[uid])
        dst.append(node_idx)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def build_resource_edges(user_index, pair_counts, user_registry, resource_registry):
    """
    Build (user, resource) edges with edge_attr = count.
    Only unique pairs; count is stored as edge attribute.
    """
    src, dst, attrs = [], [], []
    for (uid, resource), count in pair_counts.items():
        if uid not in user_registry:
            continue
        if resource not in resource_registry:
            continue
        src.append(user_registry[uid])
        dst.append(resource_registry[resource])
        attrs.append(count)

    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float).unsqueeze(1)

    return edge_index, edge_attr


def build_works_with_edges(uid_list, user_index, uid_to_supervisor):
    """
    Users sharing the same supervisor get a (canonical) undirected edge.
    No self-loops. No duplicate (A,B) / (B,A) pairs.
    """
    supervisor_to_users = defaultdict(list)
    for uid in uid_list:
        sup = uid_to_supervisor.get(uid, "Unknown")
        if sup != "Unknown":
            supervisor_to_users[sup].append(uid)

    edge_set = set()
    src, dst = [], []

    for sup, members in supervisor_to_users.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u = user_index[members[i]]
                v = user_index[members[j]]
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key not in edge_set:
                    edge_set.add(key)
                    src.append(key[0])
                    dst.append(key[1])

    if not src:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    log.info(f"  works_with edges: {edge_index.shape[1]}")
    return edge_index


# =========================================================
# STEP 7 — ASSEMBLE HeteroData
# =========================================================

def assemble_graph(
    user_x, dept_x, role_x, team_x, bu_x, pc_x, domain_x, ext_x,
    edge_belongs_to,
    edge_associated_with,
    edge_assigned_to,
    edge_member_of,
    edge_uses, edge_uses_attr,
    edge_visits, edge_visits_attr,
    edge_touches, edge_touches_attr,
    edge_works_with,
):
    data = HeteroData()

    # Nodes
    data["user"].x = user_x
    data["department"].x = dept_x
    data["role"].x = role_x
    data["team"].x = team_x
    data["business_unit"].x = bu_x
    data["pc"].x = pc_x
    data["website_domain"].x = domain_x
    data["file_extension"].x = ext_x

    # Edges
    data["user", "belongs_to", "department"].edge_index = edge_belongs_to
    data["user", "associated_with", "role"].edge_index = edge_associated_with
    data["user", "assigned_to", "team"].edge_index = edge_assigned_to
    data["user", "member_of", "business_unit"].edge_index = edge_member_of

    data["user", "uses", "pc"].edge_index = edge_uses
    data["user", "uses", "pc"].edge_attr = edge_uses_attr

    data["user", "visits", "website_domain"].edge_index = edge_visits
    data["user", "visits", "website_domain"].edge_attr = edge_visits_attr

    data["user", "touches", "file_extension"].edge_index = edge_touches
    data["user", "touches", "file_extension"].edge_attr = edge_touches_attr

    data["user", "works_with", "user"].edge_index = edge_works_with

    return data


# =========================================================
# STEP 8 — VALIDATE
# =========================================================

def validate_graph(data: HeteroData):
    log.info("Validating graph ...")
    for ntype in data.node_types:
        x = data[ntype].x
        assert x is not None and x.ndim == 2, f"Node {ntype} has bad feature tensor"
        assert not torch.isnan(x).any(), f"NaN in {ntype} features"
        log.info(f"  [{ntype}] nodes={x.shape[0]}, feat_dim={x.shape[1]}")

    for etype in data.edge_types:
        ei = data[etype].edge_index
        assert ei.shape[0] == 2, f"Bad edge_index for {etype}"
        log.info(f"  {etype} edges={ei.shape[1]}")

    log.info("Validation passed.")


# =========================================================
# MAIN
# =========================================================

def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. User features
    user_feat_matrix, user_index, uid_list = load_user_features(FUSED_FEATURES_PATH)

    # 2. LDAP
    uid_to_dept, uid_to_role, uid_to_team, uid_to_bu, uid_to_supervisor = load_ldap(
        LDAP_DIR, user_index
    )

    # 3. Stream events (single pass)
    (
        user_pc_logins,
        user_domain_visits,
        user_ext_touches,
        pc_stats,
        domain_stats,
        ext_stats,
    ) = stream_event_logs(UNIFIED_EVENTS_PATH)

    # 4. Registries
    (
        dept_registry, role_registry, team_registry, bu_registry,
        pc_registry, domain_registry, ext_registry,
    ) = build_registries(
        uid_list,
        uid_to_dept, uid_to_role, uid_to_team, uid_to_bu,
        pc_stats, domain_stats, ext_stats,
    )

    # 5. Node features
    (
        user_x, dept_x, role_x, team_x, bu_x,
        pc_x, domain_x, ext_x,
    ) = build_node_features(
        user_feat_matrix,
        uid_list, user_index,
        uid_to_dept, uid_to_role, uid_to_team, uid_to_bu,
        dept_registry, role_registry, team_registry, bu_registry,
        pc_stats, pc_registry,
        domain_stats, domain_registry,
        ext_stats, ext_registry,
    )

    # 6. Structural edges
    log.info("Building structural edges ...")
    edge_belongs_to = build_structural_edges(uid_list, user_index, uid_to_dept, dept_registry)
    edge_associated_with = build_structural_edges(uid_list, user_index, uid_to_role, role_registry)
    edge_assigned_to = build_structural_edges(uid_list, user_index, uid_to_team, team_registry)
    edge_member_of = build_structural_edges(uid_list, user_index, uid_to_bu, bu_registry)

    # 7. Resource edges
    log.info("Building resource edges ...")
    edge_uses, edge_uses_attr = build_resource_edges(
        user_index, user_pc_logins, user_index, pc_registry
    )
    edge_visits, edge_visits_attr = build_resource_edges(
        user_index, user_domain_visits, user_index, domain_registry
    )
    edge_touches, edge_touches_attr = build_resource_edges(
        user_index, user_ext_touches, user_index, ext_registry
    )

    # 8. Works-with edges
    log.info("Building works_with edges ...")
    edge_works_with = build_works_with_edges(uid_list, user_index, uid_to_supervisor)

    # 9. Assemble
    log.info("Assembling HeteroData ...")
    data = assemble_graph(
        user_x, dept_x, role_x, team_x, bu_x, pc_x, domain_x, ext_x,
        edge_belongs_to,
        edge_associated_with,
        edge_assigned_to,
        edge_member_of,
        edge_uses, edge_uses_attr,
        edge_visits, edge_visits_attr,
        edge_touches, edge_touches_attr,
        edge_works_with,
    )

    # 10. Validate
    validate_graph(data)

    # 11. Save graph
    torch.save(data, OUTPUT_GRAPH_PATH)
    log.info(f"Graph saved -> {OUTPUT_GRAPH_PATH}")

    # 12. Save artifacts
    artifacts = {
        "user_index": user_index,
        "uid_list": uid_list,
        "dept_registry": dept_registry,
        "role_registry": role_registry,
        "team_registry": team_registry,
        "bu_registry": bu_registry,
        "pc_registry": pc_registry,
        "domain_registry": domain_registry,
        "ext_registry": ext_registry,
        "uid_to_dept": uid_to_dept,
        "uid_to_role": uid_to_role,
        "uid_to_team": uid_to_team,
        "uid_to_bu": uid_to_bu,
        "uid_to_supervisor": uid_to_supervisor,
    }
    with open(OUTPUT_ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"Artifacts saved -> {OUTPUT_ARTIFACTS_PATH}")

    log.info("Graph construction complete.")


if __name__ == "__main__":
    main()