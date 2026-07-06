# Insider Threat Detection System

A production-grade graph neural network system for detecting insider threats using temporal heterogeneous graphs and attention-based architectures.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Pipeline](#data-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Components](#key-components)
- [Training Pipeline](#training-pipeline)
- [Model Architecture](#model-architecture)
- [Data Formats](#data-formats)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system analyzes user behavior from multiple data sources (logins, emails, file access, web browsing, device connections) to identify anomalous patterns that may indicate insider threats. It uses a three-stage neural network pipeline:

1. **Temporal Graph Network (TGN)** - Learns node representations from temporal event sequences
2. **Graph Attention Network (GAT)** - Contextualizes embeddings using graph structure
3. **MLP Classifier** - Produces risk scores for each user

**Key Features:**
- Streaming architecture for large-scale datasets
- Temporal modeling of user behavior over time
- Heterogeneous graph handling (users, PCs, domains, file types)
- Dynamic edge weighting for anomaly detection

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │  Logon  │ │  Email  │ │  File   │ │   HTTP  │ │ Device  │        │
│  │ Events  │ │ Events  │ │  Access │ │ Traffic │ │ Events  │        │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │
│       │           │           │           │           │              │
│       └───────────┴───────────┴───────────┴───────────┘              │
│                            │                                         │
│                    ┌───────▼───────┐                                  │
│                    │ Unified Events│                                  │
│                    └───────┬───────┘                                  │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────────┐
│                   GRAPH CONSTRUCTION                                  │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Node Types:                                                  │   │
│  │  • User (from LDAP)                                           │   │
│  │  • PC (Physical computers)                                    │   │
│  │  • WebsiteDomain (HTTP URLs)                                  │   │
│  │  • FileExtension (File types)                                 │   │
│  │  • Department/Role/Team/BusinessUnit (organizational)        │   │
│  └───────────────────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Edge Types (Temporal):                                       │   │
│  │  • User → accesses → PC (logon/device)                       │   │
│  │  • User → visits → WebsiteDomain (HTTP)                      │   │
│  │  • User → touches → FileExtension (file access)              │   │
│  └───────────────────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Edge Types (Structural):                                     │   │
│  │  • User → belongs_to → Department                            │   │
│  │  • User → has_role → Role                                    │   │
│  │  • User → member_of → Team                                   │   │
│  └───────────────────────────────────────────────────────────────┘   │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────────┐
│                    FEATURE ENGINEERING                                │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  40 scalar features + 16-dim temporal encoding per edge:     │   │
│  │  • Temporal (hour, day, weekend, after-hours)                │   │
│  │  • Behavioral (event frequency, device switching)            │   │
│  │  • Psychological (from psychometric data)                    │   │
│  │  • Organizational (department, role encoded)                 │   │
│  │  • Resource sensitivity (file sensitivity scores)            │   │
│  │  • Graph features (degree, interaction count)                │   │
│  └───────────────────────────────────────────────────────────────┘   │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────────┐
│                      NEURAL NETWORK PIPELINE                          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐        │
│  │  TGN Model   │ ────▶│  GAT Model   │ ────▶│  MLP         │        │
│  │              │      │              │      │  Classifier  │        │
│  │ • Memory     │      │ • Multi-head │      │  256 → 128   │        │
│  │ • Messages   │      │ • Attention  │      │  → 1 (risk)  │        │
│  │ • Aggregator │      │ • Residual   │      │              │        │
│  └──────────────┘      └──────────────┘      └──────────────┘        │
│                             │                                         │
│                      ┌──────▼───────┐                                 │
│                      │ Risk Score   │                                 │
│                      │ [0, 1]       │                                 │
│                      │ 0=Normal,    │                                 │
│                      │ 1=Threat     │                                 │
│                      └──────────────┘                                 │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Input Data (Raw)

| File | Description | Key Fields |
|------|-------------|------------|
| `logon.csv` | User login/logout events | id, date, user, pc, activity |
| `email.csv` | Email communications | id, date, user, to, cc, bcc, from, size, attachment_count |
| `file.csv` | File access/copy events | id, date, user, pc, filename, content |
| `http.csv` | Web browsing history | id, date, user, pc, url, content |
| `device.csv` | USB/device connection events | id, date, user, pc, activity |
| `LDAP.csv` | User organizational data | user_id, department, role, team, supervisor |
| `psychometric.csv` | Personality assessments | employee_name, user_id, O, C, E, A, N (Big 5) |

### Processed Data

| File | Description | Purpose |
|------|-------------|---------|
| `unified_events.csv` | All events normalized to common schema | Pipeline input |
| `behavior_features.csv` | Per-user behavioral statistics | Feature fusion |
| `psychology_features.csv` | Psychological scores per user | Feature fusion |
| `file_sensitivity.csv` | File sensitivity scores per extension | Feature fusion |
| `fused_features.csv` | Combined features for all users | TGN memory initialization |
| `edge_features_*.pt` | 56-dim feature vectors per edge | GNN training |

---

## Project Structure

```
InsiderThreatSystem/
├── data/
│   ├── raw/                          # Source data files
│   │   ├── logon.csv
│   │   ├── email.csv
│   │   ├── file.csv
│   │   ├── http.csv
│   │   ├── device.csv
│   │   ├── LDAP.csv
│   │   └── psychometric.csv
│   │
│   └── processed/                    # Intermediate and final features
│       ├── unified_events.csv
│       ├── behavior_features.csv
│       ├── psychology_features.csv
│       ├── file_sensitivity.csv
│       └── fused_features.csv
│
├── graph/                            # Graph construction and features
│   ├── build_event_graph.py          # Main graph builder (streaming)
│   ├── edge_features.py              # Feature engineering (40+16 dims)
│   ├── edge_weighting.py             # Dynamic edge weighting
│   └── output/                       # Generated outputs
│       ├── node_graph_skeleton.pt    # Graph structure (nodes + structural edges)
│       ├── edge_shard_manifest.json  # Shard index (O(num_shards), not O(num_edges))
│       └── edge_feature_shards/      # Sharded edge features
│           ├── User__accesses__PC/
│           ├── User__visits__WebsiteDomain/
│           └── User__touches__FileExtension/
│
├── models/                           # Neural network modules
│   ├── tgn_model.py                  # Temporal Graph Network
│   ├── gat_model.py                  # Graph Attention Network
│   ├── mlp_classifier.py             # MLP Risk Classifier
│   ├── aggregator.py                 # Message aggregation strategies
│   ├── memory.py                     # TGN memory module
│   └── time_encoder.py               # Time encoding for TGN
│
├── training/                         # Training pipeline
│   ├── train.py                      # Main training orchestration
│   └── config.py                     # Configuration management
│
├── behavior/                         # Behavior feature extraction
│   └── behavior_features.py          # Extract behavioral metrics
│
├── fusion/                           # Feature fusion pipeline
│   └── feature_fusion.py             # Merge all features
│
├── alerts/                           # Alert generation (empty - placeholder)
│   └── alert_engine.py
│
├── utils/                            # Utility functions
│
├── training/                         # Training pipeline
│   └── train.py                      # Main training orchestration
│
├── training/                         # Training pipeline
│   └── config.py                     # Configuration management
│
├── psychology/                       # Psychology feature processing
│
├── sensitivity/                      # File sensitivity scoring
│
├── evaluate.py                       # Evaluation script
├── inspect_data.py                   # Data exploration script
├── train.py                          # Wrapper training script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Installation

```bash
# Clone the repository
cd InsiderThreatSystem

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install torch torch-geometric pandas numpy scikit-learn tqdm
pip install tensorboard psutil
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torch-geometric
- pandas, numpy, scikit-learn

---

## Usage

### 1. Data Preprocessing

The system expects raw data in the `data/raw/` directory. If using the CERT dataset:

```bash
# Place all raw CSV files in data/raw/
# Files should include: logon.csv, email.csv, file.csv, http.csv, device.csv
```

### 2. Build Unified Events

Combine all event sources into a normalized schema:

```bash
python preprocessing/build_unified_events.py
```

This creates `data/processed/unified_events.csv` with columns:
- `user_id`, `timestamp`, `event_type`, `resource`, `target_user`

### 3. Build Graph Structure

```bash
python graph/build_event_graph.py \
    --events data/processed/unified_events.csv \
    --fused-features data/processed/fused_features.csv \
    --file-sensitivity data/processed/file_sensitivity.csv \
    --ldap data/raw/LDAP.csv \
    --output-dir graph/output \
    --chunk-size 500000
```

**Output:**
- `graph/output/node_graph_skeleton.pt` - Graph structure
- `graph/output/edge_shard_manifest.json` - Shard index
- `graph/output/preprocessing_artifacts.pkl` - Lookups and mappings

### 4. Compute Edge Features

```bash
python graph/edge_features.py \
    --graph-output-dir graph/output \
    --psychology-csv data/processed/psychology_features.csv \
    --fused-csv data/processed/fused_features.csv \
    --behavior-csv data/processed/behavior_features.csv
```

**Output:**
- 56-dim feature vectors per edge (40 scalar + 16 temporal encoding)
- Sharded files for memory efficiency

### 5. Train the Model

```bash
python training/train.py \
    --graph-path graph/output/node_graph_skeleton.pt \
    --edge-shard-dir graph/output/edge_feature_shards \
    --labels-dir data/labels \
    --checkpoint-dir checkpoints \
    --log-dir runs
```

**Training Process:**
1. Loads graph structure
2. Streams edge features shard by shard
3. Runs TGN → GAT → MLP forward pass
4. Computes loss and backpropagates
5. Saves checkpoints and logs metrics

### 6. Evaluate

```bash
python training/train.py --self-test
```

---

## Key Components

### Graph Builder (`graph/build_event_graph.py`)

**Key Features:**
- **Streaming architecture** - Processes events in chunks (default 500k rows)
- **No accumulation** - Shards written immediately, never held in RAM
- **Per-relation sorting** - Within-shard chronological order only
- **Two-phase construction**:
  1. Load static data (LDAP, features) into memory
  2. Stream events, write sharded temporal edges

**Output Format:**
```python
{
    "edge_index": LongTensor[2, E],      # [src_ids, dst_ids]
    "edge_time": LongTensor[E],          # Unix timestamps
    "raw_timestamp": LongTensor[E],      # Original timestamps
    "event_type": LongTensor[E],         # Event type IDs
    "event_index": LongTensor[E],        # Original CSV row index
    "resource": LongTensor[E],           # Destination node ID
    "target_user": LongTensor[E],        # Target user (if applicable)
    "src_node_type": str,
    "dst_node_type": str,
    "relation_name": str                 # e.g., "accesses", "visits"
}
```

### Feature Engineer (`graph/edge_features.py`)

**Two-pass design:**
- **Pass 1**: Compute per-user temporal state (O(num_users) memory)
- **Pass 2**: Compute full features per edge, write shards

**40 Scalar Features:**
- **Temporal (0-10)**: Hour, weekday, after-hours flags, time since last event
- **Behavior (11-17)**: User anomaly score, activity frequency, device switching
- **Psychological (18-20)**: Big 5 personality traits, behavior deviation
- **Organizational (21-25)**: Department/role/team encoded
- **Resource (26-29)**: File sensitivity, domain popularity
- **Graph (30-32)**: Node degrees, historical interaction count
- **Event/Relation One-hot (33-39)**: Event type and relation type indicators

**16-Dim Temporal Encoding:**
- Sinusoidal time encoding (TGN-style)
- Captures temporal patterns at multiple scales

### Temporal Graph Network (`models/tgn_model.py`)

**Architecture:**
```
Edge Shard
    │
    ▼
┌──────────────────────────────────┐
│ 1. Time Encoder (Δt)            │
│    - Learned sinusoidal encoding │
│    - Time since last update      │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ 2. Message Function             │
│    - Concat: [src_mem, dst_mem,  │
│      edge_features, time_enc,    │
│      edge_weight]                │
│    - MLP → message vectors       │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ 3. Message Aggregator           │
│    - Last (most recent)          │
│    - Mean (average)              │
│    - Attention (weighted)        │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ 4. GRU Memory Update            │
│    - Read memory for all nodes   │
│    - Compute new memory (GRU)    │
│    - Commit updates atomically   │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ 5. Embedding Module             │
│    - [memory, aggregated_msg,    │
│      time_encoding] → embedding  │
└──────────────────────────────────┘
```

**Key Design Decisions:**
- Memory is **state**, not parameters (reset per session)
- Updates computed from **shared pre-batch snapshot** (prevents race conditions)
- Messages flow in **both directions** (mutual influence)

### Graph Attention Network (`models/gat_model.py`)

**Architecture:**
```
Node Embeddings (TGN output)
    │
    ▼
┌──────────────────────────────────┐
│ Multi-Head Attention Layer 1    │
│ - 4 heads, 32 dim each          │
│ - Concat output                 │
│ - ELU + Dropout                 │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ Multi-Head Attention Layer 2    │
│ - 4 heads, 128 dim (averaged)   │
│ - Residual connection           │
│ - LayerNorm                     │
└──────────────────────────────────┘
    │
    ▼
Contextual Embeddings
```

**Attention Formula:**
```
e_ij = LeakyReLU( a^T [Wh_i || Wh_j || edge_weight_ij] )
alpha_ij = softmax_j(e_ij)
output_i = Σ_j alpha_ij * Wh_j
```

**Key Feature:** Edge weight incorporated directly into attention computation.

### MLP Classifier (`models/mlp_classifier.py`)

**Architecture:**
```
GAT Embeddings [N, 128]
    │
    ▼
Linear(128 → 256) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear(128 → 1) + Sigmoid
    │
    ▼
Risk Score [N, 1] ∈ [0, 1]
```

**Training Loss:**
- `BCEWithLogitsLoss` (numerically stable, preferred)
- Alternative: `BCELoss` (requires pre-sigmoid probabilities)

---

## Training Pipeline

### Configuration

```python
# See training/config.py for all options
TrainingConfig(
    graph_path="graph/output/node_graph_skeleton.pt",
    edge_shard_dir="graph/output/edge_feature_shards",
    labels_dir="data/labels",
    
    # Model dimensions
    tgn_embedding_dim=128,
    gat_embedding_dim=128,
    
    # Optimization
    batch_size=32,
    epochs=100,
    learning_rate=1e-3,
    optimizer="adamw",
    
    # Regularization
    patience=10,
    grad_clip_norm=5.0,
    
    # Hardware
    device="auto",  # cuda/cpu/mps
    seed=42,
)
```

### Pipeline Stages

1. **Load Production Graph** - Skeleton with nodes + structural edges
2. **Discover Edge Shards** - From manifest (O(num_shards), not O(num_edges))
3. **Initialize Models** - TGN → GAT → MLP
4. **Epoch Loop:**
   - Stream shards (memory-efficient)
   - Forward pass: TGN → GAT → MLP
   - Compute loss on labeled nodes
   - Backward pass + optimizer step
   - Validate on hold-out set
5. **Checkpoint** - Best model + training state
6. **Early Stopping** - Based on validation F1

### Self-Test

```bash
python training/train.py --self-test
```

Verifies:
- All production modules load correctly
- Models initialize without errors
- Forward/backward pass works
- Checkpoint saving/loading
- Metrics computation

---

## Model Architecture

### TGN Configuration
```python
TGNConfig(
    memory_dim=128,
    time_dim=32,
    embedding_dim=128,
    message_hidden_dim=128,
    aggregator="attention",  # or "last", "mean"
)
```

### GAT Configuration
```python
GATConfig(
    in_dim=128,         # TGN output dim
    hidden_dim=128,
    out_dim=128,
    heads=4,
    dropout=0.2,
)
```

### MLP Classifier Configuration
```python
MLPClassifierConfig(
    hidden_dim_1=256,
    hidden_dim_2=128,
    dropout=0.3,
)
```

---

## Data Formats

### Edge Shard Schema
```python
{
    "edge_index": LongTensor[2, E],       # COO format edge list
    "edge_time": LongTensor[E],          # Unix timestamps (seconds)
    "features": FloatTensor[E, 40],      # 40 scalar features
    "temporal_encoding": FloatTensor[E, 16],  # 16-dim sinusoidal encoding
    "feature_names": List[str],          # Feature names for debugging
    "feature_version": str,              # Schema version
    "relation": str,                     # Edge type (e.g., "User__accesses__PC")
    "src_node_type": str,                # Source node type
    "dst_node_type": str,                # Destination node type
    "shard_index": int,                  # Shard number
    "num_edges": int,                    # Number of edges
}
```

### Labels Format
Supported: `labels.pt`, `labels.csv`, `labels.pkl`

CSV format:
```csv
node_id,label
0,0.0
1,1.0
2,0.0
...
```

---

## Configuration

### Environment Variables
- `EDGE_FEATURE_SHARD_DIR` - Override default shard location

### Config File
```json
{
  "graph_path": "graph/output/node_graph_skeleton.pt",
  "edge_shard_dir": "graph/output/edge_feature_shards",
  "labels_dir": "data/labels",
  "checkpoint_dir": "checkpoints",
  "log_dir": "runs",
  "epochs": 100,
  "learning_rate": 0.001,
  "tgn_embedding_dim": 128,
  "gat_embedding_dim": 128,
  "seed": 42
}
```

---

## Troubleshooting

### Common Issues

**1. Missing Production Artifacts**
```
ERROR: Cannot start training: the following production modules could not be imported
```
**Solution:** Ensure `graph/` directory exists with all required files and is on PYTHONPATH.

**2. Node Count Mismatch**
```
ValueError: edge_attr has X features but model expects Y
```
**Solution:** Verify `edge_features.py` output matches `edge_weighting.py` input dimension (should be 56).

**3. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** 
- Reduce `chunk_size` in `build_event_graph.py`
- Use `--device cpu` for training
- Reduce batch size in config

**4. Empty Labels**
```
ProductionArtifactError: Labels file loaded but contained zero entries
```
**Solution:** Create `data/labels/labels.csv` with user IDs and risk labels (0=normal, 1=threat).

**5. Shard Not Found**
```
ProductionArtifactError: No shards matching 'edge_features_*.pt' found
```
**Solution:** Run `graph/edge_features.py` first to generate feature shards.

---

## Performance Considerations

### Memory Optimization
- **Streaming architecture** - Never holds full edge list in RAM
- **Sharded features** - O(chunk_size) peak memory, not O(num_edges)
- **Edge features** - 56 bytes per edge (40 scalar + 16 temporal encoding)

### Speed Optimization
- **GPU training** - Enable CUDA for TGN/GAT/MLP
- **Mixed precision** - Uses AMP on supported hardware
- **Gradient checkpointing** - Trade compute for memory

### Scaling
- **1M edges** - ~56 MB edge features, ~100 shards at 500k chunk size
- **10M edges** - ~560 MB features, ~1000 shards
- **100M edges** - ~5.6 GB features, ~10000 shards (requires disk space)

---

## Contributing

1. Follow existing code patterns
2. Add comments for complex logic
3. Update this README for major changes
4. Test with `--self-test` flag

---

## License

This project uses the CERT Insider Threat Dataset (Release 4, Dataset 2).
See `data/raw/license.txt` for usage restrictions.

---

## Contact

For questions or issues, review the module docstrings in individual files or run:
```bash
python graph/build_event_graph.py --help
python training/train.py --help
```
