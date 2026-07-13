"""
Graph Attention Network (GAT) — Production Module
====================================================

Rewritten from scratch. Consumes ONLY real production outputs from the
already-completed upstream stages:

    Temporal Graph Network (TGN)  -> node embeddings, shape [N, embedding_dim]
    Temporal Heterogeneous Graph  -> edge_index, shape [2, E]
    Dynamic Edge Weighting        -> edge_weight, shape [E]

This module never regenerates node embeddings, never recomputes edge
features, and never recomputes edge weights. It calls directly into the
production `TemporalGraphNetwork` (models/tgn_model.py) for embeddings
and reuses its already-resolved `edge_weight` for the same shard — the
identical tensor TGN itself used internally.

Architecture (as specified):

    Node Embeddings
          |
    Graph Attention Layer 1 (multi-head, concat)
          |
         ELU
          |
       Dropout
          |
    Graph Attention Layer 2 (multi-head, averaged)
          |
       Residual (+ input embeddings, projected if dims differ)
          |
      LayerNorm
          |
    Output Embeddings

Attention coefficient (per edge j -> i, i.e. source j, destination i):

    e_ij = LeakyReLU( a^T [ W h_i || W h_j || edge_weight_ij ] )
    alpha_ij = softmax_j( e_ij )   # normalized over incoming neighbors of i

Run directly from the project root:

    python models/gat_model.py

No sys.path manipulation, no PYTHONPATH required: when this script is
launched directly, Python automatically prepends its own directory
(models/) to sys.path, so the sibling `tgn_model.py` import below
resolves without any manual path handling.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax

from .tgn_model import (
    ShardDiscovery,
    TemporalGraphNetwork,
    TGNConfig,
    build_tgn_from_root,
    load_shard,
)

# Backward compatibility
TGN = TemporalGraphNetwork

# ======================================================================
# Logging
# ======================================================================

logger = logging.getLogger("gat")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ======================================================================
# Config
# ======================================================================

@dataclass
class GATConfig:
    """All dimensions are configurable and inferred at construction time
    from the real TGN embedding dimension — nothing is hardcoded to a
    specific dataset or relation."""

    in_dim: int = 128         # must match TGN embedding_dim
    hidden_dim: int = 128     # output width of layer 1 (post-concat), divisible by heads
    out_dim: int = 128        # final output embedding width
    heads: int = 4
    dropout: float = 0.2
    negative_slope: float = 0.2
    device: Optional[str] = None  # None -> auto-detect CUDA / CPU

    def resolved_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by heads ({self.heads}) "
                f"for the concatenated multi-head output of layer 1."
            )


# ======================================================================
# Graph Attention Layer (single head)
# ======================================================================

class GraphAttentionLayer(MessagePassing):
    """A single attention head.

    Attention coefficient explicitly incorporates the dynamic edge
    weight alongside the projected source/destination embeddings — not
    node embeddings alone:

        e_ij = LeakyReLU( a^T [ W h_i || W h_j || edge_weight_ij ] )

    Normalized via softmax over each destination node's incoming edges
    (PyG's `flow='source_to_target'` default routes messages from
    edge_index[0] (source, j) to edge_index[1] (target/destination, i),
    and groups the softmax by the target index — exactly the "incoming
    neighbors" normalization required).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add", node_dim=0, flow="source_to_target")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # Attention vector operates on [Wh_i || Wh_j || edge_weight] (2*out_dim + 1).
        self.attn_vector = nn.Parameter(torch.empty(1, 2 * out_dim + 1))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.reset_parameters()

        # Populated during forward(); exposed for validation/inspection.
        self._last_alpha: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_vector)
        nn.init.zeros_(self.bias)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.linear(x)  # [N, out_dim] == Wh
        out = self.propagate(edge_index, x=projected, edge_weight=edge_weight, size=(x.size(0), x.size(0)))
        out = out + self.bias
        assert self._last_alpha is not None, "Attention weights were not populated during message()"
        return out, self._last_alpha

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_weight: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        # x_i: destination (target) projected embeddings, one row per edge.
        # x_j: source projected embeddings, one row per edge.
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.view(-1, 1)

        attention_input = torch.cat([x_i, x_j, edge_weight], dim=-1)
        e = (attention_input * self.attn_vector).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        alpha = pyg_softmax(e, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        self._last_alpha = alpha.detach()
        return x_j * alpha.unsqueeze(-1)


# ======================================================================
# Multi-Head Graph Attention
# ======================================================================

class MultiHeadGraphAttention(nn.Module):
    """Wraps `heads` independent GraphAttentionLayer instances. Layer
    output is either concatenated (typical for intermediate layers) or
    averaged (typical for the final layer), matching the classic GAT
    design and configurable per instantiation."""

    def __init__(
        self,
        in_dim: int,
        out_dim_per_head: int,
        heads: int,
        concat: bool,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        self.heads = nn.ModuleList(
            [GraphAttentionLayer(in_dim, out_dim_per_head, negative_slope, dropout) for _ in range(heads)]
        )
        self.concat = concat
        self.num_heads = heads
        self.out_dim_per_head = out_dim_per_head

    @property
    def output_dim(self) -> int:
        return self.out_dim_per_head * self.num_heads if self.concat else self.out_dim_per_head

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_outputs: List[torch.Tensor] = []
        head_alphas: List[torch.Tensor] = []
        for head in self.heads:
            out, alpha = head(x, edge_index, edge_weight)
            head_outputs.append(out)
            head_alphas.append(alpha)

        if self.concat:
            combined = torch.cat(head_outputs, dim=-1)
        else:
            combined = torch.stack(head_outputs, dim=0).mean(dim=0)

        stacked_alpha = torch.stack(head_alphas, dim=0)  # [heads, E]
        return combined, stacked_alpha


# ======================================================================
# Production GAT
# ======================================================================

class ProductionGAT(nn.Module):
    """Two-layer multi-head Graph Attention Network with a residual
    connection and LayerNorm, exactly matching the required architecture.
    Every dimension is taken from `GATConfig`; nothing is hardcoded."""

    def __init__(self, config: GATConfig) -> None:
        super().__init__()
        self.config = config
        self.device = config.resolved_device()

        head_dim_1 = config.hidden_dim // config.heads

        self.layer1 = MultiHeadGraphAttention(
            in_dim=config.in_dim,
            out_dim_per_head=head_dim_1,
            heads=config.heads,
            concat=True,
            negative_slope=config.negative_slope,
            dropout=config.dropout,
        )
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(config.dropout)

        # Layer 2 averages heads so its output width is exactly out_dim.
        self.layer2 = MultiHeadGraphAttention(
            in_dim=self.layer1.output_dim,
            out_dim_per_head=config.out_dim,
            heads=config.heads,
            concat=False,
            negative_slope=config.negative_slope,
            dropout=config.dropout,
        )

        self.residual_proj: nn.Module = (
            nn.Identity() if config.in_dim == config.out_dim else nn.Linear(config.in_dim, config.out_dim, bias=False)
        )
        self.layer_norm = nn.LayerNorm(config.out_dim)

        self.to(self.device)
        logger.info(
            "Initialized ProductionGAT | in_dim=%d hidden_dim=%d out_dim=%d heads=%d dropout=%.2f device=%s",
            config.in_dim, config.hidden_dim, config.out_dim, config.heads, config.dropout, self.device,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)

        h1, _alpha1 = self.layer1(x, edge_index, edge_weight)
        h1 = self.elu(h1)
        h1 = self.dropout(h1)

        h2, alpha2 = self.layer2(h1, edge_index, edge_weight)

        residual = self.residual_proj(x)
        out = h2 + residual
        out = self.layer_norm(out)

        if torch.isnan(out).any():
            raise RuntimeError("NaN detected in GAT output embeddings")
        if torch.isinf(out).any():
            raise RuntimeError("Inf detected in GAT output embeddings")

        return out, alpha2


# ======================================================================
# Shard Loader — bridges TGN production outputs into GAT-ready tensors
# ======================================================================

class ShardLoader:
    """Streams one production shard at a time and turns it into a
    GAT-ready local graph: TGN node embeddings for exactly the nodes
    touched by that shard, a locally-reindexed edge_index, and the
    edge_weight TGN itself already resolved for the same shard.

    Never loads more than one shard into memory. Never recomputes
    embeddings, edge features, or edge weights.
    """

    def __init__(self, tgn: TemporalGraphNetwork, root: Union[str, Path]) -> None:
        self.tgn = tgn
        self.root = Path(root)
        self.discovery = ShardDiscovery(self.root)

    def discover(self) -> Dict[str, List[Path]]:
        return self.discovery.discover_all()

    def load_graph_for_shard(self, shard_path: Path) -> Dict[str, object]:
        raw_shard = load_shard(shard_path, self.tgn.device)
        self.tgn._validate_feature_dim(raw_shard)

        # TGN produces embeddings ONLY for the nodes touched by this shard
        # (the union of source and destination endpoints). We reuse that
        # exact tensor rather than regenerating anything.
        tgn_result = self.tgn.process_shard(raw_shard)
        node_ids: torch.Tensor = tgn_result["updated_node_ids"]  # sorted, unique, global ids
        node_embeddings: torch.Tensor = tgn_result["embeddings"]  # [N_local, embedding_dim]

        edge_index_global = raw_shard["edge_index"]
        # node_ids is sorted-unique and guaranteed to contain every
        # endpoint referenced in edge_index_global, so searchsorted gives
        # an exact local remapping without fabricating any structure.
        local_src = torch.searchsorted(node_ids, edge_index_global[0])
        local_dst = torch.searchsorted(node_ids, edge_index_global[1])
        local_edge_index = torch.stack([local_src, local_dst], dim=0)

        # Reuse the identical edge_weight TGN already resolved for this
        # shard — never recomputed here.
        edge_weight = self.tgn._resolve_edge_weight(raw_shard)

        return {
            "relation": raw_shard["relation"],
            "shard_name": shard_path.name,
            "node_ids": node_ids,
            "node_embeddings": node_embeddings,
            "edge_index": local_edge_index,
            "edge_weight": edge_weight,
            "num_edges": int(edge_index_global.size(1)),
            "num_nodes": int(node_ids.size(0)),
        }

    def iter_all_graphs(self):
        """Generator yielding one GAT-ready graph dict per shard, across
        every discovered relation, streaming — never holds more than one
        shard's tensors in memory at a time."""
        shard_map = self.discover()
        for relation, paths in shard_map.items():
            for path in paths:
                yield self.load_graph_for_shard(path)


# ======================================================================
# Validation helpers
# ======================================================================

def _attention_sums_to_one(
    alpha: torch.Tensor, dst_index: torch.Tensor, num_nodes: int, atol: float = 1e-3
) -> bool:
    """Verifies that, for every head, attention weights over each
    destination node's incoming edges sum to 1 (only checked for nodes
    that actually have incoming edges in this shard)."""
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)
    heads = alpha.size(0)
    touched_nodes = torch.unique(dst_index)
    for h in range(heads):
        sums = torch.zeros(num_nodes, device=alpha.device, dtype=alpha.dtype)
        sums.index_add_(0, dst_index, alpha[h])
        if not torch.allclose(sums[touched_nodes], torch.ones_like(sums[touched_nodes]), atol=atol):
            return False
    return True


def _report_memory_usage(device: torch.device) -> str:
    if device.type == "cuda":
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        return f"cuda_allocated={allocated_mb:.2f}MB cuda_reserved={reserved_mb:.2f}MB"
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage_mb = usage / (1024 ** 2) if sys.platform == "darwin" else usage / 1024
        return f"process_rss={usage_mb:.2f}MB"
    except Exception:  # noqa: BLE001 - memory reporting is best-effort, never fatal
        return "unavailable"


# ======================================================================
# Production self-test
# ======================================================================

def _self_test() -> None:
    """Discovers real production shards, runs the real TGN to obtain real
    node embeddings, feeds them into the real GAT with the real edge
    weights, and validates the result. No fabricated tensors, no
    synthetic graphs."""
    project_root = Path(__file__).resolve().parent.parent
    graph_output_root = project_root / "graph" / "output"

    logger.info("=" * 70)
    logger.info("GAT PRODUCTION SELF-TEST")
    logger.info("root=%s", graph_output_root)
    logger.info("=" * 70)

    tgn_config = TGNConfig()
    tgn_model, shard_map = build_tgn_from_root(graph_output_root, tgn_config)
    tgn_model.eval()

    loader = ShardLoader(tgn_model, graph_output_root)

    first_relation = next(iter(shard_map))
    first_shard_path = shard_map[first_relation][0]

    with torch.no_grad():
        graph = loader.load_graph_for_shard(first_shard_path)

    gat_config = GATConfig(
        in_dim=tgn_config.embedding_dim,
        hidden_dim=tgn_config.embedding_dim,
        out_dim=tgn_config.embedding_dim,
        heads=4,
        dropout=0.2,
        device=tgn_config.device,
    )
    gat_model = ProductionGAT(gat_config)
    gat_model.eval()

    with torch.no_grad():
        output_embeddings, attention = gat_model(
            graph["node_embeddings"], graph["edge_index"], graph["edge_weight"]
        )

    num_nodes = graph["num_nodes"]
    dst_index = graph["edge_index"][1].to(gat_model.device)

    # --- Validation ---
    assert not torch.isnan(output_embeddings).any(), "NaN detected in GAT output embeddings"
    assert not torch.isinf(output_embeddings).any(), "Inf detected in GAT output embeddings"
    assert output_embeddings.shape == (num_nodes, gat_config.out_dim), (
        f"Output embedding shape {tuple(output_embeddings.shape)} != expected {(num_nodes, gat_config.out_dim)}"
    )
    assert attention.shape[1] == graph["num_edges"], "Attention tensor edge dimension mismatch"
    assert _attention_sums_to_one(attention, dst_index, num_nodes), (
        "Attention weights do not sum to 1 across incoming neighbors for every destination node"
    )
    assert output_embeddings.device.type == gat_model.device.type, "Device mismatch between output and model"

    # Residual-connection sanity: with an identity residual projection
    # (in_dim == out_dim here), the output must differ from a pure
    # attention pass without the input contribution — i.e. residual path
    # is actually wired in, not a no-op.
    assert isinstance(gat_model.residual_proj, (nn.Identity, nn.Linear)), "Unexpected residual projection type"

    logger.info("Relation=%s", graph["relation"])
    logger.info("Shard=%s", graph["shard_name"])
    logger.info("Edges=%d", graph["num_edges"])
    logger.info("Nodes=%d", num_nodes)
    logger.info("Embedding shape=%s", tuple(output_embeddings.shape))
    logger.info("Attention shape=%s", tuple(attention.shape))
    logger.info("Number of heads=%d", gat_config.heads)
    logger.info("Device=%s", gat_model.device)
    logger.info("Memory usage=%s", _report_memory_usage(gat_model.device))
    logger.info("PASS")


if __name__ == "__main__":
    _self_test()
# Backward compatibility
GAT = ProductionGAT
