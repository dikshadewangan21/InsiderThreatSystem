"""
graph/edge_weighting.py
=======================
Production-grade Learnable Dynamic Edge Weighting for the CERT insider-threat
temporal heterogeneous graph pipeline.

This module is position-independent in the pipeline.  It receives the rich
edge feature tensor produced by ``graph/edge_features.py`` and outputs a
scalar weight in (0, 1) for every edge.  Those weights modulate message
passing in the downstream TGN memory module and GAT attention layers.

Architecture (UNCHANGED from the validated version)
-----------------------------------------------------
edge_attr [E, F]
    │
    ▼
Linear(F → hidden1)          # project to latent space
    │
LayerNorm(hidden1)           # stabilise training, esp. with mixed precision
    │
GELU                         # smooth non-linearity
    │
Dropout(p)                   # regularisation
    │
Linear(hidden1 → hidden2)    # compress
    │
GELU
    │
Linear(hidden2 → 1)          # scalar logit
    │
Sigmoid (fp32-safe, see below)
    │
edge_weight [E]

--------------------------------------------------------------------------
PRODUCTION REVIEW NOTES (see accompanying writeup for full explanation)
--------------------------------------------------------------------------
Every change below is opt-in via a new keyword argument whose default
reproduces the original module's exact numeric behavior. No existing caller,
checkpoint, or integration test is broken by upgrading to this file.

1. Sigmoid is now computed explicitly in forward() instead of as the last
   layer of nn.Sequential, always upcast to float32 immediately before the
   nonlinearity and cast back to the input dtype afterward. In fp32 this is
   bit-identical to the old behavior. Under fp16/bf16 autocast, this prevents
   the gate from saturating to exact 0.0/1.0 (a real risk given fp16's ~11
   representable logit range before exp() over/underflows), which previously
   could zero out gradients for legitimately anomalous edges and produce
   log(0) if any downstream loss/attention takes a log of the weight.
   Removing Sigmoid from nn.Sequential does NOT change any state_dict key
   (Sigmoid has no parameters), so old checkpoints load into this class with
   no changes.
2. New optional `output_eps` (default 0.0, i.e. disabled): when set > 0,
   bounds edge_weight strictly inside (output_eps, 1 - output_eps) instead of
   the mathematically-open-but-practically-saturating (0, 1). Off by default.
3. New optional `return_logits` forward() argument (default False): lets a
   training loop request the pre-sigmoid logit so it can use
   BCEWithLogitsLoss (numerically fused, stable) instead of BCELoss on an
   already-squashed probability. Default False preserves the existing return
   value exactly.
4. New optional `chunk_size` / `use_checkpoint` forward() arguments (both
   default to the old single-pass behavior): bound peak activation memory
   for very large shards, and optionally trade compute for memory via
   gradient checkpointing during training. All ops in this network are
   row-independent (Linear, LayerNorm-per-row, elementwise activations), so
   chunking produces numerically identical results in eval mode; in training
   mode dropout draws a different random mask per chunk than an unchunked
   pass would, which is statistically equivalent regularization, not a bug.
5. Checkpoint I/O (`save_weighter` / `load_weighter`) gained defensive
   validation and a `strict` passthrough, and `save_weighter` now embeds a
   schema-version tag. Existing checkpoints still load unchanged.
6. `apply_edge_weights`'s docstring previously claimed to return a
   "SparseTensor-compatible structure" -- it actually returns the plain
   (edge_index, edge_weight) COO tuple, which IS what PyG's `edge_weight`
   argument on layers like GCNConv expects, but is NOT a `torch_sparse.
   SparseTensor` object. Docstring corrected; a new, separate, additive
   `to_sparse_tensor()` helper is provided for callers who actually need a
   `torch_sparse.SparseTensor` (soft/optional dependency, only imported if
   used).
7. GATConv (named as the downstream attention layer in your pipeline) does
   NOT accept a bare scalar `edge_weight` kwarg the way GCNConv does -- its
   attention is computed from node/edge features directly. To actually gate
   GAT's attention with this module's output, the TGN/GAT integration code
   will need to either (a) concatenate edge_weight as an extra edge_attr
   column before GATConv, or (b) multiply GATConv's returned attention
   coefficients post-hoc. This is a downstream integration note, not
   something fixable inside this file.
8. The original module docstring claimed input dimensionality is "inferred
   at first forward pass" -- the implementation actually requires an
   explicit `in_features` at construction and never used LazyLinear. That is
   the right call for a checkpointed production module (deterministic
   config, no first-batch side effects), so the code is unchanged; only the
   inaccurate docstring claim was corrected.
9. NOT changed, considered and rejected: switching GELU to
   `approximate="tanh"` (marginal inference speedup) and adding a second
   Dropout after the stage-2 GELU. Both would silently alter the forward
   pass's numeric output relative to the already-validated checkpoint /
   integration tests, which is not "absolutely necessary" and was explicitly
   out of scope.

--------------------------------------------------------------------------
KNOWN INTEGRATION MISMATCH TO RESOLVE BEFORE TRAINING (not fixable here)
--------------------------------------------------------------------------
This module's docstring/example instantiate it with `in_features=40`. The
edge_features.py in this conversation's history produces 13 scalar features
+ an 8-dim temporal encoding (21 total, in two separate tensors, not one
40-wide block). Confirm which edge_features.py version actually produced the
edge feature shards this module was validated against before wiring it into
TGN -- `in_features` must exactly equal whatever width edge_attr the loader
hands this module, or forward() will raise ValueError (correctly -- see
review notes, this fail-loud behavior is correct and was NOT changed).

Usage
-----
    from graph.edge_weighting import DynamicEdgeWeighting, apply_edge_weights

    weighter = DynamicEdgeWeighting(in_features=40).to(device)
    edge_weight = weighter(edge_attr)          # [E], unchanged call site
    ei_weighted = apply_edge_weights(edge_index, edge_weight)

    # New, optional, for large shards during training:
    edge_weight = weighter(edge_attr, chunk_size=200_000, use_checkpoint=True)

    # New, optional, for BCEWithLogitsLoss-based training:
    logits = weighter(edge_attr, return_logits=True)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from torch import Tensor

log = logging.getLogger(__name__)

_CHECKPOINT_SCHEMA_VERSION = "1.1"


# ---------------------------------------------------------------------------
# DynamicEdgeWeighting
# ---------------------------------------------------------------------------

class DynamicEdgeWeighting(nn.Module):
    """
    Learnable scalar weight predictor for heterogeneous graph edges.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input edge feature vector (F). Must be set
        explicitly to match whatever edge_features.py actually emits --
        this module does not infer it lazily (see module docstring, note 8).
    hidden1 : int
        Width of the first hidden layer. Default: 64.
    hidden2 : int
        Width of the second hidden layer. Default: 32.
    dropout : float
        Dropout probability applied after the first activation. Default: 0.1.
    output_eps : float
        If > 0, bounds edge_weight strictly inside (output_eps, 1-output_eps)
        instead of relying on sigmoid's asymptotic (0, 1). Default 0.0
        (disabled -- reproduces the original module's exact output).

    Inputs
    ------
    edge_attr : Tensor [E, F]
        Edge feature matrix. E = number of edges in the current batch/shard.
        F must equal ``in_features``. Accepts float16 / bfloat16 (AMP).

    Outputs
    -------
    edge_weight : Tensor [E]  (or logits [E] if return_logits=True)
        Scalar weight in (0, 1) for every edge. Higher = stronger / more
        relevant edge for downstream message passing.
    """

    def __init__(
        self,
        in_features: int,
        hidden1:     int   = 64,
        hidden2:     int   = 32,
        dropout:     float = 0.1,
        output_eps:  float = 0.0,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not (0.0 <= output_eps < 0.5):
            raise ValueError(f"output_eps must be in [0, 0.5), got {output_eps}")

        self.in_features = in_features
        self.hidden1     = hidden1
        self.hidden2     = hidden2
        self.dropout_p   = dropout
        self.output_eps  = output_eps

        # NOTE: Sigmoid intentionally lives OUTSIDE this Sequential (applied
        # explicitly in forward()) so it can be computed in fp32 regardless
        # of ambient autocast dtype, and so return_logits can skip it
        # entirely. Sigmoid has no parameters, so removing it from the
        # Sequential does not change any state_dict key -- old checkpoints
        # load into this class unmodified.
        self.net = nn.Sequential(
            # Stage 1: project to latent space
            nn.Linear(in_features, hidden1, bias=True),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            # Stage 2: compress
            nn.Linear(hidden1, hidden2, bias=True),
            nn.GELU(),
            # Stage 3: scalar logit
            nn.Linear(hidden2, 1, bias=True),
        )

        self._init_weights()
        log.info(
            "DynamicEdgeWeighting initialized — in=%d  h1=%d  h2=%d  dropout=%.2f  "
            "output_eps=%.2e  params=%d",
            in_features, hidden1, hidden2, dropout, output_eps, self._count_params(),
        )

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        """Xavier uniform for all Linear layers; zero bias. (unchanged)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _validate_input(self, edge_attr: Tensor) -> None:
        if edge_attr.dim() != 2:
            raise ValueError(
                f"edge_attr must be 2-D [E, F], got shape {tuple(edge_attr.shape)}"
            )
        if edge_attr.shape[1] != self.in_features:
            raise ValueError(
                f"edge_attr has {edge_attr.shape[1]} features but model expects "
                f"{self.in_features}. Rebuild with the correct in_features."
            )

    def _squash(self, logit: Tensor, input_dtype: torch.dtype) -> Tensor:
        """Sigmoid computed in fp32 for numerical safety under AMP, then cast
        back to the caller's dtype. In fp32 end-to-end this is bit-identical
        to `nn.Sigmoid()(logit)`. Optionally bounded away from exact 0/1 via
        output_eps (default 0.0 -> no bounding, exact sigmoid)."""
        prob32 = torch.sigmoid(logit.float())
        if self.output_eps > 0.0:
            prob32 = self.output_eps + (1.0 - 2.0 * self.output_eps) * prob32
        return prob32.to(input_dtype)

    def _run_net(self, chunk: Tensor, use_checkpoint: bool) -> Tensor:
        if use_checkpoint and self.training:
            return torch_checkpoint.checkpoint(self.net, chunk, use_reentrant=False)
        return self.net(chunk)

    def forward(
        self,
        edge_attr: Tensor,
        chunk_size: Optional[int] = None,
        use_checkpoint: bool = False,
        return_logits: bool = False,
    ) -> Tensor:
        """
        Compute a scalar importance weight for every edge.

        Parameters
        ----------
        edge_attr : Tensor [E, F]
            Edge feature matrix. Can be float32, float16, or bfloat16.
        chunk_size : Optional[int]
            If set, processes edge_attr in row-chunks of this size to bound
            peak activation memory on very large (million-edge) shards.
            Default None reproduces the original single-pass behavior
            exactly. All ops in this network are row-independent, so results
            are numerically identical to the unchunked pass in eval mode.
        use_checkpoint : bool
            If True (and the module is in training mode), applies gradient
            checkpointing per chunk to trade recompute for activation memory
            during backward. No effect at inference (model.eval()) or when
            chunk_size is None and use_checkpoint would still apply to the
            single pass. Default False reproduces the original behavior.
        return_logits : bool
            If True, returns the pre-sigmoid logit [E] instead of the
            squashed weight -- for use with BCEWithLogitsLoss. Default False
            preserves the original return value exactly.

        Returns
        -------
        Tensor [E]
            Per-edge scalar weights in (0, 1) (or logits if return_logits).
        """
        self._validate_input(edge_attr)
        input_dtype = edge_attr.dtype

        if chunk_size is None or chunk_size >= edge_attr.shape[0]:
            logit = self._run_net(edge_attr, use_checkpoint).squeeze(-1)
        else:
            if chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {chunk_size}")
            chunks = [
                self._run_net(edge_attr[start : start + chunk_size], use_checkpoint)
                for start in range(0, edge_attr.shape[0], chunk_size)
            ]
            logit = torch.cat(chunks, dim=0).squeeze(-1)

        if return_logits:
            return logit
        return self._squash(logit, input_dtype)

    @torch.no_grad()
    def predict(self, edge_attr: Tensor, chunk_size: Optional[int] = None) -> Tensor:
        """Convenience inference wrapper: forces eval-mode-safe, no-grad
        execution. Equivalent to `model.eval(); model(edge_attr, chunk_size=...)`
        under `torch.no_grad()`, provided as a single call for inference
        pipelines. Does not mutate the module's training/eval mode."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(edge_attr, chunk_size=chunk_size)
        finally:
            self.train(was_training)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden1={self.hidden1}, "
            f"hidden2={self.hidden2}, dropout={self.dropout_p}, "
            f"output_eps={self.output_eps}"
        )

    def config_dict(self) -> Dict[str, object]:
        """Return constructor kwargs — useful for checkpointing.

        NOTE: includes `output_eps`, which older callers of `from_config`
        (predating this review) will not recognize as a constructor kwarg.
        Loading a checkpoint saved by *this* version into *this* version is
        unaffected; only loading a checkpoint saved by this version into an
        older copy of the class would need output_eps stripped first.
        """
        return {
            "in_features": self.in_features,
            "hidden1":     self.hidden1,
            "hidden2":     self.hidden2,
            "dropout":     self.dropout_p,
            "output_eps":  self.output_eps,
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, object]) -> "DynamicEdgeWeighting":
        """Reconstruct from a ``config_dict()`` snapshot. Tolerant of older
        snapshots that predate `output_eps` (falls back to the disabled
        default of 0.0, i.e. original behavior)."""
        cfg = dict(cfg)
        cfg.setdefault("output_eps", 0.0)
        return cls(**cfg)


# ---------------------------------------------------------------------------
# apply_edge_weights
# ---------------------------------------------------------------------------

def apply_edge_weights(
    edge_index:  Tensor,
    edge_weight: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Attach scalar weights to an edge index tensor, ready for downstream GNN.

    This function is deliberately minimal — it validates shapes and returns
    the pair ``(edge_index, edge_weight)`` in COO format. This is exactly
    the format PyTorch Geometric's `edge_weight` argument expects on layers
    such as `GCNConv`. It is NOT a `torch_sparse.SparseTensor` object; if you
    need one, use `to_sparse_tensor()` below. (The original docstring here
    overstated this as "SparseTensor-compatible" -- corrected.)

    GATConv does not accept a bare `edge_weight` kwarg the way GCNConv does;
    see module docstring note 7 for how to route this module's output into
    GAT attention.

    Parameters
    ----------
    edge_index : LongTensor [2, E]
        COO-format edge index (source row 0, destination row 1).
    edge_weight : Tensor [E]
        Per-edge scalar weights, typically the output of
        ``DynamicEdgeWeighting.forward()``.

    Returns
    -------
    edge_index : LongTensor [2, E]
        Unchanged — returned for call-site convenience (unpack the tuple).
    edge_weight : Tensor [E]
        Validated and contiguous weight tensor on the same device as
        ``edge_index``.

    Raises
    ------
    ValueError
        If ``edge_index`` is not shape [2, E] or ``edge_weight`` is not
        shape [E] with the same E.

    Example
    -------
    >>> ei, ew = apply_edge_weights(edge_index, weighter(edge_attr))
    >>> out = gcn_conv(x, ei, edge_weight=ew)
    """
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must be shape [2, E], got {tuple(edge_index.shape)}"
        )
    E = edge_index.shape[1]

    if edge_weight.dim() != 1:
        raise ValueError(
            f"edge_weight must be 1-D [E], got shape {tuple(edge_weight.shape)}"
        )
    if edge_weight.shape[0] != E:
        raise ValueError(
            f"edge_weight has {edge_weight.shape[0]} entries but edge_index has "
            f"{E} edges."
        )

    edge_weight = edge_weight.to(edge_index.device).contiguous()
    return edge_index, edge_weight


def to_sparse_tensor(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_nodes: Optional[int] = None,
):
    """
    Build an actual `torch_sparse.SparseTensor` from (edge_index, edge_weight)
    for PyG sparse-backend layers that want one directly, e.g. some
    `SparseTensor`-based propagate() paths. Optional dependency: only
    imports `torch_sparse` if this function is actually called, so the rest
    of the module works without it installed.

    Parameters
    ----------
    edge_index : LongTensor [2, E]
    edge_weight : Tensor [E]
    num_nodes : Optional[int]
        Total node count; inferred from edge_index.max() + 1 if omitted.

    Returns
    -------
    torch_sparse.SparseTensor

    Raises
    ------
    ImportError
        If `torch_sparse` is not installed.
    """
    try:
        from torch_sparse import SparseTensor
    except ImportError as e:
        raise ImportError(
            "to_sparse_tensor() requires the optional 'torch_sparse' package "
            "(pip install torch_sparse). apply_edge_weights() does not need it."
        ) from e

    edge_index, edge_weight = apply_edge_weights(edge_index, edge_weight)
    n = num_nodes if num_nodes is not None else int(edge_index.max().item()) + 1
    return SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(n, n),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_weighter(
    model:    DynamicEdgeWeighting,
    path:     str,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """
    Save model state + config so it can be reconstructed without the
    original constructor call.

    Parameters
    ----------
    model    : trained DynamicEdgeWeighting instance.
    path     : destination file (e.g. ``"checkpoints/edge_weighter.pt"``).
    metadata : optional dict of extra fields (epoch, loss, etc.) to embed.
    """
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "state_dict": model.state_dict(),
        "config":     model.config_dict(),
        "metadata":   metadata or {},
    }
    torch.save(payload, path)
    log.info("DynamicEdgeWeighting saved → %s  (schema v%s)", path, _CHECKPOINT_SCHEMA_VERSION)


def load_weighter(path: str, map_location: str = "cpu", strict: bool = True) -> DynamicEdgeWeighting:
    """
    Reconstruct a DynamicEdgeWeighting from a checkpoint written by
    ``save_weighter``.

    Parameters
    ----------
    path         : checkpoint file path.
    map_location : torch device string (default ``"cpu"``).
    strict       : passed through to `load_state_dict`. Default True
                    (unchanged from original behavior). Set False only if
                    loading a checkpoint from a structurally different
                    (but compatible) module version.

    Returns
    -------
    DynamicEdgeWeighting
        Model in eval mode with weights loaded.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if "state_dict" not in payload or "config" not in payload:
        raise ValueError(
            f"Checkpoint at {path} is missing required keys "
            f"('state_dict', 'config'); found keys: {list(payload.keys())}"
        )

    model = DynamicEdgeWeighting.from_config(payload["config"])
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=strict)
    if not strict and (missing or unexpected):
        log.warning(
            "Non-strict load — missing keys: %s, unexpected keys: %s", missing, unexpected
        )
    model.eval()
    log.info(
        "DynamicEdgeWeighting loaded from %s  schema=%s  config=%s",
        path, payload.get("schema_version", "<pre-versioning>"), payload["config"],
    )
    return model


# ---------------------------------------------------------------------------
# Self-test support (only used by the `if __name__ == "__main__"` block at
# the bottom of this file; importing this module never touches any of the
# functions below).
#
# The pipeline stores edge feature shards per-relation:
#   graph/output/edge_feature_shards/<RelationName>/edge_features_NNNNNN.pt
# Relation names (e.g. "User__accesses__PC") are NEVER hardcoded anywhere
# below -- everything is discovered by recursively walking the shard root
# with pathlib and grouping files by their parent directory.
# ---------------------------------------------------------------------------

MANIFEST_FILENAMES = ("edge_shard_manifest.json", "feature_manifest.json")


def _resolve_project_root() -> Path:
    """Best-effort project root: the parent of this file's containing
    'graph' directory. Diagnostic aid only -- discovery below also always
    tries the current working directory regardless of this guess, so a
    wrong guess here cannot cause a false failure."""
    return Path(__file__).resolve().parent.parent


def _candidate_shard_roots(explicit_dir: Optional[str]) -> list[Path]:
    """Every directory we're willing to treat as a candidate
    'edge_feature_shards' root, in priority order. Pure pathlib, no string
    path concatenation, so this behaves identically on Windows and Linux."""
    candidates: list[Path] = []

    if explicit_dir:
        candidates.append(Path(explicit_dir).expanduser())

    env_dir = os.environ.get("EDGE_FEATURE_SHARD_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    cwd = Path.cwd()
    project_root = _resolve_project_root()
    module_dir = Path(__file__).resolve().parent

    for base in (cwd, project_root, module_dir.parent):
        candidates.append(base / "graph" / "output" / "edge_feature_shards")
    candidates.append(cwd / "output" / "edge_feature_shards")
    candidates.append(module_dir / "output" / "edge_feature_shards")

    # De-duplicate by resolved path while preserving first-seen order.
    seen: set[str] = set()
    unique: list[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def discover_relation_shards(shard_root: Path) -> Dict[str, list[Path]]:
    """Recursively discover every `edge_features_*.pt` file under
    `shard_root` using pathlib, grouped by relation.

    The relation for a given shard file is its path relative to
    `shard_root`, minus the filename -- i.e. its parent directory name in a
    per-relation layout (`edge_feature_shards/<Relation>/edge_features_*.pt`).
    A flat layout (`edge_feature_shards/edge_features_*.pt`, no relation
    subdirectory) is also supported and grouped under the sentinel name
    "(root)", so this function works for both today's per-relation layout
    and any flat layout used elsewhere without special-casing either one.

    No relation name is ever hardcoded: this is purely a directory walk.
    """
    if not shard_root.exists():
        return {}

    relations: Dict[str, list[Path]] = {}
    for pt_file in sorted(shard_root.rglob("edge_features_*.pt")):
        rel_parent = pt_file.parent.relative_to(shard_root)
        relation_name = str(rel_parent) if str(rel_parent) != "." else "(root)"
        relations.setdefault(relation_name, []).append(pt_file)
    return relations


def _print_discovered_relations(relations: Dict[str, list[Path]]) -> None:
    print("Discovered relations:")
    for name in sorted(relations):
        print(f"  {name}")
        print(f"      {len(relations[name])} shards")


def _gather_manifest_diagnostics(
    project_root: Path, cwd: Path
) -> list[Tuple[Path, Optional[dict]]]:
    """Locate any manifest json at conventional graph/output locations and
    attempt to parse it, purely for diagnostic display. A manifest that is
    missing or unparsable is NOT an error here -- shard discovery above does
    not depend on it -- but its contents (or absence) are always reported so
    a failure is never silent."""
    search_dirs = {project_root / "graph" / "output", cwd / "graph" / "output"}
    found: list[Tuple[Path, Optional[dict]]] = []
    seen: set[str] = set()

    for base in search_dirs:
        for name in MANIFEST_FILENAMES:
            path = base / name
            key = str(path.resolve()) if path.exists() else str(path)
            if not path.exists() or key in seen:
                continue
            seen.add(key)
            try:
                with open(path, "r") as f:
                    contents = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("Found manifest at %s but could not parse it: %s", path, e)
                contents = None
            found.append((path, contents))
    return found


def _print_diagnostics(
    candidates: list[Path],
    manifests: list[Tuple[Path, Optional[dict]]],
) -> None:
    """Full diagnostic dump. Called whenever shard discovery fails, so a
    failure can never be silent: working directory, resolved project root,
    every directory searched and whether it exists, total .pt files found,
    and every manifest location + contents (or parse failure)."""
    print("-" * 70)
    print("DIAGNOSTICS")
    print("-" * 70)
    print(f"Current working directory : {Path.cwd()}")
    print(f"Resolved project root     : {_resolve_project_root()}")
    print()
    print("Shard directories searched:")
    total_pt = 0
    for c in candidates:
        exists = c.exists()
        pt_count = len(list(c.rglob("edge_features_*.pt"))) if exists else 0
        total_pt += pt_count
        shown = c.resolve() if exists else c
        print(f"  [{'EXISTS ' if exists else 'MISSING'}] {shown}  ({pt_count} .pt files)")
    print()
    print(f"Total edge_features_*.pt files discovered: {total_pt}")
    print()
    print("Manifest files:")
    if manifests:
        for path, contents in manifests:
            print(f"  location: {path}")
            if contents is not None:
                print("  contents:")
                print(textwrap.indent(json.dumps(contents, indent=2), "    "))
            else:
                print("  contents: <not found or failed to parse>")
    else:
        print("  (no manifest file found at any conventional graph/output location)")
    print("-" * 70)


def select_first_shard(relations: Dict[str, list[Path]]) -> Tuple[str, Path]:
    """Deterministically pick one shard to test: the first shard (by sorted
    filename) of the first relation (by sorted name). Relies only on the
    dict discover_relation_shards() already produced -- no new filesystem
    assumptions, no hardcoded relation names."""
    if not relations:
        raise FileNotFoundError("select_first_shard() called with no discovered relations.")
    relation_name = sorted(relations)[0]
    shard_path = relations[relation_name][0]
    return relation_name, shard_path


def load_shard_edge_attr(shard_path: Path) -> Dict[str, object]:
    """Load one edge_features_*.pt shard using the ACTUAL production schema.

    The shards written by this pipeline contain:
        edge_index, edge_time, features, temporal_encoding, feature_names,
        feature_version, relation, src_node_type, dst_node_type,
        shard_index, num_edges
    They do NOT contain 'edge_attr', 'src_idx', or 'dst_idx' -- those were
    an incorrect assumption in an earlier version of this loader and have
    been removed. edge_features.py and build_event_graph.py are untouched;
    this function only adapts to their existing output.

    The model input tensor is built as:
        edge_attr = features                                   if no temporal_encoding
        edge_attr = concat([features, temporal_encoding], dim=1) otherwise
    matching what the production edge feature engine intends (temporal
    encoding is a per-edge vector meant to travel alongside the scalar
    features, not a replacement for them). Feature dimensionality is never
    hardcoded -- it is read off the loaded tensor's own shape.

    edge_index is used exactly as stored; it is never reconstructed from
    separate src/dst columns, since none exist in this schema.

    Raises specific, descriptive exceptions on malformed input rather than
    suppressing anything -- callers should let these propagate.
    """
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard file does not exist: {shard_path}")

    shard = torch.load(shard_path, map_location="cpu")
    if not isinstance(shard, dict):
        raise TypeError(
            f"Expected {shard_path} to deserialize to a dict, got {type(shard).__name__}"
        )

    if "features" not in shard:
        raise KeyError(
            f"Shard {shard_path} is missing required key 'features'. "
            f"Keys found: {list(shard.keys())}"
        )
    edge_attr = shard["features"]
    if edge_attr.dim() != 2:
        raise ValueError(
            f"Shard {shard_path}: 'features' must be 2-D [E, F], got shape {tuple(edge_attr.shape)}"
        )

    temporal_encoding = shard.get("temporal_encoding")
    if temporal_encoding is not None:
        if temporal_encoding.dim() != 2 or temporal_encoding.shape[0] != edge_attr.shape[0]:
            raise ValueError(
                f"Shard {shard_path}: 'temporal_encoding' shape {tuple(temporal_encoding.shape)} "
                f"is incompatible with 'features' shape {tuple(edge_attr.shape)}"
            )
        edge_attr = torch.cat([edge_attr, temporal_encoding], dim=1)

    if "edge_index" not in shard:
        raise KeyError(
            f"Shard {shard_path} is missing required key 'edge_index'. "
            f"Keys found: {list(shard.keys())}"
        )
    edge_index = shard["edge_index"]
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Shard {shard_path}: 'edge_index' must be shape [2, E], got {tuple(edge_index.shape)}"
        )
    if edge_index.shape[1] != edge_attr.shape[0]:
        raise ValueError(
            f"Shard {shard_path}: edge_index has {edge_index.shape[1]} edges but "
            f"features/temporal_encoding imply {edge_attr.shape[0]} edges."
        )

    return {
        "edge_attr": edge_attr,
        "edge_index": edge_index,
        "relation": shard.get("relation"),
        "raw_keys": list(shard.keys()),
    }



def _run_integration_self_test() -> int:
    """Discover every relation's edge feature shards, load one shard, run it
    through DynamicEdgeWeighting, and report PASS/FAIL. See module docstring
    note on the self-test for the full behavioral contract.

    Returns
    -------
    int
        Process exit code: 0 on pass, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="edge_weighting.py integration self-test "
        "(discovers real edge feature shards produced by edge_features.py)"
    )
    parser.add_argument(
        "--shard-dir", type=str, default=None,
        help="edge_feature_shards root to search (containing per-relation subdirectories)",
    )
    parser.add_argument(
        "--shard-file", type=str, default=None,
        help="Exact shard file to use, bypassing relation discovery entirely",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DynamicEdgeWeighting — integration self-test")
    print("=" * 70)

    cwd = Path.cwd()
    project_root = _resolve_project_root()

    # --- Discovery --------------------------------------------------------
    if args.shard_file:
        shard_path = Path(args.shard_file).resolve()
        relation_name = shard_path.parent.name or "(root)"
        relations: Dict[str, list[Path]] = {relation_name: [shard_path]}
        candidates = [shard_path.parent]
    else:
        candidates = _candidate_shard_roots(args.shard_dir)
        relations = {}
        for root in candidates:
            found = discover_relation_shards(root)
            if found:
                relations = found
                break

        if not relations:
            print("FAIL — Could not locate any edge feature shard (edge_features_*.pt)")
            print()
            manifests = _gather_manifest_diagnostics(project_root, cwd)
            _print_diagnostics(candidates, manifests)
            print("=" * 70)
            print("RESULT: FAIL (no edge feature shard found)")
            print("=" * 70)
            return 1

    _print_discovered_relations(relations)
    print()

    manifests = _gather_manifest_diagnostics(project_root, cwd)
    if manifests:
        for path, _ in manifests:
            print(f"Manifest located: {path}")
    else:
        print("Manifest located: (none found at conventional locations — "
              "proceeded on directory discovery alone, per spec this is not fatal)")
    print()

    # --- Load one shard -----------------------------------------------------
    relation_name, shard_path = select_first_shard(relations)
    shard_path = shard_path.resolve()
    shard_id = shard_path.stem

    loaded = load_shard_edge_attr(shard_path)
    edge_attr = loaded["edge_attr"]
    edge_index = loaded["edge_index"]  # used exactly as stored, never reconstructed
    num_edges = int(edge_attr.shape[0])
    feature_dim = int(edge_attr.shape[1])

    print("Selected shard for testing:")
    print(f"  absolute path : {shard_path}")
    print(f"  relation      : {relation_name}"
          + (f"  (shard metadata relation: {loaded['relation']})" if loaded.get("relation") else ""))
    print(f"  shard id      : {shard_id}")
    print(f"  edge count    : {num_edges}")
    print(f"  feature dim   : {feature_dim}  (features"
          + ("+temporal_encoding" if "temporal_encoding" in loaded["raw_keys"] else "")
          + f", inferred from tensor shape, not hardcoded)")
    print(f"  edge_index    : {tuple(edge_index.shape)}  (used directly, not reconstructed)")
    print()

    if edge_attr is None:
        raise RuntimeError(f"edge_attr resolved to None for shard {shard_path} — cannot proceed.")
    print("✓ edge_attr present (assembled from 'features' + optional 'temporal_encoding')")

    # --- Run DynamicEdgeWeighting --------------------------------------------
    device = torch.device(args.device)
    model = DynamicEdgeWeighting(in_features=feature_dim).to(device)
    model.eval()
    print(f"Instantiated DynamicEdgeWeighting(in_features={feature_dim})  device={device}")

    edge_attr = edge_attr.to(device)
    with torch.no_grad():
        edge_weight = model(edge_attr)
    print(f"Forward pass complete. edge_weight shape={tuple(edge_weight.shape)}  device={edge_weight.device}")

    # Prove the edge_index -> edge_weight handoff that TGN will rely on
    # actually works, using edge_index exactly as loaded from the shard
    # (never reconstructed from separate src/dst columns).
    ei_checked, ew_checked = apply_edge_weights(edge_index.to(device), edge_weight)
    print(f"apply_edge_weights() OK — edge_index {tuple(ei_checked.shape)}, "
          f"edge_weight {tuple(ew_checked.shape)}, device={ew_checked.device}")
    print()

    ew_cpu = edge_weight.detach().float().cpu()
    ew_min, ew_max = ew_cpu.min().item(), ew_cpu.max().item()
    ew_mean, ew_std = ew_cpu.mean().item(), ew_cpu.std().item()
    print(f"min  = {ew_min:.6f}")
    print(f"max  = {ew_max:.6f}")
    print(f"mean = {ew_mean:.6f}")
    print(f"std  = {ew_std:.6f}")
    print()

    # --- Validation -----------------------------------------------------------
    has_nan = bool(torch.isnan(edge_weight).any().item())
    has_inf = bool(torch.isinf(edge_weight).any().item())
    in_range = bool(((edge_weight >= 0.0) & (edge_weight <= 1.0)).all().item())
    shape_ok = tuple(edge_weight.shape) == (num_edges,)

    checks = [
        ("no NaN", not has_nan),
        ("no Inf", not has_inf),
        ("output shape == number of edges", shape_ok),
        ("all values inside [0,1]", in_range),
    ]
    failures = [label for label, passed in checks if not passed]
    for label, passed in checks:
        print(f"{'✓' if passed else '✗'} {label}")

    print("=" * 70)
    if not failures:
        print(
            f"RESULT: PASS — relation={relation_name} | shard={shard_path.name} | "
            f"edges={num_edges} | in_features={feature_dim} | device={device}"
        )
    else:
        print(f"RESULT: FAIL — failed checks: {', '.join(failures)}")
    print("=" * 70)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(_run_integration_self_test())