"""
models/aggregator.py

TGN Message Aggregator.

When a node participates in multiple events within the same batch, it
receives multiple raw messages.  The aggregator reduces these into a
single aggregated message that is fed to the memory updater.

Three strategies are provided:

    LastMessageAggregator  — keep only the most recent message per node.
                             Matches the original TGN paper's default.

    MeanMessageAggregator  — element-wise mean across all messages.
                             Simple and robust to variable interaction counts.

    AttentionAggregator    — learned attention weights over messages,
                             conditioned on the node's current memory.
                             Most expressive; adds O(message_dim) parameters.

All aggregators expose the same interface: they receive an (unbatched)
list of (node_id, message, timestamp) tuples and return
``(unique_node_ids, aggregated_messages, latest_timestamps)``.

Reference:
    Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic
    Graphs" (NeurIPS 2020 Workshop)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Type alias for clarity
# ---------------------------------------------------------------------------
# Each entry: (node_id: int, message: Tensor[message_dim], timestamp: float)
NodeMessageList = List[Tuple[int, torch.Tensor, float]]


class MessageAggregator(nn.Module, ABC):
    """Abstract base class for TGN message aggregators."""

    @abstractmethod
    def aggregate(
        self,
        node_messages: Dict[int, List[Tuple[torch.Tensor, float]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate raw messages per node.

        Args:
            node_messages (Dict[int, List[Tuple[Tensor, float]]]):
                Mapping from node_id to a list of ``(message, timestamp)``
                pairs in chronological order.
            device (torch.device): Target device for output tensors.

        Returns:
            Tuple of three tensors:
                * ``node_ids``      shape ``(N,)``    — unique node indices
                * ``agg_messages``  shape ``(N, D)``  — aggregated messages
                * ``timestamps``    shape ``(N,)``    — latest event time per node
        """


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class LastMessageAggregator(MessageAggregator):
    """
    Keeps the most recent message for each node.

    No learnable parameters.  This matches the default used in the
    original TGN implementation and works well when events carry
    sufficient information individually.
    """

    def aggregate(
        self,
        node_messages: Dict[int, List[Tuple[torch.Tensor, float]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select the last (chronologically latest) message per node.

        Args:
            node_messages: See base class.
            device: See base class.

        Returns:
            See base class.
        """
        node_ids_list: List[int] = []
        agg_msgs_list: List[torch.Tensor] = []
        timestamps_list: List[float] = []

        for node_id, msg_ts_pairs in node_messages.items():
            # Sort by timestamp ascending; take the last entry
            sorted_pairs = sorted(msg_ts_pairs, key=lambda x: x[1])
            last_msg, last_ts = sorted_pairs[-1]

            node_ids_list.append(node_id)
            agg_msgs_list.append(last_msg)
            timestamps_list.append(last_ts)

        if not node_ids_list:
            d = next(iter(node_messages.values()))[0][0].shape[0] if node_messages else 1
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, d, device=device),
                torch.zeros(0, device=device),
            )

        node_ids = torch.tensor(node_ids_list, dtype=torch.long, device=device)
        agg_messages = torch.stack(agg_msgs_list).to(device)
        timestamps = torch.tensor(timestamps_list, dtype=torch.float, device=device)

        return node_ids, agg_messages, timestamps


class MeanMessageAggregator(MessageAggregator):
    """
    Computes the element-wise mean of all messages received by a node.

    No learnable parameters.  Robust when interaction order is noisy
    or when multiple simultaneous interactions carry equal weight.
    """

    def aggregate(
        self,
        node_messages: Dict[int, List[Tuple[torch.Tensor, float]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mean-pool all messages per node.

        Args:
            node_messages: See base class.
            device: See base class.

        Returns:
            See base class.
        """
        node_ids_list: List[int] = []
        agg_msgs_list: List[torch.Tensor] = []
        timestamps_list: List[float] = []

        for node_id, msg_ts_pairs in node_messages.items():
            msgs = torch.stack([m for m, _ in msg_ts_pairs])   # (K, D)
            last_ts = max(ts for _, ts in msg_ts_pairs)

            node_ids_list.append(node_id)
            agg_msgs_list.append(msgs.mean(dim=0))
            timestamps_list.append(last_ts)

        if not node_ids_list:
            d = next(iter(node_messages.values()))[0][0].shape[0] if node_messages else 1
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, d, device=device),
                torch.zeros(0, device=device),
            )

        node_ids = torch.tensor(node_ids_list, dtype=torch.long, device=device)
        agg_messages = torch.stack(agg_msgs_list).to(device)
        timestamps = torch.tensor(timestamps_list, dtype=torch.float, device=device)

        return node_ids, agg_messages, timestamps


class AttentionAggregator(MessageAggregator):
    """
    Attention-weighted aggregation of messages.

    Computes a scalar attention weight for each message conditioned on
    the node's current memory vector:

        α_k = softmax( v^T tanh(W_q * s_node + W_k * msg_k) )
        agg  = Σ_k α_k * msg_k

    This allows the model to focus on the most informative interactions.

    Args:
        memory_dim (int): Dimensionality of node memory (query dimension).
        message_dim (int): Dimensionality of incoming messages (key dimension).
        hidden_dim (int): Hidden dimension for the attention scoring MLP.
    """

    def __init__(
        self,
        memory_dim: int,
        message_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.message_dim = message_dim

        self.W_query = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(message_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_key.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def _score(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention scores for a single node.

        Args:
            query (torch.Tensor): Shape ``(1, memory_dim)`` — node memory.
            keys (torch.Tensor): Shape ``(K, message_dim)`` — messages.

        Returns:
            torch.Tensor: Shape ``(K,)`` — normalised attention weights.
        """
        q = self.W_query(query)                # (1, hidden_dim)
        k = self.W_key(keys)                   # (K, hidden_dim)
        scores = self.v(torch.tanh(q + k))     # (K, 1)
        return F.softmax(scores.squeeze(-1), dim=0)   # (K,)

    def aggregate(
        self,
        node_messages: Dict[int, List[Tuple[torch.Tensor, float]]],
        device: torch.device,
        node_memories: Dict[int, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Attention-weighted aggregation of messages per node.

        Args:
            node_messages: See base class.
            device: See base class.
            node_memories (Dict[int, Tensor] | None): Optional mapping of
                node_id to current memory vector used as attention query.
                If ``None``, uniform averaging is used as fallback.

        Returns:
            See base class.
        """
        node_ids_list: List[int] = []
        agg_msgs_list: List[torch.Tensor] = []
        timestamps_list: List[float] = []

        for node_id, msg_ts_pairs in node_messages.items():
            msgs = torch.stack([m for m, _ in msg_ts_pairs]).to(device)   # (K, D)
            last_ts = max(ts for _, ts in msg_ts_pairs)

            if node_memories is not None and node_id in node_memories:
                mem = node_memories[node_id].unsqueeze(0).to(device)      # (1, memory_dim)
                weights = self._score(mem, msgs)                          # (K,)
                agg = (weights.unsqueeze(-1) * msgs).sum(dim=0)           # (D,)
            else:
                agg = msgs.mean(dim=0)

            node_ids_list.append(node_id)
            agg_msgs_list.append(agg)
            timestamps_list.append(last_ts)

        if not node_ids_list:
            d = next(iter(node_messages.values()))[0][0].shape[0] if node_messages else 1
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, d, device=device),
                torch.zeros(0, device=device),
            )

        node_ids = torch.tensor(node_ids_list, dtype=torch.long, device=device)
        agg_messages = torch.stack(agg_msgs_list)
        timestamps = torch.tensor(timestamps_list, dtype=torch.float, device=device)

        return node_ids, agg_messages, timestamps