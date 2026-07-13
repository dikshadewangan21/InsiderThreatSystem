"""
models/message.py

TGN Message Function.

For each directed interaction (src → dst at time t), the message function
produces a raw message vector that encodes:

    * The source node's current memory
    * The destination node's current memory
    * The time encoding of the elapsed time since last update
    * The raw edge features (e.g. login counts, visit counts)

Two variants are provided:

    IdentityMessage   — concatenate all inputs, no learned projection.
                        Cheap; useful as a baseline.

    MLPMessage        — project the concatenation through a 2-layer MLP.
                        Adds expressiveness for complex edge semantics.

Reference:
    Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic
    Graphs" (NeurIPS 2020 Workshop)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from models.time_encoder import TimeEncoder


class MessageFunction(nn.Module, ABC):
    """Abstract base class for TGN message functions."""

    @abstractmethod
    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        delta_t: torch.Tensor,
        edge_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute messages for a batch of interactions.

        Args:
            src_memory (torch.Tensor): Shape ``(E, memory_dim)`` —
                source node memories at time of interaction.
            dst_memory (torch.Tensor): Shape ``(E, memory_dim)`` —
                destination node memories at time of interaction.
            delta_t (torch.Tensor): Shape ``(E,)`` —
                elapsed time since each source node's last update.
            edge_features (torch.Tensor | None): Shape ``(E, edge_feat_dim)``
                or ``None`` if no edge features are available.

        Returns:
            torch.Tensor: Shape ``(E, message_dim)``.
        """


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class IdentityMessage(MessageFunction):
    """
    Identity message function.

    Constructs messages purely by concatenating:
        [s_src || s_dst || Φ(Δt) || e_feat]

    No learned parameters.  The output dimension is:
        2 * memory_dim + time_dim + edge_feat_dim

    Args:
        memory_dim (int): Dimensionality of node memory vectors.
        time_dim (int): Dimensionality of the time encoding.
        edge_feat_dim (int): Dimensionality of edge feature vectors.
            Pass 0 if no edge features are used.
    """

    def __init__(
        self,
        memory_dim: int,
        time_dim: int,
        edge_feat_dim: int,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.time_encoder = TimeEncoder(time_dim)
        self.edge_feat_dim = edge_feat_dim
        self.message_dim = 2 * memory_dim + time_dim + edge_feat_dim

    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        delta_t: torch.Tensor,
        edge_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Concatenate inputs to produce messages.

        Returns:
            torch.Tensor: Shape ``(E, message_dim)``.
        """
        time_enc = self.time_encoder(delta_t)          # (E, time_dim)

        parts = [src_memory, dst_memory, time_enc]

        if edge_features is not None and self.edge_feat_dim > 0:
            parts.append(edge_features)

        return torch.cat(parts, dim=-1)                # (E, message_dim)


class MLPMessage(MessageFunction):
    """
    MLP message function.

    Concatenates [s_src || s_dst || Φ(Δt) || e_feat] and projects
    through a two-layer MLP with ReLU activations and layer norm.

    Args:
        memory_dim (int): Dimensionality of node memory vectors.
        time_dim (int): Dimensionality of the time encoding.
        edge_feat_dim (int): Dimensionality of edge feature vectors.
            Pass 0 if no edge features are used.
        message_dim (int): Output dimensionality of the message MLP.
        dropout (float): Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        memory_dim: int,
        time_dim: int,
        edge_feat_dim: int,
        message_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.memory_dim = memory_dim
        self.time_encoder = TimeEncoder(time_dim)
        self.edge_feat_dim = edge_feat_dim
        self.message_dim = message_dim

        in_dim = 2 * memory_dim + time_dim + edge_feat_dim
        hidden_dim = max(in_dim, message_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, message_dim),
            nn.LayerNorm(message_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        delta_t: torch.Tensor,
        edge_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Project concatenated inputs through the MLP.

        Returns:
            torch.Tensor: Shape ``(E, message_dim)``.
        """
        time_enc = self.time_encoder(delta_t)          # (E, time_dim)

        parts = [src_memory, dst_memory, time_enc]

        if edge_features is not None and self.edge_feat_dim > 0:
            parts.append(edge_features)

        raw = torch.cat(parts, dim=-1)                 # (E, in_dim)
        return self.mlp(raw)                           # (E, message_dim)