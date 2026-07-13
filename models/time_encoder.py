"""
models/time_encoder.py

Learnable time encoding using the Time2Vec formulation.
Maps a scalar timestamp delta to a dense vector representation.

Reference:
    Kazemi et al. "Time2Vec: Learning a Vector Representation of Time" (2019)
    Xu et al. "Inductive Representation Learning on Temporal Graphs" (ICLR 2020)
"""

import math
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Encodes a scalar time delta into a fixed-dimensional embedding.

    The encoding uses a learnable sinusoidal basis:
        out[0]     = w_0 * t + b_0              (linear term)
        out[1..d]  = sin(w_i * t + b_i)         (periodic terms)

    This allows the model to learn meaningful temporal patterns at
    multiple frequencies without manual feature engineering.

    Args:
        out_channels (int): Dimensionality of the output time embedding.
            Should match the memory/message dimension or be projected later.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

        # Learnable frequency and phase parameters
        self.w = nn.Linear(1, out_channels)

        # Initialise frequencies to span multiple time scales
        nn.init.uniform_(self.w.weight, 0.0, 2.0 * math.pi)
        nn.init.zeros_(self.w.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of time deltas.

        Args:
            t (torch.Tensor): Shape ``(N,)`` or ``(N, 1)``.
                Raw timestamp deltas (float, any consistent unit).

        Returns:
            torch.Tensor: Shape ``(N, out_channels)``.
                Time embeddings. The first dimension is linear; the
                remaining dimensions are sinusoidal.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)                  # (N, 1)

        out = self.w(t)                          # (N, out_channels)

        # First component is linear, rest are sinusoidal
        linear = out[:, :1]
        periodic = torch.sin(out[:, 1:])

        return torch.cat([linear, periodic], dim=-1)   # (N, out_channels)