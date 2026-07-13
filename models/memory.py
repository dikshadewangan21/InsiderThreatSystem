"""
models/memory.py

TGN Memory Module.

Each node maintains a persistent memory vector that summarises its
interaction history up to the last processed event.  Memory is updated
in strict chronological order and stored as a detach-able buffer so
gradient flow is correctly controlled across time steps.

Reference:
    Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic
    Graphs" (NeurIPS 2020 Workshop)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """
    Persistent node memory store with a GRU-based updater.

    Memory for node ``i`` is a vector ``s_i ∈ R^{memory_dim}``
    initialised to zero and updated each time node ``i`` is involved in
    an interaction.  The update equation is:

        s_i(t) = GRU(agg_message_i(t), s_i(t⁻))

    where ``agg_message_i(t)`` is the aggregated message delivered to
    node ``i`` at time ``t``.

    Args:
        num_nodes (int): Total number of nodes tracked by memory.
        memory_dim (int): Dimensionality of each node's memory vector.
        message_dim (int): Dimensionality of incoming aggregated messages.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        message_dim: int,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim

        # GRU cell: takes (message, current_memory) -> new_memory
        self.gru = nn.GRUCell(
            input_size=message_dim,
            hidden_size=memory_dim,
        )

        # Persistent buffers — not trained parameters, but saved with state_dict
        self.register_buffer(
            "memory",
            torch.zeros(num_nodes, memory_dim),
        )
        self.register_buffer(
            "last_update",
            torch.zeros(num_nodes, dtype=torch.float),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve current memory for the requested node ids.

        Args:
            node_ids (torch.Tensor): Shape ``(N,)`` — integer node indices.

        Returns:
            torch.Tensor: Shape ``(N, memory_dim)``.
        """
        return self.memory[node_ids]

    def get_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the timestamp of the most recent memory update.

        Args:
            node_ids (torch.Tensor): Shape ``(N,)`` — integer node indices.

        Returns:
            torch.Tensor: Shape ``(N,)`` — float timestamps.
        """
        return self.last_update[node_ids]

    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """
        Update memory for a batch of nodes using the GRU cell.

        Nodes may appear multiple times in ``node_ids`` (source and
        destination of the same event).  Updates are applied in the
        order they appear; callers must ensure the input is chronologically
        sorted.

        Args:
            node_ids (torch.Tensor): Shape ``(N,)`` — integer node indices.
            messages (torch.Tensor): Shape ``(N, message_dim)`` — aggregated
                messages for each node.
            timestamps (torch.Tensor): Shape ``(N,)`` — event timestamps
                used to update ``last_update``.
        """
        # Pull current memory for these nodes
        current_mem = self.memory[node_ids]          # (N, memory_dim)

        # Run GRU update
        new_mem = self.gru(messages, current_mem)    # (N, memory_dim)

        # Write back — detach to stop BPTT across separate forward passes
        self.memory[node_ids] = new_mem.detach()
        self.last_update[node_ids] = timestamps.detach()

    def reset_state(self) -> None:
        """
        Zero all memory vectors and last-update timestamps.
        Call at the start of a new epoch or when re-processing from scratch.
        """
        self.memory.zero_()
        self.last_update.zero_()

    def detach_memory(self) -> None:
        """
        Detach memory tensors from the computation graph.
        Useful when processing long event streams in mini-batches to
        prevent unbounded BPTT.
        """
        self.memory = self.memory.detach()
        self.last_update = self.last_update.detach()

    def backup_memory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a detached copy of the current memory state.
        Useful for validation — run eval on a snapshot without polluting
        the training memory.

        Returns:
            Tuple of (memory, last_update) tensors.
        """
        return self.memory.clone(), self.last_update.clone()

    def restore_memory(
        self,
        memory_backup: torch.Tensor,
        last_update_backup: torch.Tensor,
    ) -> None:
        """
        Restore memory from a previously taken backup.

        Args:
            memory_backup (torch.Tensor): Shape ``(num_nodes, memory_dim)``.
            last_update_backup (torch.Tensor): Shape ``(num_nodes,)``.
        """
        self.memory = memory_backup.clone()
        self.last_update = last_update_backup.clone()