#!/usr/bin/env python3
"""Validate that the TGN, GAT, and MLP are learning on real training data.

This is intentionally a probe, not part of the production training pipeline.
It reuses the existing trainer and real edge shards, then measures a fixed
shard before training and after each epoch.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.config import TrainingConfig  # noqa: E402
from train import (  # noqa: E402
    InsiderThreatTrainer,
    apply_edge_weighting,
    stream_edge_shards,
)


@dataclass
class ProbeSnapshot:
    tgn_embedding_var: float
    gat_attention_var: float
    gat_attention_mean: float
    mlp_weight_var: float
    mlp_weight_norm: float


def _flat_trainable_params(module: torch.nn.Module) -> torch.Tensor:
    params = [
        param.detach().float().reshape(-1).cpu()
        for param in module.parameters()
        if param.requires_grad
    ]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def _mlp_snapshot(trainer: InsiderThreatTrainer) -> tuple[float, float]:
    params = _flat_trainable_params(trainer.mlp)
    if params.numel() == 0:
        return 0.0, 0.0
    return float(params.var(unbiased=False).item()), float(params.norm().item())


def _probe_fixed_shard(trainer: InsiderThreatTrainer, shard: Dict[str, torch.Tensor]) -> ProbeSnapshot:
    trainer.tgn.eval()
    trainer.gat.eval()
    trainer.mlp.eval()
    if trainer.edge_weighter is not None:
        trainer.edge_weighter.eval()

    if hasattr(trainer.tgn, "memory") and hasattr(trainer.tgn.memory, "reset"):
        trainer.tgn.memory.reset()

    shard = apply_edge_weighting(dict(shard), trainer.edge_weighter)
    edge_index = shard["edge_index"]

    with torch.no_grad():
        tgn_result = trainer.tgn.process_shard(
            {
                "edge_index": edge_index,
                "features": shard["features"],
                "temporal_encoding": shard["temporal_encoding"],
                "edge_time": shard["edge_time"],
                "edge_weight": shard.get("edge_weight"),
            }
        )
        node_embeddings = tgn_result["embeddings"].detach().float()
        touched_ids = tgn_result["updated_node_ids"]
        local_edge_index = torch.stack(
            [
                torch.searchsorted(touched_ids, edge_index[0]),
                torch.searchsorted(touched_ids, edge_index[1]),
            ],
            dim=0,
        )
        gat_result = trainer.gat(node_embeddings, local_edge_index, shard["edge_weight"])
        attention = gat_result[1].detach().float()

    mlp_var, mlp_norm = _mlp_snapshot(trainer)
    return ProbeSnapshot(
        tgn_embedding_var=float(node_embeddings.var(unbiased=False).item()),
        gat_attention_var=float(attention.var(unbiased=False).item()),
        gat_attention_mean=float(attention.mean().item()),
        mlp_weight_var=mlp_var,
        mlp_weight_norm=mlp_norm,
    )


def _print_snapshot(label: str, snapshot: ProbeSnapshot) -> None:
    print(
        f"{label} | "
        f"tgn_embedding_var={snapshot.tgn_embedding_var:.10e} | "
        f"gat_attention_var={snapshot.gat_attention_var:.10e} | "
        f"gat_attention_mean={snapshot.gat_attention_mean:.10e} | "
        f"mlp_weight_var={snapshot.mlp_weight_var:.10e} | "
        f"mlp_weight_norm={snapshot.mlp_weight_norm:.10e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe model learning without changing the training pipeline.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = TrainingConfig(device=args.device, epochs=args.epochs)
    trainer = InsiderThreatTrainer(config)
    fixed_shard = next(stream_edge_shards(config.edge_shard_dir, trainer.device))

    before = _probe_fixed_shard(trainer, fixed_shard)
    _print_snapshot("before_training", before)

    previous_mlp = _flat_trainable_params(trainer.mlp)
    snapshots = [before]

    for epoch in range(args.epochs):
        train_loss, train_metrics, _ = trainer._run_epoch(trainer.train_ids, train=True)  # noqa: SLF001
        current_mlp = _flat_trainable_params(trainer.mlp)
        mlp_delta = float((current_mlp - previous_mlp).norm().item())
        previous_mlp = current_mlp

        snapshot = _probe_fixed_shard(trainer, fixed_shard)
        snapshots.append(snapshot)
        _print_snapshot(f"after_epoch_{epoch + 1}", snapshot)
        print(
            f"epoch_{epoch + 1}_learning | "
            f"train_loss={train_loss:.10e} | "
            f"train_grad_norm={train_metrics.get('grad_norm_mean', 0.0):.10e} | "
            f"mlp_weight_delta={mlp_delta:.10e}"
        )

    after = snapshots[-1]
    print(
        "learning_delta | "
        f"tgn_embedding_var_delta={(after.tgn_embedding_var - before.tgn_embedding_var):.10e} | "
        f"gat_attention_var_delta={(after.gat_attention_var - before.gat_attention_var):.10e} | "
        f"gat_attention_mean_delta={(after.gat_attention_mean - before.gat_attention_mean):.10e} | "
        f"mlp_weight_var_delta={(after.mlp_weight_var - before.mlp_weight_var):.10e} | "
        f"mlp_weight_norm_delta={(after.mlp_weight_norm - before.mlp_weight_norm):.10e}"
    )


if __name__ == "__main__":
    main()
