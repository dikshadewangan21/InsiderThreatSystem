import pickle
import torch
import json
from pathlib import Path

output_dir = Path(r"e:\InsiderThreatSystem\graph\output")

print("--- SKELETON GRAPH ---")
skeleton = torch.load(output_dir / "node_graph_skeleton.pt", weights_only=False)
print(skeleton)
print("Node types:", skeleton.node_types)
print("Edge types:", skeleton.edge_types)
for nt in skeleton.node_types:
    print(f"Node '{nt}' keys:", skeleton[nt].keys())
    if 'x' in skeleton[nt]:
        print(f"Node '{nt}' x shape:", skeleton[nt].x.shape)
for et in skeleton.edge_types:
    print(f"Edge {et} keys:", skeleton[et].keys())
    if 'edge_index' in skeleton[et]:
        print(f"Edge {et} edge_index shape:", skeleton[et].edge_index.shape)

print("\n--- PREPROCESSING ARTIFACTS ---")
with open(output_dir / "preprocessing_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)
print("Artifacts keys:", list(artifacts.keys()))
for k in ["dept_registry", "role_registry", "team_registry", "bu_registry", "pc_registry", "domain_registry", "ext_registry"]:
    if k in artifacts:
        print(f"Registry '{k}' size:", len(artifacts[k]))

print("\n--- SAMPLE SHARD ---")
manifest_path = output_dir / "edge_shard_manifest.json"
with open(manifest_path, "r") as f:
    manifest = json.load(f)
first_shard = manifest["shards"][0]
first_shard_path = Path(r"e:\InsiderThreatSystem") / first_shard["path"]
print("Loading sample shard from:", first_shard_path)
shard = torch.load(first_shard_path, weights_only=False)
print("Shard keys:", list(shard.keys()))
for k, v in shard.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: tensor of shape {v.shape}, dtype {v.dtype}")
    else:
        print(f"  {k}: {type(v)} = {v}")
