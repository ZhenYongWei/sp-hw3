"""Evaluate official ACT checkpoints: per-dimension L1 and success rates on env D."""
import io, os, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
from PIL import Image
from torchvision import transforms
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType


class CalvinEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, envs, chunk_size):
        self.chunk_size = chunk_size
        self.samples = []
        env_map = {"A": "splitA", "B": "splitB", "C": "splitC", "D": "splitD"}
        for env in envs:
            self._load_split(Path(data_root) / env_map[env])
        print(f"  Loaded {len(self.samples)} eval samples from {envs}")

    def _load_split(self, split_dir):
        chunks_dir = split_dir / "data"
        n_ep = 0
        for chunk_dir in sorted(chunks_dir.glob("chunk-*")):
            for pf in sorted(chunk_dir.glob("episode_*.parquet")):
                table = pq.read_table(pf)
                ep_len = table.num_rows
                if ep_len < self.chunk_size + 1:
                    continue
                img_col = table.column("image")
                states_np = np.stack(table.column("state").to_pylist())
                actions_np = np.stack(table.column("actions").to_pylist())
                # sample a few start points per episode
                for start in range(0, ep_len - self.chunk_size, max(1, ep_len // 4)):
                    self.samples.append((img_col[start].as_py()["bytes"], states_np[start],
                                         actions_np[start:start + self.chunk_size]))
                n_ep += 1
        print(f"    {split_dir.name}: {n_ep} episodes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, state, action_chunk = self.samples[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return t(img), torch.tensor(state, dtype=torch.float32), \
               torch.tensor(action_chunk, dtype=torch.float32)


def load_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    config = ACTConfig(**{k: v for k, v in config_dict.items() if not k.startswith("_")})
    policy = ACTPolicy(config).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    chunk_size = config.chunk_size
    print(f"  Loaded {ckpt_path}: epoch={ckpt['epoch']}, chunk_size={chunk_size}")
    return policy, chunk_size


def evaluate(policy, chunk_size, eval_loader, device):
    all_per_dim_l1 = []
    all_sample_l1 = []
    n_batches = 0
    with torch.no_grad():
        for images, states, actions in eval_loader:
            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)
            batch = {
                "observation.images.image": images,
                "observation.state": states,
                "observation.images": [images],
            }
            pred, _ = policy.model(batch)
            # pred: (B, chunk_size, 7), actions: (B, chunk_size, 7)
            per_dim_l1 = torch.abs(pred - actions).mean(dim=(0, 1))  # (7,)
            sample_l1 = torch.abs(pred - actions).mean(dim=(1, 2))   # (B,)
            all_per_dim_l1.append(per_dim_l1.cpu().numpy())
            all_sample_l1.append(sample_l1.cpu().numpy())
            n_batches += 1

    per_dim = np.concatenate(all_per_dim_l1).reshape(-1, 7).mean(axis=0)
    samples = np.concatenate(all_sample_l1)
    overall_l1 = samples.mean()

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
    sr = {}
    for t in thresholds:
        sr[t] = float((samples < t).mean())

    return {
        "overall_l1": float(overall_l1),
        "per_dim_l1": per_dim.tolist(),
        "joint_l1": float(per_dim[:6].mean()),
        "gripper_l1": float(per_dim[6]),
        "success_rates": sr,
        "n_samples": len(samples),
    }


def main():
    data_root = "/root/public/data/calvin-lerobot"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task2_output_official"

    models = [
        ("envA", f"{base}/envA/checkpoint_epoch_199.pt"),
        ("envABC", f"{base}/envABC/checkpoint_epoch_199.pt"),
    ]

    results = {}
    for name, ckpt_path in models:
        print(f"\n=== Evaluating {name} ===")
        policy, chunk_size = load_policy(ckpt_path, device)
        eval_ds = CalvinEvalDataset(data_root, ["D"], chunk_size)
        eval_loader = DataLoader(eval_ds, batch_size=48, shuffle=False, num_workers=8, pin_memory=True)
        res = evaluate(policy, chunk_size, eval_loader, device)
        results[name] = res
        print(f"  Overall L1: {res['overall_l1']:.4f}")
        print(f"  Joint L1 (dim 1-6): {res['joint_l1']:.4f}")
        print(f"  Gripper L1 (dim 7): {res['gripper_l1']:.4f}")
        print(f"  Per-dim L1: {[f'{v:.4f}' for v in res['per_dim_l1']]}")
        print(f"  Success rates: " + ", ".join(f"@{t}={res['success_rates'][t]*100:.1f}%" for t in [0.05, 0.1, 0.2, 0.3, 0.5]))

    print("\n=== SUMMARY ===")
    print(f"{'Metric':<25} {'envA':>12} {'envABC':>12} {'Diff':>12}")
    print("-" * 61)
    for key in ["overall_l1", "joint_l1", "gripper_l1"]:
        a, b = results["envA"][key], results["envABC"][key]
        print(f"{key:<25} {a:>12.4f} {b:>12.4f} {b-a:>+12.4f}")
    for t in [0.05, 0.1, 0.2, 0.3, 0.5]:
        a = results["envA"]["success_rates"][t] * 100
        b = results["envABC"]["success_rates"][t] * 100
        print(f"{'SR@'+str(t):<25} {a:>11.1f}% {b:>11.1f}% {b-a:>+11.1f}%")

    # Save results as JSON for figure generation
    import json
    out_path = f"{base}/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
