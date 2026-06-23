"""Evaluate chunk-size experiment checkpoints on env D."""
import io, json
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
        print(f"  Loaded {len(self.samples)} eval samples from {envs} (chunk={chunk_size})")

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
    print(f"  Loaded {ckpt_path}: chunk_size={config.chunk_size}")
    return policy, config.chunk_size


def evaluate(policy, chunk_size, eval_loader, device):
    all_per_dim = []
    all_sample_l1 = []
    with torch.no_grad():
        for images, states, actions in eval_loader:
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            batch = {"observation.images.image": images, "observation.state": states,
                     "observation.images": [images]}
            pred, _ = policy.model(batch)
            per_dim = torch.abs(pred - actions).mean(dim=(0, 1))
            sample_l1 = torch.abs(pred - actions).mean(dim=(1, 2))
            all_per_dim.append(per_dim.cpu().numpy())
            all_sample_l1.append(sample_l1.cpu().numpy())
    per_dim = np.concatenate(all_per_dim).reshape(-1, 7).mean(axis=0)
    samples = np.concatenate(all_sample_l1)
    return {
        "overall_l1": float(samples.mean()),
        "per_dim_l1": per_dim.tolist(),
        "n_samples": len(samples),
    }


def main():
    data_root = "/root/public/data/calvin-lerobot"
    device = torch.device("cuda")
    base = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task2_output_official"

    models = [
        ("chunk10", f"{base}/envA_chunk10/checkpoint_epoch_199.pt"),
        ("chunk50", f"{base}/envA_chunk50/checkpoint_epoch_199.pt"),
    ]

    results = {}
    for name, ckpt_path in models:
        print(f"\n=== {name} ===")
        policy, chunk_size = load_policy(ckpt_path, device)
        eval_ds = CalvinEvalDataset(data_root, ["D"], chunk_size)
        eval_loader = DataLoader(eval_ds, batch_size=48, shuffle=False, num_workers=8, pin_memory=True)
        res = evaluate(policy, chunk_size, eval_loader, device)
        results[name] = res
        print(f"  Overall L1: {res['overall_l1']:.4f}")
        print(f"  Per-dim: {[f'{v:.4f}' for v in res['per_dim_l1']]}")

    # Also load existing chunk30 result for comparison
    existing_path = f"{base}/eval_results.json"
    if Path(existing_path).exists():
        with open(existing_path) as f:
            existing = json.load(f)
        results["chunk30"] = existing["envA"]
        print(f"\n=== chunk30 (existing) ===")
        print(f"  Overall L1: {results['chunk30']['overall_l1']:.4f}")

    print("\n=== CHUNK COMPARISON ===")
    print(f"{'Chunk':<10} {'Eval L1':>10} {'Joint L1':>10} {'Gripper L1':>12} {'N samples':>10}")
    print("-" * 52)
    for cs in [10, 30, 50]:
        key = f"chunk{cs}"
        if key in results:
            r = results[key]
            jl = sum(r["per_dim_l1"][:6]) / 6
            gl = r["per_dim_l1"][6]
            print(f"{cs:<10} {r['overall_l1']:>10.4f} {jl:>10.4f} {gl:>12.4f} {r['n_samples']:>10}")

    out_path = f"{base}/eval_chunk_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
