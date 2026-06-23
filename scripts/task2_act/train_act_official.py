"""Train ACT on CALVIN using official LeRobot ACT implementation."""
import argparse, io, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from PIL import Image
from torchvision import transforms
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType


class CalvinDataset(Dataset):
    def __init__(self, data_root, envs, chunk_size=100):
        self.chunk_size = chunk_size
        self.samples = []
        env_map = {"A": "splitA", "B": "splitB", "C": "splitC", "D": "splitD"}
        for env in envs:
            self._load_split(Path(data_root) / env_map[env])
        print(f"  Loaded {len(self.samples)} samples from {envs}")

    def _load_split(self, split_dir):
        chunks_dir = split_dir / "data"
        for chunk_dir in sorted(chunks_dir.glob("chunk-*")):
            for pf in sorted(chunk_dir.glob("episode_*.parquet")):
                table = pq.read_table(pf)
                ep_len = table.num_rows
                if ep_len < self.chunk_size + 1: continue
                img_col = table.column("image")
                states_np = np.stack(table.column("state").to_pylist())
                actions_np = np.stack(table.column("actions").to_pylist())
                for start in range(0, ep_len - self.chunk_size, max(1, ep_len // 8)):
                    self.samples.append((img_col[start].as_py()["bytes"], states_np[start],
                                         actions_np[start:start + self.chunk_size]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_bytes, state, action_chunk = self.samples[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self._transform(img), torch.tensor(state, dtype=torch.float32), \
               torch.tensor(action_chunk, dtype=torch.float32)

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ACTTrainWrapper(nn.Module):
    """Wrapper that returns only loss tensor for DataParallel compatibility."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        loss, loss_dict = self.policy(batch)
        # Store loss_dict as attribute for logging (only from first replica)
        if not hasattr(self, '_last_loss_dict') or self._last_loss_dict is None:
            self._last_loss_dict = loss_dict
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, default="A")
    parser.add_argument("--data-root", type=str, default="/root/public/data/calvin-lerobot")
    parser.add_argument("--output-dir", type=str, default="task2_output_official")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-envs", type=str, default="D")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()

    config = ACTConfig(
        chunk_size=args.chunk_size, kl_weight=args.kl_weight,
        n_action_steps=args.chunk_size,  # must be <= chunk_size
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = ACTPolicy(config).to(device)
    print(f"Official ACT parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Multi-GPU support
    if n_gpus > 1:
        train_wrapper = ACTTrainWrapper(policy).to(device)
        train_wrapper = nn.DataParallel(train_wrapper)
        raw_policy = train_wrapper.module.policy  # access underlying policy
        print(f"Using DataParallel on {n_gpus} GPUs (effective batch={args.batch_size * n_gpus})")
    else:
        train_wrapper = policy
        raw_policy = policy
        print("Single GPU mode")

    train_ds = CalvinDataset(args.data_root, list(args.envs), args.chunk_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    eval_ds = CalvinDataset(args.data_root, list(args.eval_envs), args.chunk_size)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_file = os.path.join(args.output_dir, "training_log.txt")
    for epoch in range(args.epochs):
        t0 = time.time()
        policy.train()
        tl, ta, tk, n = 0, 0, 0, 0
        for bi, (images, states, actions) in enumerate(train_loader):
            images, states, actions = images.to(device), states.to(device), actions.to(device)
            action_is_pad = torch.zeros(images.size(0), args.chunk_size, dtype=torch.bool, device=device)
            batch = {"observation.images.image": images, "observation.state": states,
                     "action": actions, "action_is_pad": action_is_pad}

            if n_gpus > 1:
                # DataParallel: forward returns loss tensor only
                if hasattr(train_wrapper, 'module'):
                    train_wrapper.module._last_loss_dict = None
                loss = train_wrapper(batch)
                loss = loss.mean()  # gather from all GPUs → scalar
                ld = train_wrapper.module._last_loss_dict or {}
            else:
                loss, ld = policy(batch)

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); ta += ld.get("l1_loss", 0); tk += ld.get("kld_loss", 0); n += 1
            if bi % 100 == 0:
                print(f"  E{epoch} B{bi}/{len(train_loader)}: loss={loss.item():.4f} "
                      f"l1={ld.get('l1_loss',0):.4f} kl={ld.get('kld_loss',0):.4f}")
        scheduler.step()

        # Evaluate
        policy.eval()
        el = 0; en = 0
        with torch.no_grad():
            for images, states, actions in eval_loader:
                images, states, actions = images.to(device), states.to(device), actions.to(device)
                batch = {"observation.images.image": images, "observation.state": states,
                         "observation.images": [images]}
                pred, _ = raw_policy.model(batch)
                el += F.l1_loss(pred, actions).item(); en += 1
        eval_l1 = el / max(1, en)

        dt = time.time() - t0
        line = (f"Epoch {epoch}: train_loss={tl/n:.4f} train_action={ta/n:.4f} "
                f"train_kl={tk/n:.4f} eval_action={eval_l1:.4f} time={dt:.0f}s\n")
        print(line.strip())
        with open(log_file, "a") as f: f.write(line)

        if (epoch+1) % args.save_every == 0 or epoch == args.epochs-1:
            ckpt = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({"epoch": epoch, "model_state_dict": policy.state_dict(),
                        "config": config.__dict__}, ckpt)
            print(f"  Saved {ckpt}")
    print("Done!")


if __name__ == "__main__":
    main()
