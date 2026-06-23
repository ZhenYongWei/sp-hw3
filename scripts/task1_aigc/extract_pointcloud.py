import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/tmp/threestudio")

def extract_pointcloud_from_nerf(ckpt_path, output_path, n_points=50000, resolution=128):
    from threestudio.utils.config import load_config
    from pytorch_lightning import Trainer

    ckpt = torch.load(ckpt_path, map_location="cpu")

    system_state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("system."):
            system_state[k[7:]] = v

    geometry_type = ckpt["hyper_parameters"].get("system.geometry_type", "")

    print(f"Geometry type: {geometry_type}")

    if "implicit-volume" in geometry_type:
        from threestudio.models.geometry.implicit_volume import ImplicitVolume
        geo_config = ckpt["hyper_parameters"]["system"]["geometry"]
        geometry = ImplicitVolume(**geo_config)
        geometry.load_state_dict(
            {k.replace("geometry.", ""): v for k, v in system_state.items() if k.startswith("geometry.")}
        )
    else:
        print(f"Unknown geometry type: {geometry_type}")
        return

    geometry.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = geometry.to(device)

    radius = ckpt["hyper_parameters"]["system"]["geometry"]["radius"]

    x = torch.linspace(-radius, radius, resolution, device=device)
    y = torch.linspace(-radius, radius, resolution, device=device)
    z = torch.linspace(-radius, radius, resolution, device=device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    batch_size = 10000
    densities = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        with torch.no_grad():
            density = geometry.forward_density(batch)
            densities.append(density.cpu())

    densities = torch.cat(densities)

    threshold = 0.5
    mask = densities > threshold

    valid_points = points[mask].cpu().numpy()

    print(f"Points above threshold {threshold}: {len(valid_points)} / {len(points)}")

    if len(valid_points) > n_points:
        indices = np.random.choice(len(valid_points), n_points, replace=False)
        valid_points = valid_points[indices]

    write_ply(output_path, valid_points)
    print(f"Saved {len(valid_points)} points to {output_path}")


def write_ply(path, points):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with open(path, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-points", type=int, default=50000)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()

    extract_pointcloud_from_nerf(args.ckpt, args.output, args.n_points, args.resolution)
