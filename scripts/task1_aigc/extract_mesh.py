import sys
import os
import torch
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from omegaconf import OmegaConf

sys.path.insert(0, "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/threestudio")

from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.models.geometry.base import contract_to_unisphere


def load_geometry(ckpt_path, config_yaml_path):
    import yaml

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    geo_state = {k.replace("geometry.", ""): v for k, v in state.items() if k.startswith("geometry.")}

    with open(config_yaml_path) as f:
        cfg = yaml.safe_load(f)
    geo_cfg = cfg["system"]["geometry"]

    radius = geo_cfg.get("radius", 2.0)
    n_input_dims = 3
    n_feature_dims = geo_cfg.get("n_feature_dims", 3)
    pos_enc_cfg = OmegaConf.create(geo_cfg["pos_encoding_config"])

    mlp_cfg = OmegaConf.create(geo_cfg.get("mlp_network_config", {
        "otype": "VanillaMLP",
        "activation": "ReLU",
        "output_activation": "none",
        "n_neurons": 64,
        "n_hidden_layers": 1,
    }))

    encoding = get_encoding(n_input_dims, pos_enc_cfg)
    density_network = get_mlp(encoding.n_output_dims, 1, mlp_cfg)
    feature_network = get_mlp(encoding.n_output_dims, n_feature_dims, mlp_cfg)

    enc_keys = {}
    for k, v in geo_state.items():
        if "encoding" in k:
            # checkpoint has extra nesting: encoding.encoding.encoding.params -> encoding.encoding.params
            parts = k.split(".")
            if parts[0] == "encoding" and parts[1] == "encoding" and parts[2] == "encoding":
                new_key = "encoding.encoding." + ".".join(parts[3:])
            elif parts[0] == "encoding" and parts[1] == "encoding":
                new_key = k
            else:
                new_key = k
            enc_keys[new_key] = v

    # Remove "encoding." prefix for load_state_dict on the encoding module
    enc_load = {k: v for k, v in enc_keys.items()}
    encoding.load_state_dict(enc_load)
    density_network.load_state_dict(
        {k.replace("density_network.", ""): v for k, v in geo_state.items() if k.startswith("density_network.")}
    )
    feature_network.load_state_dict(
        {k.replace("feature_network.", ""): v for k, v in geo_state.items() if k.startswith("feature_network.")}
    )

    bbox = geo_state.get("bbox", None)
    if bbox is not None:
        bbox = bbox.float()

    density_bias = geo_cfg.get("density_bias", "blob_magic3d")
    density_blob_scale = geo_cfg.get("density_blob_scale", 10.0)
    density_blob_std = geo_cfg.get("density_blob_std", 0.5)
    density_activation = geo_cfg.get("density_activation", "softplus")

    return {
        "encoding": encoding,
        "density_network": density_network,
        "feature_network": feature_network,
        "radius": radius,
        "bbox": bbox,
        "density_bias": density_bias,
        "density_blob_scale": density_blob_scale,
        "density_blob_std": density_blob_std,
        "density_activation": density_activation,
    }


def compute_density(model, points):
    encoding = model["encoding"]
    density_net = model["density_network"]
    radius = model["radius"]
    bbox = model["bbox"]
    density_bias_type = model["density_bias"]
    density_blob_scale = model["density_blob_scale"]
    density_blob_std = model["density_blob_std"]
    density_act = model["density_activation"]

    points_unscaled = points
    unbounded = bbox is None or (bbox[1] > radius).any()

    points_norm = contract_to_unisphere(points, bbox, unbounded)

    enc = encoding(points_norm.view(-1, 3))
    density = density_net(enc).view(*points.shape[:-1], 1)

    if density_bias_type == "blob_magic3d":
        density_bias = density_blob_scale * (
            1 - torch.sqrt((points_unscaled ** 2).sum(dim=-1)) / density_blob_std
        )[..., None]
    elif density_bias_type == "blob_dreamfusion":
        density_bias = density_blob_scale * torch.exp(
            -0.5 * (points_unscaled ** 2).sum(dim=-1) / density_blob_std ** 2
        )[..., None]
    else:
        density_bias = density_bias_type

    raw_density = density + density_bias
    density = get_activation(density_act)(raw_density)

    return density


def compute_features(model, points):
    encoding = model["encoding"]
    feature_net = model["feature_network"]
    bbox = model["bbox"]
    radius = model["radius"]

    unbounded = bbox is None or (bbox[1] > radius).any()
    points_norm = contract_to_unisphere(points, bbox, unbounded)

    enc = encoding(points_norm.view(-1, 3))
    features = feature_net(enc).view(*points.shape[:-1], 3)

    return features


def extract_mesh(ckpt_path, config_yaml_path, output_obj_path, resolution=256, threshold=10.0):
    device = torch.device("cuda")
    model = load_geometry(ckpt_path, config_yaml_path)

    for k in ["encoding", "density_network", "feature_network"]:
        model[k] = model[k].to(device).eval()

    if model["bbox"] is not None:
        model["bbox"] = model["bbox"].to(device)

    radius = model["radius"]

    N = resolution
    x = np.linspace(-radius, radius, N)
    y = np.linspace(-radius, radius, N)
    z = np.linspace(-radius, radius, N)

    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)

    density_grid = np.zeros((N, N, N), dtype=np.float32)

    batch_size = 100000
    flat_points = grid.reshape(-1, 3)

    for i in range(0, len(flat_points), batch_size):
        batch = torch.from_numpy(flat_points[i:i + batch_size]).float().to(device)
        with torch.no_grad():
            d = compute_density(model, batch)
            density_grid_flat = d.cpu().numpy().flatten()
        start_idx = i
        end_idx = min(i + batch_size, len(flat_points))
        density_grid.reshape(-1)[start_idx:end_idx] = density_grid_flat

    vertices, faces, normals, values = marching_cubes(density_grid, threshold)

    vertices = vertices / (N - 1) * (2 * radius) - radius

    print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")

    vertex_points = torch.from_numpy(vertices).float().to(device)
    colors = []
    for i in range(0, len(vertex_points), batch_size):
        with torch.no_grad():
            feat = compute_features(model, vertex_points[i:i + batch_size])
            colors.append(feat.cpu().numpy())
    colors = np.concatenate(colors)
    colors = torch.sigmoid(torch.from_numpy(colors)).numpy()  # albedo_activation: sigmoid (config)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.export(output_obj_path)
    print(f"Saved mesh to {output_obj_path}")

    ply_path = output_obj_path.replace(".obj", ".ply")
    mesh.export(ply_path)
    print(f"Saved PLY to {ply_path}")

    return mesh


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()

    extract_mesh(args.ckpt, args.config, args.output, args.resolution, args.threshold)
