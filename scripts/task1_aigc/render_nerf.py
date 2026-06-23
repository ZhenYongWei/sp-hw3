"""Volume-render the DreamFusion NeRF directly (shows the actual learned appearance,
even when marching-cubes geometry collapses to the blob prior sphere).

Loads geometry (encoding+density+feature nets) from a threestudio checkpoint and
does proper alpha-compositing volume rendering from orbit cameras.
"""
import argparse, os, sys
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "threestudio"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_mesh import load_geometry, compute_density, compute_features

SH_C0 = 0.28209479177387814


def render_nerf(ckpt, config_yaml, out_path, n_views=8, res=400, radius=2.6):
    dev = torch.device("cuda")
    model = load_geometry(ckpt, config_yaml)
    for k in ["encoding", "density_network", "feature_network"]:
        model[k] = model[k].to(dev).eval()
    if model["bbox"] is not None:
        model["bbox"] = model["bbox"].to(dev)

    frames = []
    for i in range(n_views):
        deg = 2 * np.pi * i / n_views
        eye = np.array([radius * np.cos(deg), radius * np.sin(deg) * 0.5, radius * 0.35])
        frames.append(_render_one(model, eye, res, dev))
        print(f"  view {i+1}/{n_views} done")
    mosaic = np.concatenate(frames, axis=1)
    Image.fromarray(mosaic).save(out_path)
    print(f"saved {out_path}  ({mosaic.shape})")


def _render_one(model, eye, res, dev, target=(0, 0, 0), near=1.0, far=4.0, n_samples=128):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    up = np.array([0., 0., 1.]); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    W = H = res
    fov = np.radians(40); fx = W / (2 * np.tan(fov / 2))
    # ray dirs (camera looks +f, x=right, y=up)
    xs = (np.arange(W) - W / 2) / fx; ys = (np.arange(H) - H / 2) / fx
    ix, iy = np.meshgrid(xs, ys)
    dirs = ix[..., None] * r + iy[..., None] * u + f[None, None, :]
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = torch.tensor(dirs, dtype=torch.float32, device=dev).reshape(-1, 3)
    origins = torch.tensor(eye, dtype=torch.float32, device=dev).expand_as(dirs)

    t_vals = torch.linspace(near, far, n_samples, device=dev)
    pts = origins[:, None, :] + dirs[:, None, :] * t_vals[None, :, None]  # (P, S, 3)
    P, S, _ = pts.shape
    pts_flat = pts.reshape(-1, 3)

    # density + features in chunks
    dens = []; feat = []
    bs = 200000
    with torch.no_grad():
        for i in range(0, len(pts_flat), bs):
            chunk = pts_flat[i:i+bs]
            dens.append(compute_density(model, chunk).reshape(-1))
            feat.append(compute_features(model, chunk))
    density = torch.cat(dens).reshape(P, S)
    features = torch.cat(feat).reshape(P, S, 3)
    color = torch.sigmoid(features)  # albedo

    # volume render (alpha compositing)
    deltas = torch.full((S,), (far - near) / n_samples, device=dev)
    alpha = 1.0 - torch.exp(-density * deltas[None, :])  # (P, S)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    T = torch.cat([torch.ones(P, 1, device=dev), T[:, :-1]], dim=1)
    w = T * alpha  # (P, S)
    rgb = (w[..., None] * color).sum(1)  # (P, 3)
    acc = w.sum(1, keepdim=True)  # (P, 1)
    rgb = rgb / acc.clamp(min=1e-4)

    img = rgb.reshape(H, W, 3).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--res", type=int, default=400)
    args = ap.parse_args()
    render_nerf(args.ckpt, args.config, args.out, args.n_views, args.res)
