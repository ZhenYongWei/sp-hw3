"""Retrain the Mip-NeRF360 garden background with the corrected v3 3DGS pipeline.

The original garden model was trained with the buggy v1 script (random init, no
densification) and learned essentially no color (all gaussians = 0.5 +/- 0.003).
This script rebuilds it properly. No COLMAP points3D is available, so we init by
ray-casting a grid from each camera at multiple depths.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gsplat_v3 import (Gaussians, _ssim, make_viewmat_K, _save)
from gsplat import rasterization


def load_frames(data_dir, downscale=4):
    meta = json.load(open(os.path.join(data_dir, "transforms_train.json")))
    fov_x = float(meta["camera_angle_x"])
    frames = []
    for fr in meta["frames"]:
        fpath = os.path.join(data_dir, "images", fr["file_path"].split("/")[-1])
        if not os.path.exists(fpath):
            fpath = os.path.join(data_dir, fr["file_path"])
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        W, H = img.size
        if downscale > 1:
            W, H = W // downscale, H // downscale
            img = img.resize((W, H), Image.LANCZOS)
        frames.append({"image": np.asarray(img, dtype=np.float32) / 255.0,
                       "c2w": np.array(fr["transform_matrix"], dtype=np.float64)})
    return frames, fov_x


def ray_init_points(frames, fov_x, n_per_cam=900, n_depths=3, depth_range=(1.0, 8.0)):
    """Cast a grid of rays from each camera; sample points at several depths."""
    pts = []
    for f in frames:
        c2w = f["c2w"]
        cam_pos = c2w[:3, 3]
        # forward = -z (OpenGL/Blender convention typical for NeRF transforms)
        fwd = -c2w[:3, 2]
        right = c2w[:3, 0]
        up = c2w[:3, 1]
        H, W = f["image"].shape[:2]
        fl = W / (2 * np.tan(fov_x / 2))
        g = int(math.sqrt(n_per_cam))
        xs = np.linspace(-0.45, 0.45, g) * W
        ys = np.linspace(-0.45, 0.45, g) * H
        depths = np.linspace(depth_range[0], depth_range[1], n_depths)
        for dx in xs:
            for dy in ys:
                d_ray = (fwd + (dx - W / 2) / fl * right + (dy - H / 2) / fl * up)
                d_ray = d_ray / np.linalg.norm(d_ray)
                for dp in depths:
                    pts.append(cam_pos + d_ray * dp)
    pts = np.array(pts, dtype=np.float32)
    # subsample
    if len(pts) > 60000:
        idx = np.random.choice(len(pts), 60000, replace=False)
        pts = pts[idx]
    return pts


def nn_scale(pts, k=3):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return float(np.clip(d[:, 1:].mean(), 1e-4, None))


def train(data_dir, output_dir, iterations=20000, downscale=4, device="cuda:0",
          sh_degree=3, save_every=7000):
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)
    frames, fov_x = load_frames(data_dir, downscale=downscale)
    H, W = frames[0]["image"].shape[:2]
    print(f"Garden: {len(frames)} frames, {W}x{H}, fov={fov_x:.3f}")

    # normalize scene using camera positions (object of interest is around there)
    cam_pos = np.array([f["c2w"][:3, 3] for f in frames])
    center = cam_pos.mean(0)
    scale = np.max(np.linalg.norm(cam_pos - center, axis=1))
    for f in frames:
        c2w = f["c2w"].copy()
        c2w[:3, 3] = (c2w[:3, 3] - center) / scale
        f["c2w_n"] = c2w
    print(f"normalized: scale={scale:.2f}")

    # init points via ray casting (in normalized coords, depths scaled)
    init_pts = ray_init_points(frames, fov_x,
                               depth_range=(1.0 / scale * 0.5, 8.0 / scale))
    # re-center the rays using normalized cam positions
    init_pts = (init_pts - center) / scale
    # colors: neutral gray, will be learned
    init_cols = np.full((len(init_pts), 3), 0.5, dtype=np.float32)
    iscale = nn_scale(init_pts)
    print(f"init: {len(init_pts)} points, nn-scale={iscale:.5f}")

    g = Gaussians(init_pts, init_cols, iscale, dev, sh_degree=sh_degree)

    cams = []
    for f in frames:
        vm, K = make_viewmat_K(f["c2w_n"], fov_x, W, H)
        cams.append({"gt": torch.tensor(f["image"], device=dev, dtype=torch.float32),
                     "viewmat": torch.tensor(vm, device=dev),
                     "K": torch.tensor(K, device=dev)})
    print(f"prepared {len(cams)} cameras")

    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh": 2.5e-3}
    densify_from, densify_until, densify_interval = 500, 12000, 200
    cfg = {"grad_thresh": 0.0002, "split_scale": max(iscale, 0.01),
           "min_opacity": 0.02, "max_scale_abs": 0.5,
           "max_gaussians": 400000, "grad_quantile": 0.88}
    warmup = 1000
    adam_t = 0

    for step in range(1, iterations + 1):
        lr_now = {k: v * (0.32 if step < warmup else 1.0) for k, v in lrs.items()}
        if step == densify_until:
            lr_now = {k: v * 0.1 for k, v in lr_now.items()}
        cd = cams[np.random.randint(len(cams))]
        q = g.quats / g.quats.norm(dim=-1, keepdim=True)
        s = g.scales.exp().clamp(max=1e4)
        o = g.opacities.sigmoid()
        renders, _, _ = rasterization(
            means=g.means, quats=q, scales=s, opacities=o, colors=g.sh,
            viewmats=cd["viewmat"].unsqueeze(0), Ks=cd["K"].reshape(1, 3, 3),
            width=W, height=H, sh_degree=sh_degree, render_mode="RGB", packed=True)
        rendered = renders[0]
        gt = cd["gt"]
        l1 = torch.abs(rendered - gt).mean()
        ssim_val = _ssim(rendered, gt)
        loss = (1 - ssim_val) * 0.2 + l1 * 0.8

        g.zero_grad()
        loss.backward()
        adam_t += 1
        g.adam_step(lr_now, adam_t)
        g.accumulate_grad()

        if densify_from <= step < densify_until and step % densify_interval == 0:
            n0, n1 = g.densify_and_prune(cfg)
            if step % 2000 == 0:
                print(f"  densify@{step}: {n0} -> {n1}")

        if step % 1000 == 0 or step == iterations:
            with torch.no_grad():
                psnr_v = -10.0 * torch.log10(((rendered - gt) ** 2).mean()).item()
            print(f"step {step}/{iterations}: L1={l1.item():.4f} SSIM={ssim_val.item():.4f} "
                  f"PSNR={psnr_v:.2f} N={g.means.shape[0]}")

        if step % save_every == 0 or step == iterations:
            _save(g, os.path.join(output_dir, f"point_cloud_{step}.ply"))

    _save(g, os.path.join(output_dir, "point_cloud.ply"))
    torch.save({"means": g.means.detach(),
                "quats": (g.quats / g.quats.norm(dim=-1, keepdim=True)).detach(),
                "scales": g.scales.exp().detach(),
                "opacities": g.opacities.sigmoid().detach(),
                "sh": g.sh.detach(), "fov_x": fov_x, "img_w": W, "img_h": H,
                "frames": [f["c2w_n"] for f in frames], "center": center, "scale": scale},
               os.path.join(output_dir, "splats_final.pth"))
    print(f"Done. {g.means.shape[0]} gaussians -> {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="task1_data/garden_3dgs")
    ap.add_argument("--output_dir", default="task1_data/garden_output_v3")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    train(args.data_dir, args.output_dir, iterations=args.iterations,
          downscale=args.downscale, device=args.device)
