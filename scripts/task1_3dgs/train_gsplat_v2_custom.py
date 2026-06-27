import json, os, struct
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

SH_C0 = 0.28209479177387814


def load_colmap_points(sparse_dir):
    pts = []
    colors = []
    with open(os.path.join(sparse_dir, "points3D.txt"), "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
            pts.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(pts, dtype=np.float32), np.array(colors, dtype=np.float32)


def load_data(data_dir, downscale=2):
    with open(os.path.join(data_dir, "transforms_train.json")) as f:
        meta = json.load(f)

    frames = []
    fov_x = meta["camera_angle_x"]

    for frame in meta["frames"]:
        fpath = os.path.join(data_dir, "images", frame["file_path"])
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        W, H = img.size
        if downscale > 1:
            W, H = W // downscale, H // downscale
            img = img.resize((W, H), Image.LANCZOS)
        c2w = np.array(frame["transform_matrix"])
        frames.append({"image": np.array(img) / 255.0, "c2w": c2w})

    return frames, fov_x


def c2w_to_viewmat(c2w, fov_x, W, H):
    c2w = c2w.astype(np.float32)
    fov_x = np.float32(fov_x)
    fl_x = np.float32(W) / (2 * np.tan(fov_x / 2))
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = T
    K = np.array([[fl_x, 0, W / 2], [0, fl_x, H / 2], [0, 0, 1]], dtype=np.float32)
    return viewmat, K


def train(
    data_dir,
    output_dir,
    colmap_sparse_dir,
    iterations=30000,
    batch_size=4,
    downscale=2,
    device="cuda:0",
):
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)

    frames, fov_x = load_data(data_dir, downscale=downscale)
    H, W = frames[0]["image"].shape[:2]
    print(f"Data: {len(frames)} frames, {W}x{H} (downscale={downscale}), fov={fov_x:.3f}")

    cam_data = []
    for f in frames:
        viewmat, K = c2w_to_viewmat(f["c2w"], fov_x, W, H)
        cam_data.append({
            "gt": torch.tensor(f["image"], device=dev, dtype=torch.float32),
            "viewmat": torch.tensor(viewmat, device=dev, dtype=torch.float32),
            "K": torch.tensor(K, device=dev, dtype=torch.float32),
        })
    print(f"Prepared {len(cam_data)} cameras")

    # === 关键改进1: 从COLMAP点云初始化 ===
    colmap_pts, colmap_colors = load_colmap_points(colmap_sparse_dir)

    # 归一化场景到单位球
    scene_center = colmap_pts.mean(axis=0).astype(np.float32)
    scene_scale = np.float32(np.max(np.linalg.norm(colmap_pts - scene_center, axis=1)))
    colmap_pts = ((colmap_pts - scene_center) / scene_scale).astype(np.float32)
    # 同时归一化相机位姿
    for f in frames:
        c2w = f["c2w"].astype(np.float32)
        c2w[:3, 3] = (c2w[:3, 3] - scene_center) / scene_scale
        f["c2w"] = c2w
    # 重建 cam_data
    cam_data = []
    for f in frames:
        viewmat, K = c2w_to_viewmat(f["c2w"], fov_x, W, H)
        cam_data.append({
            "gt": torch.tensor(f["image"], device=dev, dtype=torch.float32),
            "viewmat": torch.tensor(viewmat, device=dev, dtype=torch.float32),
            "K": torch.tensor(K, device=dev, dtype=torch.float32),
        })
    print(f"Scene normalized: center={scene_center}, scale={scene_scale:.2f}")

    n_init = len(colmap_pts)
    print(f"COLMAP init: {n_init} points, range=[{colmap_pts.min():.2f}, {colmap_pts.max():.2f}]")

    means = torch.nn.Parameter(torch.tensor(colmap_pts, dtype=torch.float32, device=dev))

    dists = np.linalg.norm(colmap_pts[:, None] - colmap_pts[None, :], axis=-1)
    avg_dist = np.median(dists[dists > 0]) if np.sum(dists > 0) > 0 else 0.01
    init_scale = avg_dist * 0.3  # 归一化后用更小的scale
    scales = torch.nn.Parameter(torch.ones(n_init, 3, device=dev) * init_scale)

    quats = torch.nn.Parameter(torch.tile(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev), (n_init, 1)
    ))

    raw_opacities = torch.nn.Parameter(
        torch.tensor(np.log(0.1 / (1 - 0.1)), dtype=torch.float32, device=dev).repeat(n_init)
    )

    sh_dc = torch.tensor(colmap_colors, dtype=torch.float32, device=dev)
    sh_dc = (sh_dc - 0.5) / SH_C0
    n_sh = 4  # (degree+1)^2 for degree=1
    colors = torch.nn.Parameter(torch.zeros(n_init, n_sh, 3, device=dev))
    colors.data[:, 0, :] = sh_dc

    print(f"Init: {n_init} Gaussians, scale={init_scale:.4f}")

    # === 关键改进2: 参数分组 + 不同学习率 ===
    param_groups = [
        {"params": [means], "lr": 1.6e-4, "name": "means"},
        {"params": [scales], "lr": 5e-3, "name": "scales"},
        {"params": [quats], "lr": 1e-3, "name": "quats"},
        {"params": [raw_opacities], "lr": 5e-2, "name": "opacities"},
        {"params": [colors], "lr": 2.5e-3, "name": "colors"},
    ]
    optimizer = torch.optim.Adam(param_groups, eps=1e-7)

    # === densification 控制 ===
    densify_from_step = 500
    densify_until_step = 15000
    densify_interval = 100
    densify_grad_threshold = 0.0002
    prune_opacity_threshold = 0.005
    densify_screen_size_threshold = 0.05

    best_ssim = -1

    for step in range(iterations):
        batch_idx = np.random.choice(len(cam_data), min(batch_size, len(cam_data)), replace=False)

        q = quats / quats.norm(dim=-1, keepdim=True)
        s = scales.abs().clamp(min=1e-6)  # 直接用绝对值，不exp
        o = raw_opacities.sigmoid()

        total_loss = 0
        for idx in batch_idx:
            cd = cam_data[idx]
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=q,
                scales=s,
                opacities=o,
                colors=colors,
                viewmats=cd["viewmat"].unsqueeze(0),
                Ks=cd["K"].reshape(1, 3, 3),
                width=W,
                height=H,
                sh_degree=1,
                packed=True,
            )

            rendered = render_colors[0]
            gt = cd["gt"]

            l1 = torch.abs(rendered - gt).mean()
            mu1, mu2 = rendered.mean(), gt.mean()
            s1_sq = ((rendered - mu1) ** 2).mean()
            s2_sq = ((gt - mu2) ** 2).mean()
            s12 = ((rendered - mu1) * (gt - mu2)).mean()
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (s1_sq + s2_sq + C2)
            )
            total_loss = total_loss + (1 - ssim) * 0.2 + l1 * 0.8

        loss = total_loss / len(batch_idx)

        optimizer.zero_grad()
        loss.backward()

        # 保存梯度范数用于densification
        if means.grad is not None:
            saved_grad_norms = torch.norm(means.grad, dim=-1).detach().clone()
        else:
            saved_grad_norms = None

        optimizer.step()

        # === 关键改进3: Densification (clone + split) ===
        if (
            step >= densify_from_step
            and step < densify_until_step
            and step % densify_interval == 0
        ):
            with torch.no_grad():
                n_before = means.shape[0]

                # Prune low opacity
                o_cur = raw_opacities.sigmoid()
                keep = o_cur > prune_opacity_threshold
                if keep.sum() < n_before:
                    idx_keep = keep.nonzero(as_tuple=True)[0]
                    means = torch.nn.Parameter(means.data[idx_keep])
                    scales = torch.nn.Parameter(scales.data[idx_keep])
                    quats = torch.nn.Parameter(quats.data[idx_keep])
                    raw_opacities = torch.nn.Parameter(raw_opacities.data[idx_keep])
                    colors = torch.nn.Parameter(colors.data[idx_keep])
                    if saved_grad_norms is not None:
                        saved_grad_norms = saved_grad_norms[idx_keep]

                n_after_prune = means.shape[0]

                # Clone + Split using saved gradients
                if saved_grad_norms is not None and len(saved_grad_norms) == means.shape[0]:
                    s_cur = scales.data.abs().clamp(min=1e-8)
                    max_scale = s_cur.max(dim=-1).values

                    clone_mask = (saved_grad_norms > densify_grad_threshold) & (
                        max_scale < init_scale
                    )
                    if clone_mask.sum() > 0:
                        clone_idx = clone_mask.nonzero(as_tuple=True)[0]
                        means = torch.nn.Parameter(torch.cat([means.data, means.data[clone_idx]]))
                        scales = torch.nn.Parameter(torch.cat([scales.data, scales.data[clone_idx]]))
                        quats = torch.nn.Parameter(torch.cat([quats.data, quats.data[clone_idx]]))
                        raw_opacities = torch.nn.Parameter(torch.cat([raw_opacities.data, raw_opacities.data[clone_idx]]))
                        colors = torch.nn.Parameter(torch.cat([colors.data, colors.data[clone_idx]]))

                    # Split
                    split_mask = (saved_grad_norms > densify_grad_threshold) & (
                        max_scale >= init_scale
                    )
                    if split_mask.sum() > 0:
                        split_idx = split_mask.nonzero(as_tuple=True)[0]
                        n_split = len(split_idx)
                        new_means = means.data[split_idx].clone()
                        new_scales = scales.data[split_idx].clone()
                        new_quats = quats.data[split_idx].clone()
                        new_opa = raw_opacities.data[split_idx].clone()
                        new_colors = colors.data[split_idx].clone()

                        s_split = new_scales.abs()
                        offsets = torch.randn(n_split, 3, device=dev) * s_split * 0.5
                        new_means = new_means + offsets
                        new_scales = new_scales * 0.6

                        o_split = raw_opacities.data[split_idx].sigmoid()
                        new_opa_val = torch.log((o_split * 0.5) / (1 - o_split * 0.5 + 1e-8))
                        raw_opacities = torch.nn.Parameter(raw_opacities.data.clone())
                        raw_opacities.data[split_idx] = new_opa_val

                        means = torch.nn.Parameter(torch.cat([means.data, new_means]))
                        scales = torch.nn.Parameter(torch.cat([scales.data, new_scales]))
                        quats = torch.nn.Parameter(torch.cat([quats.data, new_quats]))
                        raw_opacities = torch.nn.Parameter(torch.cat([raw_opacities.data, new_opa]))
                        colors = torch.nn.Parameter(torch.cat([colors.data, new_colors]))

                # 重建optimizer
                optimizer = torch.optim.Adam([
                    {"params": [means], "lr": 1.6e-4 if step < 15000 else 1.6e-5, "name": "means"},
                    {"params": [scales], "lr": 5e-3, "name": "scales"},
                    {"params": [quats], "lr": 1e-3, "name": "quats"},
                    {"params": [raw_opacities], "lr": 5e-2, "name": "opacities"},
                    {"params": [colors], "lr": 2.5e-3, "name": "colors"},
                ], eps=1e-7)

        # Reduce learning rate after 15000
        if step == 15000:
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

        if step % 500 == 0:
            n = means.shape[0]
            print(
                f"Step {step}/{iterations}: L1={l1.item():.4f}, SSIM={ssim.item():.4f}, Loss={loss.item():.4f}, N={n}"
            )
            if ssim.item() > best_ssim:
                best_ssim = ssim.item()

    # === Save ===
    q_final = (quats / quats.norm(dim=-1, keepdim=True)).detach()
    s_final = scales.abs().clamp(min=1e-8).detach()
    o_final = raw_opacities.sigmoid().detach()

    torch.save(
        {
            "means": means.detach(),
            "quats": q_final,
            "scales": s_final,
            "opacities": o_final,
            "colors": colors.detach(),
            "fov_x": fov_x,
            "img_w": W,
            "img_h": H,
            "frames": [f["c2w"] for f in frames],
        },
        os.path.join(output_dir, "splats_final.pth"),
    )

    _export_ply(
        means.detach(),
        q_final,
        s_final,
        o_final,
        colors.detach(),
        os.path.join(output_dir, "point_cloud.ply"),
    )
    print(f"Done! {means.shape[0]} Gaussians, best_ssim={best_ssim:.4f}")


def _export_ply(means, quats, scales, opacities, colors, path):
    from plyfile import PlyData, PlyElement

    n = means.shape[0]
    rgb = (colors[:, 0, :] * SH_C0 + 0.5).clip(0, 1)
    rgb_np = rgb.cpu().numpy()
    rgb_u8 = (rgb_np * 255).astype(np.uint8)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    el = np.empty(n, dtype=dtype)
    m = means.cpu().numpy()
    el["x"], el["y"], el["z"] = m[:, 0], m[:, 1], m[:, 2]
    c = colors.cpu().numpy()
    el["f_dc_0"], el["f_dc_1"], el["f_dc_2"] = c[:, 0, 0], c[:, 0, 1], c[:, 0, 2]
    el["opacity"] = opacities.cpu().numpy()
    s = scales.cpu().numpy()
    el["scale_0"], el["scale_1"], el["scale_2"] = s[:, 0], s[:, 1], s[:, 2]
    q = quats.cpu().numpy()
    el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    el["red"], el["green"], el["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    PlyData([PlyElement.describe(el, "vertex")]).write(path)
    print(f"PLY: {n} gaussians -> {path}")


if __name__ == "__main__":
    train(
        "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/real_object",
        "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/real_object_output_v2",
        "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/real_object/sparse/0",
        iterations=30000,
        batch_size=1,
        downscale=4,
        device="cuda:0",
    )
