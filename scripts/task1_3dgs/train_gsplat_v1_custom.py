import json, os
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

SH_C0 = 0.28209479177387814

def load_nerf_data(data_dir, split="train"):
    with open(os.path.join(data_dir, f"transforms_{split}.json")) as f:
        meta = json.load(f)
    frames = []
    for frame in meta["frames"]:
        fpath = os.path.join(data_dir, "images", frame["file_path"])
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        c2w = np.array(frame["transform_matrix"])
        frames.append({"image": np.array(img) / 255.0, "c2w": c2w})
    fov_x = meta["camera_angle_x"]
    return frames, fov_x

def c2w_to_w2c(c2w, fov_x, img_w, img_h):
    fl_x = img_w / (2 * np.tan(fov_x / 2))
    fl_y = fl_x
    cx, cy = img_w / 2.0, img_h / 2.0
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
    return R, T, K

def train(data_dir, output_dir, iterations=7000, n_gaussians=30000, lr_init=1e-3, device="cuda:0"):
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)
    
    frames, fov_x = load_nerf_data(data_dir, "train")
    H, W = frames[0]["image"].shape[:2]
    print(f"Data: {len(frames)} frames, {W}x{H}, fov={fov_x:.3f} rad ({fov_x*180/np.pi:.1f} deg)")
    
    # Precompute all camera params
    cam_data = []
    for f in frames:
        R, T, K = c2w_to_w2c(f["c2w"], fov_x, W, H)
        cam_data.append({
            "gt": torch.tensor(f["image"], device=dev, dtype=torch.float32),
            "viewmat": torch.tensor(np.eye(4), device=dev, dtype=torch.float32),
            "K": torch.tensor(K, device=dev, dtype=torch.float32),
        })
        cam_data[-1]["viewmat"][:3, :3] = torch.tensor(R, device=dev, dtype=torch.float32)
        cam_data[-1]["viewmat"][:3, 3] = torch.tensor(T, device=dev, dtype=torch.float32)
    
    # Init gaussians - random positions around origin
    means = torch.nn.Parameter(torch.randn(n_gaussians, 3, device=dev) * 0.3)
    scales = torch.nn.Parameter(torch.ones(n_gaussians, 3, device=dev) * 0.01)
    quats = torch.nn.Parameter(torch.randn(n_gaussians, 4, device=dev))
    raw_opacities = torch.nn.Parameter(torch.zeros(n_gaussians, device=dev))
    colors = torch.nn.Parameter(torch.zeros(n_gaussians, 12, 3, device=dev))
    
    optimizer = torch.optim.Adam([
        {"params": [means], "lr": lr_init, "name": "means"},
        {"params": [quats], "lr": lr_init, "name": "quats"},
        {"params": [scales], "lr": 5e-3, "name": "scales"},
        {"params": [raw_opacities], "lr": 5e-2, "name": "opacities"},
        {"params": [colors], "lr": lr_init, "name": "colors"},
    ], eps=1e-7)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01**(1/iterations))
    
    best_loss = float("inf")
    
    for step in range(iterations):
        idx = np.random.randint(len(cam_data))
        cd = cam_data[idx]
        
        q = quats / quats.norm(dim=-1, keepdim=True)
        s = scales.abs().clamp(min=1e-6)
        o = raw_opacities.sigmoid()
        
        render_colors, render_alphas, info = rasterization(
            means=means, quats=q, scales=s, opacities=o, colors=colors,
            viewmats=cd["viewmat"].unsqueeze(0),
            Ks=cd["K"].reshape(1, 3, 3),
            width=W, height=H,
            sh_degree=1,
            packed=True,
        )
        
        rendered = render_colors[0]
        gt = cd["gt"]
        
        l1 = torch.abs(rendered - gt).mean()
        mu1, mu2 = rendered.mean(), gt.mean()
        s1_sq = ((rendered - mu1)**2).mean()
        s2_sq = ((gt - mu2)**2).mean()
        s12 = ((rendered - mu1) * (gt - mu2)).mean()
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1_sq+s2_sq+C2))
        loss = 0.8 * l1 + 0.2 * (1 - ssim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Periodic pruning of low-opacity gaussians
        if step > 0 and step % 1000 == 0:
            with torch.no_grad():
                keep = o > 0.005
                n_keep = keep.sum().item()
                if n_keep < n_gaussians * 0.5:
                    print(f"  Pruning: {n_gaussians} -> {n_keep} gaussians (opacity > 0.005)")
                    idx_keep = keep.nonzero(as_tuple=True)[0]
                    means.data = means.data[idx_keep]
                    quats.data = quats.data[idx_keep]
                    scales.data = scales.data[idx_keep]
                    raw_opacities.data = raw_opacities.data[idx_keep]
                    colors.data = colors.data[idx_keep]
                    n_gaussians = n_keep
        
        if step % 500 == 0:
            n = means.shape[0]
            print(f"Step {step}/{iterations}: L1={l1.item():.4f}, SSIM={ssim.item():.4f}, Loss={loss.item():.4f}, N={n}")
            if loss.item() < best_loss:
                best_loss = loss.item()
    
    # Save
    q_final = (quats / quats.norm(dim=-1, keepdim=True)).detach()
    s_final = scales.abs().clamp(min=1e-6).detach()
    o_final = raw_opacities.sigmoid().detach()
    
    torch.save({
        "means": means.detach(), "quats": q_final, "scales": s_final,
        "opacities": o_final, "colors": colors.detach(),
        "fov_x": fov_x, "img_w": W, "img_h": H,
        "frames": [f["c2w"] for f in frames],
    }, os.path.join(output_dir, "splats_final.pth"))
    
    _export_ply(means.detach(), q_final, s_final, o_final, colors.detach(),
                os.path.join(output_dir, "point_cloud.ply"))
    print(f"Done! Saved to {output_dir}, best_loss={best_loss:.4f}")

def _export_ply(means, quats, scales, opacities, colors, path):
    from plyfile import PlyData, PlyElement
    n = means.shape[0]
    rgb = (colors[:, 0, :] * SH_C0 + 0.5).clip(0, 1)
    rgb_np = rgb.cpu().numpy()
    rgb_u8 = (rgb_np * 255).astype(np.uint8)
    dtype = [("x","f4"),("y","f4"),("z","f4"),
             ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
             ("opacity","f4"),
             ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
             ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
             ("red","u1"),("green","u1"),("blue","u1")]
    el = np.empty(n, dtype=dtype)
    m = means.cpu().numpy(); el["x"],el["y"],el["z"] = m[:,0],m[:,1],m[:,2]
    c = colors.cpu().numpy(); el["f_dc_0"],el["f_dc_1"],el["f_dc_2"] = c[:,0,0],c[:,0,1],c[:,0,2]
    el["opacity"] = opacities.cpu().numpy()
    s = scales.cpu().numpy(); el["scale_0"],el["scale_1"],el["scale_2"] = s[:,0],s[:,1],s[:,2]
    q = quats.cpu().numpy(); el["rot_0"],el["rot_1"],el["rot_2"],el["rot_3"] = q[:,0],q[:,1],q[:,2],q[:,3]
    el["red"],el["green"],el["blue"] = rgb_u8[:,0],rgb_u8[:,1],rgb_u8[:,2]
    PlyData([PlyElement.describe(el,"vertex")]).write(path)
    print(f"PLY: {n} gaussians -> {path}")

if __name__ == "__main__":
    train(
        "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectA_3dgs",
        "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectA_output",
        iterations=7000, n_gaussians=30000, device="cuda:0"
    )
