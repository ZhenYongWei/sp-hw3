"""Render trained 3DGS and evaluate quality (PSNR/SSIM/LPIPS) + novel-view mosaic."""
import argparse
import os

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

SH_C0 = 0.28209479177387814


def load_splats(path):
    d = torch.load(path, map_location="cuda")
    return d


def make_viewmat_K(c2w, fov_x, W, H):
    fl = W / (2.0 * np.tan(fov_x / 2.0))
    w2c = np.linalg.inv(c2w)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = w2c[:3, :3]
    vm[:3, 3] = w2c[:3, 3]
    K = np.array([[fl, 0, W / 2], [0, fl, H / 2], [0, 0, 1]], dtype=np.float32)
    return vm, K


def render(d, c2w, W, H, sh_degree=3):
    vm, K = make_viewmat_K(c2w, d["fov_x"], W, H)
    vm = torch.tensor(vm, device="cuda")
    Ks = torch.tensor(K, device="cuda").reshape(1, 3, 3)
    q = d["quats"]
    s = d["scales"].clamp(max=1e4)
    o = d["opacities"]
    with torch.no_grad():
        r, _, _ = rasterization(
            means=d["means"], quats=q, scales=s, opacities=o, colors=d["sh"],
            viewmats=vm.unsqueeze(0), Ks=Ks, width=W, height=H,
            sh_degree=sh_degree, render_mode="RGB", packed=True)
    return r[0]


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 100.0 if mse == 0 else -10.0 * np.log10(mse)


def ssim_torch(x, y):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    return fn(x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]).item()


_LPIPS = None
def lpips_torch(x, y):
    global _LPIPS
    try:
        if _LPIPS is None:
            import lpips
            _LPIPS = lpips.LPIPS(net="vgg").cuda()
        return _LPIPS(x.permute(2, 0, 1)[None] * 2 - 1,
                      y.permute(2, 0, 1)[None] * 2 - 1).item()
    except Exception as e:
        return float("nan")


def main(splats_path, data_dir, output_dir, n_novel=8, sh_degree=3,
         sharp_thresh=80.0, downscale=2):
    os.makedirs(output_dir, exist_ok=True)
    d = load_splats(splats_path)
    W, H = int(d["img_w"]), int(d["img_h"])
    saved_frames = d["frames"]
    print(f"Splats: {d['means'].shape[0]} gaussians, {W}x{H}, {len(saved_frames)} cameras")

    # reload frames with the SAME filter/order used in training so GT[i] aligns
    # exactly with saved_frames[i] (the json index path would mis-align because
    # blurry frames were skipped)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from train_gsplat_v3 import load_frames
    gt_frames, _ = load_frames(data_dir, sharp_thresh=sharp_thresh, downscale=downscale)
    n = min(len(saved_frames), len(gt_frames))
    print(f"aligned views: {n} (saved={len(saved_frames)}, gt={len(gt_frames)})")

    psnrs, ssims, lpipses = [], [], []
    for i in range(n):
        gt_t = torch.tensor(gt_frames[i]["image"], device="cuda", dtype=torch.float32)
        rend = render(d, saved_frames[i], W, H, sh_degree)
        psnrs.append(psnr(rend, gt_t))
        ssims.append(ssim_torch(rend, gt_t))
        lpipses.append(lpips_torch(rend, gt_t))

    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    lpipses = np.array(lpipses)
    print(f"\n=== Reconstruction quality ({n} views) ===")
    print(f"PSNR:  {psnrs.mean():.3f} +/- {psnrs.std():.3f}")
    print(f"SSIM:  {ssims.mean():.4f} +/- {ssims.std():.4f}")
    print(f"LPIPS: {lpipses.mean():.4f} +/- {lpipses[np.isfinite(lpipses)].std():.4f}")

    # mosaic of GT vs render for 8 evenly spaced views
    idxs = np.linspace(0, n - 1, 8).astype(int)
    pads = []
    for i in idxs:
        gt = (gt_frames[i]["image"] * 255).astype(np.uint8)
        rend = render(d, saved_frames[i], W, H, sh_degree)
        rend_np = (rend.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pad = np.concatenate([gt, rend_np], axis=1)
        pads.append(pad)
    mosaic = np.concatenate(pads, axis=0)
    Image.fromarray(mosaic).save(os.path.join(output_dir, "gt_vs_render.png"))
    print(f"saved gt_vs_render.png ({mosaic.shape})")

    # novel views: interpolate between consecutive poses
    novel = []
    for k in range(n_novel):
        t = (k + 1) / (n_novel + 1)
        i0 = int((len(saved_frames) - 1) * (k / n_novel))
        i1 = min(i0 + 1, len(saved_frames) - 1)
        c0 = saved_frames[i0]; c1 = saved_frames[i1]
        c2w = (1 - t) * np.array(c0) + t * np.array(c1)
        c2w[:3, :3] /= np.linalg.norm(c2w[:3, :3], axis=0, keepdims=True) + 1e-8
        rend = render(d, c2w, W, H, sh_degree)
        novel.append((rend.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    rows = [np.concatenate(novel[i:i+4], axis=1) for i in range(0, len(novel), 4)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(output_dir, "novel_views.png"))
    print(f"saved novel_views.png")

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"PSNR {psnrs.mean():.3f} {psnrs.std():.3f}\n")
        f.write(f"SSIM {ssims.mean():.4f} {ssims.std():.4f}\n")
        f.write(f"LPIPS {lpipses.mean():.4f}\n")
        f.write(f"N_gaussians {d['means'].shape[0]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splats", default="task1_data/real_object_output_v3/splats_final.pth")
    ap.add_argument("--data_dir", default="task1_data/real_object")
    ap.add_argument("--output_dir", default="task1_data/real_object_output_v3/eval")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--sharp_thresh", type=float, default=80.0)
    ap.add_argument("--downscale", type=int, default=2)
    args = ap.parse_args()
    main(args.splats, args.data_dir, args.output_dir, sh_degree=args.sh_degree,
         sharp_thresh=args.sharp_thresh, downscale=args.downscale)
