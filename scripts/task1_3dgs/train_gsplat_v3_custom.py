"""3DGS training (gsplat) — corrected version v3.

Fixes vs v1/v2:
  1. Correct camera convention: viewmat uses true w2c[:3, :3] (NOT transposed).
     v2's transposed form scrambled most camera poses (only 5/5000 pts projected
     correctly at frame 30 vs 2295/5000 with the correct convention).
  2. SfM init from COLMAP points3D instead of random.
  3. Proper adaptive density control (clone/split/prune) with Adam-state
     preservation (no optimizer reset each step).
  4. SH degree 3 (was 1), standard 3DGS hyperparameters, 30k iterations.
  5. Blurry-frame filtering.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------- data
def load_frames(data_dir, sharp_thresh=0.0, downscale=1):
    meta = json.load(open(os.path.join(data_dir, "transforms_train.json")))
    fov_x = float(meta["camera_angle_x"])
    frames, n_skip = [], 0
    for fr in meta["frames"]:
        fpath = os.path.join(data_dir, "images", fr["file_path"])
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        if sharp_thresh > 0:
            lap = cv2.Laplacian(np.asarray(img.convert("L")), cv2.CV_64F).var()
            if lap < sharp_thresh:
                n_skip += 1
                continue
        W, H = img.size
        if downscale > 1:
            W, H = W // downscale, H // downscale
            img = img.resize((W, H), Image.LANCZOS)
        frames.append({"image": np.asarray(img, dtype=np.float32) / 255.0,
                       "c2w": np.array(fr["transform_matrix"], dtype=np.float64)})
    if n_skip:
        print(f"  skipped {n_skip} blurry frames (< {sharp_thresh})")
    return frames, fov_x


def load_colmap_points(sparse_dir):
    pts, cols = [], []
    with open(os.path.join(sparse_dir, "points3D.txt")) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
            cols.append([float(p[4]), float(p[5]), float(p[6])])
    return (np.array(pts, dtype=np.float32),
            np.array(cols, dtype=np.float32) / 255.0)


def make_viewmat_K(c2w, fov_x, W, H):
    fl = W / (2.0 * np.tan(fov_x / 2.0))
    w2c = np.linalg.inv(c2w)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = w2c[:3, :3]            # TRUE w2c rotation (fixed)
    vm[:3, 3] = w2c[:3, 3]
    K = np.array([[fl, 0, W / 2], [0, fl, H / 2], [0, 0, 1]], dtype=np.float32)
    return vm, K


def nn_scale(pts, k=3):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return float(np.clip(d[:, 1:].mean(), 1e-4, None))


# ---------------------------------------------------------------- model
class Gaussians:
    NAMES = ["means", "scales", "quats", "opacities", "sh"]

    def __init__(self, pts, colors, init_scale, device, sh_degree=3):
        n = len(pts)
        self.sh_degree = sh_degree
        self.n_sh = (sh_degree + 1) ** 2
        self.device = device
        self.init_scale = init_scale

        self.means = torch.tensor(pts, dtype=torch.float32, device=device)
        self.scales = torch.full((n, 3), math.log(max(init_scale, 1e-6)),
                                 dtype=torch.float32, device=device)
        self.quats = torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)),
                                  dtype=torch.float32, device=device)
        self.opacities = torch.full((n,), math.log(0.1 / 0.9),
                                    dtype=torch.float32, device=device)
        sh = torch.zeros((n, self.n_sh, 3), dtype=torch.float32, device=device)
        sh[:, 0, :] = torch.tensor((colors - 0.5) / SH_C0,
                                   dtype=torch.float32, device=device)
        self.sh = sh

        # mark autograd
        for p in self._params():
            p.requires_grad_(True)

        # Adam moments, preserved across densification
        self.state = {nm: {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
                      for nm, p in self._zipped()}
        self._reset_accum()

    def _params(self):
        return [self.means, self.scales, self.quats, self.opacities, self.sh]

    def _zipped(self):
        return zip(self.NAMES, self._params())

    def _reset_accum(self):
        self.grad_accum = torch.zeros(self.means.shape[0], device=self.device)
        self.denom = torch.zeros(self.means.shape[0], device=self.device)

    def zero_grad(self):
        for p in self._params():
            p.grad = None

    def adam_step(self, lrs, step, beta1=0.9, beta2=0.999, eps=1e-7):
        b1 = 1 - beta1 ** step
        b2 = 1 - beta2 ** step
        for nm, p in self._zipped():
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[nm]
            st["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
            st["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            m_hat = st["exp_avg"] / b1
            v_hat = st["exp_avg_sq"] / b2
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lrs[nm])

    def accumulate_grad(self):
        if self.means.grad is not None:
            ag = self.means.grad.detach().abs().sum(dim=-1)
            visible = ag > 0
            self.grad_accum[visible] += ag[visible]
            self.denom[visible] += 1.0

    def densify_and_prune(self, cfg):
        # average gradient when visible (denom counts visibility, not steps)
        grads = self.grad_accum / self.denom.clamp(min=1)
        scales_exp = self.scales.data.exp().clamp(max=1e4)
        max_scale = scales_exp.max(dim=-1).values
        opa = self.opacities.data.sigmoid()

        prune_mask = (opa < cfg["min_opacity"]) | (max_scale > cfg["max_scale_abs"])
        keep = ~prune_mask
        # auto-calibrated threshold: my 3D gradient median ~0.002 (vs official
        # 0.0002 for 2D screen grad), so the fixed floor is meaningless. Use a
        # dynamic percentile so only genuinely under-reconstructed gaussians
        # (top ~15%) densify, preventing exponential count explosion.
        vis = self.denom > 0
        if vis.any():
            floor = max(cfg["grad_thresh"], float(grads[vis].quantile(cfg["grad_quantile"])))
        else:
            floor = cfg["grad_thresh"]
        grad_mask = grads > floor
        clone_mask = grad_mask & (max_scale < cfg["split_scale"]) & keep
        split_mask = grad_mask & (max_scale >= cfg["split_scale"]) & keep

        # hard cap to avoid runaway growth
        n0 = self.means.shape[0]
        if n0 >= cfg["max_gaussians"]:
            if prune_mask.any():                      # still allow prune-only
                self._apply(prune_mask, torch.zeros_like(prune_mask), torch.zeros_like(prune_mask))
            self._reset_accum()
            return n0, self.means.shape[0]

        if prune_mask.any() or clone_mask.any() or split_mask.any():
            self._apply(prune_mask, clone_mask, split_mask)
        self._reset_accum()
        return n0, self.means.shape[0]

    def _apply(self, prune_mask, clone_mask, split_mask):
        keep_idx = (~prune_mask).nonzero(as_tuple=True)[0]
        clone_idx = clone_mask.nonzero(as_tuple=True)[0]
        split_idx = split_mask.nonzero(as_tuple=True)[0]
        n_keep = len(keep_idx)

        new_means = self.means.data[keep_idx].clone()
        new_scales = self.scales.data[keep_idx].clone()
        new_quats = self.quats.data[keep_idx].clone()
        new_opa = self.opacities.data[keep_idx].clone()
        new_sh = self.sh.data[keep_idx].clone()

        # source-row mapping for Adam state inheritance (-1 = fresh/zeroed)
        src = -torch.ones(keep_idx.shape[0], dtype=torch.long, device=self.device)
        src[:] = keep_idx

        if clone_idx.numel() > 0:                      # clone: inherit state
            new_means = torch.cat([new_means, self.means.data[clone_idx]])
            new_scales = torch.cat([new_scales, self.scales.data[clone_idx]])
            new_quats = torch.cat([new_quats, self.quats.data[clone_idx]])
            new_opa = torch.cat([new_opa, self.opacities.data[clone_idx]])
            new_sh = torch.cat([new_sh, self.sh.data[clone_idx]])
            src = torch.cat([src, clone_idx])

        if split_idx.numel() > 0:                      # split: shrink+shake
            ns = split_idx.numel()
            s_pl = self.scales.data[split_idx].exp()
            offset = torch.randn(ns, 3, device=self.device) * s_pl
            new_means = torch.cat([new_means, self.means.data[split_idx] + offset])
            new_scales = torch.cat([new_scales, torch.log(s_pl * 0.6 + 1e-8)])
            new_quats = torch.cat([new_quats, self.quats.data[split_idx]])
            new_opa = torch.cat([new_opa, self.opacities.data[split_idx]])
            new_sh = torch.cat([new_sh, self.sh.data[split_idx]])
            src = torch.cat([src, split_idx])

        self.means = new_means.requires_grad_(True)
        self.scales = new_scales.requires_grad_(True)
        self.quats = new_quats.requires_grad_(True)
        self.opacities = new_opa.requires_grad_(True)
        self.sh = new_sh.requires_grad_(True)

        valid = src >= 0
        for nm, p in self._zipped():
            old = self.state[nm]
            ea = torch.zeros_like(p)
            eas = torch.zeros_like(p)
            if valid.any():
                ea[valid] = old["exp_avg"][src[valid]]
                eas[valid] = old["exp_avg_sq"][src[valid]]
            self.state[nm] = {"exp_avg": ea, "exp_avg_sq": eas}


# ---------------------------------------------------------------- train
def _ssim(x, y):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = x.mean(), y.mean()
    s1 = ((x - mu1) ** 2).mean()
    s2 = ((y - mu2) ** 2).mean()
    s12 = ((x - mu1) * (y - mu2)).mean()
    return ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)
            / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)))


def train(data_dir, output_dir, sparse_dir, iterations=30000, downscale=2,
          sharp_thresh=80.0, device="cuda:0", sh_degree=3, save_every=7000):
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)

    frames, fov_x = load_frames(data_dir, sharp_thresh=sharp_thresh, downscale=downscale)
    H, W = frames[0]["image"].shape[:2]
    print(f"Data: {len(frames)} frames, {W}x{H}, fov={fov_x:.3f} ({math.degrees(fov_x):.1f} deg)")

    pts, cols = load_colmap_points(sparse_dir)
    center = pts.mean(0)
    scale = np.max(np.linalg.norm(pts - center, axis=1))
    pts_n = ((pts - center) / scale).astype(np.float32)
    for f in frames:
        c2w = f["c2w"].copy()
        c2w[:3, 3] = (c2w[:3, 3] - center) / scale
        f["c2w_n"] = c2w
    print(f"Scene normalized: scale={scale:.2f}, pts={len(pts_n)}")

    init_scale = nn_scale(pts_n)
    print(f"Init gaussian scale (nn-dist): {init_scale:.5f}")

    g = Gaussians(pts_n, cols, init_scale, dev, sh_degree=sh_degree)

    cams = []
    for f in frames:
        vm, K = make_viewmat_K(f["c2w_n"], fov_x, W, H)
        cams.append({"gt": torch.tensor(f["image"], device=dev, dtype=torch.float32),
                     "viewmat": torch.tensor(vm, device=dev),
                     "K": torch.tensor(K, device=dev)})
    print(f"Prepared {len(cams)} cameras")

    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh": 2.5e-3}
    densify_from, densify_until, densify_interval = 500, 13000, 200
    cfg = {"grad_thresh": 0.0002,
           "split_scale": max(init_scale, 0.01),
           "min_opacity": 0.02,
           "max_scale_abs": 0.5,
           "max_gaussians": 250000,
           "grad_quantile": 0.88}
    opacity_reg, scale_reg = 1e-5, 1e-5
    warmup = 1000

    adam_t = 0
    history = []

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
            viewmats=cd["viewmat"].unsqueeze(0),
            Ks=cd["K"].reshape(1, 3, 3),
            width=W, height=H,
            sh_degree=sh_degree,
            render_mode="RGB",
            packed=True,
        )
        rendered = renders[0]
        gt = cd["gt"]

        l1 = torch.abs(rendered - gt).mean()
        ssim_val = _ssim(rendered, gt)
        loss = (1 - ssim_val) * 0.2 + l1 * 0.8
        loss = loss + opacity_reg * o.mean() + scale_reg * (s.mean() - 1.0).abs()

        g.zero_grad()
        loss.backward()
        adam_t += 1
        g.adam_step(lr_now, adam_t)
        g.accumulate_grad()

        if densify_from <= step < densify_until and step % densify_interval == 0:
            n0, n1 = g.densify_and_prune(cfg)
            if step % 1000 == 0:
                print(f"  densify@{step}: {n0} -> {n1} gaussians")
        if step % 500 == 0 or step == iterations:
            with torch.no_grad():
                psnr_v = -10.0 * torch.log10(((rendered - gt) ** 2).mean()).item()
            n = g.means.shape[0]
            print(f"step {step}/{iterations}: L1={l1.item():.4f} SSIM={ssim_val.item():.4f} "
                  f"PSNR={psnr_v:.2f} N={n}")
            history.append((step, l1.item(), ssim_val.item(), psnr_v, n))

        if step % save_every == 0 or step == iterations:
            _save(g, os.path.join(output_dir, f"point_cloud_{step}.ply"))

    _save(g, os.path.join(output_dir, "point_cloud.ply"))
    torch.save({
        "means": g.means.detach(),
        "quats": (g.quats / g.quats.norm(dim=-1, keepdim=True)).detach(),
        "scales": g.scales.exp().detach(),
        "opacities": g.opacities.sigmoid().detach(),
        "sh": g.sh.detach(), "fov_x": fov_x, "img_w": W, "img_h": H,
        "frames": [f["c2w_n"] for f in frames], "center": center, "scale": scale,
        "history": history,
    }, os.path.join(output_dir, "splats_final.pth"))
    print(f"Done. Final gaussians: {g.means.shape[0]}, saved -> {output_dir}")


def _save(g, path):
    from plyfile import PlyData, PlyElement
    q = (g.quats / g.quats.norm(dim=-1, keepdim=True)).detach()
    s = g.scales.exp().clamp(max=1e4).detach()
    o = g.opacities.sigmoid().detach()
    rgb = (g.sh[:, 0, :].detach() * SH_C0 + 0.5).clamp(0, 1)
    m = g.means.detach().cpu().numpy()
    c = g.sh[:, 0, :].detach().cpu().numpy()
    sn, qn, on = s.cpu().numpy(), q.cpu().numpy(), o.cpu().numpy()
    rgb_u8 = (rgb.cpu().numpy() * 255).astype(np.uint8)
    n = m.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
             ("opacity", "f4"),
             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    el = np.empty(n, dtype=dtype)
    el["x"], el["y"], el["z"] = m[:, 0], m[:, 1], m[:, 2]
    el["f_dc_0"], el["f_dc_1"], el["f_dc_2"] = c[:, 0], c[:, 1], c[:, 2]
    el["opacity"] = on
    el["scale_0"], el["scale_1"], el["scale_2"] = sn[:, 0], sn[:, 1], sn[:, 2]
    el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"] = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    el["red"], el["green"], el["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    PlyData([PlyElement.describe(el, "vertex")], byte_order="<").write(path)
    print(f"  saved PLY: {n} gaussians -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="task1_data/real_object")
    ap.add_argument("--sparse_dir", default="task1_data/real_object/sparse/0")
    ap.add_argument("--output_dir", default="task1_data/real_object_output_v3")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--sharp_thresh", type=float, default=80.0)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    train(args.data_dir, args.output_dir, args.sparse_dir,
          iterations=args.iterations, downscale=args.downscale,
          sharp_thresh=args.sharp_thresh, device=args.device, sh_degree=args.sh_degree)
