import numpy as np
import torch
import cv2
from pathlib import Path

SH_C0 = 0.28209479177387814


def read_3dgs_ply(path):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]

    with open(path, "rb") as f:
        header_end = 0
        while True:
            line = f.readline()
            header_end += len(line)
            if line.decode().strip() == "end_header":
                break

    data = np.fromfile(path, dtype=dtype, offset=header_end)
    return data


def load_for_gsplat(path, device):
    data = read_3dgs_ply(path).copy()

    means = torch.tensor(np.stack([data["x"], data["y"], data["z"]], axis=-1), dtype=torch.float32, device=device)
    quats = torch.tensor(np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1), dtype=torch.float32, device=device)
    scales = torch.tensor(np.stack([data["scale_0"].copy(), data["scale_1"].copy(), data["scale_2"].copy()], axis=-1), dtype=torch.float32, device=device)
    opacities = torch.tensor(data["opacity"].copy(), dtype=torch.float32, device=device)

    f_dc = torch.tensor(np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1), dtype=torch.float32, device=device)
    rgbs = (f_dc.unsqueeze(1) * SH_C0 + 0.5).clip(0, 1).squeeze(1)

    return means, quats, scales, opacities, rgbs


def render_flythrough(ply_path, output_dir, n_frames=120):
    device = torch.device("cuda")
    from gsplat.rendering import rasterization

    means, quats, scales, opacities, rgbs = load_for_gsplat(ply_path, device)

    W, H = 800, 600
    fov_x = 60.0
    fx = W / (2 * np.tan(np.radians(fov_x / 2)))
    fy = fx

    center = np.array([0.0, 0.0, 0.0])
    look_at = np.array([0.0, 0.0, 0.0])
    radius = 2.5
    height_offset = 1.0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames

        cam_x = center[0] + radius * np.cos(angle)
        cam_y = center[1] + radius * np.sin(angle)
        cam_z = center[2] + height_offset

        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)

        # OpenCV convention: X-right, Y-down, Z-forward (into scene)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = -up_corrected  # Y is down
        c2w[:3, 2] = forward  # Z is forward (into scene)
        c2w[:3, 3] = cam_pos

        w2c = np.linalg.inv(c2w)

        viewmats = torch.from_numpy(w2c).float().to(device).unsqueeze(0)

        Ks = torch.tensor([[fx, 0.0, W / 2.0],
                           [0.0, fy, H / 2.0],
                           [0.0, 0.0, 1.0]], dtype=torch.float32, device=device).unsqueeze(0)

        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=rgbs,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
        )

        img = render_colors[0, :, :, :3].cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if i % 20 == 0:
            print(f"Frame {i}/{n_frames}")

    print("Creating video...")
    video_path = str(output_dir / "flythrough.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (W, H))
    for i in range(n_frames):
        img = cv2.imread(str(output_dir / f"{i:04d}.png"))
        writer.write(img)
    writer.release()
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    base = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data"
    render_flythrough(
        ply_path=f"{base}/fused_scene.ply",
        output_dir=f"{base}/flythrough_output",
        n_frames=120,
    )
