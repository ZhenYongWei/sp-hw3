"""Clean mesh renderer using nvdiffrast (CUDA rasterizer, headless).

Produces properly-shaded renders of triangle meshes — far clearer than the
oversized-gaussian blob rendering used for fusion preview.
"""
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image


def _look_at(eye, target, up=(0, 0, 1)):
    """World->camera view matrix (OpenCV: +z forward, +x right, +y down)."""
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)      # forward
    r = np.cross(f, up); r /= np.linalg.norm(r)   # right
    u = np.cross(r, f)                             # up
    M = np.eye(4)
    M[0, :3] = r                                   # row 0 = right
    M[1, :3] = u                                   # row 1 = up
    M[2, :3] = f                                   # row 2 = forward
    M[:3, 3] = -M[:3, :3] @ eye                    # translation
    return M


def _perspective(fovy_deg, aspect, near=0.01, far=100.0):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    P = np.array([[f / aspect, 0, 0, 0],
                  [0, f, 0, 0],
                  [0, 0, far / (far - near), -(far * near) / (far - near)],
                  [0, 0, 1, 0]], float)
    return P


def render_mesh(vertices, faces, colors=None, normals=None,
                cam_eye=(2.5, 2.5, 1.5), cam_target=(0, 0, 0),
                fovy=35, resolution=768, light_dir=(0.4, 0.3, 0.8),
                bg_color=(0.92, 0.92, 0.94)):
    """Render a mesh from one viewpoint. vertices/faces: numpy. colors: (N,3) 0-1."""
    dev = torch.device("cuda")
    ctx = dr.RasterizeCudaContext()

    v = np.asarray(vertices, float)
    c = (v.min(0) + v.max(0)) / 2
    v = v - c
    # homogeneous clip-space positions
    V = _look_at(cam_eye, cam_target)
    P = _perspective(fovy, 1.0)
    mvp = (P @ V).astype(np.float32)
    pos = np.concatenate([v, np.ones((len(v), 1))], 1) @ mvp.T
    pos = torch.tensor(pos, device=dev, dtype=torch.float32).unsqueeze(0)  # (1,N,4)
    tri = torch.tensor(np.asarray(faces, np.int32), device=dev)

    H = W = resolution
    rast_out, _ = dr.rasterize(ctx, pos, tri, resolution=(H, W))
    # interpolated attributes
    if colors is None:
        colors = np.ones((len(v), 3)) * 0.7
    col_attr = torch.tensor(np.asarray(colors, np.float32), device=dev)
    col, _ = dr.interpolate(col_attr[None], rast_out, tri)
    # normals for shading
    if normals is None:
        vn = _compute_normals(v, np.asarray(faces))
    else:
        vn = np.asarray(normals, float)
    nrm_attr = torch.tensor(vn.astype(np.float32), device=dev)
    nrm, _ = dr.interpolate(nrm_attr[None], rast_out, tri)
    nrm = torch.nn.functional.normalize(nrm, dim=-1)
    # Lambertian
    ld = torch.tensor(np.asarray(light_dir, float), device=dev, dtype=torch.float32)
    ld = ld / ld.norm()
    diff = (nrm @ ld).clamp(min=0.15)  # ambient floor 0.15
    shaded = col * diff[..., None]
    # mask
    mask = (rast_out[..., 3] > 0).float()[..., None]
    bg = torch.tensor(bg_color, device=dev)
    img = shaded * mask + bg * (1 - mask)
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _compute_normals(v, f):
    vn = np.zeros_like(v)
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    fn = np.cross(a, b)
    np.add.at(vn, f[:, 0], fn); np.add.at(vn, f[:, 1], fn); np.add.at(vn, f[:, 2], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True); n[n == 0] = 1
    return vn / n


def render_mosaic(vertices, faces, colors=None, out_path=None,
                  resolution=600, cam_dist=None, label="object"):
    """Render 4 orbit views into a single mosaic image."""
    v = np.asarray(vertices, float)
    extent = v.ptp(0).max()
    if cam_dist is None:
        cam_dist = extent * 1.6
    frames = []
    for deg in [0, 90, 180, 270]:
        a = np.radians(deg)
        eye = [cam_dist * np.cos(a), cam_dist * np.sin(a) * 0.5, extent * 0.55]
        img = render_mesh(vertices, faces, colors, cam_eye=eye, resolution=resolution)
        frames.append(img)
    mosaic = np.concatenate(frames, axis=1)
    if out_path:
        Image.fromarray(mosaic).save(out_path)
        print(f"  saved {label}: {out_path}  (4 views {resolution}x{resolution})")
    return mosaic


if __name__ == "__main__":
    import trimesh, sys
    for name, path in [("objectC_stool", "task1_data/objectC_stool_output/objectC_stool_mesh.ply"),
                       ("objectB_car", "task1_data/objectB_output/objectB_mesh.obj")]:
        m = trimesh.load(path)
        colors = None
        if hasattr(m.visual, 'vertex_colors'):
            colors = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
            if colors.std() < 0.01:
                colors = None  # flat gray, skip
        render_mosaic(np.asarray(m.vertices), np.asarray(m.faces), colors,
                      out_path=f"task1_data/object_isolation/{name}_clean.png", label=name)
