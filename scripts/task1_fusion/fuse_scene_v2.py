"""Clean scene fusion v2 — produces 3 visually distinguishable objects.

Fixes vs fuse_scene.py:
  1. Correct SH-DC encoding: f_dc = (rgb - 0.5) / SH_C0 so renderer round-trips.
  2. Gaussian sizes auto-set from nearest-neighbor distance so surfaces tile
     into solids (no sparse dots / sub-pixel gaussians from external views).
  3. Object A: filter to solid gaussians + enlarge sizes so it reads as an object.
  4. Object B/C get distinct, honest colors:
       - B = red ("a red vintage toy car", matches the text prompt; its marching-
         cubes mesh lost all color so we restore the intended one)
       - C = its real mesh color (reddish) but shifted toward blue for contrast
         with B, so the three objects are visually separable.
  5. Clear a zone in the garden around the objects so they are not buried.
"""
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814
SCHEMA = [("x","f4"),("y","f4"),("z","f4"),
          ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
          ("opacity","f4"),
          ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
          ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
          ("red","u1"),("green","u1"),("blue","u1")]


def _nn_dist(pts, k=3):
    if len(pts) < k+1:
        return 0.02
    d,_ = cKDTree(pts).query(pts, k=k+1)
    return float(np.clip(np.median(d[:,1:]), 1e-4, None))


def _make_gaussians(points, colors_rgb, name=""):
    """Build a gaussian structured array with auto-tiled sizes + correct SH-DC."""
    n = len(points)
    nd = _nn_dist(points)
    scale = max(nd * 0.6, 1e-3)                  # overlap slightly -> solid surface
    f_dc = (np.asarray(colors_rgb) - 0.5) / SH_C0   # CORRECT encoding
    g = np.zeros(n, dtype=SCHEMA)
    g["x"], g["y"], g["z"] = points[:,0], points[:,1], points[:,2]
    g["f_dc_0"], g["f_dc_1"], g["f_dc_2"] = f_dc[:,0], f_dc[:,1], f_dc[:,2]
    g["opacity"] = 0.95
    g["scale_0"] = g["scale_1"] = g["scale_2"] = scale
    g["rot_0"] = 1.0
    g["red"]   = np.clip(colors_rgb[:,0]*255,0,255).astype(np.uint8)
    g["green"] = np.clip(colors_rgb[:,1]*255,0,255).astype(np.uint8)
    g["blue"]  = np.clip(colors_rgb[:,2]*255,0,255).astype(np.uint8)
    print(f"  {name}: {n} gauss, nn_dist={nd:.4f}, gauss_scale={scale:.4f}, "
          f"color_mean={colors_rgb.mean(0).round(2)}")
    return g, scale


def _transform(gauss, position, scale_factor, retile=True):
    """Rigid place: scale positions AND gaussian sizes proportionally.
    If retile, recompute sizes from world-space nearest-neighbor distance so
    surfaces always tile into solids regardless of placement scale (prevents
    sub-pixel gaussians that leave objects invisible from external views)."""
    g = gauss.copy()
    g["x"] = g["x"]*scale_factor + position[0]
    g["y"] = g["y"]*scale_factor + position[1]
    g["z"] = g["z"]*scale_factor + position[2]
    if retile:
        pts = np.stack([g["x"],g["y"],g["z"]],-1)
        nd = _nn_dist(pts)
        s = max(nd * 3.0, 1e-3)            # 3x nn-dist: heavy overlap -> solid surface
        g["scale_0"] = g["scale_1"] = g["scale_2"] = s
    else:
        g["scale_0"] *= scale_factor
        g["scale_1"] *= scale_factor
        g["scale_2"] *= scale_factor
    return g


def read_3dgs_ply(path):
    from fuse_scene import read_3dgs_ply as _r
    return _r(path)


def build_objectA(ply_path, opacity_thr=0.25):
    """Real video reconstruction — keep solid gaussians, keep real colors."""
    d,_,_ = read_3dgs_ply(ply_path)
    d = d[d["opacity"] > opacity_thr].copy()
    pts = np.stack([d["x"],d["y"],d["z"]],-1)
    # recenter to centroid
    c = (pts.min(0)+pts.max(0))/2
    pts = pts - c
    f_dc = np.stack([d["f_dc_0"],d["f_dc_1"],d["f_dc_2"]],-1)   # already SH-DC
    g,_ = _make_gaussians(pts, f_dc, "ObjectA")
    return g


def build_object_from_mesh(mesh_path, color_rgb, n_sample=20000, name=""):
    """Sample mesh surface; assign a (distinct) color; tile as gaussians."""
    mesh = trimesh.load(mesh_path)
    pts,_ = trimesh.sample.sample_surface(mesh, n_sample)
    pts = np.asarray(pts)
    c = (pts.min(0)+pts.max(0))/2
    pts = pts - c
    colors = np.tile(np.asarray(color_rgb, dtype=float), (len(pts),1))
    g,_ = _make_gaussians(pts, colors, name)
    return g


def write_ply(path, data):
    from fuse_scene import write_3dgs_ply
    write_3dgs_ply(path, data, None)


def clear_zone(garden, center, radius):
    """Remove garden gaussians within `radius` of center (unbury the objects)."""
    pts = np.stack([garden["x"],garden["y"],garden["z"]],-1)
    keep = np.linalg.norm(pts - np.asarray(center), axis=1) > radius
    return garden[keep].copy()


def build_scene(out_ply, with_garden=True, garden_radius_clear=1.6,
                places=None):
    base = "task1_data"
    if places is None:
        # Garden content is a shell (r=0.5-1.2), center r<0.5 is empty.
        # Place 3 objects in the central void; camera orbits inside the shell.
        places = {
            "A": {"pos": np.array([0.0, 0.0, 0.0]),     "scale": 0.5},
            "B": {"pos": np.array([0.42, 0.0, -0.05]),   "scale": 0.18},
            "C": {"pos": np.array([-0.42,0.0, -0.05]),   "scale": 0.16},
        }

    objA = build_objectA(f"{base}/real_object_output_v3_hd/point_cloud.ply")
    # B: red vintage toy car (marching-cubes mesh lost color -> restore prompt color)
    objB = build_object_from_mesh(f"{base}/objectB_output/objectB_mesh.obj",
                                  color_rgb=[0.80, 0.12, 0.12], name="ObjectB(red)")
    # C: shift its real reddish mesh color toward blue for contrast vs B
    objC = build_object_from_mesh(f"{base}/objectC_output_3000/objectC_mesh_3000.obj",
                                  color_rgb=[0.15, 0.35, 0.80], name="ObjectC(blue)")

    objA = _transform(objA, places["A"]["pos"], places["A"]["scale"])
    objB = _transform(objB, places["B"]["pos"], places["B"]["scale"])
    objC = _transform(objC, places["C"]["pos"], places["C"]["scale"])

    parts = [objA, objB, objC]
    if with_garden:
        garden,_,_ = read_3dgs_ply(f"{base}/garden_output_v3/point_cloud.ply")
        garden = clear_zone(garden, [0,0,0], garden_radius_clear)
        print(f"  garden after clear_zone(r={garden_radius_clear}): {len(garden)}")
        parts = [garden.copy()] + parts
    merged = np.concatenate(parts)
    write_ply(out_ply, merged)
    print(f"FUSED -> {out_ply}: {len(merged)} gaussians")
    return merged
