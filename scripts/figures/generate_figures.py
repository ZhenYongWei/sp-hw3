import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def parse_log(log_path):
    epochs = []
    train_loss = []
    train_action = []
    train_kl = []
    eval_action = []
    cl_l1 = []
    cl_sr = []

    with open(log_path) as f:
        for line in f:
            m = re.search(
                r"Epoch (\d+): train_loss=(\S+) train_action=(\S+) "
                r"train_kl=(\S+) eval_action=(\S+)",
                line,
            )
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_action.append(float(m.group(3)))
                train_kl.append(float(m.group(4)))
                eval_action.append(float(m.group(5)))
                m2 = re.search(r"cl_l1=(\S+) cl_sr@0.05=(\S+) cl_sr@0.1=(\S+) cl_sr@0.2=(\S+) cl_sr@0.5=(\S+)", line)
                if m2:
                    cl_l1.append(float(m2.group(1)))
                    cl_sr.append(float(m2.group(4)))  # use cl_sr@0.2 as primary metric
                else:
                    cl_l1.append(None)
                    cl_sr.append(None)

    return {
        "epochs": np.array(epochs),
        "train_loss": np.array(train_loss),
        "train_action": np.array(train_action),
        "train_kl": np.array(train_kl),
        "eval_action": np.array(eval_action),
        "cl_l1": cl_l1,
        "cl_sr": cl_sr,
    }


def plot_loss_curves(envA_data, envABC_data, envA_klfix, envABC_klfix, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    colors_orig = {"A": "#4C72B0", "ABC": "#DD8452"}
    colors_klfix = {"A": "#55A868", "ABC": "#C44E52"}

    ax = axes[0, 0]
    ax.plot(envA_data["epochs"], envA_data["train_action"], label="A (KL=0)", color=colors_orig["A"], linestyle="--")
    ax.plot(envA_klfix["epochs"], envA_klfix["train_action"], label="A (klfix)", color=colors_klfix["A"])
    ax.plot(envABC_data["epochs"], envABC_data["train_action"], label="ABC (KL=0)", color=colors_orig["ABC"], linestyle="--")
    ax.plot(envABC_klfix["epochs"], envABC_klfix["train_action"], label="ABC (klfix)", color=colors_klfix["ABC"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Action Loss")
    ax.legend(fontsize=7)
    ax.set_title("Training Action Loss")

    ax = axes[0, 1]
    ax.plot(envA_data["epochs"], envA_data["eval_action"], label="A (KL=0)", color=colors_orig["A"], linestyle="--")
    ax.plot(envA_klfix["epochs"], envA_klfix["eval_action"], label="A (klfix)", color=colors_klfix["A"])
    ax.plot(envABC_data["epochs"], envABC_data["eval_action"], label="ABC (KL=0)", color=colors_orig["ABC"], linestyle="--")
    ax.plot(envABC_klfix["epochs"], envABC_klfix["eval_action"], label="ABC (klfix)", color=colors_klfix["ABC"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval Action Loss (D)")
    ax.legend(fontsize=7)
    ax.set_title("Zero-Shot Eval on Env D")

    ax = axes[0, 2]
    ax.plot(envA_data["epochs"], envA_data["train_kl"], label="A (KL=0)", color=colors_orig["A"], linestyle="--")
    ax.plot(envA_klfix["epochs"], envA_klfix["train_kl"], label="A (klfix)", color=colors_klfix["A"])
    ax.plot(envABC_data["epochs"], envABC_data["train_kl"], label="ABC (KL=0)", color=colors_orig["ABC"], linestyle="--")
    ax.plot(envABC_klfix["epochs"], envABC_klfix["train_kl"], label="ABC (klfix)", color=colors_klfix["ABC"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.legend(fontsize=7)
    ax.set_title("CVAE KL Divergence")

    ax = axes[1, 0]
    valid_l1_A = [(e, v) for e, v in zip(envA_klfix["epochs"], envA_klfix["cl_l1"]) if v is not None]
    valid_l1_ABC = [(e, v) for e, v in zip(envABC_klfix["epochs"], envABC_klfix["cl_l1"]) if v is not None]
    if valid_l1_A:
        ax.plot([x[0] for x in valid_l1_A], [x[1] for x in valid_l1_A], label="A (klfix)", color=colors_klfix["A"])
    if valid_l1_ABC:
        ax.plot([x[0] for x in valid_l1_ABC], [x[1] for x in valid_l1_ABC], label="ABC (klfix)", color=colors_klfix["ABC"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Closed-loop L1 Error")
    ax.legend(fontsize=7)
    ax.set_title("Closed-loop Rollout L1")

    ax = axes[1, 1]
    chunk_sizes = [10, 30, 50]
    train_losses = [0.075, 0.106, 0.118]
    eval_losses = [0.091, 0.128, 0.132]
    x = np.arange(len(chunk_sizes))
    width = 0.35
    ax.bar(x - width / 2, train_losses, width, label="Train Loss", color="#4C72B0")
    ax.bar(x + width / 2, eval_losses, width, label="Eval Loss (D)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in chunk_sizes])
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Chunk Size Robustness")

    ax = axes[1, 2]
    models = ["A\n(KL=0)", "ABC\n(KL=0)", "A\n(klfix)", "ABC\n(klfix)"]
    eval_vals = [envA_data["eval_action"][-1], envABC_data["eval_action"][-1],
                 envA_klfix["eval_action"][-1], envABC_klfix["eval_action"][-1]]
    cl_vals = [None, None, envA_klfix["cl_l1"][-1] if envA_klfix["cl_l1"][-1] is not None else 0,
               envABC_klfix["cl_l1"][-1] if envABC_klfix["cl_l1"][-1] is not None else 0]
    bar_colors = [colors_orig["A"], colors_orig["ABC"], colors_klfix["A"], colors_klfix["ABC"]]
    x = np.arange(len(models))
    ax.bar(x, eval_vals, color=bar_colors, alpha=0.7, label="Eval Loss (D)")
    if all(v is not None for v in cl_vals):
        ax.bar(x, cl_vals, color=bar_colors, alpha=0.4, hatch="//", label="Closed-loop L1")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error")
    ax.legend(fontsize=7)
    ax.set_title("Final Performance Summary")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/act_loss_curves.png")
    print(f"Saved {output_dir}/act_loss_curves.png")


def plot_chunk_comparison(output_dir):
    chunk_sizes = [10, 30, 50]
    train_losses = [0.075, 0.106, 0.118]
    eval_losses = [0.091, 0.128, 0.132]

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(chunk_sizes))
    width = 0.35
    ax.bar(x - width / 2, train_losses, width, label="Train Loss", color="#4C72B0")
    ax.bar(x + width / 2, eval_losses, width, label="Eval Loss (D)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"chunk={c}" for c in chunk_sizes])
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.set_title("Action Chunking Robustness (Env A → D)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/chunk_comparison.png")
    print(f"Saved {output_dir}/chunk_comparison.png")


def plot_mesh_comparison(output_dir):
    try:
        import trimesh
        meshB = trimesh.load("/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectB_output/objectB_mesh.obj")
        meshC = trimesh.load("/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectC_output/objectC_mesh.obj")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].triplot(meshB.vertices[:, 0], meshB.vertices[:, 1], meshB.faces, color="gray", linewidth=0.3)
        axes[0].set_title("Object B (DreamFusion)")
        axes[0].set_aspect("equal")

        axes[1].triplot(meshC.vertices[:, 0], meshC.vertices[:, 1], meshC.faces, color="gray", linewidth=0.3)
        axes[1].set_title("Object C (Zero123)")
        axes[1].set_aspect("equal")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/mesh_comparison.png")
        print(f"Saved {output_dir}/mesh_comparison.png")
    except Exception as e:
        print(f"Mesh plot failed: {e}")


def plot_rendered_frames(output_dir):
    import cv2
    fly_dir = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/flythrough_output"

    frames = [0, 30, 60, 90]
    fig, axes = plt.subplots(1, len(frames), figsize=(12, 3))

    for i, fidx in enumerate(frames):
        path = f"{fly_dir}/{fidx:04d}.png"
        img = cv2.imread(path)
        if img is not None:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Frame {fidx}")
        else:
            axes[i].set_title(f"Frame {fidx} (missing)")
        axes[i].axis("off")

    plt.suptitle("Fused Scene Flythrough", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/flythrough_frames.png")
    print(f"Saved {output_dir}/flythrough_frames.png")


def plot_object_renders(output_dir):
    import cv2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Object A - no separate renders, use a placeholder
    axes[0].text(0.5, 0.5, "Object A\n3DGS (24 views)\nSSIM=0.88\n30K Gaussians",
                 ha="center", va="center", fontsize=10, transform=axes[0].transAxes)
    axes[0].set_title("Object A: Multi-view 3DGS")
    axes[0].axis("off")

    # Object B renders
    objB_dir = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectB_output/dreamfusion-sd-v1-5"
    subdirs = sorted([d for d in os.listdir(objB_dir) if os.path.isdir(os.path.join(objB_dir, d))])
    if subdirs:
        save_dir = os.path.join(objB_dir, subdirs[0], "save")
        img_path = os.path.join(save_dir, "it5000-0.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[1].text(0.5, 0.5, "Object B\nDreamFusion (text)", ha="center", va="center", fontsize=10, transform=axes[1].transAxes)
    axes[1].set_title("Object B: DreamFusion")
    axes[1].axis("off")

    # Object C renders
    objC_dir = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectC_output/zero123-sai-custom"
    subdirs = sorted([d for d in os.listdir(objC_dir) if os.path.isdir(os.path.join(objC_dir, d))])
    if subdirs:
        save_dir = os.path.join(objC_dir, subdirs[0], "save", "it5000-test")
        img_path = os.path.join(save_dir, "0.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            axes[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[2].text(0.5, 0.5, "Object C\nZero123 (1 image)", ha="center", va="center", fontsize=10, transform=axes[2].transAxes)
    axes[2].set_title("Object C: Zero123")
    axes[2].axis("off")

    plt.suptitle("Three 3D Generation Pathways", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/object_comparison.png")
    print(f"Saved {output_dir}/object_comparison.png")


import os

if __name__ == "__main__":
    output_dir = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/report/figures"

    envA_log = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task2_output/envA/training_log.txt"
    envABC_log = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task2_output/envABC/training_log.txt"
    envA_klfix_log = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task2_output/envA_klfix_train.log"
    envABC_klfix_log = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task2_output/envABC_klfix_train.log"

    envA_data = parse_log(envA_log)
    envABC_data = parse_log(envABC_log)
    envA_klfix = parse_log(envA_klfix_log)
    envABC_klfix = parse_log(envABC_klfix_log)

    plot_loss_curves(envA_data, envABC_data, envA_klfix, envABC_klfix, output_dir)
    plot_chunk_comparison(output_dir)
    plot_mesh_comparison(output_dir)
    plot_rendered_frames(output_dir)
    plot_object_renders(output_dir)
