"""Generate act_official_results.png from real training logs and eval results."""
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

BASE = Path("/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task2_output_official")
OUT = Path("/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/report/figures")


def parse_log(path):
    epochs, tl, el = [], [], []
    with open(path) as f:
        for line in f:
            m = re.search(r"Epoch (\d+): train_loss=(\S+) \S+ \S+ eval_action=(\S+)", line)
            if m:
                epochs.append(int(m.group(1)))
                tl.append(float(m.group(2)))
                el.append(float(m.group(3)))
    return np.array(epochs), np.array(tl), np.array(el)


ea_ep, ea_tl, ea_el = parse_log(BASE / "envA" / "training_log.txt")
ec_ep, ec_tl, ec_el = parse_log(BASE / "envABC" / "training_log.txt")

with open(BASE / "eval_results.json") as f:
    ev = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# (a) Training loss
ax = axes[0, 0]
ax.plot(ea_ep, ea_tl, label="Env A", color="#4C72B0")
ax.plot(ec_ep, ec_tl, label="Env A+B+C", color="#DD8452")
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("(a) Training Loss Convergence"); ax.legend(fontsize=8)
ax.set_ylim(0, 1)

# (b) Eval L1 on D
ax = axes[0, 1]
ax.plot(ea_ep, ea_el, label="Env A", color="#4C72B0")
ax.plot(ec_ep, ec_el, label="Env A+B+C", color="#DD8452")
ax.set_xlabel("Epoch"); ax.set_ylabel("Eval L1 on Env D")
ax.set_title("(b) Zero-Shot Eval on Env D"); ax.legend(fontsize=8)
ax.set_ylim(0.26, 0.30)

# (c) Per-dimension L1
ax = axes[1, 0]
dims = np.arange(1, 8)
ea_pd = ev["envA"]["per_dim_l1"]
ec_pd = ev["envABC"]["per_dim_l1"]
width = 0.35
ax.bar(dims - width / 2, ea_pd, width, label="Env A", color="#4C72B0")
ax.bar(dims + width / 2, ec_pd, width, label="Env A+B+C", color="#DD8452")
ax.axvline(6.5, color="gray", linestyle="--", alpha=0.5)
ax.text(6.5, 0.9, "gripper", ha="center", fontsize=8, color="gray")
ax.set_xticks(dims)
ax.set_xticklabels([f"d{i}" for i in dims])
ax.set_xlabel("Action Dimension"); ax.set_ylabel("L1 Error")
ax.set_title("(c) Per-Dimension L1 on Env D"); ax.legend(fontsize=8)

# (d) Success rates
ax = axes[1, 1]
thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
ea_sr = [ev["envA"]["success_rates"][str(t)] * 100 for t in thresholds]
ec_sr = [ev["envABC"]["success_rates"][str(t)] * 100 for t in thresholds]
x = np.arange(len(thresholds))
ax.bar(x - width / 2, ea_sr, width, label="Env A", color="#4C72B0")
ax.bar(x + width / 2, ec_sr, width, label="Env A+B+C", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels([f"@{t}" for t in thresholds])
ax.set_xlabel("L1 Threshold"); ax.set_ylabel("Success Rate (%)")
ax.set_title("(d) Success Rate on Env D"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "act_official_results.png")
print(f"Saved {OUT / 'act_official_results.png'}")
