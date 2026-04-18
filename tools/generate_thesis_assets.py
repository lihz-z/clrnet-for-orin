import csv
import json
import math
import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_log import parse_log


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "paper_assets"
CH4 = OUT_ROOT / "chapter4"
CH5 = OUT_ROOT / "chapter5"
DEPLOY = ROOT / "deploy_artifacts"
QUAL = DEPLOY / "qualitative_outputs"


ABLATION_SUMMARY = [
    {
        "method": "CLRNet baseline",
        "display": "CLRNet Baseline",
        "branch": "None",
        "best_metric": 0.595567,
        "final_metric": 0.593964,
        "best_epoch": 20,
        "duration_hours": 4.58,
        "source": "readme_summary",
        "work_dir": "work_dirs/clr/r34_rainlane_baseline",
    },
    {
        "method": "CLRNet + Frequency Gate",
        "display": "Frequency Gate",
        "branch": "FG only",
        "best_metric": 0.596509,
        "final_metric": 0.592280,
        "best_epoch": 20,
        "duration_hours": 4.6214,
        "source": "readme_summary",
        "work_dir": "work_dirs/clr/r34_rainlane_fg_only",
    },
    {
        "method": "CLRNet + Directional Attention",
        "display": "Directional Attention",
        "branch": "DA only",
        "best_metric": 0.598182,
        "final_metric": 0.596832,
        "best_epoch": 22,
        "duration_hours": 4.3864,
        "source": "raw_log+readme_summary",
        "work_dir": "work_dirs/clr/r34_rainlane_da_only",
    },
    {
        "method": "CLRNet + FGM",
        "display": "FGM",
        "branch": "FG + DA",
        "best_metric": 0.597021,
        "final_metric": 0.592778,
        "best_epoch": 22,
        "duration_hours": 4.6636,
        "source": "readme_summary",
        "work_dir": "work_dirs/clr/r34_rainlane_new",
    },
]


PRECISION_RECALL_MAP = {
    "CLRNet Baseline": {"precision": 59.02, "recall": 60.10},
    "Frequency Gate": {"precision": 59.43, "recall": 59.88},
    "Directional Attention": {"precision": 59.57, "recall": 60.06},
    "FGM": {"precision": 59.48, "recall": 59.93},
}


def ensure_dirs():
    CH4.mkdir(parents=True, exist_ok=True)
    CH5.mkdir(parents=True, exist_ok=True)


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows, path, title=None):
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = []
    if title:
        lines.append(f"# {title}\n\n")
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_da_log_summary():
    return parse_log(str(ROOT / "da_only.log"))


def get_stage2_status():
    text = (ROOT / "fgm_stage2.log").read_text(encoding="utf-8", errors="ignore")
    crashed = "FileNotFoundError" in text
    first_error = None
    if crashed:
        match = re.search(r"FileNotFoundError: (.+)", text)
        first_error = match.group(1) if match else "FileNotFoundError"
    return {
        "run_name": "CLRNet-FGM stage2",
        "work_dir": "work_dirs/clr/r34_rainlane_fgm_stage2",
        "epochs_planned": 10,
        "epochs_completed_with_metric": 0,
        "status": "interrupted" if crashed else "unknown",
        "issue": first_error or "N/A",
        "note": "Stage-2 finetuning log exists, but validation crashed before a stable metric was written.",
    }


def make_ablation_tables():
    rows = []
    extended_rows = []
    for item in ABLATION_SUMMARY:
        display = item["display"]
        f1 = item["best_metric"] * 100
        precision = PRECISION_RECALL_MAP[display]["precision"]
        recall = PRECISION_RECALL_MAP[display]["recall"]
        rows.append({
            "Method": item["method"],
            "F1 (%)": round(f1, 2),
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "Best Epoch": item["best_epoch"],
            "Source": item["source"],
        })
        extended_rows.append({
            "Method": item["method"],
            "Best F1 (%)": round(item["best_metric"] * 100, 2),
            "Final F1 (%)": round(item["final_metric"] * 100, 2),
            "Final-Best (pp)": round((item["final_metric"] - item["best_metric"]) * 100, 3),
            "Duration (h)": item["duration_hours"],
            "Work Dir": item["work_dir"],
            "Source": item["source"],
        })

    write_csv(rows, CH4 / "table_4_1_ablation_real.csv")
    write_md(rows, CH4 / "table_4_1_ablation_real.md", title="Table 4.1 Real Ablation Results")
    write_csv(extended_rows, CH4 / "table_4_1_ablation_extended.csv")
    write_md(extended_rows, CH4 / "table_4_1_ablation_extended.md", title="Extended Ablation Summary")


def plot_ablation_bars():
    labels = [item["display"] for item in ABLATION_SUMMARY]
    f1 = [item["best_metric"] * 100 for item in ABLATION_SUMMARY]
    precision = [PRECISION_RECALL_MAP[item["display"]]["precision"] for item in ABLATION_SUMMARY]
    recall = [PRECISION_RECALL_MAP[item["display"]]["recall"] for item in ABLATION_SUMMARY]

    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width, f1, width=width, label="F1", color="#355070")
    plt.bar(x, precision, width=width, label="Precision", color="#6d597a")
    plt.bar(x + width, recall, width=width, label="Recall", color="#b56576")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Metric (%)")
    plt.title("Figure 4.3 Real Ablation Comparison of FGM Branch Variants")
    plt.ylim(58.8, 60.4)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CH4 / "figure_4_3_ablation_metrics.png", dpi=220)
    plt.close()


def plot_ablation_stability():
    labels = [item["display"] for item in ABLATION_SUMMARY]
    best = [item["best_metric"] * 100 for item in ABLATION_SUMMARY]
    final = [item["final_metric"] * 100 for item in ABLATION_SUMMARY]
    duration = [item["duration_hours"] for item in ABLATION_SUMMARY]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax1.plot(x, best, marker="o", linewidth=2, color="#1d3557", label="Best F1")
    ax1.plot(x, final, marker="s", linewidth=2, color="#e76f51", label="Final F1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("F1 (%)")
    ax1.set_ylim(59.1, 60.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(x, duration, alpha=0.18, color="#2a9d8f", label="Duration (h)")
    ax2.set_ylabel("Training Time (h)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    plt.title("Ablation Stability and Training Cost Across Module Variants")
    plt.tight_layout()
    plt.savefig(CH4 / "figure_4_x_ablation_stability.png", dpi=220)
    plt.close()


def plot_deployed_model_training_curves():
    summary = get_da_log_summary()
    metrics = summary["metrics"]
    train_records = summary["train_records"]

    # Keep the last training snapshot before each validation metric.
    per_epoch_loss = []
    for metric_record in metrics:
        snapshot = metric_record.get("train_snapshot") or {}
        per_epoch_loss.append({
            "epoch": metric_record["epoch_human"],
            "metric": metric_record["metric"] * 100,
            "loss": snapshot.get("loss"),
            "cls_loss": snapshot.get("cls_loss"),
            "iou_loss": snapshot.get("iou_loss"),
        })

    epochs = [item["epoch"] for item in per_epoch_loss]
    f1 = [item["metric"] for item in per_epoch_loss]
    total_loss = [item["loss"] for item in per_epoch_loss]
    cls_loss = [item["cls_loss"] for item in per_epoch_loss]
    iou_loss = [item["iou_loss"] for item in per_epoch_loss]
    best_epoch = summary["best_metric"]["epoch_human"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, total_loss, marker="o", color="#264653", label="Train Loss")
    axes[0].plot(epochs, cls_loss, marker="s", color="#e76f51", label="Cls Loss")
    axes[0].plot(epochs, iou_loss, marker="^", color="#2a9d8f", label="IoU Loss")
    axes[0].axvline(best_epoch, linestyle="--", color="#6d597a", label=f"Best Epoch = {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Curve of Deployed DA-only Model")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, f1, marker="o", color="#1d3557", label="Validation F1")
    axes[1].axvline(best_epoch, linestyle="--", color="#6d597a", label=f"Best Epoch = {best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 (%)")
    axes[1].set_title("Validation F1 Curve of Deployed DA-only Model")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(CH4 / "figure_5_4_deployed_model_training_curves.png", dpi=220)
    plt.close()

    rows = []
    for item in per_epoch_loss:
        rows.append({
            "Epoch": item["epoch"],
            "Validation F1 (%)": round(item["metric"], 4),
            "Train Loss": round(item["loss"], 4) if item["loss"] is not None else "N/A",
            "Cls Loss": round(item["cls_loss"], 4) if item["cls_loss"] is not None else "N/A",
            "IoU Loss": round(item["iou_loss"], 4) if item["iou_loss"] is not None else "N/A",
        })
    write_csv(rows, CH4 / "table_4_x_deployed_model_curve_data.csv")
    write_md(rows, CH4 / "table_4_x_deployed_model_curve_data.md", title="Deployed Model Epoch Curves")


def make_stage2_and_param_tables():
    stage2_status = get_stage2_status()
    one_stage = next(item for item in ABLATION_SUMMARY if item["display"] == "FGM")
    rows = [
        {
            "Run": "FGM one-stage",
            "Work Dir": one_stage["work_dir"],
            "Best F1 (%)": round(one_stage["best_metric"] * 100, 2),
            "Best Epoch": one_stage["best_epoch"],
            "Duration (h)": one_stage["duration_hours"],
            "Status": "completed",
            "Source": one_stage["source"],
        },
        {
            "Run": stage2_status["run_name"],
            "Work Dir": stage2_status["work_dir"],
            "Best F1 (%)": "N/A",
            "Best Epoch": "N/A",
            "Duration (h)": "N/A",
            "Status": stage2_status["status"],
            "Source": "raw_log",
        },
    ]
    write_csv(rows, CH4 / "table_4_x_two_stage_status.csv")
    write_md(rows, CH4 / "table_4_x_two_stage_status.md", title="One-stage vs Stage-2 Status")

    issue_rows = [
        {
            "Run": stage2_status["run_name"],
            "Status": stage2_status["status"],
            "Issue": stage2_status["issue"],
            "Note": stage2_status["note"],
        }
    ]
    write_csv(issue_rows, CH4 / "table_4_x_stage2_issue_log.csv")
    write_md(issue_rows, CH4 / "table_4_x_stage2_issue_log.md", title="Stage-2 Current Status")

    param_rows = [
        {
            "Config": "FG only",
            "enable_freq": True,
            "enable_dir": False,
            "freq_gate_threshold": 0.22,
            "direction_bins": 4,
            "apply_levels": "[1, 2]",
            "res_scale_init": 0.08,
        },
        {
            "Config": "DA only",
            "enable_freq": False,
            "enable_dir": True,
            "freq_gate_threshold": 0.22,
            "direction_bins": 4,
            "apply_levels": "[1, 2]",
            "res_scale_init": 0.08,
        },
        {
            "Config": "FGM one-stage",
            "enable_freq": True,
            "enable_dir": True,
            "freq_gate_threshold": 0.22,
            "direction_bins": 4,
            "apply_levels": "[1, 2]",
            "res_scale_init": 0.08,
        },
        {
            "Config": "FGM stage2 target",
            "enable_freq": True,
            "enable_dir": True,
            "freq_gate_threshold": 0.22,
            "direction_bins": 4,
            "apply_levels": "[1, 2]",
            "res_scale_init": 0.05,
        },
    ]
    write_csv(param_rows, CH4 / "table_4_x_parameter_settings.csv")
    write_md(param_rows, CH4 / "table_4_x_parameter_settings.md", title="Current Parameter Settings")


def make_environment_table():
    rows = [
        {"Item": "Target Platform", "Value": "Jetson Orin", "Source": "nvidia-smi"},
        {"Item": "Driver Version", "Value": "540.4.0", "Source": "nvidia-smi"},
        {"Item": "JetPack / L4T", "Value": "R36.4.3", "Source": "/etc/nv_tegra_release"},
        {"Item": "TensorRT", "Value": "10.3.0", "Source": "python tensorrt"},
        {"Item": "PyTorch", "Value": "2.5.0a0+872d972e41.nv24.08", "Source": ".venv"},
        {"Item": "ONNX Runtime GPU", "Value": "1.23.0", "Source": ".venv"},
        {"Item": "Input Resolution", "Value": "320 x 800", "Source": "deployment config"},
        {"Item": "Batch Size", "Value": "1", "Source": "benchmark setup"},
    ]
    write_csv(rows, CH5 / "table_5_x_environment.csv")
    write_md(rows, CH5 / "table_5_x_environment.md", title="Deployment Environment")


def make_artifact_table():
    files = [
        ROOT / "check_point" / "21.pth",
        DEPLOY / "clrnet_da_only.onnx",
        DEPLOY / "clrnet_da_only_fp16.engine",
        DEPLOY / "clrnet_da_only_int8.engine",
        DEPLOY / "clrnet_da_only_int8.calib",
    ]
    labels = {
        "21.pth": "Trained PyTorch checkpoint",
        "clrnet_da_only.onnx": "Exported ONNX graph",
        "clrnet_da_only_fp16.engine": "TensorRT FP16 engine",
        "clrnet_da_only_int8.engine": "TensorRT INT8 engine",
        "clrnet_da_only_int8.calib": "INT8 calibration cache",
    }
    rows = []
    for file in files:
        size_mb = file.stat().st_size / (1024 ** 2)
        rows.append({
            "Artifact": file.name,
            "Type": labels[file.name],
            "Size (MB)": round(size_mb, 3),
            "Path": str(file.relative_to(ROOT)),
        })
    write_csv(rows, CH5 / "table_5_x_artifacts.csv")
    write_md(rows, CH5 / "table_5_x_artifacts.md", title="Deployment Artifacts")


def make_speedup_tables():
    modes = [
        read_json(DEPLOY / "benchmark_pytorch_fp32.json"),
        read_json(DEPLOY / "benchmark_onnxruntime_fp32.json"),
        read_json(DEPLOY / "benchmark_tensorrt_fp16.json"),
        read_json(DEPLOY / "benchmark_tensorrt_int8.json"),
    ]
    base = modes[0]
    rows = []
    for mode in modes:
        rows.append({
            "Mode": mode["mode"],
            "Latency (ms)": mode["avg_latency_ms"],
            "FPS": mode["fps"],
            "Peak GPU Mem (MB)": mode["peak_gpu_mem_mb"],
            "Speedup vs PyTorch": round(base["avg_latency_ms"] / mode["avg_latency_ms"], 3),
            "FPS Gain vs PyTorch": round(mode["fps"] / base["fps"], 3),
            "Latency Reduction (%)": round((1 - mode["avg_latency_ms"] / base["avg_latency_ms"]) * 100, 2),
        })
    write_csv(rows, CH5 / "table_5_x_speedup.csv")
    write_md(rows, CH5 / "table_5_x_speedup.md", title="Deployment Speedup Summary")

    labels = [row["Mode"] for row in rows]
    mins = [read_json(DEPLOY / f"benchmark_{'pytorch_fp32' if 'PyTorch' in row['Mode'] else 'onnxruntime_fp32' if 'ONNX Runtime' in row['Mode'] else 'tensorrt_fp16' if 'FP16' in row['Mode'] else 'tensorrt_int8'}.json")["min_latency_ms"] for row in rows]
    meds = [read_json(DEPLOY / f"benchmark_{'pytorch_fp32' if 'PyTorch' in row['Mode'] else 'onnxruntime_fp32' if 'ONNX Runtime' in row['Mode'] else 'tensorrt_fp16' if 'FP16' in row['Mode'] else 'tensorrt_int8'}.json")["median_latency_ms"] for row in rows]
    maxs = [read_json(DEPLOY / f"benchmark_{'pytorch_fp32' if 'PyTorch' in row['Mode'] else 'onnxruntime_fp32' if 'ONNX Runtime' in row['Mode'] else 'tensorrt_fp16' if 'FP16' in row['Mode'] else 'tensorrt_int8'}.json")["max_latency_ms"] for row in rows]

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.plot(x, mins, marker="o", label="Min", color="#264653")
    plt.plot(x, meds, marker="s", label="Median", color="#e76f51")
    plt.plot(x, maxs, marker="^", label="Max", color="#2a9d8f")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Latency (ms)")
    plt.title("Latency Distribution Across Deployment Backends")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CH5 / "figure_5_x_latency_distribution.png", dpi=220)
    plt.close()


def make_deployment_flow_figure():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    boxes = [
        ("21.pth\nTrained checkpoint", 0.07, "#355070"),
        ("ONNX export\nonnx + onnxsim", 0.29, "#6d597a"),
        ("TensorRT build\nFP16 / INT8", 0.51, "#b56576"),
        ("Orin inference\nORT / TRT runtime", 0.73, "#2a9d8f"),
        ("Qualitative & latency\nfigures / tables", 0.93, "#e76f51"),
    ]
    for text, x, color in boxes:
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            bbox=dict(boxstyle="round,pad=0.55", fc=color, ec="none"),
            transform=ax.transAxes,
        )
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.08
        x1 = boxes[i + 1][1] - 0.08
        ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5), xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="#444"))
    plt.tight_layout()
    plt.savefig(CH5 / "figure_5_1_deployment_flow_real.png", dpi=220)
    plt.close()


def overlay_mask(original, rendered):
    diff = cv2.absdiff(rendered, original)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return (gray > 20).astype(np.uint8)


def backend_consistency():
    images = [
        "o1_001166.00_aug4",
        "o2_06_12_aug1",
        "o1_001186.00_aug3",
    ]
    backends = [
        ("pytorch_fp32", "PyTorch FP32"),
        ("onnxruntime_fp32", "ONNX Runtime FP32"),
        ("tensorrt_fp16", "TensorRT FP16"),
        ("tensorrt_int8", "TensorRT INT8"),
    ]
    rows = []
    for name in images:
        original = cv2.imread(str(ROOT.parent / f"{name}.jpg"))
        ref = cv2.imread(str(QUAL / f"{name}_pytorch_fp32.jpg"))
        ref_mask = overlay_mask(original, ref)
        ref_pixels = int(ref_mask.sum())
        for suffix, display in backends:
            img = cv2.imread(str(QUAL / f"{name}_{suffix}.jpg"))
            mask = overlay_mask(original, img)
            inter = int((mask & ref_mask).sum())
            union = int((mask | ref_mask).sum())
            iou = inter / union if union else 1.0
            pixel_diff = int(np.abs(mask.astype(np.int16) - ref_mask.astype(np.int16)).sum())
            rows.append({
                "Image": name,
                "Backend": display,
                "Overlay Pixels": int(mask.sum()),
                "PyTorch Ref Pixels": ref_pixels,
                "Overlay IoU vs PyTorch": round(iou, 4),
                "Different Pixels vs PyTorch": pixel_diff,
            })
    write_csv(rows, CH5 / "table_5_x_backend_consistency.csv")
    write_md(rows, CH5 / "table_5_x_backend_consistency.md", title="Backend Consistency on Three Qualitative Samples")


def load_font(size):
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def make_qualitative_montage():
    rows = [
        "o1_001166.00_aug4",
        "o2_06_12_aug1",
        "o1_001186.00_aug3",
    ]
    cols = [
        ("pytorch_fp32", "PyTorch FP32"),
        ("onnxruntime_fp32", "ONNX Runtime FP32"),
        ("tensorrt_fp16", "TensorRT FP16"),
        ("tensorrt_int8", "TensorRT INT8"),
    ]

    sample = Image.open(QUAL / f"{rows[0]}_{cols[0][0]}.jpg")
    w, h = sample.size
    margin = 24
    header_h = 56
    row_label_w = 210
    canvas = Image.new("RGB", (row_label_w + len(cols) * (w + margin) + margin,
                               header_h + len(rows) * (h + margin) + margin), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26)
    header_font = load_font(20)
    row_font = load_font(18)

    draw.text((margin, 12), "Figure 5.6 Qualitative Comparison Across Four Deployment Backends", fill=(20, 20, 20), font=title_font)

    for col_idx, (_, title) in enumerate(cols):
        x = row_label_w + margin + col_idx * (w + margin)
        draw.text((x + 8, header_h - 30), title, fill=(30, 30, 30), font=header_font)

    for row_idx, row_name in enumerate(rows):
        y = header_h + margin + row_idx * (h + margin)
        draw.text((margin, y + h // 2 - 10), row_name, fill=(30, 30, 30), font=row_font)
        for col_idx, (suffix, _) in enumerate(cols):
            x = row_label_w + margin + col_idx * (w + margin)
            img = Image.open(QUAL / f"{row_name}_{suffix}.jpg")
            canvas.paste(img, (x, y))

    out_path = CH5 / "figure_5_6_backend_qualitative_grid.png"
    canvas.save(out_path)


def make_index():
    rows = [
        {"Section": "Chapter 4", "Asset": "table_4_1_ablation_real.md", "Description": "Real ablation table based on experiment summaries"},
        {"Section": "Chapter 4", "Asset": "figure_4_3_ablation_metrics.png", "Description": "Ablation bar chart for F1 / Precision / Recall"},
        {"Section": "Chapter 4", "Asset": "figure_5_4_deployed_model_training_curves.png", "Description": "Real training curves of deployed DA-only model"},
        {"Section": "Chapter 4", "Asset": "table_4_x_two_stage_status.md", "Description": "One-stage vs stage-2 current status"},
        {"Section": "Chapter 5", "Asset": "table_5_x_environment.md", "Description": "Actual Orin deployment environment"},
        {"Section": "Chapter 5", "Asset": "table_5_x_artifacts.md", "Description": "Model conversion artifacts and sizes"},
        {"Section": "Chapter 5", "Asset": "table_5_x_speedup.md", "Description": "Latency / FPS / speedup summary"},
        {"Section": "Chapter 5", "Asset": "table_5_x_backend_consistency.md", "Description": "Consistency check across four backends"},
        {"Section": "Chapter 5", "Asset": "figure_5_1_deployment_flow_real.png", "Description": "Actual deployment flow figure"},
        {"Section": "Chapter 5", "Asset": "figure_5_6_backend_qualitative_grid.png", "Description": "12-image qualitative montage"},
    ]
    write_md(rows, OUT_ROOT / "index.md", title="Generated Thesis Assets")


def main():
    ensure_dirs()
    make_ablation_tables()
    plot_ablation_bars()
    plot_ablation_stability()
    plot_deployed_model_training_curves()
    make_stage2_and_param_tables()
    make_environment_table()
    make_artifact_table()
    make_speedup_tables()
    make_deployment_flow_figure()
    backend_consistency()
    make_qualitative_montage()
    make_index()
    print(f"Saved thesis assets under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
