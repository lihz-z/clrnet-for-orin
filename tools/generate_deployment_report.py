import argparse
import csv
import json
import os

import matplotlib.pyplot as plt


REMARKS = {
    "PyTorch FP32": "桌面/原生 PyTorch 基线",
    "ONNX Runtime FP32": "ONNX 图优化 + CUDA EP",
    "TensorRT FP16": "精度与速度平衡最优",
    "TensorRT INT8": "极限吞吐配置",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate table 5.6 and figure 5.5 assets")
    parser.add_argument("--results", nargs="+", required=True, help="Result JSON files")
    parser.add_argument("--table_csv", required=True, type=str)
    parser.add_argument("--table_md", required=True, type=str)
    parser.add_argument("--figure", required=True, type=str)
    return parser.parse_args()


def load_results(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "部署模式": data["mode"],
            "平均延迟（ms）": data["avg_latency_ms"],
            "FPS": data["fps"],
            "显存占用（MB）": data.get("peak_gpu_mem_mb", 0),
            "备注": REMARKS.get(data["mode"], ""),
        })
    order = ["PyTorch FP32", "ONNX Runtime FP32", "TensorRT FP16", "TensorRT INT8"]
    rows.sort(key=lambda item: order.index(item["部署模式"]) if item["部署模式"] in order else 99)
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = "| 部署模式 | 平均延迟（ms） | FPS | 显存占用（MB） | 备注 |\n"
    sep = "| --- | ---: | ---: | ---: | --- |\n"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['部署模式']} | {row['平均延迟（ms）']:.4f} | {row['FPS']:.4f} | "
            f"{row['显存占用（MB）']:.4f} | {row['备注']} |\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def plot(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    plt.figure(figsize=(8, 5))
    colors = {
        "PyTorch FP32": "#4c78a8",
        "ONNX Runtime FP32": "#f58518",
        "TensorRT FP16": "#54a24b",
        "TensorRT INT8": "#e45756",
    }

    for row in rows:
        mode = row["部署模式"]
        x = row["平均延迟（ms）"]
        y = row["FPS"]
        plt.scatter(x, y, s=120, color=colors.get(mode, "#333333"))
        plt.annotate(mode, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10)

    plt.xlabel("Average Latency (ms)")
    plt.ylabel("Throughput (FPS)")
    plt.title("Latency vs Throughput Across Deployment Modes")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def main():
    args = parse_args()
    rows = load_results(args.results)
    write_csv(rows, args.table_csv)
    write_md(rows, args.table_md)
    plot(rows, args.figure)
    print(f"Saved table csv: {args.table_csv}")
    print(f"Saved table md : {args.table_md}")
    print(f"Saved figure   : {args.figure}")


if __name__ == "__main__":
    main()
