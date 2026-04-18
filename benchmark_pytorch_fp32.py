import os
import sys
import json
import time
import argparse
from statistics import mean, median

import torch

sys.path.insert(0, os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CLRNet PyTorch FP32 inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="benchmark_pytorch_fp32.json")
    return parser.parse_args()


def patch_mmcv_config_compat():
    """
    给 mmcv.Config / ConfigDict 动态补 haskey 方法，避免项目源码里调用 cfg.haskey(...) 报错。
    """
    from mmcv import Config
    from mmcv.utils import ConfigDict

    if not hasattr(Config, "haskey"):
        Config.haskey = lambda self, key: key in self

    if not hasattr(ConfigDict, "haskey"):
        ConfigDict.haskey = lambda self, key: key in self


def disable_pretrained(cfg):
    """
    关闭 benchmark 期间的 pretrained 下载，避免重复联网或覆盖你自己的 checkpoint。
    """
    # 顶层 pretrained
    if "pretrained" in cfg:
        cfg.pretrained = None

    # 常见 net.pretrained
    if hasattr(cfg, "net") and cfg.net is not None and "pretrained" in cfg.net:
        cfg.net.pretrained = None

    # 常见 backbone.pretrained
    if hasattr(cfg, "backbone") and cfg.backbone is not None and "pretrained" in cfg.backbone:
        cfg.backbone.pretrained = None

    if hasattr(cfg, "net") and cfg.net is not None:
        if "backbone" in cfg.net and cfg.net.backbone is not None and "pretrained" in cfg.net.backbone:
            cfg.net.backbone.pretrained = None


def load_model(config_path: str, checkpoint_path: str, device: str):
    patch_mmcv_config_compat()

    from mmcv import Config
    from clrnet.models.registry import build_net

    cfg = Config.fromfile(config_path)
    disable_pretrained(cfg)

    model = build_net(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint should be dict, got {type(ckpt)}")

    if "net" in ckpt:
        state_dict = ckpt["net"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model, cfg, missing, unexpected


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark is intended for GPU inference.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print("=" * 80)
    print("PyTorch FP32 Benchmark")
    print("=" * 80)
    print(f"config      : {args.config}")
    print(f"checkpoint  : {args.checkpoint}")
    print(f"device      : {args.device}")
    print(f"input shape : ({args.batch_size}, 3, {args.height}, {args.width})")
    print(f"warmup      : {args.warmup}")
    print(f"iters       : {args.iters}")
    print("=" * 80)

    model, cfg, missing, unexpected = load_model(args.config, args.checkpoint, args.device)

    print(f"load_state_dict missing keys   : {len(missing)}")
    print(f"load_state_dict unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("first missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("first unexpected keys:", unexpected[:10])

    x = torch.randn(
        args.batch_size, 3, args.height, args.width,
        device=device, dtype=torch.float32
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x)
        torch.cuda.synchronize(device)

    times_ms = []

    with torch.inference_mode():
        for _ in range(args.iters):
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            _ = model(x)

            torch.cuda.synchronize(device)
            end = time.perf_counter()

            times_ms.append((end - start) * 1000.0)

    avg_latency_ms = mean(times_ms)
    med_latency_ms = median(times_ms)
    min_latency_ms = min(times_ms)
    max_latency_ms = max(times_ms)
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    current_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

    result = {
        "mode": "PyTorch FP32",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "warmup_iters": args.warmup,
        "benchmark_iters": args.iters,
        "avg_latency_ms": round(avg_latency_ms, 4),
        "median_latency_ms": round(med_latency_ms, 4),
        "min_latency_ms": round(min_latency_ms, 4),
        "max_latency_ms": round(max_latency_ms, 4),
        "fps": round(fps, 4),
        "peak_gpu_mem_mb": round(peak_mem_mb, 4),
        "current_gpu_mem_mb": round(current_mem_mb, 4),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Benchmark Result")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 80)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
