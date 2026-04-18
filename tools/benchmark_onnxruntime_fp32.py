import argparse
import json
import os
import time
from statistics import mean, median

import numpy as np
import onnxruntime as ort
import torch

from deploy_common import current_gpu_mem_used_mb, parse_input_shape


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CLRNet ONNX Runtime FP32 inference")
    parser.add_argument("--onnx", required=True, type=str)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", type=str, default="benchmark_onnxruntime_fp32.json")
    return parser.parse_args()


def main():
    args = parse_args()
    input_shape = parse_input_shape(args.batch_size, args.height, args.width)

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider is not available: {providers}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    mem_before = current_gpu_mem_used_mb()
    session = ort.InferenceSession(
        args.onnx,
        sess_options=so,
        providers=["CUDAExecutionProvider"],
    )
    mem_after_session = current_gpu_mem_used_mb()

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    x_np = np.random.randn(*input_shape).astype(np.float32)
    x_ort = ort.OrtValue.ortvalue_from_numpy(x_np, "cuda", 0)

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input(input_name, x_ort)
    for name in output_names:
        io_binding.bind_output(name, "cuda", 0)

    with torch.inference_mode():
        for _ in range(args.warmup):
            session.run_with_iobinding(io_binding)
        torch.cuda.synchronize()

    times_ms = []
    peak_mem_mb = mem_after_session
    with torch.inference_mode():
        for _ in range(args.iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            session.run_with_iobinding(io_binding)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
            peak_mem_mb = max(peak_mem_mb, current_gpu_mem_used_mb())

    avg_latency_ms = mean(times_ms)
    med_latency_ms = median(times_ms)
    result = {
        "mode": "ONNX Runtime FP32",
        "onnx": args.onnx,
        "providers": session.get_providers(),
        "input_shape": list(input_shape),
        "warmup_iters": args.warmup,
        "benchmark_iters": args.iters,
        "avg_latency_ms": round(avg_latency_ms, 4),
        "median_latency_ms": round(med_latency_ms, 4),
        "min_latency_ms": round(min(times_ms), 4),
        "max_latency_ms": round(max(times_ms), 4),
        "fps": round(1000.0 / avg_latency_ms, 4),
        "peak_gpu_mem_mb": round(peak_mem_mb - mem_before, 4),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
