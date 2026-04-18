import argparse
import json
import os
import time
from statistics import mean, median

import tensorrt as trt
import torch

from deploy_common import current_gpu_mem_used_mb, parse_input_shape


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CLRNet TensorRT inference")
    parser.add_argument("--onnx", required=True, type=str)
    parser.add_argument("--engine", required=True, type=str)
    parser.add_argument("--mode", choices=["fp16", "int8"], required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--workspace_gb", type=int, default=2)
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def torch_dtype_from_trt(dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class RandomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_shape, cache_file, num_batches):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.cache_file = cache_file
        self.num_batches = num_batches
        self.batch_idx = 0
        self.buffer = torch.empty(self.input_shape, device="cuda", dtype=torch.float32)

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):
        if self.batch_idx >= self.num_batches:
            return None
        self.buffer.copy_(torch.rand_like(self.buffer))
        self.batch_idx += 1
        return [int(self.buffer.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(args, input_shape):
    os.makedirs(os.path.dirname(os.path.abspath(args.engine)), exist_ok=True)

    with trt.Builder(TRT_LOGGER) as builder:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(args.onnx, "rb") as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"Failed to parse ONNX:\n" + "\n".join(map(str, errors)))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))

        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

        if args.mode == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            cache_file = os.path.splitext(args.engine)[0] + ".calib"
            config.int8_calibrator = RandomEntropyCalibrator(
                input_shape=input_shape,
                cache_file=cache_file,
                num_batches=args.calib_batches,
            )

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT engine")

    with open(args.engine, "wb") as f:
        f.write(serialized)


def load_engine(engine_path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    return engine


def allocate_io(engine, context, input_shape):
    tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, input_shape)

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        dtype = torch_dtype_from_trt(engine.get_tensor_dtype(name))
        tensors[name] = torch.empty(shape, device="cuda", dtype=dtype)
        context.set_tensor_address(name, int(tensors[name].data_ptr()))

    return tensors


def main():
    args = parse_args()
    input_shape = parse_input_shape(args.batch_size, args.height, args.width)
    if not os.path.exists(args.engine):
        build_engine(args, input_shape)

    mem_before = current_gpu_mem_used_mb()
    engine = load_engine(args.engine)
    context = engine.create_execution_context()
    tensors = allocate_io(engine, context, input_shape)
    input_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    )
    tensors[input_name].normal_()
    stream = torch.cuda.current_stream()
    peak_mem_mb = current_gpu_mem_used_mb()

    for _ in range(args.warmup):
        ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT warmup inference failed")
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
        peak_mem_mb = max(peak_mem_mb, current_gpu_mem_used_mb())

    avg_latency_ms = mean(times_ms)
    result = {
        "mode": f"TensorRT {args.mode.upper()}",
        "onnx": args.onnx,
        "engine": args.engine,
        "input_shape": list(input_shape),
        "warmup_iters": args.warmup,
        "benchmark_iters": args.iters,
        "avg_latency_ms": round(avg_latency_ms, 4),
        "median_latency_ms": round(median(times_ms), 4),
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
