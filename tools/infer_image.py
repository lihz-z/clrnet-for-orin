import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

from benchmark_tensorrt import allocate_io, load_engine
from deploy_common import load_model
from clrnet.utils.visualization import imshow_lanes


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image inference and save deployment visualization")
    parser.add_argument("--backend", choices=["pytorch", "onnxruntime", "tensorrt"], required=True)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def preprocess(img, cfg):
    cropped = img[cfg.cut_height:, :, :]
    resized = cv2.resize(cropped, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).copy()


def infer_pytorch(model, inp):
    x = torch.from_numpy(inp).cuda(non_blocking=True)
    with torch.inference_mode():
        return model(x)


def infer_onnxruntime(onnx_path, inp):
    session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    x_ort = ort.OrtValue.ortvalue_from_numpy(inp, "cuda", 0)
    io = session.io_binding()
    io.bind_ortvalue_input(input_name, x_ort)
    io.bind_output(output_name, "cuda", 0)
    session.run_with_iobinding(io)
    outputs = io.copy_outputs_to_cpu()
    return torch.from_numpy(outputs[0]).cuda()


def infer_tensorrt(engine_path, inp):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    tensors = allocate_io(engine, context, tuple(inp.shape))
    input_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    )
    tensors[input_name].copy_(torch.from_numpy(inp).to(device="cuda", dtype=tensors[input_name].dtype))
    ok = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT inference failed")
    torch.cuda.synchronize()
    output_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
    )
    return tensors[output_name]


def main():
    args = parse_args()
    model, cfg, missing, unexpected = load_model(args.config, args.checkpoint, args.device)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    inp = preprocess(img, cfg)
    if args.backend == "pytorch":
        pred = infer_pytorch(model, inp)
    elif args.backend == "onnxruntime":
        if not args.onnx:
            raise ValueError("--onnx is required for onnxruntime backend")
        pred = infer_onnxruntime(args.onnx, inp)
    else:
        if not args.engine:
            raise ValueError("--engine is required for tensorrt backend")
        pred = infer_tensorrt(args.engine, inp)

    lanes = model.heads.get_lanes(pred)[0]
    lanes = [lane.to_array(cfg) for lane in lanes]
    imshow_lanes(img.copy(), lanes, out_file=args.output)
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
