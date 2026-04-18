import argparse
import os

import onnx
import torch
from onnxsim import simplify

from deploy_common import ExportWrapper, load_model, parse_input_shape


def parse_args():
    parser = argparse.ArgumentParser(description="Export CLRNet checkpoint to ONNX")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no_simplify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    model, _, missing, unexpected = load_model(args.config, args.checkpoint, args.device)
    wrapper = ExportWrapper(model).eval()

    input_shape = parse_input_shape(args.batch_size, args.height, args.width)
    dummy = torch.randn(*input_shape, device=args.device, dtype=torch.float32)

    print("=" * 80)
    print("Export ONNX")
    print("=" * 80)
    print(f"config      : {args.config}")
    print(f"checkpoint  : {args.checkpoint}")
    print(f"output      : {args.output}")
    print(f"input shape : {input_shape}")
    print(f"opset       : {args.opset}")
    print(f"missing keys: {len(missing)}")
    print(f"unexpected  : {len(unexpected)}")
    print("=" * 80)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["predictions"],
            opset_version=args.opset,
        )

    model_onnx = onnx.load(args.output)
    onnx.checker.check_model(model_onnx)

    if not args.no_simplify:
        simplified, ok = simplify(model_onnx)
        if not ok:
            raise RuntimeError("onnxsim failed to validate the simplified model")
        onnx.save(simplified, args.output)

    print(f"Saved ONNX to: {args.output}")


if __name__ == "__main__":
    main()
