import ctypes
import os
from typing import Tuple

import torch


def patch_mmcv_config_compat():
    from mmcv import Config
    from mmcv.utils import ConfigDict

    if not hasattr(Config, "haskey"):
        Config.haskey = lambda self, key: key in self

    if not hasattr(ConfigDict, "haskey"):
        ConfigDict.haskey = lambda self, key: key in self


def disable_pretrained(cfg):
    if "pretrained" in cfg:
        cfg.pretrained = None

    if hasattr(cfg, "net") and cfg.net is not None and "pretrained" in cfg.net:
        cfg.net.pretrained = None

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


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def resolve_cudart():
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def current_gpu_mem_used_mb() -> float:
    cudart = resolve_cudart()
    if cudart is None:
        return 0.0

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    ret = cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    if ret != 0:
        return 0.0
    return (total.value - free.value) / (1024 ** 2)


def parse_input_shape(batch_size: int, height: int, width: int) -> Tuple[int, int, int, int]:
    return (batch_size, 3, height, width)
