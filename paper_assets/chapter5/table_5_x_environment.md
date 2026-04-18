# Deployment Environment

| Item | Value | Source |
| --- | --- | --- |
| Target Platform | Jetson Orin | nvidia-smi |
| Driver Version | 540.4.0 | nvidia-smi |
| JetPack / L4T | R36.4.3 | /etc/nv_tegra_release |
| TensorRT | 10.3.0 | python tensorrt |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 | .venv |
| ONNX Runtime GPU | 1.23.0 | .venv |
| Input Resolution | 320 x 800 | deployment config |
| Batch Size | 1 | benchmark setup |
