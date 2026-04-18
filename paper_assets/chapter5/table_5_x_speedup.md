# Deployment Speedup Summary

| Mode | Latency (ms) | FPS | Peak GPU Mem (MB) | Speedup vs PyTorch | FPS Gain vs PyTorch | Latency Reduction (%) |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 39.3418 | 25.4183 | 127.6870 | 1.0000 | 1.0000 | 0.0000 |
| ONNX Runtime FP32 | 29.8266 | 33.5271 | 499.7422 | 1.3190 | 1.3190 | 24.1900 |
| TensorRT FP16 | 7.5435 | 132.5641 | 29.3867 | 5.2150 | 5.2150 | 80.8300 |
| TensorRT INT8 | 5.9027 | 169.4154 | 10.8242 | 6.6650 | 6.6650 | 85.0000 |
