# Backend Consistency on Three Qualitative Samples

| Image | Backend | Overlay Pixels | PyTorch Ref Pixels | Overlay IoU vs PyTorch | Different Pixels vs PyTorch |
| --- | --- | --- | --- | --- | --- |
| o1_001166.00_aug4 | PyTorch FP32 | 12812 | 12812 | 1.0000 | 0 |
| o1_001166.00_aug4 | ONNX Runtime FP32 | 12812 | 12812 | 1.0000 | 0 |
| o1_001166.00_aug4 | TensorRT FP16 | 12823 | 12812 | 0.9718 | 367 |
| o1_001166.00_aug4 | TensorRT INT8 | 12901 | 12812 | 0.8279 | 2421 |
| o2_06_12_aug1 | PyTorch FP32 | 10087 | 10087 | 1.0000 | 0 |
| o2_06_12_aug1 | ONNX Runtime FP32 | 10087 | 10087 | 1.0000 | 0 |
| o2_06_12_aug1 | TensorRT FP16 | 10088 | 10087 | 0.9610 | 401 |
| o2_06_12_aug1 | TensorRT INT8 | 9157 | 10087 | 0.8532 | 1524 |
| o1_001186.00_aug3 | PyTorch FP32 | 9190 | 9190 | 1.0000 | 0 |
| o1_001186.00_aug3 | ONNX Runtime FP32 | 9190 | 9190 | 1.0000 | 0 |
| o1_001186.00_aug3 | TensorRT FP16 | 9195 | 9190 | 0.9678 | 301 |
| o1_001186.00_aug3 | TensorRT INT8 | 8888 | 9190 | 0.8562 | 1400 |
