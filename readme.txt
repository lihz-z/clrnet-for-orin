重新开机只需要执行下列步骤:
bash
运行
# 1. 激活你之前创建的conda环境（重启后默认会回到base环境，切回来即可）
conda activate clrnet

# 2. 进入文件目录
cd autodl-tmp/clrnet

# 3. （仅当你需要重新编译算子时才需要，单纯运行训练可以跳过这步）
# export TORCH_CUDA_ARCH_LIST="8.9"
做完之后，就可以直接启动训练了，和你关机前的状态完全一样：
bash
运行训练
python main.py configs/clrnet/clr_resnet34_rainlane.py --gpus 0 

测试
python main.py configs/clrnet/clr_resnet34_rainlane.py --test --load_from work_dirs/clr/r34_rainlane/20260327_195343_lr_6e-04_b_24/ckpt/0.pth --gpus 0

可视化输出
python main.py configs/clrnet/clr_resnet34_rainlane.py --test --load_from work_dirs/clr/r34_rainlane/20260327_195343_lr_6e-04_b_24/ckpt/0.pth --gpus 0 --view

#现在原始训练的模型输出
autodl-tmp/clrnet/work_dirs/clr/r34_rainlane/20260327_195343_lr_6e-04_b_24

#车道线图片
autodl-tmp/clrnet/work_dirs/clr/r34_rainlane/20260402_145343_lr_6e-04_b_24/visualization

#删除缓存文件（如果之前有缓存，可能跳过 load_annotations，导致 img_name 来自缓存）
rm -f cache/culane_test.pkl

#加载新的数据集时清除缓存
rm -f cache/culane_train.pkl



4.18，1：52
1.现在已经在增强数据集上跑完了消融实验：-----数据在各自最新的work_dirs最新文件
clr_resnet34_rainlane_baseline_matched.py
clr_resnet34_rainlane_da_only.py
clr_resnet34_rainlane_fg_only.py
clr_resnet34_culane.py（原数据集但是效果好）

2.想调优再跑一轮，两阶段训练（用最好的pth）
clr_resnet34_culane.py
clr_resnet34_rainlane_fgm.py
clr_resnet34_rainlane_da_only.py
clr_resnet34_rainlane_fg_only.py

中的apply_levels=[1, 2]---->apply_levels=[2]（只在最高层做抗干扰）


python main.py configs/clrnet/clr_resnet34_rainlane.py --gpus 0 ; \
python main.py configs/clrnet/clr_resnet34_rainlane_fgm.py --gpus 0 ; \
python main.py configs/clrnet/clr_resnet34_rainlane_da_only.py --gpus 0 ; \
python main.py configs/clrnet/clr_resnet34_rainlane_fg_only.py --gpus 0

4.18
最终实验数据
(clrnet) root@autodl-container-578e4b9b8e-6edc601e:~/autodl-tmp/clrnet# python tools/analyze_log.py work_dirs/clr/r34_rainlane_baseline/20260416_232618_lr_6e-04_b_24/log.txt
========================================================================================
log: /root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_baseline/20260416_232618_lr_6e-04_b_24/log.txt
config:
  work_dirs=work_dirs/clr/r34_rainlane_baseline
  epochs=25, batch_size=24, base_lr=0.0006
time:
  start=2026-04-16 23:26:18
  end=2026-04-17 04:01:06
  duration_hours=4.58
metrics:
  eval_count=25, mean=0.570397, std=0.026427
  best=epoch=20 (internal=19), metric=0.595567, step=14040, lr=5.7e-05, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_baseline/20260416_232618_lr_6e-04_b_24/ckpt/19.pth
  final=epoch=25 (internal=24), metric=0.593964, step=17550, lr=0, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_baseline/20260416_232618_lr_6e-04_b_24/ckpt/24.pth
  final_minus_best=-0.001603
  trend=tail stays close to the best metric, training looks relatively stable
best_epoch_train_snapshot:
  loss=0.8001
  cls_loss=0.1306
  reg_xytl_loss=0.2387
  seg_loss=0.0541
  iou_loss=0.3767
  stage_0_acc=99.3316
  stage_1_acc=99.3571
  stage_2_acc=99.3707
  batch=0.4954
  data=0.0508
(clrnet) root@autodl-container-578e4b9b8e-6edc601e:~/autodl-tmp/clrnet# python tools/analyze_log.py work_dirs/clr/r34_rainlane_da_only/20260417_131916_lr_6e-04_b_24/log.txt
========================================================================================
log: /root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_da_only/20260417_131916_lr_6e-04_b_24/log.txt
config:
  work_dirs=work_dirs/clr/r34_rainlane_da_only
  epochs=25, batch_size=24, base_lr=0.0006
time:
  start=2026-04-17 13:19:16
  end=2026-04-17 17:42:27
  duration_hours=4.3864
metrics:
  eval_count=25, mean=0.572748, std=0.022697
  best=epoch=22 (internal=21), metric=0.598182, step=15444, lr=1.1e-05, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_da_only/20260417_131916_lr_6e-04_b_24/ckpt/21.pth
  final=epoch=25 (internal=24), metric=0.596832, step=17550, lr=0, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_da_only/20260417_131916_lr_6e-04_b_24/ckpt/24.pth
  final_minus_best=-0.001350
  trend=tail stays close to the best metric, training looks relatively stable
best_epoch_train_snapshot:
  loss=0.8104
  cls_loss=0.1334
  reg_xytl_loss=0.2374
  seg_loss=0.0584
  iou_loss=0.3812
  stage_0_acc=99.3153
  stage_1_acc=99.349
  stage_2_acc=99.3591
  batch=0.4628
  data=0.0516
(clrnet) root@autodl-container-578e4b9b8e-6edc601e:~/autodl-tmp/clrnet# python tools/analyze_log.py work_dirs/clr/r34_rainlane_fg_only/20260417_192752_lr_6e-04_b_24/log.txt
========================================================================================
log: /root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_fg_only/20260417_192752_lr_6e-04_b_24/log.txt
config:
  work_dirs=work_dirs/clr/r34_rainlane_fg_only
  epochs=25, batch_size=24, base_lr=0.0006
time:
  start=2026-04-17 19:27:52
  end=2026-04-18 00:05:09
  duration_hours=4.6214
metrics:
  eval_count=25, mean=0.573899, std=0.020765
  best=epoch=20 (internal=19), metric=0.596509, step=14040, lr=2.9e-05, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_fg_only/20260417_192752_lr_6e-04_b_24/ckpt/19.pth
  final=epoch=25 (internal=24), metric=0.592280, step=17550, lr=0, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_fg_only/20260417_192752_lr_6e-04_b_24/ckpt/24.pth
  final_minus_best=-0.004229
  trend=tail stays close to the best metric, training looks relatively stable
best_epoch_train_snapshot:
  loss=0.8589
  cls_loss=0.1409
  reg_xytl_loss=0.2548
  seg_loss=0.0639
  iou_loss=0.3993
  stage_0_acc=99.3316
  stage_1_acc=99.3598
  stage_2_acc=99.3667
  batch=0.4486
  data=0.0527
(clrnet) root@autodl-container-578e4b9b8e-6edc601e:~/autodl-tmp/clrnet# python tools/analyze_log.py work_dirs/clr/r34_rainlane_new/20260418_023310_lr_6e-04_b_24/log.txt
========================================================================================
log: /root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_new/20260418_023310_lr_6e-04_b_24/log.txt
config:
  work_dirs=work_dirs/clr/r34_rainlane_new
  epochs=25, batch_size=24, base_lr=0.0006
time:
  start=2026-04-18 02:33:10
  end=2026-04-18 07:12:59
  duration_hours=4.6636
metrics:
  eval_count=25, mean=0.572446, std=0.023189
  best=epoch=22 (internal=21), metric=0.597021, step=15444, lr=1.1e-05, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_new/20260418_023310_lr_6e-04_b_24/ckpt/21.pth
  final=epoch=25 (internal=24), metric=0.592778, step=17550, lr=0, pth=/root/autodl-tmp/clrnet/work_dirs/clr/r34_rainlane_new/20260418_023310_lr_6e-04_b_24/ckpt/24.pth
  final_minus_best=-0.004244
  trend=tail stays close to the best metric, training looks relatively stable
best_epoch_train_snapshot:
  loss=0.7965
  cls_loss=0.1378
  reg_xytl_loss=0.2301
  seg_loss=0.0582
  iou_loss=0.3705
  stage_0_acc=99.2893
  stage_1_acc=99.324
  stage_2_acc=99.337
  batch=0.4428
  data=0.0532
(clrnet) root@autodl-container-578e4b9b8e-6edc601e:~/autodl-tmp/clrnet# 





# ========== 1. 初始化环境 ==========
# 1.1 创建并激活conda环境
conda create -n clrnet python=3.8 -y
conda activate clrnet

# 1.2 查看GPU信息（确认4090正常，不是看CPU）
nvidia-smi


# ========== 2. 第一步先装PyTorch！所有操作的基础 ==========
# 卸载残留的旧torch（如果有的话）
pip uninstall torch torchvision torchaudio -y
pip cache purge

# 安装支持CUDA11.8、适配sm89的PyTorch 2.2版本
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# ========== 3. 验证PyTorch是否安装正确，这步必须先过！ ==========
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda); print('GPU型号:', torch.cuda.get_device_name(0)); print('支持的架构:', torch.cuda.get_arch_list())"
# 这步输出正常了，再进行下一步，否则环境有问题


# ========== 4. 安装其他依赖 ==========
# 4.1 安装CLRNet的其他python依赖
pip install -r requirements.txt

# 4.2 修复依赖版本问题（如果有报错再执行，正常可以跳过）
pip install pathspec==0.9.0

# 4.3 修复imgaug的numpy兼容问题（用你修改好的版本覆盖即可）
# 把你修改好的meta.py覆盖到: /root/autodl-tmp/conda/envs/clrnet/lib/python3.8/site-packages/imgaug/augmenters/meta.py


# ========== 5. 编译CLRNet的自定义CUDA算子，适配4090 ==========
# 5.1 清理旧的编译残留
rm -rf build/ dist/ *.egg-info/ clrnet/ops/*.so
python setup.py clean --all

# 5.2 指定编译架构为sm_89，适配4090
export TORCH_CUDA_ARCH_LIST="8.9"

# 5.3 重新编译安装算子
pip install -e . --no-cache-dir
# 或者用这个也可以: python setup.py build develop


# ========== 6. 最后验证算子是否编译成功 ==========
# 检查编译好的算子是否支持sm89
cuobjdump -elf clrnet/ops/nms_impl.so | grep sm_

# ========== 7. 启动训练 ==========
python main.py configs/clrnet/clr_resnet34_rainlane.py --gpus 0