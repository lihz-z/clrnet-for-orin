#!/bin/bash
CURRENT_PID=887752
echo "等待主进程 $CURRENT_PID 结束..."
while kill -0 $CURRENT_PID 2>/dev/null; do sleep 10; done
echo "开始后续三个训练..."
python main.py configs/clrnet/clr_resnet34_rainlane_da_only.py --gpus 0 > da_only.log 2>&1
python main.py configs/clrnet/clr_resnet34_rainlane_fgm_stage2.py --gpus 0 > fgm_stage2.log 2>&1
python main.py configs/clrnet/clr_resnet34_rainlane_fg_only.py --gpus 0 > fg_only.log 2>&1
echo "全部完成！"
