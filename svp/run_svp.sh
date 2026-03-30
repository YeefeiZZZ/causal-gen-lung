#!/bin/bash

python -m svp.mimic coreset \
  --datasets-dir "/home/yifei/Documents/SVP/data" \
  --arch resnet50 \
  --batch-size 128 \
  --num-workers 5 \
  --subset 6000 \
  --selection-method forgetting_events \
  --proxy-arch resnet18 \
  --proxy-batch-size 128 \
  --proxy-scale-learning-rates

if [ $? -eq 0 ]; then
  echo "任务完成，系统即将关机..."
  sudo shutdown -h now
else
  echo "任务失败，未执行关机"
fi
