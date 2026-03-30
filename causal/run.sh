#!/bin/bash

exp_name="hvae_clean_uncer_20%"
run_cmd="python main.py \
    --exp_name=$exp_name \
    --data_dir=/home/yifei/Documents/MIMIC_data/60k_clean \
    --csv_dir=/home/yifei/Documents/MIMIC_data/60k_clean \
    --hps mimic192 \
    --parents_x age race sex finding \
    --context_dim=8 \
    --concat_pa \
    --lr=0.001 \
    --bs=32 \
    --wd=0.05 \
    --beta=9 \
    --x_like=ydiag_dgauss \
    --z_max_res=96 \
    --eval_freq=4"

# 启用核心转储
ulimit -c unlimited

if [ "$2" = "nohup" ]; then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  pid=$!
  echo "Started training in background with nohup, PID: $pid"
  wait $pid
  if [ -f core ]; then
    echo "核心转储文件已生成，启动gdb进行调试..."
    gdb python3 core
  else
    echo "未生成核心转储文件。"
  fi
else
  ${run_cmd}
  if [ -f core ]; then
    echo "核心转储文件已生成，启动gdb进行调试..."
    gdb python3 core
  else
    echo "未生成核心转储文件。"
  fi
fi

