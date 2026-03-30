#!/bin/bash
model_name='aux'
exp_name=$model_name'-age_wcc_tttttt'
parents='a_r_s_f'
mkdir -p "../../checkpoints/$parents/$exp_name"


run_cmd="python train_pgm.py \
    --exp_name=$exp_name \
    --dataset mimic \
    --data_dir=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/MIMIC_CXR/wcc \
    --hps mimic384 \
    --setup sup_aux \
    --parents_x age wcc race sex finding \
    --context_dim=16 \
    --concat_pa \
    --lr=0.001 \
    --bs=32 \ "

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
