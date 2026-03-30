#!/bin/bash

model_name='cf_mimic'
exp_name=$model_name'-age_wcc_80_bl'
parents='a_r_s_f'
mkdir -p "../../checkpoints/$parents/$exp_name"


run_cmd="python train_cf.py \
    --exp_name=$exp_name \
    --dataset mimic \
    --data_dir=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/MIMIC_CXR/wcc  \
    --pgm_path=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/causal-gen1-main/checkpoints/a_w_r_s_f/pgm_age_wcc_80/checkpoint.pt  \
    --predictor_path=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/causal-gen1-main/checkpoints/a_w_r_s_f/aux_age_wcc_80/checkpoint.pt  \
    --vae_path=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/causal-gen1-main/checkpoints/a_w_r_s_f/hvae_age_wcc_80/checkpoint.pt  \
    --parents_x = age wcc race sex finding \
    --lr=5e-5 \
    --bs=8 \
    --wd=0.1 \
    --eval_freq=1 \
    --plot_freq=2 \
    --epochs=4 \
    --do_pa=wcc \
    --alpha=0.1 \
    --seed=7"

#${run_cmd}

if [ "$2" = "nohup" ]
then
 nohup ${run_cmd} > $exp_name.out 2>&1 &
 echo "Started training in background with nohup, PID: $!"
else
 ${run_cmd}
fi
