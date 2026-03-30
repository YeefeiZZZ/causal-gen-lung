#!/bin/bash
model_name='cf_mnist'
exp_name=$model_name'_0%'
parents='a_r_s_f'
mkdir -p "../../checkpoints/$parents/$exp_name"


run_cmd="python train_cf_mnist.py \
    --exp_name=$exp_name \
    --dataset morphomnist \
    --data_dir=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/causal-gen1-main/datasets/morphomnist  \
    --pgm_path=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/causal-gen1-main/checkpoints/t_i_d/pgm_mnist_0%/checkpoint.pt  \
    --predictor_path=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/causal-gen1-main/checkpoints/t_i_d/aux_mnist_0%/checkpoint.pt  \
    --vae_path=/mnt/de1dcd1c-9be8-42ed-aa06-bb73570121ac/causal-gen1-main/checkpoints/t_i_d/mnist_vae_0%/checkpoint.pt  \
    --parents_x = age race sex finding \
    --lr=1e-3 \
    --bs=8 \
    --wd=0.01 \
    --eval_freq=1 \
    --plot_freq=2 \
    --epochs=50 \
    --do_pa=finding \
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

# sudo shutdown -h now

# srun python train_cf.py \
#     --data_dir='../ukbb' \
#     --exp_name=$exp_name \
#     --pgm_path='../../checkpoints/sup_pgm/checkpoint.pt' \
#     --predictor_path='../../checkpoints/sup_aux_prob/checkpoint.pt' \
#     --vae_path='../../checkpoints/$parents/$model_name/checkpoint.pt' \
#     --lr=1e-4 \
#     --bs=32 \
#     --wd=0.1 \
#     --eval_freq=1 \
#     --plot_freq=500 \
#     --do_pa=None \
#     --alpha=0.1 \
#     --seed=7
