#!/bin/bash

ARGSTR_CBD="--topk 50 --openai_model LOCAL:meta-llama/Llama-3.3-70B-Instruct --prompt_type confound_w_descriptors_with_conversational_history --distance_type confusion --subselect -1 --decay_factor 10 --classwise_topk 10 --num_iters 100 --perc_labels 0.0 --perc_initial_descriptors 1.00                   --salt a.local_run"
ARGSTR_LM4CV="--topk 50 --openai_model LOCAL:meta-llama/Llama-3.3-70B-Instruct --prompt_type confound_w_descriptors_with_conversational_history --distance_type confusion --subselect -1 --decay_factor 10 --classwise_topk 10 --num_iters 100 --perc_labels 0.0 --perc_initial_descriptors 1.00 --algorithm lm4cv --salt a.local_run_lm4cv"
commands=(
    "python escher/iteration.py --dataset cifar100 $ARGSTR_CBD"
    "python escher/iteration.py --dataset cars $ARGSTR_CBD"
    "python escher/iteration.py --dataset flower $ARGSTR_CBD"
    "python escher/iteration.py --dataset food101 $ARGSTR_CBD"
    "python escher/iteration.py --dataset cub $ARGSTR_CBD"
    "python escher/iteration.py --dataset nabirds $ARGSTR_CBD"
)

# Array of GPU IDs
gpu_ids=(4 5 6 7)

# remove existing scripts
rm -f scripts/run_on_gpu_*.sh
mkdir -p scripts/
mkdir -p scripts/logs
logfile_iter=200
# Distribute commands across GPUs
for i in "${!commands[@]}"; do
    offset=$((i % ${#gpu_ids[@]}))
    gpu_id=${gpu_ids[$offset]}
    echo "set -x" >> "scripts/run_on_gpu_$gpu_id.sh"
    echo "CUDA_VISIBLE_DEVICES=$gpu_id ${commands[$i]} &> scripts/logs/logfile_$logfile_iter.log" >> "scripts/run_on_gpu_$gpu_id.sh"
    logfile_iter=$((logfile_iter+1))
done

# Provide permission to execute scripts
chmod +x scripts/run_on_gpu_*.sh

echo "Scripts to run commands on different GPUs have been created."

# run the scripts in the scripts/* directory
for i in $(ls scripts/run_on_gpu_*.sh); do
    echo "Running $i"
    bash $i &> $i.log &
done