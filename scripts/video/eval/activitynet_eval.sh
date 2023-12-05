#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llama-vid/llama-vid-7b-full-224-video-fps-1"
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamavid/eval/model_activitynet_qa.py \
    --model-path ./work_dirs/$CKPT \
    --video_dir ./data/LLaMA-VID-Eval/ActivityNet-QA/all_test \
    --gt_file_question ./data/LLaMA-VID-Eval/ActivityNet-QA/test_q.json \
    --gt_file_answers ./data/LLaMA-VID-Eval/ActivityNet-QA/test_a.json \
    --output_dir ./work_dirs/eval_activitynet/$CKPT \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &

done

wait

python llamavid/eval/eval_activitynet_qa.py \
    --pred_path ./work_dirs/eval_activitynet/$CKPT \
    --output_dir ./work_dirs/eval_activitynet/$CKPT/results \
    --output_json ./work_dirs/eval_activitynet/$CKPT/results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE