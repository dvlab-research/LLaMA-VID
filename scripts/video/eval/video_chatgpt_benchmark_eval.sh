#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llama-vid/llama-vid-7b-full-224-video-fps-1"
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamavid/eval/model_video_chatgpt_general.py \
        --model-path ./work_dirs/$CKPT \
        --video_dir ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/Test_Videos \
        --gt_file ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/generic_qa.json \
        --output_dir ./work_dirs/eval_video_chatgpt/$CKPT \
        --output_name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &

done

wait

python llamavid/eval/evaluate_benchmark_1_correctness.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/correctness_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/correctness_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE

python llamavid/eval/evaluate_benchmark_2_detailed_orientation.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/detail_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/detail_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE

python llamavid/eval/evaluate_benchmark_3_context.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/context_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/context_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16\
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamavid/eval/model_video_chatgpt_general.py \
        --model-path ./work_dirs/$CKPT \
        --video_dir ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/Test_Videos \
        --gt_file ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/temporal_qa.json \
        --output_dir ./work_dirs/eval_video_chatgpt/$CKPT \
        --output_name pred_temporal \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &

done

wait

python llamavid/eval/evaluate_benchmark_4_temporal.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/temporal_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/temporal_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamavid/eval/model_video_chatgpt_consistency.py \
        --model-path ./work_dirs/$CKPT \
        --video_dir ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/Test_Videos \
        --gt_file ./data/LLaMA-VID-Eval/video-chatgpt/evaluation/consistency_qa.json \
        --output_dir ./work_dirs/eval_video_chatgpt/$CKPT \
        --output_name pred_consistency \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &

done

wait

python llamavid/eval/evaluate_benchmark_5_consistency.py \
    --pred_path ./work_dirs/eval_video_chatgpt/$CKPT \
    --output_dir ./work_dirs/eval_video_chatgpt/$CKPT/consistency_results \
    --output_json ./work_dirs/eval_video_chatgpt/$CKPT/consistency_results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE
