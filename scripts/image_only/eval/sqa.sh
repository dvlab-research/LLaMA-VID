#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llama-vid/llama-vid-7b-full-336"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llamavid.eval.model_vqa_science \
        --model-path work_dirs/$CKPT \
        --question-file data/LLaMA-VID-Eval/scienceqa/llava_test_CQM-A.json \
        --image-folder data/LLaMA-VID-Eval/scienceqa/images/test \
        --answers-file work_dirs/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./work_dirs/scienceqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./work_dirs/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_science_qa.py \
    --base-dir data/LLaVA-Eval/scienceqa \
    --result-file $output_file \
    --output-file work_dirs/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result work_dirs/scienceqa/answers/${CKPT}_result.json
