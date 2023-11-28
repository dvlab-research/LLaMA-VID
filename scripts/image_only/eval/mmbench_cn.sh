#!/bin/bash

CKPT="llama-vid/llama-vid-7b-full-336"
SPLIT="mmbench_dev_cn_20231003"

CUDA_VISIBLE_DEVICES=0 python -m llamavid.eval.model_vqa_mmbench \
    --model-path ./work_dirs/$CKPT \
    --question-file ./data/LLaMA-VID-Eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./data/LLaMA-VID-Eval/mmbench_cn/answers/$SPLIT/$CKPT.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 

mkdir -p ./data/LLaMA-VID-Eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./data/LLaMA-VID-Eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./data/LLaMA-VID-Eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./data/LLaMA-VID-Eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment $CKPT
