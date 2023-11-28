#!/bin/bash

CKPT="llama-vid/llama-vid-7b-full-336"
CUDA_VISIBLE_DEVICES=0 python -m llamavid.eval.model_vqa_loader \
    --model-path work_dirs/$CKPT \
    --question-file data/LLaMA-VID-Eval/MME/llava_mme.jsonl \
    --image-folder data/LLaMA-VID-Eval/MME/MME_Benchmark_release_version \
    --answers-file data/LLaMA-VID-Eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 

cd data/LLaMA-VID-Eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
