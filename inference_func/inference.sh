#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"

cd "$PROJECT_PARENT"

INFERENCE_ARGS=(
    --input_dir "/home/jl_fs/train_test_dataset/demo"
    --model_path "/home/jl_fs/DOVE/pretrained_models/DOVE"
    --output_path "/home/jl_fs/MajorProject_VSR/outputs"
    --init_from "/home/jl_fs/DOVE/checkpoints/transformer.pt"
    --lora_path "/home/jl_fs/MajorProject_VSR/checkpoints/stage_2/checkpoint-100"
    --dtype "float16"
    --is_vae_st
    --save_format yuv420p
)

python -m MajorProject_VSR.inference_func.dove_inference \
    "${INFERENCE_ARGS[@]}"