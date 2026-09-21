#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B"
    --dtype "bfloat16"  # ["float16", "bfloat16", "float32"]
    --gradient_checkpointing
    # --init_from "checkpoint/DOVE-s1/final"  # dir containing transformer.pt to initialise from
)

# LoRA Configuration (leave empty for full fine-tuning)
LORA_ARGS=(
    # --use_lora
    # --lora_rank 64
    # --lora_alpha 64
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "checkpoint/DOVE-s1"
)

# Data Configuration
DATA_ARGS=(
    --video_dir "../datasets/train"
    # --prompt_json "prompts.json"
    --num_frames 25  # frames should be 8N+1
    --crop_size 320 640  # HR crop size (height width)
    --upscale 4
    --num_workers 8
)

# Degradation Configuration
DEGRADATION_ARGS=(
    --blur_prob 0.8
    --noise_prob 0.5
    --jpeg_prob 0.7
    --video_compress_prob 0.0
)

# Training Configuration
TRAIN_ARGS=(
    --num_epochs 1000  # upper bound; training stops at max_train_steps
    --max_train_steps 10000
    --seed 42
    --batch_size 2
    --gradient_accumulation_steps 1
    --learning_rate 2e-5
)

# SR parameters
SR_ARGS=(
    --sr_noise_step 399
    --noise_step 0
    # --empty_prompt_embedding "pretrained_models/prompt_embeddings/<hash>.safetensors"
)

# TCG module
TCG_ARGS=(
    --drop_ratio 0.35
    --block_intervals 20 25 32 37
)

# Checkpointing / Logging Configuration
CHECKPOINT_ARGS=(
    --save_steps 400  # save checkpoint every x steps
    --log_steps 10
    --log_level "INFO"  # ["DEBUG", "INFO", "WARNING"]; DEBUG adds per-tensor stats (slower)
    --param_report "summary"  # ["none", "summary", "detailed"]
)

# Combine all arguments and launch training
python train_stage1.py \
    "${MODEL_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${DEGRADATION_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${TCG_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}"