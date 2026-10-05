#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model Configuration
MODEL_ARGS=(
    --model_path "/home/jl_fs/DOVE/pretrained_models/DOVE"
    --dtype "bfloat16"  # ["float16", "bfloat16", "float32"]
    --gradient_checkpointing
    --enable_slicing
    --init_from "/home/jl_fs/checkpoint/MAT-s1/checkpoint-750"  # dir containing transformer.pt to initialise from
)

# LoRA Configuration (leave empty for full fine-tuning)
LORA_ARGS=(
    --use_lora
    --lora_rank 512
    --lora_alpha 512
    --target_modules "to_q" "to_k" "to_v" "to_out.0"
    --fp32_master_weights
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "checkpoint/MAT-s1"
)

# Data Configuration
DATA_ARGS=(
    --video_dir "/home/jl_fs/train_test_dataset/HQ-VSR"
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
    --num_epochs 10  # upper bound; training stops at max_train_steps
    --max_train_steps 1000
    --seed 3114121
    --batch_size 1
    --gradient_accumulation_steps 8
    --learning_rate 1.22e-06
)

VALIDATION_ARGS=(
    --val_lr_dir "/home/jl_fs/train_test_dataset/small_val_set/LQ-Video"
    --val_gt_dir "/home/jl_fs/train_test_dataset/small_val_set/GT-Video"
    --val_metrics "psnr,ssim,lpips,dists"
    --val_steps 100
    --val_fps 8
)

# SR parameters
SR_ARGS=(
    --sr_noise_step 399
    --noise_step 0
    # --empty_prompt_embedding "pretrained_models/prompt_embeddings/<hash>.safetensors"
    --empty_prompt_embedding "/home/jl_fs/DOVE/pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
)

# TCG module
TCG_ARGS=(
    --drop_ratio 0.42
    --block_intervals 14 21 32 39
)

# Checkpointing / Logging Configuration
CHECKPOINT_ARGS=(
    --save_steps 150  # save checkpoint every x steps
    --log_steps 1
    --log_level "INFO"  # ["DEBUG", "INFO", "WARNING"]; DEBUG adds per-tensor stats (slower)
    --param_report "summary"  # ["none", "summary", "detailed"]
)

# Combine all arguments and launch training
python -m MajorProject_VSR.train.train \
    "${MODEL_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${DEGRADATION_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${TCG_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}"