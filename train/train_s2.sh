#!/bin/bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Setup project paths (matches inference script structure)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"

cd "$PROJECT_PARENT"

# Model Configuration
MODEL_ARGS=(
    --model_path "/home/jl_fs/DOVE/pretrained_models/DOVE"
    --dtype "bfloat16"
    --gradient_checkpointing
    --init_from "/home/jl_fs/checkpoint/MAT-s1/checkpoint-300"
    --empty_prompt_embedding "/home/jl_fs/DOVE/pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
    --enable_slicing
    --enable_tiling
)

# LoRA Configuration (leave empty for full fine-tuning)
LORA_ARGS=(
    --train_lora
    --use_lora
    --lora_rank 512
    --lora_alpha 512
    --target_modules "to_q" "to_k" "to_v" "to_out.0"
    --fp32_master_weights
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/home/jl_fs/checkpoint/MAT-S2-0.42"
)

TCG_ARGS=(
    --drop_ratio 0.42
    --block_intervals 14 21 32 39
)

# Data Configuration
DATA_ARGS=(
    --video_dir "/home/jl_fs/train_test_dataset/HQ-VSR"
    --num_frames 6
    --crop_size 320 640
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

# Diffusion / SR Configuration
SR_ARGS=(
    --sr_noise_step 399
    --noise_step 0
)

# Loss Weights Configuration
LOSS_ARGS=(
    --ea_dists_weight 1.0
    --dists_weight 0.0
    --ea_lpips_weight 0.0
    --lpips_weight 0.0
    --frame_diff_weight 1.0
)

# Training / Optimisation Configuration
TRAIN_ARGS=(
    --max_train_steps 1000
    --seed 291749217
    --batch_size 1
    --gradient_accumulation_steps 8
    --learning_rate 9.6e-6
    --max_grad_norm 1.0
)

# Checkpointing / Logging Configuration
CHECKPOINT_ARGS=(
    --save_steps 80
    --log_steps 1
    --log_level "INFO"
    --param_report "summary"
)

# Periodic Validation Configuration
VALIDATION_ARGS=(
    --val_lr_dir "/home/jl_fs/train_test_dataset/UDM10/LQ-Video"
    --val_gt_dir "/home/jl_fs/train_test_dataset/UDM10/GT-Video"
    --val_metrics "psnr,ssim,lpips,dists"
    --val_steps 50
    --val_fps 8
)

RESUME_ARGS=(
    --resume_from "/home/jl_fs/checkpoint/MAT-S2-0.42/checkpoint-60"
)

# Combine all arguments and launch training
# Note: If train_stage2.py is in the root directory, change the line below to: python train_stage2.py \
python -m MajorProject_VSR.train.trainS2 \
    "${MODEL_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${DEGRADATION_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${TCG_ARGS[@]}"\
    "${LOSS_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"