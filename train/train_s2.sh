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
    --init_from "checkpoint/DOVE-s1"  # Stage-1 checkpoint dir to initialise from
    --empty_prompt_embedding "pretrained_models/prompt_embeddings/empty_prompt.safetensors"
    --enable_slicing
    --enable_tiling
)

# LoRA Configuration (leave empty for full fine-tuning)
LORA_ARGS=(
    # --use_lora
    # --lora_rank 64
    # --lora_alpha 64
    # --target_modules "to_q" "to_k" "to_v" "to_out.0"
    # --fp32_master_weights
    # --no_fp32_master_weights  # Uncomment to disable fp32 master weights
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "checkpoint/DOVE-s2"
)

# Data Configuration
DATA_ARGS=(
    --video_dir "../datasets/train"
    # --prompt_json "prompts.json"
    --num_frames 9  # Clip length; must satisfy (F-1) % 8 == 0 (e.g., 9, 17, 25)
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

# Diffusion / SR Configuration
SR_ARGS=(
    --sr_noise_step 399
    --noise_step 0
)

# Loss Weights Configuration (first perceptual weight > 0 wins)
LOSS_ARGS=(
    --ea_dists_weight 1.0   # Edge-aware DISTS (paper's default)
    --dists_weight 0.0
    --ea_lpips_weight 0.0
    --lpips_weight 0.0
    --frame_diff_weight 1.0
)

# Training / Optimisation Configuration
TRAIN_ARGS=(
    --max_train_steps 500   # Optimizer steps (paper: 500)
    --seed 42
    --batch_size 1          # Kept at 1 for video VRAM constraints
    --gradient_accumulation_steps 1
    --learning_rate 5e-6    # Default from argparse
    --max_grad_norm 1.0
)

# Checkpointing / Logging Configuration
CHECKPOINT_ARGS=(
    --save_steps 100
    --log_steps 10
    --log_level "INFO"      # ["DEBUG", "INFO", "WARNING"]; DEBUG adds per-tensor stats (slower)
    --param_report "summary" # ["none", "summary", "detailed"]
)

# Periodic Validation Configuration (skipped unless both dirs are set)
VALIDATION_ARGS=(
    # --val_lr_dir "../datasets/val_lr"
    # --val_gt_dir "../datasets/val_gt"
    --val_metrics "psnr,ssim,lpips,dists"
    --val_steps 100         # Validate every N optimizer steps (0 disables)
    --val_fps 8
)

# Combine all arguments and launch training
python train_stage2.py \
    "${MODEL_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${DEGRADATION_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${LOSS_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"