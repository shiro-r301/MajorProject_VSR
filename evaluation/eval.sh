#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m MajorProject_VSR.evaluation.memory_benchmark \
  --mode memory \
  --model_path "/home/jl_fs/DOVE/pretrained_models/DOVE" \
  --load_s1 "/home/jl_fs/checkpoint/MAT_S1-dr_0.42-alpha_128-r_64/checkpoint-500" \
  --load_s2 "/home/jl_fs/checkpoint/MAT-S2-0.42_alpha-192_rank-128/checkpoint-400" \
  --empty_prompt_embedding "/home/jl_fs/DOVE/pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"\
  --dtype bf16 \
  --upscale 4 \
  --noise_step 0 \
  --sr_noise_step 399 \
  --block_intervals 14 21 32 39 \
  --mem_frames 33 49 65\
  --mem_lr_size 180 320 \
  --mem_drop_ratios 0.35 0.42 0.5 \
  --mem_reps 3 \
  --mem_warmup 1 \
  --mem_per_block \
  --seed 0 \
  --output_json out/mem.json \
  --log_file out/mem.log \
  --log_level INFO

# python -m MajorProject_VSR.evaluation.validation \
#     --model_path "/home/jl_fs/DOVE/pretrained_models/DOVE" \
#     --empty_prompt_embedding "/home/jl_fs/DOVE/pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors" \
#     --load_s1 /home/jl_fs/checkpoint/MAT_S1-dr_0.42-alpha_128-r_64/checkpoint-500 \
#     --load_s2 /home/jl_fs/checkpoint/MAT-S2-0.42_alpha-192_rank-128/checkpoint-400 \
#     --lr_path /home/Datasets/MVSR4x/LQ-Video \
#     --gt_path /home/Datasets/MVSR4x/GT-Video \
#     --pred_path /home/Final_eval/MVSR4x/S2/MAT_S2-dr_0.42-alpha_196-r_128 \
#     --drop_ratio 0.42 \
#     --block_intervals 14 21 32 39 \
#     --upscale 4 \
#     --noise_step 0 \
#     --sr_noise_step 399 \
#     --metrics psnr ssim lpips dists clipiqa dover\
#     --dtype bf16 \
#     --dover_root /home/jl_fs/MajorProject_VSR/DOVER \
#     --dover_opt /home/jl_fs/MajorProject_VSR/DOVER/dover.yml \

# python -m MajorProject_VSR.evaluation.validation \
#     --model_path "/home/jl_fs/DOVE/pretrained_models/DOVE" \
#     --empty_prompt_embedding "/home/jl_fs/DOVE/pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors" \
#     --lr_path /home/Datasets/YouHQ40/LQ-Video/sub_LQ \
#     --gt_path /home/Datasets/YouHQ40/GT-Video/sub_GT \
#     --pred_path /home/Final_eval/YouHQ40/baseline/ \
#     --drop_ratio 0.42 \
#     --block_intervals 14 21 32 39 \
#     --upscale 4 \
#     --noise_step 0 \
#     --sr_noise_step 399 \
#     --metrics psnr ssim lpips dists clipiqa dover\
#     --dtype bf16 \
#     --dover_root /home/jl_fs/MajorProject_VSR/DOVER \
#     --dover_opt /home/jl_fs/MajorProject_VSR/DOVER/dover.yml \
    
# python -m MajorProject_VSR.evaluation.validation \
#     --model_path "/home/jl_fs/DOVE/pretrained_models/DOVE" \
#     --empty_prompt_embedding "/home/jl_fs/DOVE/pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors" \
#     --load_s1 /home/jl_fs/checkpoint/MAT_S1-dr_0.42-alpha_128-r_64/checkpoint-500 \
#     --load_s2 outputs/stage2_run \
#     --lr_path /home/Datasets/UDM10/LQ-Video \
#     --gt_path /home/Datasets/UDM10/GT-Video \
#     --pred_path /home/Final_eval/UDM10 \
#     # --output_json /home/Final_eval/UDM10/ \
#     --upscale 4 \
#     --noise_step 0 \
#     --sr_noise_step 399 \
#     --metrics psnr ssim lpips dists \
#     --dtype bf16