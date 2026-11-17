MajorProject_VSR
A research codebase for video super-resolution (VSR) built around the DOVE / CogVideoX video-diffusion pipeline. The repository adds Temporal Curvature Guidance (TCG) to the CogVideoX 3D transformer and provides two training stages, inference utilities, quality evaluation, and GPU-memory benchmarking.
Contents
Overview
Method and model components
Repository layout
Requirements and external assets
Dataset format
Training
Inference
Evaluation and memory benchmarking
Token capture and TCG tests
Configuration reference
Notes and limitations
Overview
The project adapts a CogVideoX transformer-based diffusion pipeline for video super-resolution. Low-resolution (LR) video is spatially upscaled and passed through the diffusion transformer, while the high-resolution (HR) video provides the training target. The implementation is organized into two training stages:
Stage 1 — diffusion/transformer adaptation: trains the transformer (optionally with LoRA) using an on-the-fly real-world degradation pipeline and an x0 prediction objective.
Stage 2 — pixel-space refinement: initializes from Stage 1 and optimizes a decoded video using pixel and perceptual/temporal losses. The VAE parameters are kept frozen, while the decoder remains in the gradient path so gradients can reach the latent/transformer output.
At inference time, the pipeline loads the base CogVideoX/DOVE model and optional Stage-1/Stage-2 or LoRA weights, applies TCG in selected transformer blocks, and reconstructs an upscaled video. Long or large videos can be processed using temporal chunks and spatial tiles.
Method and model components
CogVideoX transformer integration
patchTransformer.py replaces the CogVideoX transformer class exposed by Diffusers with the repository implementation. The patch imports CogVideoXTransformer3DModel and TemporalCurvatureGuidance from temporal_cogvideox_transformer_3d_batch_proccessing.py and installs the replacement before the pipeline is loaded.
The active implementation therefore uses the batch-processing transformer file. The repository also contains other transformer/TCG implementations:
temporal_cogvideox_transformer_3d.py
tram_vsr_cogvideox3d_transformer_3d.py
These are separate code variants; the patch module does not select them by default.
Temporal Curvature Guidance (TCG)
TCG operates on transformer tokens arranged on a temporal/spatial patch grid. It estimates temporal change using the angle between consecutive token-displacement vectors, normalizes the resulting curvature scores, and uses those scores to select tokens to retain locally and tokens to merge into temporal/spatial summaries. The transformer integrates the resulting token sequence and corresponding positional information into its attention blocks.
The main controls are:
drop_ratio: fraction of tokens selected for merging rather than retaining individually.
block_intervals: transformer block indices at which TCG is applied.
use_temporal_grouping: selects the temporal-grouping path in the TCG implementation.
The batch-processing implementation supports batch-vectorized token selection and temporal grouping. Its non-temporal-grouping reference/fallback path explicitly guards against batch sizes greater than one.
Stage 1
Implemented in train/train.py.
Loads a CogVideoXPipeline from the supplied pretrained model directory.
Encodes HR and LR video through the VAE; LR is resized to the HR spatial dimensions before model use.
Uses 3D rotary positional embeddings and the CogVideoX DPM scheduler.
Supports full transformer training or LoRA training. The supplied Stage-1 shell script enables LoRA and FP32 master weights.
Supports gradient checkpointing, VAE slicing/tiling, periodic validation, logging, and checkpoint saving.
Stage 2
Implemented in train/trainS2.py.
Initializes from a base model and Stage-1 transformer weights.
Supports LoRA training and resuming optimizer/scheduler state from a Stage-2 checkpoint.
Encodes and decodes frames independently through the VAE; the VAE weights remain frozen.
Computes pixel MSE, a configurable perceptual term, and frame-difference loss. The perceptual implementation supports the configured pyiqa metrics; the supplied Stage-2 shell configuration selects edge-aware DISTS (ea_dists) and frame-difference loss.
Repository layout
MajorProject_VSR/
├── dataloading/
│   └── VideoFileLoader.py
├── evaluation/
│   ├── eval.sh
│   ├── memory_benchmark.py
│   ├── metric_utils.py
│   └── validation.py
├── inference_func/
│   ├── dove_inference.py
│   ├── inference.py
│   └── inference.sh
├── train/
│   ├── train.py
│   ├── trainS2.py
│   ├── train_s1.sh
│   └── train_s2.sh
├── patchTransformer.py
├── temporal_cogvideox_transformer_3d.py
├── temporal_cogvideox_transformer_3d_batch_proccessing.py
├── tram_vsr_cogvideox3d_transformer_3d.py
├── test_ToMe.py
└── plot_loss.ipynb
__pycache__ directories and compiled Python files are omitted from this overview.
Path
Purpose
dataloading/VideoFileLoader.py
Video dataset, random temporal windows/crops, augmentations, and on-the-fly LR degradation.
patchTransformer.py
Installs the repository CogVideoX transformer implementation into Diffusers.
temporal_cogvideox_transformer_3d_batch_proccessing.py
Active batch-oriented CogVideoX transformer and TCG implementation.
temporal_cogvideox_transformer_3d.py
Alternative transformer/TCG implementation.
tram_vsr_cogvideox3d_transformer_3d.py
Additional TRaM-VSR transformer/TCG implementation variant.
train/train.py
Stage-1 training entry point.
train/trainS2.py
Stage-2 training entry point.
inference_func/inference.py
Pipeline loading, padding, and callable inference wrapper used by validation.
inference_func/dove_inference.py
Video/image-folder inference, chunking, tiling, output writing, optional token capture.
evaluation/validation.py
Video super-resolution validation and metric reporting.
evaluation/metric_utils.py
Image/video quality metric utilities.
evaluation/memory_benchmark.py
Evaluation and GPU-memory profiling modes.
test_ToMe.py
Shape, merge/restore, RoPE, and batch-independence tests for TCG behavior.
plot_loss.ipynb
Notebook for plotting training loss.

Requirements and external assets
The code imports the following principal Python packages (this is not a pinned or exhaustive environment specification):
Python and PyTorch
Hugging Face diffusers and transformers
safetensors
decord
torchvision
opencv-python (cv2)
numpy
Pillow
tqdm
pyiqa
imageio (and video-writing support used by the scripts)
psutil
The exact compatible versions depend on the Diffusers/CogVideoX implementation used for the pretrained model. This repository does not contain a requirements.txt, environment.yml, or pinned dependency lockfile. Install and validate a compatible environment before running the scripts; in particular, the custom transformer imports internal Diffusers modules and may require a compatible Diffusers version.
Required external assets
Prepare these assets separately:
A local pretrained DOVE / CogVideoX model directory accepted by CogVideoXPipeline.from_pretrained.
Training videos (HR clips) for Stage 1 and Stage 2.
Optional validation LR and GT video directories.
Optional precomputed empty-prompt embedding file (.safetensors or .pt, depending on the entry point).
Stage-1 and Stage-2 checkpoints for inference/evaluation.
Optional DOVER repository and its pretrained weights if using the DOVER video-quality metric.
The default prompt embedding path referenced in the Python code is:
pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors
The shell scripts use machine-specific absolute paths such as /home/jl_fs/.... Update these paths to match your environment before execution.
Dataset format
The training dataset loader expects a directory containing video files directly (not a nested directory tree). Supported extensions are:
.mp4  .avi  .mov  .mkv  .webm
Example:
data/
└── train_hr/
    ├── clip_0001.mp4
    ├── clip_0002.mp4
    └── clip_0003.mp4
VideoFileSRDataset reads HR frames using Decord, selects a temporal window, takes a spatial crop, applies random horizontal/vertical/temporal flips during training, and generates the LR sequence using RealWorldDegradation.
The degradation pipeline exposes configurable probabilities for blur, noise, JPEG compression, and video compression. The training scripts pass these values through command-line arguments. LR videos do not need to be pre-generated for this training loader.
An optional prompt_json maps video filename stems to text prompts. For example:
{
  "clip_0001": "A car driving along a city street",
  "clip_0002": "People walking in a park"
}
The loader returns the prompt associated with each clip, or an empty string if no mapping is present. Stage-1 batch preparation currently uses the first prompt in the batch for the batch forward pass; use batch size 1 if per-video prompt separation is required.
Video length and crop considerations
Stage 1's provided script uses --num_frames 25.
Stage 2's Python argument default is 9 frames; the supplied Stage-2 shell script overrides this to 7.
The inference code pads temporal length so that (number_of_frames - 1) is divisible by 8.
HR crop dimensions are provided as height width; the supplied scripts use 320 640 and an upscale factor of 4.
Choose clip lengths and spatial dimensions that are compatible with the selected VAE/transformer configuration and verify them on a small run before launching a full training job.
Training
Run commands from the parent directory of the MajorProject_VSR Python package so imports such as MajorProject_VSR.train.train resolve correctly. The shell scripts assume the repository directory is named MajorProject_VSR and is importable from that parent directory.
Stage 1
Edit the paths and settings in train/train_s1.sh, then run:
bash MajorProject_VSR/train/train_s1.sh
The supplied script configures:
Setting
Value in supplied script
Precision
bfloat16
LoRA
Enabled; rank 64, alpha 128
LoRA target modules
to_q, to_k, to_v, to_out.0
HR crop
320 × 640
Upscale factor
4×
Clip length
25 frames
Batch size
1
Gradient accumulation
8 steps
Learning rate
1e-4
Maximum optimizer steps
1000
Checkpoint interval
100 steps
TCG drop ratio
0.42
TCG block intervals
14, 21, 32, 39

The script also enables gradient checkpointing and VAE slicing, configures periodic validation, and sets the degradation probabilities. These are example experiment settings, not universal defaults.
A minimal direct invocation has the following form (replace paths with real paths):
python -m MajorProject_VSR.train.train \
  --model_path /path/to/pretrained/DOVE \
  --video_dir /path/to/hr_training_videos \
  --output_dir /path/to/output/stage1 \
  --num_frames 25 \
  --crop_size 320 640 \
  --upscale 4 \
  --dtype bfloat16 \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --max_train_steps 1000
Stage 2
Stage 2 is intended to refine a Stage-1 model. Set --init_from to the Stage-1 checkpoint and update the data/model/output paths in train/train_s2.sh. Then run:
bash MajorProject_VSR/train/train_s2.sh
The supplied script configures LoRA rank 128 and alpha 192, uses ea_dists_weight=1.0 and frame_diff_weight=1.0, and sets a maximum of 1000 training steps. It also includes a --resume_from path; update or remove this option depending on whether you are resuming an existing Stage-2 run.
A minimal direct invocation has the following form:
python -m MajorProject_VSR.train.trainS2 \
  --model_path /path/to/pretrained/DOVE \
  --init_from /path/to/stage1/checkpoint \
  --video_dir /path/to/hr_training_videos \
  --output_dir /path/to/output/stage2 \
  --upscale 4 \
  --dtype bfloat16 \
  --use_lora \
  --train_lora \
  --max_train_steps 1000
For exact supported options, run either entry point with --help from the package parent directory.
Inference
The primary command-line inference entry point is inference_func/dove_inference.py. It accepts a directory of input videos or image sequences and can save generated videos or PNG frame sequences.
Edit inference_func/inference.sh to set the input, base model, checkpoint, and output paths, then run:
bash MajorProject_VSR/inference_func/inference.sh
A direct invocation can be structured as follows:
python -m MajorProject_VSR.inference_func.dove_inference \
  --input_dir /path/to/low_resolution_inputs \
  --model_path /path/to/pretrained/DOVE \
  --init_from /path/to/transformer.pt \
  --lora_path /path/to/lora_checkpoint \
  --output_path /path/to/results \
  --dtype bfloat16 \
  --upscale 4 \
  --noise_step 0 \
  --sr_noise_step 399 \
  --drop_ratio 0.42 \
  --block_intervals 14 21 32 39
Useful inference options include:
Option
Description
--input_dir
Directory containing input videos or image-sequence folders.
--input_json
Optional JSON input specification.
--model_path
Local pretrained DOVE/CogVideoX pipeline directory.
--init_from
Optional full custom transformer weights (transformer.pt).
--lora_path
Optional LoRA checkpoint directory.
--output_path
Output directory.
--dtype
float16, bfloat16, or float32.
--upscale
Spatial upscale factor; default is 4.
--drop_ratio
TCG token drop/merge ratio.
--block_intervals
Transformer block indices where TCG is applied.
--tile_size_hw H W
Optional spatial tile dimensions.
--overlap_hw H W
Spatial tile overlap; default is 32 by 32.
--chunk_len
Optional temporal chunk length; 0 disables chunking.
--overlap_t
Temporal overlap between chunks; default is 8.
--is_cpu_offload
Enable sequential CPU offload.
--is_vae_st
Enable VAE slicing and tiling.
--png_save
Save output as PNG frames.
--save_format
Video pixel format passed to the output writer.

The inference implementation also supports optional transformer-token capture. See Token capture and TCG tests.
Evaluation and memory benchmarking
Quality evaluation
evaluation/validation.py runs super-resolution inference over LR inputs and compares generated outputs with corresponding GT videos. It supports video files and frame-folder inputs, prediction saving, JSON reporting, and metric selection.
Example:
python -m MajorProject_VSR.evaluation.validation \
  --model_path /path/to/pretrained/DOVE \
  --load_s1 /path/to/stage1/checkpoint \
  --load_s2 /path/to/stage2/checkpoint \
  --lr_path /path/to/lr_videos \
  --gt_path /path/to/gt_videos \
  --pred_path /path/to/predictions \
  --dtype bf16 \
  --upscale 4 \
  --drop_ratio 0.42 \
  --block_intervals 14 21 32 39 \
  --metrics psnr ssim lpips dists
The validation CLI also accepts --output_json, --empty_prompt_embedding, --tile_size_hw, --overlap_hw, --crop, --test_y_channel, --batch_mode, --fps, and --no_save_preds. The implementation initializes metrics through pyiqa; metrics that cannot be initialized are skipped with a warning.
Implemented full-reference metrics include PSNR, SSIM, LPIPS, and DISTS. Additional no-reference/quality metrics can be requested where supported by the installed pyiqa setup. The official DOVER scorer is optional and requires a separate DOVER checkout and its pretrained weights; pass --dover_root and, if needed, --dover_opt.
GPU-memory benchmarking
evaluation/memory_benchmark.py supports both evaluation and memory-profiling modes. The supplied evaluation/eval.sh demonstrates memory mode across multiple clip lengths and TCG drop ratios.
Example:
python -m MajorProject_VSR.evaluation.memory_benchmark \
  --mode memory \
  --model_path /path/to/pretrained/DOVE \
  --load_s1 /path/to/stage1/checkpoint \
  --load_s2 /path/to/stage2/checkpoint \
  --dtype bf16 \
  --upscale 4 \
  --mem_frames 33 49 65 \
  --mem_lr_size 180 320 \
  --mem_drop_ratios 0.35 0.42 0.5 \
  --mem_reps 3 \
  --mem_warmup 1 \
  --output_json outputs/memory_benchmark.json
Memory-mode controls include --mem_frames, --mem_lr_size, --mem_drop_ratios, --mem_reps, --mem_warmup, --mem_per_block, --mem_no_baseline, and --mem_no_vae_tiling.
Token capture and TCG tests
Capture transformer tokens
dove_inference.py can register forward pre-hooks on transformer blocks and save the hidden states entering selected layers. Enable capture with:
--capture_tokens \
--capture_layers 14 21 32 39 \
--token_save_path /path/to/token_dumps
If --capture_layers is omitted, the implementation captures all transformer layers. If --token_save_path is omitted, the default is a token_dumps directory under the output path. Captured layer files are named tokens_layer_<index>.pt and contain the captured hidden states and encoder hidden states (when available).
Run TCG tests
test_ToMe.py contains tests for curvature-score shapes, token split and temporal-group shapes, RoPE handling after merging, identity restoration, batch independence, and guards for reference paths that do not support batch sizes greater than one.
Run from the repository package parent directory:
python -m MajorProject_VSR.test_ToMe
Configuration reference
The most frequently adjusted settings are:
Parameter
Meaning
Code default / supplied script value
model_path
Pretrained DOVE/CogVideoX pipeline directory.
Required.
video_dir
Directory of HR training videos.
Required.
upscale
Spatial scale factor.
4.
num_frames
Number of frames sampled per training clip.
Stage 1: 49 in Python; 25 in shell script. Stage 2: 9 in Python; 7 in shell script.
crop_size
HR crop height and width.
Stage 1: 256×256 in Python; 320×640 in shell script. Stage 2: 320×640.
dtype
Model compute dtype.
bfloat16.
use_lora
Enable LoRA adapters.
Disabled in Python defaults; enabled in supplied scripts.
lora_rank, lora_alpha
LoRA rank and scaling parameter.
Stage 1 script: 64/128; Stage 2 script: 128/192.
drop_ratio
TCG token merge/drop ratio.
0.35 in Python defaults; 0.42 in supplied scripts.
block_intervals
Transformer blocks using TCG.
[20, 25, 32, 37] in Python defaults; [14, 21, 32, 39] in supplied scripts.
noise_step, sr_noise_step
Diffusion/SR timestep controls.
0 and 399 in supplied scripts.

The Python CLI defaults and shell-script experiment values are not identical. Treat the shell scripts as experiment configurations and inspect them before running.
Notes and limitations
The repository does not provide a dependency lockfile or a complete environment installation script.
The pretrained base model, prompt embeddings, datasets, DOVER assets, and trained checkpoints are external dependencies and are not included in this source snapshot.
The shell scripts contain environment-specific absolute paths and should be edited before use.
patchTransformer.py patches Diffusers globally in the Python process. The project imports it before loading the CogVideoX pipeline; preserve that order when creating new entry points.
The active transformer patch points to temporal_cogvideox_transformer_3d_batch_proccessing.py. The other transformer files are present as variants and are not automatically selected by the patch.
Stage-1 batch preparation uses the first prompt in a batch for the entire batch.
The inference path pads frame counts and spatial dimensions to satisfy model/VAE constraints, then removes padding from the generated result.
The code is research software. Validate tensor dimensions, model/checkpoint compatibility, and memory use with a small test run before launching long experiments.
Acknowledgements
This codebase builds on the Hugging Face Diffusers CogVideoX implementation and the DOVE video-diffusion super-resolution pipeline. Please consult the upstream projects and their licenses when using or redistributing their code, model weights, or datasets.

