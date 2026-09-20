"""
TRaM-VSR / DOVE Stage-1 training with On-the-Fly Real-World Degradation.
(Includes Pristine Diagnostic Logging)
"""

import argparse
import json
import logging
import math
import os
import random
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from safetensors.torch import load_file
from transformers import set_seed

import decord  # isort:skip
decord.bridge.set_bridge("torch")

try:
    from torchvision.io import write_video, read_video
    _VIDEO_IO_AVAILABLE = True
except ImportError:
    _VIDEO_IO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tram_vsr_stage1")

# --------------------------------------------------------------------------- #
# Pristine Logging Utilities
# --------------------------------------------------------------------------- #
def log_tensor_stats(tensor: torch.Tensor, name: str, level=logging.DEBUG):
    """Logs shape, dtype, device, and value distribution. Flags NaNs/Infs."""
    if tensor is None:
        logger.log(level, f"[{name}] is None")
        return
    if not torch.is_tensor(tensor):
        logger.log(level, f"[{name}] is not a tensor: {type(tensor)}")
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    shape_str = f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    
    if has_nan or has_inf:
        logger.log(level, f"[{name}] {shape_str} | !! WARNING: HAS_NAN={has_nan}, HAS_INF={has_inf} !!")
    else:
        t_float = tensor.detach().float()
        logger.log(level, f"[{name}] {shape_str} | min={t_float.min().item():.4f}, max={t_float.max().item():.4f}, mean={t_float.mean().item():.4f}, std={t_float.std().item():.4f}")

def log_memory(tag: str):
    """Logs GPU VRAM usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        logger.debug(f"[MEM] {tag}: Alloc={alloc:.2f}GB, Reserved={res:.2f}GB, MaxAlloc={max_alloc:.2f}GB")


def spatial_upsample_video(
    video: torch.Tensor,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Spatially upsamples a video without changing the temporal dimension.

    Input:
        video: [B, C, T, H, W]

    Output:
        video: [B, C, T, target_H, target_W]
    """
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor [B,C,T,H,W], got shape {tuple(video.shape)}")

    b, c, t, h, w = video.shape
    target_h, target_w = target_hw

    # [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
    video = video.permute(0, 2, 1, 3, 4).contiguous()
    video = video.reshape(b * t, c, h, w)

    # 2D spatial interpolation per frame
    video = F.interpolate(
        video,
        size=(target_h, target_w),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None,
    )

    # [B*T, C, target_H, target_W] -> [B, T, C, target_H, target_W]
    video = video.reshape(b, t, c, target_h, target_w)

    # [B, T, C, H, W] -> [B, C, T, H, W]
    video = video.permute(0, 2, 1, 3, 4).contiguous()

    return video

# --------------------------------------------------------------------------- #
# 1. Rotary embeddings & Stage-1 Forward Pass
# --------------------------------------------------------------------------- #
def prepare_rotary_positional_embeddings(
    height: int, width: int, num_frames: int,
    transformer_config: Dict, vae_scale_factor_spatial: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    p = transformer_config.patch_size
    p_t = transformer_config.patch_size_t

    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p 

    if transformer_config.patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim, crops_coords=None,
        grid_size=(grid_height, grid_width), temporal_size=base_num_frames,
        grid_type="slice", max_size=(base_size_height, base_size_width), device=device,
    )
    return freqs_cos, freqs_sin

def forward_stage1(
    pipe: CogVideoXPipeline, lr_video: torch.Tensor, hr_video: torch.Tensor,
    prompt: str, noise_step: int, sr_noise_step: int,
    empty_prompt_embedding: torch.Tensor, freeze_vae: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    try:
        logger.debug("--- [FORWARD STAGE 1 START] ---")
        log_tensor_stats(lr_video, "Input LR Video")
        log_tensor_stats(hr_video, "Input HR Video")
        log_memory("Before VAE Encode")
        
        device = pipe.vae.device
        dtype = pipe.vae.dtype

        lr_video = lr_video.to(device, dtype=dtype)
        hr_video = hr_video.to(device, dtype=dtype)

        with torch.no_grad():
            actual_latent = pipe.vae.encode(hr_video).latent_dist.sample() * pipe.vae.config.scaling_factor
            actual_latent = actual_latent.permute(0, 2, 1, 3, 4)
        log_tensor_stats(actual_latent, "Actual Latent (HR Target)")

        vae_ctx = torch.no_grad() if freeze_vae else torch.enable_grad()
        with vae_ctx:
            lr_latent = pipe.vae.encode(lr_video).latent_dist.sample() * pipe.vae.config.scaling_factor

        patch_size_t = pipe.transformer.config.patch_size_t
        ncopy = 0
        if patch_size_t is not None:
            ncopy = lr_latent.shape[2] % patch_size_t
            if ncopy > 0:
                first_frame = lr_latent[:, :, :1, :, :]
                lr_latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), lr_latent], dim=2)

        batch_size, num_channels, num_frames, height, width = lr_latent.shape
        log_tensor_stats(lr_latent, "LR Latent (Pre-Pad)")

        if prompt == "" and empty_prompt_embedding is not None:
            prompt_embedding = empty_prompt_embedding.to(device, dtype=dtype)
            if prompt_embedding.shape[0] != batch_size:
                prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)
        else:
            prompt_token_ids = pipe.tokenizer(
                prompt, padding="max_length", max_length=pipe.transformer.config.max_text_seq_length,
                truncation=True, add_special_tokens=True, return_tensors="pt",
            ).input_ids
            with torch.no_grad():
                prompt_embedding = pipe.text_encoder(prompt_token_ids.to(device))[0]
            _, seq_len, _ = prompt_embedding.shape
            prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=dtype)
            
        log_tensor_stats(prompt_embedding, "Prompt Embedding")

        lr_latent = lr_latent.permute(0, 2, 1, 3, 4)

        if noise_step != 0:
            noise = torch.randn_like(lr_latent)
            add_timesteps = torch.full((batch_size,), noise_step, dtype=torch.long, device=device)
            lr_latent = pipe.scheduler.add_noise(lr_latent, noise, add_timesteps)

        timesteps = torch.full((batch_size,), sr_noise_step, dtype=torch.long, device=device)
        vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        transformer_config = pipe.transformer.config
        
        rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial, width=width * vae_scale_factor_spatial,
                num_frames=num_frames, transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial, device=device,
            ) if transformer_config.use_rotary_positional_embeddings else None
        )

        log_tensor_stats(lr_latent, "LR Latent (DiT Input)")
        log_memory("Before DiT Forward")
        
        predicted_noise = pipe.transformer(
            hidden_states=lr_latent, encoder_hidden_states=prompt_embedding,
            timestep=timesteps, image_rotary_emb=rotary_emb, return_dict=False,
        )[0]
        
        log_tensor_stats(predicted_noise, "DiT Output (Predicted Noise/Velocity)")

        final_latent = pipe.scheduler.get_velocity(predicted_noise, lr_latent, timesteps)
        log_tensor_stats(final_latent, "Final Latent (x0_pred)")

        if patch_size_t is not None and ncopy > 0:
            final_latent = final_latent[:, ncopy:, :, :, :]

        if final_latent.shape != actual_latent.shape:
            raise RuntimeError(f"Shape mismatch: predicted {tuple(final_latent.shape)} vs target {tuple(actual_latent.shape)}")

        logger.debug("--- [FORWARD STAGE 1 END] ---")
        return final_latent, actual_latent
        
    except Exception as e:
        logger.error(f"!!! CRITICAL ERROR in forward_stage1: {str(e)} !!!")
        logger.error("Dumping tensor states at time of crash:")
        for k, v in locals().items():
            if torch.is_tensor(v):
                log_tensor_stats(v, f"CRASH_STATE -> {k}", level=logging.ERROR)
        raise


# --------------------------------------------------------------------------- #
# 2. On-the-Fly Real-World Degradation Pipeline
# --------------------------------------------------------------------------- #
class RealWorldDegradation:
    def __init__(self, scale: int = 4, blur_prob: float = 0.8, blur_kernel_range: Tuple[int, int] = (7, 21),
                 blur_sigma_range: Tuple[float, float] = (0.2, 3.0), resize_modes: Tuple[str, ...] = ("bilinear", "bicubic", "area"),
                 noise_prob: float = 0.5, noise_sigma_range: Tuple[float, float] = (0.0, 15.0),
                 jpeg_prob: float = 0.7, jpeg_quality_range: Tuple[int, int] = (30, 95),
                 video_compress_prob: float = 0.0, video_compress_crf_range: Tuple[int, int] = (23, 35),
                 fps: int = 24, shuffle_order: bool = True):
        self.scale = scale
        self.blur_prob, self.blur_kernel_range, self.blur_sigma_range = blur_prob, blur_kernel_range, blur_sigma_range
        self.resize_modes = resize_modes
        self.noise_prob, self.noise_sigma_range = noise_prob, noise_sigma_range
        self.jpeg_prob, self.jpeg_quality_range = jpeg_prob, jpeg_quality_range
        self.video_compress_prob, self.video_compress_crf_range = video_compress_prob, video_compress_crf_range
        self.fps, self.shuffle_order = fps, shuffle_order

    def _gaussian_blur(self, frames: torch.Tensor) -> torch.Tensor:
        k = random.randrange(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)
        sigma = random.uniform(*self.blur_sigma_range)
        return transforms.GaussianBlur(kernel_size=k, sigma=sigma)(frames)

    def _downsample(self, frames: torch.Tensor) -> torch.Tensor:
        mode = random.choice(self.resize_modes)
        _, _, h, w = frames.shape
        new_h, new_w = max(1, h // self.scale), max(1, w // self.scale)
        kwargs = {} if mode == "area" else {"align_corners": False}
        return F.interpolate(frames, size=(new_h, new_w), mode=mode, **kwargs).clamp(0, 1)

    def _add_noise(self, frames: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(*self.noise_sigma_range) / 255.0
        if sigma <= 0: return frames
        return (frames + torch.randn_like(frames) * sigma).clamp(0, 1)

    def _jpeg_compress(self, frames: torch.Tensor) -> torch.Tensor:
        quality = random.randint(*self.jpeg_quality_range)
        device = frames.device
        out = []
        for f in frames:
            img = (f.permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                out.append(f); continue
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            out.append(torch.from_numpy(dec).permute(2, 0, 1).float() / 255.0)
        return torch.stack(out).to(device)

    def _video_compress(self, frames: torch.Tensor) -> torch.Tensor:
        if not _VIDEO_IO_AVAILABLE: return frames
        try:
            crf = random.randint(*self.video_compress_crf_range)
            vid_uint8 = (frames.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).to(torch.uint8).cpu()
            with tempfile.TemporaryDirectory() as td:
                tmp_path = os.path.join(td, "degrade_tmp.mp4")
                write_video(tmp_path, vid_uint8, fps=self.fps, options={'crf': str(crf)})
                vid_out, _, _ = read_video(tmp_path, pts_unit='sec')
            out = vid_out.float() / 255.0
            out = out.permute(0, 3, 1, 2)
            if out.shape[0] != frames.shape[0]:
                out = out[:frames.shape[0]] if out.shape[0] > frames.shape[0] else frames
            return out.to(frames.device)
        except Exception:
            return frames

    def __call__(self, hr_frames: torch.Tensor) -> torch.Tensor:
        steps = []
        if random.random() < self.blur_prob: steps.append(self._gaussian_blur)
        if random.random() < self.noise_prob: steps.append(self._add_noise)
        if random.random() < self.jpeg_prob: steps.append(self._jpeg_compress)
        if random.random() < self.video_compress_prob: steps.append(self._video_compress)
        if self.shuffle_order: random.shuffle(steps)
        steps.append(self._downsample)

        applied_ops = [step.__name__.replace('_', '') for step in steps]
        logger.debug(f"[DEGRADE] Pipeline: {' -> '.join(applied_ops)} | Input HR: {tuple(hr_frames.shape)}")

        lr = hr_frames
        for step in steps: lr = step(lr)

        t, c, h, w = hr_frames.shape
        target_h, target_w = max(1, h // self.scale), max(1, w // self.scale)
        if lr.shape[-2:] != (target_h, target_w):
            lr = F.interpolate(lr, size=(target_h, target_w), mode='bicubic', align_corners=False).clamp(0, 1)
        return lr.clamp(0, 1)


# --------------------------------------------------------------------------- #
# 3. VideoFileSRDataset
# --------------------------------------------------------------------------- #
class VideoFileSRDataset(Dataset):
    DEFAULT_VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    def __init__(self, video_dir: str, num_frames: int = 25, hr_crop_size: Tuple[int, int] = (320, 640),
                 scale: int = 4, is_train: bool = True, video_extensions: Optional[Tuple[str, ...]] = None,
                 degradation_kwargs: Optional[dict] = None, prompt_json: Optional[str] = None):
        super().__init__()
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.hr_crop_size = hr_crop_size
        self.scale = scale
        self.is_train = is_train
        self.video_extensions = tuple(e.lower() for e in (video_extensions or self.DEFAULT_VIDEO_EXTS))

        self.video_paths = sorted([
            os.path.join(video_dir, f) for f in os.listdir(video_dir)
            if f.lower().endswith(self.video_extensions)
        ])
        if not self.video_paths:
            raise ValueError(f"No video files found in {video_dir}")

        degradation_kwargs = dict(degradation_kwargs or {})
        degradation_kwargs.setdefault('scale', scale)
        self.degrade = RealWorldDegradation(**degradation_kwargs)
        
        self.prompts = {}
        if prompt_json is not None and os.path.exists(prompt_json):
            with open(prompt_json, "r") as f:
                self.prompts = json.load(f)
            logger.info(f"Loaded {len(self.prompts)} prompts from {prompt_json}")

        logger.info(f"VideoFileSRDataset initialized: {len(self.video_paths)} clips found in {video_dir}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def _read_hr_window(self, video_path: str) -> Tuple[torch.Tensor, int]:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total < self.num_frames:
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
            start_idx = 0
        else:
            start_idx = random.randint(0, total - self.num_frames) if self.is_train else 0
            indices = list(range(start_idx, start_idx + self.num_frames))
        return vr.get_batch(indices).float() / 255.0, start_idx

    def _spatial_crop(self, frames: torch.Tensor) -> torch.Tensor:
        crop_h, crop_w = self.hr_crop_size
        T, H, W, C = frames.shape
        if H < crop_h or W < crop_w:
            up = max(crop_h / H, crop_w / W)
            new_h, new_w = math.ceil(H * up), math.ceil(W * up)
            frames = F.interpolate(frames.permute(0, 3, 1, 2), size=(new_h, new_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            T, H, W, C = frames.shape

        crop_h -= crop_h % self.scale
        crop_w -= crop_w % self.scale
        if self.is_train:
            top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
        else:
            top, left = (H - crop_h) // 2, (W - crop_w) // 2
        return frames[:, top:top + crop_h, left:left + crop_w, :]

    def _apply_augmentations(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.is_train: return frames
        if random.random() > 0.5: frames = torch.flip(frames, [2])
        if random.random() > 0.5: frames = torch.flip(frames, [1])
        if random.random() > 0.5: frames = torch.flip(frames, [0])
        return frames

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_path = self.video_paths[index]
        clip_name = Path(video_path).stem

        hr_frames, _ = self._read_hr_window(video_path)
        hr_frames = self._spatial_crop(hr_frames)
        hr_frames = self._apply_augmentations(hr_frames)

        hr_sequence = hr_frames.permute(0, 3, 1, 2).contiguous()
        lr_sequence = self.degrade(hr_sequence)

        return {
            'lr': lr_sequence,
            'hr': hr_sequence,
            'clip_name': clip_name,
            'prompt': self.prompts.get(clip_name, ""),
        }


# --------------------------------------------------------------------------- #
# 4. Training Loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="TRaM-VSR / DOVE Stage-1 training (On-the-fly Degradation)")

    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./stage1_ckpt")

    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--crop_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--upscale", type=int, default=4)

    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--sr_noise_step", type=int, default=399)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=None)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--train_vae", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if (args.num_frames - 1) % 8 != 0:
        raise ValueError(f"--num_frames must satisfy (F - 1) % 8 == 0, got {args.num_frames}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    freeze_vae = not args.train_vae

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add file handler for pristine debug logging
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train_debug.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    logger.info("=" * 64)
    logger.info("TRaM-VSR Stage-1 training (On-the-fly Real-World Degradation)")
    logger.info(f"  Video dir         : {args.video_dir}")
    logger.info(f"  Model             : {args.model_path}")
    logger.info(f"  Output dir        : {args.output_dir}")
    logger.info(f"  dtype             : {args.dtype}")
    logger.info(f"  num_frames        : {args.num_frames}")
    logger.info(f"  crop_size         : {tuple(args.crop_size)}")
    logger.info(f"  sr_noise_step     : {args.sr_noise_step} | noise_step: {args.noise_step}")
    logger.info(f"  learning_rate     : {args.learning_rate}")
    logger.info(f"  batch_size        : {args.batch_size} (grad_accum={args.gradient_accumulation_steps})")
    logger.info(f"  freeze_vae        : {freeze_vae}")
    logger.info(f"  use_lora          : {args.use_lora}")
    logger.info("=" * 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available — training on CPU will be extremely slow.")

    logger.info(f"Loading CogVideoX pipeline from {args.model_path} ...")
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    logger.info("Pipeline loaded.")

    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()

    if freeze_vae:
        pipe.vae.requires_grad_(False)
        pipe.vae.eval()
    else:
        pipe.vae.requires_grad_(True)
        pipe.vae.train()

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"], init_lora_weights="gaussian",
        )
        pipe.transformer = get_peft_model(pipe.transformer, lora_config)
        trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    else:
        pipe.transformer.requires_grad_(True)
        trainable_params = list(pipe.transformer.parameters())

    if not freeze_vae:
        trainable_params += list(pipe.vae.parameters())

    pipe.transformer.train()
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

        # =========================================================================
    # 100% VERIFICATION: Parameter & LoRA Layer Logging
    # =========================================================================
    logger.info("=" * 85)
    logger.info("PARAMETER & LORA VERIFICATION REPORT")
    logger.info("=" * 85)

    total_params = 0
    trainable_params_count = 0
    lora_trainable_params_count = 0
    original_trainable_params_count = 0
    
    trainable_layer_details = []
    frozen_layer_details = []

    # 1. Analyze Transformer Parameters
    for name, param in pipe.transformer.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params_count += num_params
            # Check if it's a LoRA parameter (peft names them with 'lora_A', 'lora_B', etc.)
            if "lora" in name.lower():
                lora_trainable_params_count += num_params
                trainable_layer_details.append(f"  [LoRA TRAINABLE] {name:<65} | shape: {str(tuple(param.shape)):<25} | {num_params:,}")
            else:
                original_trainable_params_count += num_params
                trainable_layer_details.append(f"  [ORIG TRAINABLE] {name:<65} | shape: {str(tuple(param.shape)):<25} | {num_params:,}")
        else:
            frozen_layer_details.append(f"  [FROZEN]         {name:<65} | shape: {str(tuple(param.shape)):<25} | {num_params:,}")

    # 2. Analyze VAE Parameters (if trained)
    if not freeze_vae:
        for name, param in pipe.vae.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params_count += num_params
                trainable_layer_details.append(f"  [VAE TRAINABLE]  VAE.{name:<60} | shape: {str(tuple(param.shape)):<25} | {num_params:,}")
            else:
                frozen_layer_details.append(f"  [VAE FROZEN]     VAE.{name:<60} | shape: {str(tuple(param.shape)):<25} | {num_params:,}")

    # 3. Print Summary
    logger.info(f"Total Parameters (Transformer + VAE if applicable) : {total_params:,}")
    logger.info(f"Total Trainable Parameters                         : {trainable_params_count:,} ({100 * trainable_params_count / max(total_params, 1):.4f}%)")
    
    if args.use_lora:
        logger.info(f"  -> New LoRA Trainable Parameters                 : {lora_trainable_params_count:,} ({100 * lora_trainable_params_count / max(total_params, 1):.4f}%)")
        if original_trainable_params_count > 0:
            logger.error(f"  -> 🚨 WARNING: Original Transformer Trainable Params : {original_trainable_params_count:,} (Should be 0 for pure LoRA!)")
        else:
            logger.info(f"  -> Original Transformer Trainable Params         : 0 (Correctly frozen)")
    else:
        logger.info(f"  -> Full Fine-Tuning: All Transformer params trainable ({trainable_params_count:,})")

    # 4. Print Detailed Breakdown
    logger.info("-" * 85)
    logger.info("DETAILED TRAINABLE LAYERS BREAKDOWN:")
    if not trainable_layer_details:
        logger.info("  (None)")
    for detail in trainable_layer_details:
        logger.info(detail)
        
    logger.info("-" * 85)
    logger.info(f"FROZEN LAYERS SUMMARY (Total: {len(frozen_layer_details)} layers):")
    # Printing ALL frozen layers to guarantee 100% visibility as requested.
    # If the list is massive, it will still be safely written to your train_debug.log file.
    for detail in frozen_layer_details:
        logger.info(detail)
        
    logger.info("=" * 85)
    
    empty_prompt_embedding = None
    empty_prompt_path = Path("pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors")
    if empty_prompt_path.exists():
        empty_prompt_embedding = load_file(str(empty_prompt_path))["prompt_embedding"]

    dataset = VideoFileSRDataset(
        video_dir=args.video_dir,
        num_frames=args.num_frames,
        hr_crop_size=tuple(args.crop_size),
        scale=args.upscale,
        is_train=True,
        prompt_json=args.prompt_json,
        degradation_kwargs=dict(
            blur_prob=0.8, noise_prob=0.5, jpeg_prob=0.7, video_compress_prob=0.0
        ),
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    steps_per_epoch = max(len(dataloader) // args.gradient_accumulation_steps, 1)
    total_steps = args.max_train_steps or steps_per_epoch * args.num_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()
    stop_training = False

    for epoch in range(args.num_epochs):
        logger.info(f"---- Epoch {epoch + 1}/{args.num_epochs} ----")
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}")
        optimizer.zero_grad()

        for micro_step, batch in enumerate(progress):
            try:
                t_batch_start = time.time()

                lr_video = batch['lr']
                hr_video = batch['hr']
                prompts = batch['prompt']

                logger.debug(f"[DATA] Batch {micro_step} fetched in {time.time() - t_batch_start:.3f}s")
                log_tensor_stats(lr_video, "Batch LR (CPU)")
                log_tensor_stats(hr_video, "Batch HR (CPU)")

                # Dataset gives [B, T, C, H, W]
                # Convert to [B, C, T, H, W]
                lr_video = lr_video.permute(0, 2, 1, 3, 4).contiguous()
                hr_video = hr_video.permute(0, 2, 1, 3, 4).contiguous()

                # Normalize [0, 1] -> [-1, 1]
                lr_video = lr_video * 2.0 - 1.0
                hr_video = hr_video * 2.0 - 1.0

                # Spatially upsample LR from e.g. [B,3,49,64,64]
                # to [B,3,49,256,256]
                lr_video = spatial_upsample_video(
                    lr_video,
                    target_hw=(hr_video.shape[3], hr_video.shape[4]),
                    mode="bilinear",
                )

                log_tensor_stats(lr_video, "LR Video (Upsampled)")
                log_tensor_stats(hr_video, "HR Video")

                prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts

                final_latent, actual_latent = forward_stage1(
                    pipe=pipe,
                    lr_video=lr_video,
                    hr_video=hr_video,
                    prompt=prompt,
                    noise_step=args.noise_step,
                    sr_noise_step=args.sr_noise_step,
                    empty_prompt_embedding=empty_prompt_embedding,
                    freeze_vae=freeze_vae,
                )

                loss = F.mse_loss(final_latent.float(), actual_latent.float())  

                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"!!! LOSS IS NaN/Inf at step {global_step} !!!")
                    log_tensor_stats(final_latent, "NaN_Loss -> final_latent", logging.ERROR)
                    log_tensor_stats(actual_latent, "NaN_Loss -> actual_latent", logging.ERROR)
                    
                log_tensor_stats(loss, "Loss")
                
                (loss / args.gradient_accumulation_steps).backward()
                log_memory("After Backward")

                running_loss += loss.item()
                epoch_loss_sum += loss.item()
                epoch_loss_count += 1

                if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    logger.info(f"[UPDATING GRADIENTS]: Gradients updated. step {micro_step:06d}")
                    if global_step % args.log_steps == 0:
                        avg_loss = running_loss / (args.log_steps * args.gradient_accumulation_steps)
                        current_lr = lr_scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        logger.info(f"step {global_step:06d}/{total_steps} | loss {avg_loss:.6f} | grad_norm {grad_norm:.4f} | lr {current_lr:.2e} | elapsed {elapsed / 60:.1f}min")
                        progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")
                        running_loss = 0.0

                    if global_step % args.save_steps == 0:
                        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(ckpt_path, exist_ok=True)
                        if args.use_lora:
                            pipe.transformer.save_pretrained(ckpt_path)
                        else:
                            torch.save(pipe.transformer.state_dict(), os.path.join(ckpt_path, "transformer.pt"))

                    if args.max_train_steps and global_step >= args.max_train_steps:
                        stop_training = True
                        break
            except Exception as e:
                logger.error(f"!!! CRITICAL ERROR in training loop micro_step {micro_step}: {str(e)} !!!")
                logger.error(f"Prompt: {prompts}")
                raise

        if stop_training: break

    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    if args.use_lora:
        pipe.transformer.save_pretrained(final_path)
    else:
        torch.save(pipe.transformer.state_dict(), os.path.join(final_path, "transformer.pt"))

    logger.info(f"Training complete in {(time.time() - start_time) / 60:.1f}min. Final weights saved to {final_path}")

if __name__ == "__main__":
    main()