"""
TRaM-VSR / DOVE Stage-2 training: pixel-space refinement.

Mirrors Stage-1 (`train_S1.py`) for everything shared — degradation, dataset,
rotary embeddings, diagnostics, parameter report — and matches the paper's
reference trainer (DOVES2Trainer.compute_loss) for everything stage-2 specific.

PATCHED vs the previous version, marked [FIX] inline:

  [FIX 1] RoPE: inherited from the patched Stage-1, which now has the correct
          CogVideoX 1.0 / 1.5 branches.
  [FIX 2] LOSS DOMAIN. The reference maps BOTH prediction and GT to [0, 1] with
          (x * 0.5 + 0.5).clamp(0, 1) before computing anything. The previous
          version computed MSE in [-1, 1], making it 4x larger relative to the
          perceptual term, and skipped the clamp (so saturated pixels kept
          producing gradient).
  [FIX 3] Perceptual loss is pyiqa, not piq, and supports all four reference
          modes selected by whichever weight is > 0:
              ea_dists > dists > ea_lpips > lpips
          Edge-aware variants add metric(edge(pred), edge(gt)) and divide the
          per-frame sum by (F * 2) instead of F.
  [FIX 4] Trainable params kept in fp32 + autocast (see Stage-1 [FIX 2]).
  [FIX 5] --max_grad_norm, --enable_slicing/--enable_tiling, LoRA
          init_lora_weights=True, configurable --target_modules.
  [FIX 6] Frame-diff loss is gated on the decoded frame count (shape[2] > 1),
          like the reference, not on the image/video branch flag.

Arg names now follow the reference: --ea_dists_weight / --dists_weight /
--ea_lpips_weight / --lpips_weight and --frame_diff_weight replace the old
--lambda1 / --lambda2.

MEMORY NOTE: the VAE decoder runs WITH grad here. Decoding 9 frames at 320x640
with activations retained is expensive — reduce --num_frames, --crop_size, or
push --image_ratio toward 1.0 if you OOM, and try --enable_slicing.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from safetensors.torch import load_file
from transformers import set_seed

try:
    from diffusers.training_utils import cast_training_params
except ImportError:  # pragma: no cover
    def cast_training_params(models, dtype=torch.float32):
        if not isinstance(models, list):
            models = [models]
        for m in models:
            for p in m.parameters():
                if p.requires_grad:
                    p.data = p.data.to(dtype)

# --------------------------------------------------------------------------- #
# Everything shared with Stage-1 comes from the Stage-1 module.
# --------------------------------------------------------------------------- #
try:
    from train_S1 import (
        RealWorldDegradation,
        VideoFileSRDataset,
        autocast_ctx,
        encode_text_cached,
        log_memory,
        log_tensor_stats,
        prepare_rotary_positional_embeddings,
        spatial_upsample_video,
    )
except ImportError:  # stage-1 file named train.py
    from for_git.train import (
        RealWorldDegradation,
        VideoFileSRDataset,
        autocast_ctx,
        encode_text_cached,
        log_memory,
        log_tensor_stats,
        prepare_rotary_positional_embeddings,
        spatial_upsample_video,
    )

import decord  # isort:skip

decord.bridge.set_bridge("torch")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tram_vsr_stage2")


# --------------------------------------------------------------------------- #
# 1. Frame-by-frame VAE encode/decode (Sec. 3.2, Eq. 5)
# --------------------------------------------------------------------------- #
def encode_frames_independently(
    vae, video: torch.Tensor, scaling_factor: float, freeze_vae: bool
) -> torch.Tensor:
    """
    video: [B, C, T, H, W] in [-1, 1].
    Returns latent [B, C', T, H', W'] — same layout stage-1's vae.encode gives,
    so the padding / permute code downstream is identical to forward_stage1.
    """
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor [B,C,T,H,W], got {tuple(video.shape)}")

    num_frames = video.shape[2]
    vae_ctx = torch.no_grad() if freeze_vae else torch.enable_grad()
    latents = []
    with vae_ctx:
        for t in range(num_frames):
            frame_clip = video[:, :, t : t + 1, :, :]
            frame_latent = vae.encode(frame_clip).latent_dist.sample() * scaling_factor
            latents.append(frame_latent)
    return torch.cat(latents, dim=2)


def decode_frames_independently(vae, latent: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    latent: [B, T, C', H', W'] (the layout the DiT works in).
    Grad ALWAYS flows here — the decoder sits on the loss path in stage 2 even
    though its weights are frozen.
    Returns pixels [B, C, T, H, W] in roughly [-1, 1].
    """
    latent = latent.permute(0, 2, 1, 3, 4)  # [B, C', T, H', W']
    latent = latent / scaling_factor
    num_frames = latent.shape[2]
    frames = []
    for t in range(num_frames):
        frames.append(vae.decode(latent[:, :, t : t + 1, :, :]).sample)
    return torch.cat(frames, dim=2)


# --------------------------------------------------------------------------- #
# 2. Perceptual losses — pyiqa, with the reference's four modes  [FIX 3]
# --------------------------------------------------------------------------- #
class SobelEdgeDetectionModel(nn.Module):
    """
    Stand-in for the reference's `finetune.utils.metric_utils.EdgeDetectionModel`,
    used only if that module isn't importable. Per-channel Sobel gradient
    magnitude, output kept 3-channel in [0, 1] so it can be fed straight back
    into a pyiqa metric. Fully differentiable.

    If you have the paper's repo, the real EdgeDetectionModel is imported
    automatically and this class is unused — the two are NOT guaranteed to be
    numerically identical, so weights tuned against one may need retuning.
    """

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        ky = kx.t().contiguous()
        kernel = torch.stack([kx, ky]).unsqueeze(1)  # [2, 1, 3, 3]
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] in [0, 1]
        b, c, h, w = x.shape
        weight = self.kernel.to(x.dtype).repeat(c, 1, 1, 1)  # [2C, 1, 3, 3]
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        g = F.conv2d(x_pad, weight, groups=c)  # [B, 2C, H, W], order (gx, gy) per channel
        gx, gy = g[:, 0::2], g[:, 1::2]
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        return (mag / 4.0).clamp(0.0, 1.0)  # /4 keeps typical Sobel response in range


def build_edge_model(device: torch.device) -> nn.Module:
    try:
        from finetune.utils.metric_utils import EdgeDetectionModel  # paper's repo
        logger.info("Using the reference EdgeDetectionModel from finetune.utils.metric_utils")
        model = EdgeDetectionModel()
    except ImportError:
        logger.warning(
            "finetune.utils.metric_utils.EdgeDetectionModel not importable — falling back to "
            "a Sobel edge detector. Numerically similar in spirit, not identical."
        )
        model = SobelEdgeDetectionModel()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class PerceptualLoss:
    """
    Reimplements the reference's perceptual-loss block verbatim: one mode,
    picked by the first weight that is > 0, in the order
    ea_dists > dists > ea_lpips > lpips. Per-frame accumulation, then divide by
    (F * 2) for edge-aware modes or F otherwise, then scale by the weight.
    """

    def __init__(self, args, device: torch.device):
        self.device = device
        self.mode = None
        self.weight = 0.0
        self.metric = None
        self.edge_model = None

        if args.ea_dists_weight > 0:
            self.mode, self.weight, metric_name, edge = "ea_dists", args.ea_dists_weight, "dists", True
        elif args.dists_weight > 0:
            self.mode, self.weight, metric_name, edge = "dists", args.dists_weight, "dists", False
        elif args.ea_lpips_weight > 0:
            self.mode, self.weight, metric_name, edge = "ea_lpips", args.ea_lpips_weight, "lpips", True
        elif args.lpips_weight > 0:
            self.mode, self.weight, metric_name, edge = "lpips", args.lpips_weight, "lpips", False
        else:
            logger.warning("All perceptual weights are 0 — training with MSE (+ frame-diff) only.")
            return

        try:
            import pyiqa
        except ImportError as e:
            raise ImportError(
                "Stage-2 needs pyiqa for the perceptual loss (the paper uses "
                "pyiqa.create_metric(..., as_loss=True)). Install with `pip install pyiqa`, "
                "or set all perceptual weights to 0."
            ) from e

        self.metric = pyiqa.create_metric(metric_name, device=device, as_loss=True)
        self.edge_model = build_edge_model(device) if edge else None
        logger.info(f"Perceptual loss: mode={self.mode}, metric={metric_name}, weight={self.weight}")

    def __call__(self, video_generate: torch.Tensor, hq_videos: torch.Tensor) -> torch.Tensor:
        """Both inputs [B, C, F, H, W] in [0, 1]."""
        if self.metric is None:
            return torch.zeros((), device=video_generate.device)

        num_frames = video_generate.shape[2]
        total = 0.0
        for f in range(num_frames):
            pred_frame = video_generate[:, :, f, :, :].to(dtype=torch.float32, device=self.device)
            gt_frame = hq_videos[:, :, f, :, :].to(dtype=torch.float32, device=self.device)

            total = total + self.metric(pred_frame, gt_frame)
            if self.edge_model is not None:
                total = total + self.metric(
                    self.edge_model(pred_frame), self.edge_model(gt_frame)
                )

        divisor = num_frames * 2 if self.edge_model is not None else num_frames
        return (total / divisor) * self.weight


def frame_difference_loss(x_sr: torch.Tensor, x_hr: torch.Tensor, weight: float) -> torch.Tensor:
    """
    Eq. 6: L1 between consecutive-frame differences. Inputs [B, C, F, H, W] in
    [0, 1]. [FIX 6] gated on frame count like the reference.
    """
    if x_sr.shape[2] < 2:
        return torch.zeros((), device=x_sr.device)
    x_sr = x_sr.to(dtype=torch.float32)
    x_hr = x_hr.to(dtype=torch.float32)
    diff_gen = x_sr[:, :, 1:, :, :] - x_sr[:, :, :-1, :, :]
    diff_gt = x_hr[:, :, 1:, :, :] - x_hr[:, :, :-1, :, :]
    return F.l1_loss(diff_gen, diff_gt) * weight


# --------------------------------------------------------------------------- #
# 3. Stage-2 forward pass
# --------------------------------------------------------------------------- #
def forward_stage2(
    pipe: CogVideoXPipeline,
    lr_video: torch.Tensor,
    hr_video: torch.Tensor,
    prompt: str,
    noise_step: int,
    sr_noise_step: int,
    empty_prompt_embedding: torch.Tensor,
    freeze_vae: bool,
    autocast_dtype: Optional[torch.dtype] = None,
    prompt_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    lr_video, hr_video: [B, C, T, H, W] in [-1, 1], LR already spatially
    upsampled to HR size by the caller. T == 1 for an image batch.

    Returns (x_sr, x_hr) in [-1, 1]; the CALLER converts to [0, 1] for the loss
    (see [FIX 2] in main).
    """
    try:
        logger.debug("--- [FORWARD STAGE 2 START] ---")
        log_tensor_stats(lr_video, "Input LR Video")
        log_tensor_stats(hr_video, "Input HR Video")
        log_memory("Before VAE Encode")

        device = pipe.vae.device
        dtype = pipe.vae.dtype
        scaling_factor = pipe.vae.config.scaling_factor

        lr_video = lr_video.to(device, dtype=dtype)
        hr_video = hr_video.to(device, dtype=dtype)
        orig_num_frames = lr_video.shape[2]

        lr_latent = encode_frames_independently(pipe.vae, lr_video, scaling_factor, freeze_vae)

        patch_size_t = pipe.transformer.config.patch_size_t
        ncopy = 0
        if patch_size_t is not None:
            ncopy = lr_latent.shape[2] % patch_size_t
            if ncopy > 0:
                first_frame = lr_latent[:, :, :1, :, :]
                lr_latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), lr_latent], dim=2)
            assert lr_latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = lr_latent.shape
        log_tensor_stats(lr_latent, "LR Latent (Pre-Pad)")

        if prompt == "" and empty_prompt_embedding is not None:
            prompt_embedding = empty_prompt_embedding.to(device, dtype=dtype)
            if prompt_embedding.shape[0] != batch_size:
                prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)
        else:
            prompt_embedding = encode_text_cached(pipe, prompt, device, prompt_cache)
            _, seq_len, _ = prompt_embedding.shape
            prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=dtype)

        log_tensor_stats(prompt_embedding, "Prompt Embedding")

        lr_latent = lr_latent.permute(0, 2, 1, 3, 4)  # [B, T, C', H', W']

        if noise_step != 0:
            noise = torch.randn_like(lr_latent)
            add_timesteps = torch.full((batch_size,), noise_step, dtype=torch.long, device=device)
            lr_latent = pipe.scheduler.add_noise(lr_latent, noise, add_timesteps)

        timesteps = torch.full((batch_size,), sr_noise_step, dtype=torch.long, device=device)
        vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        transformer_config = pipe.transformer.config

        rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        log_tensor_stats(lr_latent, "LR Latent (DiT Input)")
        log_memory("Before DiT Forward")

        with autocast_ctx(device, autocast_dtype):  # [FIX 4]
            predicted_noise = pipe.transformer(
                hidden_states=lr_latent,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]

        log_tensor_stats(predicted_noise, "DiT Output (Predicted Noise/Velocity)")

        sr_latent = pipe.scheduler.get_velocity(predicted_noise, lr_latent, timesteps)
        log_tensor_stats(sr_latent, "SR Latent (x0_pred)")

        if patch_size_t is not None and ncopy > 0:
            sr_latent = sr_latent[:, ncopy:, :, :, :]

        if sr_latent.shape[1] != orig_num_frames:
            raise RuntimeError(
                f"Frame count mismatch after de-padding: latent has {sr_latent.shape[1]} "
                f"frames, expected {orig_num_frames}."
            )

        log_memory("Before VAE Decode")
        x_sr = decode_frames_independently(pipe.vae, sr_latent, scaling_factor)
        log_tensor_stats(x_sr, "x_sr (Decoded Pixels, [-1,1])")

        if x_sr.shape != hr_video.shape:
            raise RuntimeError(
                f"Shape mismatch: reconstructed {tuple(x_sr.shape)} vs target {tuple(hr_video.shape)}"
            )

        logger.debug("--- [FORWARD STAGE 2 END] ---")
        return x_sr, hr_video

    except Exception as e:
        logger.error(f"!!! CRITICAL ERROR in forward_stage2: {str(e)} !!!")
        logger.error("Dumping tensor states at time of crash:")
        for k, v in locals().items():
            if torch.is_tensor(v):
                log_tensor_stats(v, f"CRASH_STATE -> {k}", level=logging.ERROR)
        raise


# --------------------------------------------------------------------------- #
# 4. ImageFileSRDataset — an image is a 1-frame video
# --------------------------------------------------------------------------- #
# class ImageFileSRDataset(VideoFileSRDataset):
#     DEFAULT_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

#     def __init__(
#         self,
#         image_dir: str,
#         hr_crop_size: Tuple[int, int] = (320, 640),
#         scale: int = 4,
#         is_train: bool = True,
#         image_extensions: Optional[Tuple[str, ...]] = None,
#         degradation_kwargs: Optional[dict] = None,
#         prompt_json: Optional[str] = None,
#     ):
#         # Not calling VideoFileSRDataset.__init__ (it scans for video files);
#         # every other inherited method behaves identically.
#         Dataset.__init__(self)
#         self.video_dir = image_dir
#         self.num_frames = 1
#         self.hr_crop_size = hr_crop_size
#         self.scale = scale
#         self.is_train = is_train
#         self.video_extensions = tuple(
#             e.lower() for e in (image_extensions or self.DEFAULT_IMAGE_EXTS)
#         )

#         self.video_paths = sorted(
#             [
#                 os.path.join(image_dir, f)
#                 for f in os.listdir(image_dir)
#                 if f.lower().endswith(self.video_extensions)
#             ]
#         )
#         if not self.video_paths:
#             raise ValueError(f"No image files found in {image_dir}")

#         degradation_kwargs = dict(degradation_kwargs or {})
#         degradation_kwargs.setdefault("scale", scale)
#         degradation_kwargs["video_compress_prob"] = 0.0  # meaningless for 1 frame
#         self.degrade = RealWorldDegradation(**degradation_kwargs)

#         self.prompts = {}
#         if prompt_json is not None and os.path.exists(prompt_json):
#             with open(prompt_json, "r") as f:
#                 self.prompts = json.load(f)
#             logger.info(f"Loaded {len(self.prompts)} prompts from {prompt_json}")

#         logger.info(f"ImageFileSRDataset initialized: {len(self.video_paths)} images found in {image_dir}")

#     def _read_hr_window(self, video_path: str) -> Tuple[torch.Tensor, int]:
#         """Returns [1, H, W, C] float in [0, 1] — same contract as the video
#         version with T == 1, so the inherited __getitem__ works unchanged."""
#         img = cv2.imread(video_path, cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError(f"Failed to read image {video_path}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         frames = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
#         return frames.unsqueeze(0), 0


def infinite_loader(dataloader: DataLoader):
    """Cycles a DataLoader forever, reshuffling each pass."""
    while True:
        for batch in dataloader:
            yield batch


# --------------------------------------------------------------------------- #
# 5. Training Loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="TRaM-VSR / DOVE Stage-2 training (pixel-space refinement, on-the-fly degradation)"
    )

    parser.add_argument("--video_dir", type=str, required=True, help="HR video clips; LR generated on the fly")
    # parser.add_argument("--image_dir", type=str, required=True, help="HR images; LR generated on the fly")
    parser.add_argument("--prompt_json", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--init_from", type=str, default=None, help="Stage-1 checkpoint dir (transformer.pt or LoRA adapter dir)")
    parser.add_argument("--output_dir", type=str, default="./stage2_ckpt")
    parser.add_argument("--num_frames", type=int, default=9, help="Video-branch clip length; must satisfy (F-1) %% 8 == 0")
    parser.add_argument("--crop_size", type=int, nargs=2, default=(320, 640))
    parser.add_argument("--upscale", type=int, default=4)
    # parser.add_argument("--image_ratio", type=float, default=0.8, help="phi: fraction of iterations from the image branch")

    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--sr_noise_step", type=int, default=399)

    # [FIX 3] reference loss weights; first one > 0 wins, in this order.
    parser.add_argument("--ea_dists_weight", type=float, default=1.0, help="Edge-aware DISTS (paper's default perceptual loss)")
    parser.add_argument("--dists_weight", type=float, default=0.0)
    parser.add_argument("--ea_lpips_weight", type=float, default=0.0)
    parser.add_argument("--lpips_weight", type=float, default=0.0)
    parser.add_argument("--frame_diff_weight", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500, help="Optimizer steps (paper: 500)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)  # [FIX 5]

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--target_modules", type=str, nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])

    parser.add_argument("--fp32_master_weights", action="store_true", default=True)  # [FIX 4]
    parser.add_argument("--no_fp32_master_weights", dest="fp32_master_weights", action="store_false")

    parser.add_argument("--enable_slicing", action="store_true")  # [FIX 5]
    parser.add_argument("--enable_tiling", action="store_true")

    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    if (args.num_frames - 1) % 8 != 0:
        raise ValueError(f"--num_frames must satisfy (F - 1) % 8 == 0, got {args.num_frames}")
    # if not (0.0 <= args.image_ratio <= 1.0):
    #    raise ValueError(f"--image_ratio (phi) must be in [0, 1], got {args.image_ratio}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    freeze_vae = True  # stage 2 always keeps VAE weights frozen
    autocast_dtype = dtype if (args.fp32_master_weights and dtype != torch.float32) else None

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train_debug.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    logger.info("=" * 64)
    logger.info("TRaM-VSR Stage-2 training (pixel-space refinement)")
    logger.info(f"  Video dir         : {args.video_dir}")
    # logger.info(f"  Image dir         : {args.image_dir}")
    logger.info(f"  Model             : {args.model_path}")
    logger.info(f"  Init from         : {args.init_from}")
    logger.info(f"  Output dir        : {args.output_dir}")
    logger.info(f"  dtype             : {args.dtype}")
    logger.info(f"  fp32 master wts   : {args.fp32_master_weights} (autocast={autocast_dtype})")
    logger.info(f"  num_frames        : {args.num_frames} | crop_size: {tuple(args.crop_size)}")
    logger.info(f"  image_ratio (phi) : {args.image_ratio}")
    logger.info(f"  sr_noise_step     : {args.sr_noise_step} | noise_step: {args.noise_step}")
    logger.info(
        f"  loss weights      : ea_dists={args.ea_dists_weight}, dists={args.dists_weight}, "
        f"ea_lpips={args.ea_lpips_weight}, lpips={args.lpips_weight}, frame_diff={args.frame_diff_weight}"
    )
    logger.info(f"  learning_rate     : {args.learning_rate}")
    logger.info(f"  batch_size        : {args.batch_size} (grad_accum={args.gradient_accumulation_steps})")
    logger.info(f"  max_train_steps   : {args.max_train_steps} | max_grad_norm: {args.max_grad_norm}")
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

    if args.enable_slicing:
        pipe.vae.enable_slicing()
        logger.info("VAE slicing enabled.")
    if args.enable_tiling:
        pipe.vae.enable_tiling()
        logger.info("VAE tiling enabled.")

    tc = pipe.transformer.config
    logger.info(
        f"RoPE branch: {'CogVideoX 1.5 (slice)' if tc.patch_size_t is not None else 'CogVideoX 1.0 (linear+crop)'} | "
        f"patch_size={tc.patch_size}, patch_size_t={tc.patch_size_t}, "
        f"native grid = {tc.sample_height // tc.patch_size} x {tc.sample_width // tc.patch_size}"
    )

    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()

    if args.use_lora:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as e:
            raise ImportError("--use_lora requires `peft`: pip install peft") from e
        if args.init_from:
            logger.info(f"Loading stage-1 LoRA adapter from {args.init_from}")
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.init_from, is_trainable=True)
        else:
            logger.info(f"Wrapping transformer with a fresh LoRA adapter (rank={args.lora_rank}, alpha={args.lora_alpha})")
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                init_lora_weights=True,  # [FIX 5]
            )
            pipe.transformer = get_peft_model(pipe.transformer, lora_config)
        trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    else:
        if args.init_from:
            ckpt_path = os.path.join(args.init_from, "transformer.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"--init_from given but no transformer.pt found at {ckpt_path}")
            logger.info(f"Loading stage-1 transformer weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"{len(missing)} missing key(s) loading stage-1 ckpt (up to 5): {missing[:5]}")
            if unexpected:
                logger.warning(f"{len(unexpected)} unexpected key(s) loading stage-1 ckpt (up to 5): {unexpected[:5]}")
        pipe.transformer.requires_grad_(True)
        trainable_params = list(pipe.transformer.parameters())

    # [FIX 4]
    if args.fp32_master_weights and dtype != torch.float32:
        cast_training_params([pipe.transformer], dtype=torch.float32)
        trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
        if not args.use_lora:
            logger.warning(
                "Full fine-tune with fp32 master weights roughly doubles weight memory. "
                "Use --use_lora or --no_fp32_master_weights if you OOM."
            )
        logger.info(f"Trainable params cast to fp32; compute runs under autocast({dtype}).")

    pipe.transformer.train()
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled (transformer only; VAE decoder stays un-checkpointed).")

    # =========================================================================
    # PARAMETER & LORA VERIFICATION REPORT
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

    for name, param in pipe.transformer.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params_count += num_params
            if "lora" in name.lower():
                lora_trainable_params_count += num_params
                trainable_layer_details.append(
                    f"  [LoRA TRAINABLE] {name:<65} | shape: {str(tuple(param.shape)):<25} | dtype: {str(param.dtype):<15} | {num_params:,}"
                )
            else:
                original_trainable_params_count += num_params
                trainable_layer_details.append(
                    f"  [ORIG TRAINABLE] {name:<65} | shape: {str(tuple(param.shape)):<25} | dtype: {str(param.dtype):<15} | {num_params:,}"
                )
        else:
            frozen_layer_details.append(
                f"  [FROZEN]         {name:<65} | shape: {str(tuple(param.shape)):<25} | dtype: {str(param.dtype):<15} | {num_params:,}"
            )

    vae_trainable = sum(p.numel() for p in pipe.vae.parameters() if p.requires_grad)

    logger.info(f"Total Transformer Parameters                       : {total_params:,}")
    logger.info(
        f"Total Trainable Parameters                         : {trainable_params_count:,} "
        f"({100 * trainable_params_count / max(total_params, 1):.4f}%)"
    )
    if args.use_lora:
        logger.info(
            f"  -> New LoRA Trainable Parameters                 : {lora_trainable_params_count:,} "
            f"({100 * lora_trainable_params_count / max(total_params, 1):.4f}%)"
        )
        if original_trainable_params_count > 0:
            logger.error(f"  -> 🚨 WARNING: Original Transformer Trainable Params : {original_trainable_params_count:,} (Should be 0 for pure LoRA!)")
        else:
            logger.info("  -> Original Transformer Trainable Params         : 0 (Correctly frozen)")
    else:
        logger.info(f"  -> Full Fine-Tuning: All Transformer params trainable ({trainable_params_count:,})")

    if vae_trainable > 0:
        logger.error(f"  -> 🚨 WARNING: VAE Trainable Params : {vae_trainable:,} (Stage 2 must keep the VAE frozen!)")
    else:
        logger.info("  -> VAE Trainable Params                          : 0 (Correctly frozen, decoder on loss path)")

    bad_dtype = [n for n, p in pipe.transformer.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if args.fp32_master_weights and bad_dtype:
        logger.error(f"  -> 🚨 {len(bad_dtype)} trainable param(s) are NOT fp32 (up to 5): {bad_dtype[:5]}")
    elif args.fp32_master_weights:
        logger.info("  -> All trainable params are fp32 ✓")

    logger.info("-" * 85)
    logger.info("DETAILED TRAINABLE LAYERS BREAKDOWN:")
    if not trainable_layer_details:
        logger.info("  (None)")
    for detail in trainable_layer_details:
        logger.info(detail)

    logger.info("-" * 85)
    logger.info(f"FROZEN LAYERS SUMMARY (Total: {len(frozen_layer_details)} layers):")
    for detail in frozen_layer_details:
        logger.info(detail)
    logger.info("=" * 85)

    empty_prompt_embedding = None
    empty_prompt_path = Path(
        "pretrained_models/prompt_embeddings/"
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
    )
    if empty_prompt_path.exists():
        empty_prompt_embedding = load_file(str(empty_prompt_path))["prompt_embedding"]
        logger.info(f"Loaded empty prompt embedding from {empty_prompt_path}")
    else:
        logger.warning(f"Empty prompt embedding not found at {empty_prompt_path}; prompts encoded on the fly.")

    prompt_cache: Dict[str, torch.Tensor] = {}

    degradation_kwargs = dict(blur_prob=0.8, noise_prob=0.5, jpeg_prob=0.7, video_compress_prob=0.0)

    video_dataset = VideoFileSRDataset(
        video_dir=args.video_dir,
        num_frames=args.num_frames,
        hr_crop_size=tuple(args.crop_size),
        scale=args.upscale,
        is_train=True,
        prompt_json=args.prompt_json,
        degradation_kwargs=degradation_kwargs,
    )
   # image_dataset = ImageFileSRDataset(
   #    image_dir=args.image_dir,
   #     hr_crop_size=tuple(args.crop_size),
   #     scale=args.upscale,
   #     is_train=True,
   #     prompt_json=args.prompt_json,
   #     degradation_kwargs=degradation_kwargs,
   # )

    video_loader = infinite_loader(
        DataLoader(video_dataset, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    )
    # image_loader = infinite_loader(
    #     DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True,
    #                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # )
    logger.info(f"Video clips: {len(video_dataset)}")

    perceptual_loss_fn = PerceptualLoss(args, torch.device(device))  # [FIX 3]

    # Paper lists beta1=0.9, beta2=0.95, beta3=0.98. beta3 is only consumed by
    # Prodigy-style optimizers in the reference's get_optimizer, so plain AdamW
    # correctly takes only the first two.
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)

    global_step = 0
    running_loss, running_mse, running_perc, running_frame = 0.0, 0.0, 0.0, 0.0
    optimizer.zero_grad()
    start_time = time.time()

    total_micro_steps = args.max_train_steps * args.gradient_accumulation_steps
    progress = tqdm(range(total_micro_steps), desc="stage-2")

    for micro_step in progress:
        try:
            t_batch_start = time.time()

            # use_image = random.random() < args.image_ratio
            batch = next(video_loader)

            lr_video = batch["lr"]
            hr_video = batch["hr"]
            prompts = batch["prompt"]

            logger.debug(
                # f"[DATA] Batch {micro_step} ({'IMAGE' if use_image else 'VIDEO'}) "
                f"fetched in {time.time() - t_batch_start:.3f}s | clips={batch['clip_name']}"
            )
            log_tensor_stats(lr_video, "Batch LR (CPU)")
            log_tensor_stats(hr_video, "Batch HR (CPU)")

            # [B, T, C, H, W] -> [B, C, T, H, W]
            lr_video = lr_video.permute(0, 2, 1, 3, 4).contiguous()
            hr_video = hr_video.permute(0, 2, 1, 3, 4).contiguous()

            # [0, 1] -> [-1, 1] for the VAE
            lr_video = lr_video * 2.0 - 1.0
            hr_video = hr_video * 2.0 - 1.0

            lr_video = spatial_upsample_video(
                lr_video, target_hw=(hr_video.shape[3], hr_video.shape[4]), mode="bilinear"
            )

            log_tensor_stats(lr_video, "LR Video (Upsampled)")
            log_tensor_stats(hr_video, "HR Video")

            prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts

            x_sr, x_hr = forward_stage2(
                pipe=pipe,
                lr_video=lr_video,
                hr_video=hr_video,
                prompt=prompt,
                noise_step=args.noise_step,
                sr_noise_step=args.sr_noise_step,
                empty_prompt_embedding=empty_prompt_embedding,
                freeze_vae=freeze_vae,
                autocast_dtype=autocast_dtype,
                prompt_cache=prompt_cache,
            )

            # ================== [FIX 2] LOSS DOMAIN ==================
            # The reference computes EVERY loss term in [0, 1], on clamped
            # tensors. MSE in [-1, 1] would be 4x larger relative to the
            # perceptual term, and skipping the clamp keeps pushing gradient
            # into pixels that can never be displayed.
            video_generate = (x_sr * 0.5 + 0.5).clamp(0.0, 1.0)
            hq_videos = (x_hr * 0.5 + 0.5).clamp(0.0, 1.0)
            # =========================================================

            log_tensor_stats(video_generate, "video_generate ([0,1])")

            mse_loss = F.mse_loss(video_generate.float(), hq_videos.float(), reduction="mean")
            perceptual_loss = perceptual_loss_fn(video_generate, hq_videos)
            frame_diff_loss = frame_difference_loss(video_generate, hq_videos, args.frame_diff_weight)

            loss = mse_loss + perceptual_loss + frame_diff_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"!!! LOSS IS NaN/Inf at step {global_step} !!!")
                log_tensor_stats(video_generate, "NaN_Loss -> video_generate", logging.ERROR)
                log_tensor_stats(hq_videos, "NaN_Loss -> hq_videos", logging.ERROR)

            log_tensor_stats(loss, "Loss")
            logger.debug(
                f"[LOSS] total={loss.item():.6f} | mse={mse_loss.item():.6f} | "
                f"perceptual({perceptual_loss_fn.mode})={float(perceptual_loss):.6f} | "
                f"frame_diff={float(frame_diff_loss):.6f}"
            )

            (loss / args.gradient_accumulation_steps).backward()
            log_memory("After Backward")

            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_perc += float(perceptual_loss)
            running_frame += float(frame_diff_loss)

            if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                logger.info(f"[UPDATING GRADIENTS]: Gradients updated. step {micro_step:06d}")

                if global_step % args.log_steps == 0:
                    n = args.log_steps * args.gradient_accumulation_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    logger.info(
                        f"step {global_step:06d}/{args.max_train_steps} | "
                        f"loss {running_loss / n:.6f} (mse {running_mse / n:.6f}, "
                        f"perc {running_perc / n:.6f}, frame {running_frame / n:.6f}) | "
                        f"grad_norm {grad_norm:.4f} | lr {current_lr:.2e} | elapsed {elapsed / 60:.1f}min"
                    )
                    progress.set_postfix(loss=f"{running_loss / n:.4f}", lr=f"{current_lr:.2e}")
                    running_loss = running_mse = running_perc = running_frame = 0.0

                if global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    if args.use_lora:
                        pipe.transformer.save_pretrained(ckpt_path)
                    else:
                        torch.save(pipe.transformer.state_dict(), os.path.join(ckpt_path, "transformer.pt"))
                    logger.info(f"Saved checkpoint to {ckpt_path}")

        except Exception as e:
            logger.error(f"!!! CRITICAL ERROR in training loop micro_step {micro_step}: {str(e)} !!!")
            raise

    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    if args.use_lora:
        pipe.transformer.save_pretrained(final_path)
    else:
        torch.save(pipe.transformer.state_dict(), os.path.join(final_path, "transformer.pt"))

    logger.info(
        f"Training complete in {(time.time() - start_time) / 60:.1f}min. Final weights saved to {final_path}"
    )


if __name__ == "__main__":
    main()