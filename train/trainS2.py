"""
TRaM-VSR / DOVE Stage-2 training: pixel-space refinement.

Shares degradation, dataset, rotary embeddings and diagnostics with Stage-1
(imported from the Stage-1 module) and follows the paper's reference trainer
(DOVES2Trainer.compute_loss) for everything Stage-2 specific:

  * The VAE encodes/decodes frame by frame; its weights stay frozen but the
    DECODER sits on the loss path, so gradients flow through it.
  * Every loss term is computed in [0, 1] on clamped tensors (see
    `to_unit_range`). MSE in [-1, 1] would be 4x larger relative to the
    perceptual term, and skipping the clamp keeps pushing gradient into pixels
    that can never be displayed.
  * Perceptual loss is pyiqa. The mode is picked by the first weight > 0 in
    the order  ea_dists > dists > ea_lpips > lpips.
  * Trainable params are kept in fp32 and the DiT runs under autocast.

Layout
------
1. Shared Stage-1 imports
2. Logging utilities
3. Frame-by-frame VAE encode/decode
4. Losses
5. Stage-2 forward pass
6. Model setup
7. Validation bridge
8. Data / checkpointing
9. Training loop
10. CLI + entry point

MEMORY NOTE: the VAE decoder runs WITH grad. Decoding many frames at 320x640
with activations retained is expensive; if you OOM, reduce --num_frames or
--crop_size, or try --enable_slicing.
"""
from MajorProject_VSR.patchTransformer import patch_CogVideoXTransformer3DModel
patch_CogVideoXTransformer3DModel()

from __future__ import annotations

import argparse
import importlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from MajorProject_VSR.dataloading.VideoFileLoader import VideoFileSRDataset
from MajorProject_VSR.inference_func.inference import DOVEInferenceFn
from train import log_memory, log_tensor_stats, prepare_rotary_positional_embeddings, spatial_upsample_video
from contextlib import nullcontext

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

import decord  # isort:skip

decord.bridge.set_bridge("torch")

try:
    from MajorProject_VSR.evaluation.validation import validation_pred
except ImportError:  # pragma: no cover
    validation_pred = None

logger = logging.getLogger("tram_vsr_stage2")

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
DEFAULT_EMPTY_PROMPT_PATH = (
    "pretrained_models/prompt_embeddings/"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
)
FREEZE_VAE = True  # Stage 2 always keeps the VAE weights frozen
LOSS_KEYS = ("total", "mse", "perceptual", "frame_diff")

# Loggers that must share handlers: the Stage-1 helpers imported below log to
# THEIR logger, so without this their DEBUG diagnostics never reach our log file.
SHARED_LOGGER_NAMES = ("tram_vsr_stage1", "tram_vsr_stage2")

def autocast_ctx(device, dtype: Optional[torch.dtype]):
    """bf16/fp16 autocast context; a no-op when dtype is None (fp32 run)."""
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)

def encode_text_cached(pipe, prompt: str, device, cache: Optional[Dict[str, torch.Tensor]]):
    if cache is not None and prompt in cache:
        return cache[prompt]
    token_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids
    with torch.no_grad():
        embedding = pipe.text_encoder(token_ids.to(device))[0]
    if cache is not None:
        cache[prompt] = embedding
    return embedding
# --------------------------------------------------------------------------- #
# 2. Logging utilities
# --------------------------------------------------------------------------- #
def setup_logging(output_dir: str, level: str) -> None:
    """Console shows INFO+; the log file receives everything the logger level allows."""
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(os.path.join(output_dir, "train_debug.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    for name in SHARED_LOGGER_NAMES:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(console)
        lg.addHandler(file_handler)
        lg.setLevel(getattr(logging, level))
        lg.propagate = False


def dump_tensors(state: Dict[str, Any], prefix: str) -> None:
    for key, value in state.items():
        if torch.is_tensor(value):
            log_tensor_stats(value, f"{prefix} -> {key}", level=logging.ERROR)


def log_parameter_summary(pipe: CogVideoXPipeline, use_lora: bool, expect_fp32: bool, detailed: bool) -> None:
    """Trainable/frozen counts plus sanity checks. Per-layer lines go to DEBUG (log file only)."""
    total = trainable = lora_trainable = orig_trainable = 0
    non_fp32: List[str] = []

    for name, param in pipe.transformer.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            is_lora = "lora" in name.lower()
            if is_lora:
                lora_trainable += n
            else:
                orig_trainable += n
            if param.dtype != torch.float32:
                non_fp32.append(name)
            tag = "LoRA" if is_lora else "TRAINABLE"
        else:
            tag = "FROZEN"
        if detailed:
            logger.debug(f"  [{tag:<9}] {name:<65} | shape={tuple(param.shape)} | dtype={param.dtype} | {n:,}")

    vae_trainable = sum(p.numel() for p in pipe.vae.parameters() if p.requires_grad)

    logger.info(f"Transformer params: total={total:,}, trainable={trainable:,} "
                f"({100 * trainable / max(total, 1):.4f}%)")
    if use_lora:
        logger.info(f"  LoRA trainable: {lora_trainable:,}")
        if orig_trainable > 0:
            logger.warning(f"  {orig_trainable:,} non-LoRA transformer params are trainable (expected 0)")
    if vae_trainable > 0:
        logger.warning(f"  VAE has {vae_trainable:,} trainable params (Stage 2 must keep the VAE frozen)")
    if expect_fp32 and non_fp32:
        logger.warning(f"  {len(non_fp32)} trainable param(s) are not fp32 (up to 5): {non_fp32[:5]}")


# --------------------------------------------------------------------------- #
# 3. Frame-by-frame VAE encode/decode (Sec. 3.2, Eq. 5)
# --------------------------------------------------------------------------- #
def encode_frames_independently(vae, video: torch.Tensor, scaling_factor: float, freeze_vae: bool) -> torch.Tensor:
    """[B, C, T, H, W] in [-1, 1] -> latent [B, C', T, H', W'] (same layout as Stage-1's vae.encode)."""
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor [B,C,T,H,W], got {tuple(video.shape)}")

    latents = []
    with torch.no_grad() if freeze_vae else torch.enable_grad():
        for t in range(video.shape[2]):
            frame = video[:, :, t : t + 1]
            latents.append(vae.encode(frame).latent_dist.sample() * scaling_factor)
    return torch.cat(latents, dim=2)


def decode_frames_independently(vae, latent: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """[B, T, C', H', W'] (DiT layout) -> pixels [B, C, T, H, W] in roughly [-1, 1].

    Grad ALWAYS flows here: the decoder is on the loss path even though frozen.
    """
    latent = latent.permute(0, 2, 1, 3, 4) / scaling_factor  # [B, C', T, H', W']
    frames = [vae.decode(latent[:, :, t : t + 1]).sample for t in range(latent.shape[2])]
    return torch.cat(frames, dim=2)


def pad_latent_frames(latent: torch.Tensor, patch_size_t: Optional[int]) -> Tuple[torch.Tensor, int]:
    """Left-pad by repeating frame 0 so T is divisible by patch_size_t. Returns (latent, n_pad)."""
    if patch_size_t is None:
        return latent, 0
    n_pad = latent.shape[2] % patch_size_t
    if n_pad > 0:
        latent = torch.cat([latent[:, :, :1].repeat(1, 1, n_pad, 1, 1), latent], dim=2)
    assert latent.shape[2] % patch_size_t == 0
    return latent, n_pad


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> clamped [0, 1]; the domain every Stage-2 loss term is computed in."""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# 4. Losses
# --------------------------------------------------------------------------- #
class SobelEdgeDetectionModel(nn.Module):
    """Fallback for the reference's `EdgeDetectionModel` (used only if that isn't importable).

    Per-channel Sobel gradient magnitude, 3-channel output in [0, 1], fully
    differentiable. NOT guaranteed numerically identical to the reference
    model, so loss weights tuned against one may need retuning for the other.
    """

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("kernel", torch.stack([kx, kx.t().contiguous()]).unsqueeze(1))  # [2,1,3,3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        weight = self.kernel.to(x.dtype).repeat(c, 1, 1, 1)  # [2C, 1, 3, 3]
        g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), weight, groups=c)
        gx, gy = g[:, 0::2], g[:, 1::2]
        return (torch.sqrt(gx * gx + gy * gy + 1e-12) / 4.0).clamp(0.0, 1.0)


def build_edge_model(device: torch.device) -> nn.Module:
    try:
        from finetune.utils.metric_utils import EdgeDetectionModel  # paper's repo

        logger.info("Using the reference EdgeDetectionModel from finetune.utils.metric_utils")
        model = EdgeDetectionModel()
    except ImportError:
        logger.warning("Reference EdgeDetectionModel not importable; falling back to a Sobel edge detector.")
        model = SobelEdgeDetectionModel()
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


# (weight arg, mode name, pyiqa metric, edge-aware). First weight > 0 wins.
_PERCEPTUAL_MODES = (
    ("ea_dists_weight", "ea_dists", "dists", True),
    ("dists_weight", "dists", "dists", False),
    ("ea_lpips_weight", "ea_lpips", "lpips", True),
    ("lpips_weight", "lpips", "lpips", False),
)


class PerceptualLoss:
    """Per-frame pyiqa loss. Edge-aware modes add metric(edge(pred), edge(gt)) and divide by 2F, else F."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.mode: Optional[str] = None
        self.weight = 0.0
        self.metric = None
        self.edge_model: Optional[nn.Module] = None
        self.device = device

        selected = None
        for weight_attr, mode, metric_name, edge_aware in _PERCEPTUAL_MODES:
            weight = getattr(args, weight_attr)
            if weight > 0:
                selected = (mode, weight, metric_name, edge_aware)
                break
        if selected is None:
            logger.warning("All perceptual weights are 0; training with MSE (+ frame-diff) only.")
            return

        self.mode, self.weight, metric_name, edge_aware = selected
        try:
            import pyiqa
        except ImportError as e:
            raise ImportError(
                "Stage-2 needs pyiqa for the perceptual loss (`pip install pyiqa`), "
                "or set all perceptual weights to 0."
            ) from e

        self.metric = pyiqa.create_metric(metric_name, device=device, as_loss=True)
        self.edge_model = build_edge_model(device) if edge_aware else None
        logger.info(f"Perceptual loss: mode={self.mode}, metric={metric_name}, weight={self.weight}")

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Both [B, C, F, H, W] in [0, 1]."""
        if self.metric is None:
            return torch.zeros((), device=pred.device)

        num_frames = pred.shape[2]
        total = 0.0
        for f in range(num_frames):
            p = pred[:, :, f].to(dtype=torch.float32, device=self.device)
            g = target[:, :, f].to(dtype=torch.float32, device=self.device)
            # .mean() is a no-op for a scalar metric and reduces a per-sample one, so backward() always gets a scalar.
            total = total + self.metric(p, g).mean()
            if self.edge_model is not None:
                total = total + self.metric(self.edge_model(p), self.edge_model(g)).mean()

        divisor = num_frames * 2 if self.edge_model is not None else num_frames
        return (total / divisor) * self.weight


def frame_difference_loss(pred: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
    """Eq. 6: L1 between consecutive-frame differences. Inputs [B, C, F, H, W] in [0, 1].

    Gated on the decoded frame count, like the reference.
    """
    if pred.shape[2] < 2:
        return torch.zeros((), device=pred.device)
    pred, target = pred.float(), target.float()
    diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
    diff_target = target[:, :, 1:] - target[:, :, :-1]
    return F.l1_loss(diff_pred, diff_target) * weight


def compute_stage2_losses(
    x_sr: torch.Tensor,
    x_hr: torch.Tensor,
    perceptual_fn: PerceptualLoss,
    frame_diff_weight: float,
) -> Dict[str, torch.Tensor]:
    """x_sr / x_hr in [-1, 1]. Returns total/mse/perceptual/frame_diff."""
    pred = to_unit_range(x_sr)
    target = to_unit_range(x_hr)
    log_tensor_stats(pred, "video_generate ([0,1])")

    mse = F.mse_loss(pred.float(), target.float(), reduction="mean")
    perceptual = perceptual_fn(pred, target)
    frame_diff = frame_difference_loss(pred, target, frame_diff_weight)
    return {"total": mse + perceptual + frame_diff, "mse": mse, "perceptual": perceptual, "frame_diff": frame_diff}


# --------------------------------------------------------------------------- #
# 5. Stage-2 forward pass
# --------------------------------------------------------------------------- #
def resolve_prompt_embedding(
    pipe: CogVideoXPipeline,
    prompt: str,
    batch_size: int,
    empty_prompt_embedding: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    prompt_cache: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Embedding for `prompt`, broadcast across the batch."""
    if prompt == "" and empty_prompt_embedding is not None:
        embedding = empty_prompt_embedding.to(device, dtype=dtype)

    if embedding.shape[0] != batch_size:
        if embedding.shape[0] != 1:
            raise ValueError(f"Cannot broadcast prompt embedding {tuple(embedding.shape)} to batch {batch_size}")
        embedding = embedding.repeat(batch_size, 1, 1)
    return embedding


def forward_stage2(
    pipe: CogVideoXPipeline,
    lr_video: torch.Tensor,
    hr_video: Optional[torch.Tensor],
    prompt: str,
    noise_step: int,
    sr_noise_step: int,
    empty_prompt_embedding: Optional[torch.Tensor],
    freeze_vae: bool,
    autocast_dtype: Optional[torch.dtype] = None,
    prompt_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    lr_video / hr_video: [B, C, T, H, W] in [-1, 1], LR already upsampled to HR size.
    `hr_video` may be None (inference): the output shape check is then skipped.

    Returns (x_sr, hr_video) in [-1, 1]; the caller maps both to [0, 1] for the loss.
    """
    try:
        logger.debug("--- [FORWARD STAGE 2 START] ---")
        log_tensor_stats(lr_video, "Input LR Video")
        log_tensor_stats(hr_video, "Input HR Video")
        log_memory("Before VAE Encode")

        device, dtype = pipe.vae.device, pipe.vae.dtype
        scaling_factor = pipe.vae.config.scaling_factor

        lr_video = lr_video.to(device, dtype=dtype)
        if hr_video is not None:
            hr_video = hr_video.to(device, dtype=dtype)
        num_input_frames = lr_video.shape[2]

        lr_latent = encode_frames_independently(pipe.vae, lr_video, scaling_factor, freeze_vae)
        transformer_config = pipe.transformer.config
        lr_latent, n_pad = pad_latent_frames(lr_latent, transformer_config.patch_size_t)
        batch_size, _, num_latent_frames, latent_h, latent_w = lr_latent.shape
        log_tensor_stats(lr_latent, "LR Latent (padded)")

        prompt_embedding = resolve_prompt_embedding(
            pipe, prompt, batch_size, empty_prompt_embedding, device, dtype, prompt_cache
        )
        log_tensor_stats(prompt_embedding, "Prompt Embedding")

        lr_latent = lr_latent.permute(0, 2, 1, 3, 4)  # [B, T', C', H', W']

        if noise_step != 0:
            noise = torch.randn_like(lr_latent)
            add_timesteps = torch.full((batch_size,), noise_step, dtype=torch.long, device=device)
            lr_latent = pipe.scheduler.add_noise(lr_latent, noise, add_timesteps)

        timesteps = torch.full((batch_size,), sr_noise_step, dtype=torch.long, device=device)

        rotary_emb = None
        if transformer_config.use_rotary_positional_embeddings:
            scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
            rotary_emb = prepare_rotary_positional_embeddings(
                height=latent_h * scale,
                width=latent_w * scale,
                num_frames=num_latent_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=scale,
                device=device,
            )

        log_tensor_stats(lr_latent, "LR Latent (DiT Input)")
        log_memory("Before DiT Forward")

        with autocast_ctx(device, autocast_dtype):
            model_output = pipe.transformer(
                hidden_states=lr_latent,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]
        log_tensor_stats(model_output, "DiT Output")

        sr_latent = pipe.scheduler.get_velocity(model_output, lr_latent, timesteps)
        log_tensor_stats(sr_latent, "SR Latent (x0)")

        if n_pad > 0:
            sr_latent = sr_latent[:, n_pad:]
        if sr_latent.shape[1] != num_input_frames:
            raise RuntimeError(
                f"Frame count mismatch after de-padding: latent has {sr_latent.shape[1]} frames, "
                f"expected {num_input_frames}."
            )

        log_memory("Before VAE Decode")
        x_sr = decode_frames_independently(pipe.vae, sr_latent, scaling_factor)
        log_tensor_stats(x_sr, "x_sr (decoded pixels, [-1,1])")

        if hr_video is not None and x_sr.shape != hr_video.shape:
            raise RuntimeError(f"Shape mismatch: reconstructed {tuple(x_sr.shape)} vs target {tuple(hr_video.shape)}")

        logger.debug("--- [FORWARD STAGE 2 END] ---")
        return x_sr, hr_video

    except Exception as e:
        logger.error(f"!!! CRITICAL ERROR in forward_stage2: {e} !!!")
        dump_tensors(locals(), "CRASH_STATE")
        raise


# --------------------------------------------------------------------------- #
# 6. Model setup
# --------------------------------------------------------------------------- #
def load_pipeline(args: argparse.Namespace, dtype: torch.dtype, device: str) -> CogVideoXPipeline:
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)

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
    return pipe


def load_full_transformer_weights(pipe: CogVideoXPipeline, init_from: str) -> None:
    ckpt_path = os.path.join(init_from, "transformer.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"--init_from given but no transformer.pt found at {ckpt_path}")

    logger.info(f"Loading stage-1 transformer weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"{len(missing)} missing key(s) loading stage-1 ckpt (up to 5): {missing[:5]}")
    if unexpected:
        logger.warning(f"{len(unexpected)} unexpected key(s) loading stage-1 ckpt (up to 5): {unexpected[:5]}")


def setup_trainable_modules(
    pipe: CogVideoXPipeline, args: argparse.Namespace, dtype: torch.dtype
) -> List[torch.nn.Parameter]:
    """Freeze text encoder + VAE; set up full-FT or LoRA on the transformer; return trainable params."""
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
                init_lora_weights=True,
            )
            pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    else:
        if args.init_from:
            load_full_transformer_weights(pipe, args.init_from)
        pipe.transformer.requires_grad_(True)

    if args.fp32_master_weights and dtype != torch.float32:
        cast_training_params([pipe.transformer], dtype=torch.float32)
        if not args.use_lora:
            logger.warning("Full fine-tune with fp32 master weights roughly doubles weight memory; "
                           "use --use_lora or --no_fp32_master_weights if you OOM.")
        logger.info(f"Trainable params cast to fp32; compute runs under autocast({dtype}).")

    pipe.transformer.train()
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled (transformer only; VAE decoder is not checkpointed).")

    return [p for p in pipe.transformer.parameters() if p.requires_grad]


def load_empty_prompt_embedding(path: str) -> Optional[torch.Tensor]:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Empty prompt embedding not found at {p}; prompts encoded on the fly.")
        return None
    logger.info(f"Loaded empty prompt embedding from {p}")
    return load_file(str(p))["prompt_embedding"]


# --------------------------------------------------------------------------- #
# 7. Validation bridge (validation.py)
# --------------------------------------------------------------------------- #
def make_stage2_inference_fn(
    pipe: CogVideoXPipeline,
    args: argparse.Namespace,
    autocast_dtype: Optional[torch.dtype],
    empty_prompt_embedding: Optional[torch.Tensor],
    prompt_cache: Dict[str, torch.Tensor],
) -> Callable:
    """Builds `inference_fn(model, lr_video, device) -> pred_video` for validation.generate_sr_videos.

    lr_video / pred_video: [F, C, H, W] float32 in [0, 1]. The `model` argument
    is ignored: inference runs the real Stage-2 forward pass through `pipe`, so
    it matches training exactly (same noise / prompt settings).
    """

    def inference_fn(model, lr_video: torch.Tensor, device: torch.device) -> torch.Tensor:
        lr = lr_video.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [1, C, F, H, W]
        lr = (lr * 2.0 - 1.0).to(device)
        _, _, _, h, w = lr.shape
        lr = spatial_upsample_video(lr, target_hw=(h * args.upscale, w * args.upscale), mode="bilinear")

        was_training = pipe.transformer.training
        pipe.transformer.eval()
        try:
            with torch.no_grad():
                x_sr, _ = forward_stage2(
                    pipe=pipe,
                    lr_video=lr,
                    hr_video=None,
                    prompt="",
                    noise_step=args.noise_step,
                    sr_noise_step=args.sr_noise_step,
                    empty_prompt_embedding=empty_prompt_embedding,
                    freeze_vae=FREEZE_VAE,
                    autocast_dtype=autocast_dtype,
                    prompt_cache=prompt_cache,
                )
        finally:
            if was_training:
                pipe.transformer.train()

        return to_unit_range(x_sr).squeeze(0).permute(1, 0, 2, 3).contiguous()  # [F, C, H, W]

    return inference_fn


def run_periodic_validation(
    pipe: CogVideoXPipeline,
    args: argparse.Namespace,
    global_step: int,
    autocast_dtype: Optional[torch.dtype],
    empty_prompt_embedding: Optional[torch.Tensor],
    prompt_cache: Dict[str, torch.Tensor],
) -> None:
    """Runs validation.validation_pred on --val_lr_dir/--val_gt_dir, if configured."""
    if not args.val_lr_dir or not args.val_gt_dir:
        return
    if validation_pred is None:
        logger.warning("validation.py (or its metric_utils dependency) not importable; skipping validation.")
        return

    inference_fn = DOVEInferenceFn(
        pipe=pipe, 
        upscale=4,
        upscale_mode="bilinear",
        noise_step=args.noise_step, 
        sr_noise_step=args.sr_noise_step,
        empty_prompt_embedding=empty_prompt_embedding,
    )   

    pipe.transformer.eval()
    eval_metrics = [m.strip().lower() for m in args.val_metrics.split(",") if m.strip()]
    pred_dir = os.path.join(args.output_dir, "val_preds", f"step_{global_step}")

    logger.info(f"[VALIDATION] step {global_step}: generating + scoring against {args.val_gt_dir}")
    try:
        report = validation_pred(
            pred_path=pred_dir,
            gt_path=args.val_gt_dir,
            eval_metrics=eval_metrics,
            model=pipe.transformer,
            lr_path=args.val_lr_dir,
            inference_fn=inference_fn,
            output_json=os.path.join(pred_dir, "validation_metrics.json"),
            fps=args.val_fps,
            overwrite_preds=True,
        )
        logger.info(f"[VALIDATION] step {global_step}: {report['average']}")
    except Exception as e:
        logger.error(f"[VALIDATION] step {global_step} failed: {e}")
    finally:
        pipe.transformer.train()  # validation leaves the model in eval mode


# --------------------------------------------------------------------------- #
# 8. Data / checkpointing
# --------------------------------------------------------------------------- #
def build_batch_iterator(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    """Endless, reshuffled-per-pass iterator over degraded video clips."""
    dataset = VideoFileSRDataset(
        video_dir=args.video_dir,
        num_frames=args.num_frames,
        hr_crop_size=tuple(args.crop_size),
        scale=args.upscale,
        is_train=True,
        prompt_json=args.prompt_json,
        degradation_kwargs=dict(
            blur_prob=args.blur_prob,
            noise_prob=args.noise_prob,
            jpeg_prob=args.jpeg_prob,
            video_compress_prob=args.video_compress_prob,
        ),
    )
    logger.info(f"Video clips: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    while True:
        yield from loader


def prepare_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Dataset [B,T,C,H,W] in [0,1] -> model layout [B,C,T,H,W] in [-1,1], LR upsampled to HR size."""
    lr_video, hr_video, prompts = batch["lr"], batch["hr"], batch["prompt"]
    log_tensor_stats(lr_video, "Batch LR (CPU)")
    log_tensor_stats(hr_video, "Batch HR (CPU)")

    lr_video = lr_video.permute(0, 2, 1, 3, 4).contiguous() * 2.0 - 1.0
    hr_video = hr_video.permute(0, 2, 1, 3, 4).contiguous() * 2.0 - 1.0
    lr_video = spatial_upsample_video(lr_video, target_hw=(hr_video.shape[3], hr_video.shape[4]), mode="bilinear")

    log_tensor_stats(lr_video, "LR Video (upsampled)")
    log_tensor_stats(hr_video, "HR Video")

    # NOTE: only the first prompt is used for the whole batch.
    prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
    return lr_video, hr_video, prompt


def save_checkpoint(pipe: CogVideoXPipeline, path: str, use_lora: bool) -> None:
    os.makedirs(path, exist_ok=True)
    if use_lora:
        pipe.transformer.save_pretrained(path)
    else:
        torch.save(pipe.transformer.state_dict(), os.path.join(path, "transformer.pt"))
    logger.info(f"Saved checkpoint to {path}")


# --------------------------------------------------------------------------- #
# 9. Training loop
# --------------------------------------------------------------------------- #
def compute_losses(
    pipe: CogVideoXPipeline,
    batch: Dict[str, Any],
    args: argparse.Namespace,
    perceptual_fn: PerceptualLoss,
    autocast_dtype: Optional[torch.dtype],
    empty_prompt_embedding: Optional[torch.Tensor],
    prompt_cache: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    lr_video, hr_video, prompt = prepare_batch(batch)
    x_sr, x_hr = forward_stage2(
        pipe=pipe,
        lr_video=lr_video,
        hr_video=hr_video,
        prompt=prompt,
        noise_step=args.noise_step,
        sr_noise_step=args.sr_noise_step,
        empty_prompt_embedding=empty_prompt_embedding,
        freeze_vae=FREEZE_VAE,
        autocast_dtype=autocast_dtype,
        prompt_cache=prompt_cache,
    )
    losses = compute_stage2_losses(x_sr, x_hr, perceptual_fn, args.frame_diff_weight)

    if not torch.isfinite(losses["total"]):
        logger.error("!!! Loss is NaN/Inf !!!")
        log_tensor_stats(x_sr, "NaN_Loss -> x_sr", logging.ERROR)
        log_tensor_stats(x_hr, "NaN_Loss -> x_hr", logging.ERROR)
    log_tensor_stats(losses["total"], "Loss")
    logger.debug(
        f"[LOSS] total={losses['total'].item():.6f} | mse={losses['mse'].item():.6f} | "
        f"perceptual({perceptual_fn.mode})={float(losses['perceptual']):.6f} | "
        f"frame_diff={float(losses['frame_diff']):.6f}"
    )
    return losses


def train(
    pipe: CogVideoXPipeline,
    batches: Iterator[Dict[str, Any]],
    trainable_params: List[torch.nn.Parameter],
    perceptual_fn: PerceptualLoss,
    args: argparse.Namespace,
    autocast_dtype: Optional[torch.dtype],
    empty_prompt_embedding: Optional[torch.Tensor],
    prompt_cache: Dict[str, torch.Tensor],
) -> None:
    accum = args.gradient_accumulation_steps

    # The paper lists beta3=0.98 as well, but that is only consumed by
    # Prodigy-style optimizers in the reference; plain AdamW takes two betas.
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)

    running = {key: 0.0 for key in LOSS_KEYS}
    global_step = 0
    start_time = time.time()
    optimizer.zero_grad()

    progress = tqdm(range(args.max_train_steps * accum), desc="stage-2")
    for micro_step in progress:
        try:
            t_fetch = time.time()
            batch = next(batches)
            logger.debug(f"[DATA] micro_step {micro_step}: fetched in {time.time() - t_fetch:.3f}s | "
                         f"clips={batch.get('clip_name')}")

            losses = compute_losses(pipe, batch, args, perceptual_fn, autocast_dtype,
                                    empty_prompt_embedding, prompt_cache)
            (losses["total"] / accum).backward()
            log_memory("After Backward")
        except Exception as e:
            logger.error(f"!!! CRITICAL ERROR in training loop micro_step {micro_step}: {e} !!!")
            raise

        for key in LOSS_KEYS:
            running[key] += float(losses[key])

        if (micro_step + 1) % accum != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        logger.debug(f"[OPT] optimizer step {global_step} (micro_step {micro_step})")

        if global_step % args.log_steps == 0:
            n = args.log_steps * accum
            avg = {key: value / n for key, value in running.items()}
            current_lr = lr_scheduler.get_last_lr()[0]
            logger.info(
                f"step {global_step:06d}/{args.max_train_steps} | "
                f"loss {avg['total']:.6f} (mse {avg['mse']:.6f}, perc {avg['perceptual']:.6f}, "
                f"frame {avg['frame_diff']:.6f}) | grad_norm {float(grad_norm):.4f} | "
                f"lr {current_lr:.2e} | elapsed {(time.time() - start_time) / 60:.1f}min"
            )
            progress.set_postfix(loss=f"{avg['total']:.4f}", lr=f"{current_lr:.2e}")
            running = {key: 0.0 for key in LOSS_KEYS}

        if global_step % args.save_steps == 0:
            save_checkpoint(pipe, os.path.join(args.output_dir, f"checkpoint-{global_step}"), args.use_lora)

        if args.val_steps > 0 and global_step % args.val_steps == 0:
            run_periodic_validation(pipe, args, global_step, autocast_dtype, empty_prompt_embedding, prompt_cache)

    save_checkpoint(pipe, os.path.join(args.output_dir, "final"), args.use_lora)
    logger.info(f"Training complete in {(time.time() - start_time) / 60:.1f}min.")


# --------------------------------------------------------------------------- #
# 10. CLI + entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TRaM-VSR / DOVE Stage-2 training (pixel-space refinement, on-the-fly degradation)"
    )

    # Paths
    p.add_argument("--video_dir", type=str, required=True, help="HR video clips; LR generated on the fly")
    p.add_argument("--prompt_json", type=str, default=None)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--init_from", type=str, default=None,
                   help="Stage-1 checkpoint dir (transformer.pt, or the LoRA adapter dir with --use_lora)")
    p.add_argument("--output_dir", type=str, default="./stage2_ckpt")
    p.add_argument("--empty_prompt_embedding", type=str, default=DEFAULT_EMPTY_PROMPT_PATH)

    # Data
    p.add_argument("--num_frames", type=int, default=9, help="Clip length; must satisfy (F-1) %% 8 == 0")
    p.add_argument("--crop_size", type=int, nargs=2, default=(320, 640))
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--blur_prob", type=float, default=0.8)
    p.add_argument("--noise_prob", type=float, default=0.5)
    p.add_argument("--jpeg_prob", type=float, default=0.7)
    p.add_argument("--video_compress_prob", type=float, default=0.0)

    # Diffusion
    p.add_argument("--noise_step", type=int, default=0)
    p.add_argument("--sr_noise_step", type=int, default=399)

    # Loss weights (first perceptual weight > 0 wins, in this order)
    p.add_argument("--ea_dists_weight", type=float, default=1.0, help="Edge-aware DISTS (paper's default)")
    p.add_argument("--dists_weight", type=float, default=0.0)
    p.add_argument("--ea_lpips_weight", type=float, default=0.0)
    p.add_argument("--lpips_weight", type=float, default=0.0)
    p.add_argument("--frame_diff_weight", type=float, default=1.0)

    # Optimisation
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=500, help="Optimizer steps (paper: 500)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Model
    p.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES))
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--target_modules", type=str, nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])
    p.add_argument("--fp32_master_weights", action="store_true", default=True)
    p.add_argument("--no_fp32_master_weights", dest="fp32_master_weights", action="store_false")
    p.add_argument("--enable_slicing", action="store_true")
    p.add_argument("--enable_tiling", action="store_true")

    p.add_argument("--drop_ratio", type=float, default=0.35)
    p.add_argument("--block_intervals", type=int, nargs="+", default=[20, 25, 32, 37])

    # Checkpointing / logging
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING"],
                   help="DEBUG enables per-tensor stats (GPU syncs); use INFO for speed.")
    p.add_argument("--param_report", type=str, default="summary", choices=["none", "summary", "detailed"],
                   help="'detailed' writes a per-layer table to the log file.")

    # Periodic validation (skipped unless both dirs are set)
    p.add_argument("--val_lr_dir", type=str, default=None, help="LR videos for periodic validation")
    p.add_argument("--val_gt_dir", type=str, default=None, help="GT videos matching --val_lr_dir by filename stem")
    p.add_argument("--val_metrics", type=str, default="psnr,ssim,lpips,dists")
    p.add_argument("--val_steps", type=int, default=100, help="Validate every N optimizer steps (0 disables)")
    p.add_argument("--val_fps", type=int, default=8)

    args = p.parse_args()
    if (args.num_frames - 1) % 8 != 0:
        p.error(f"--num_frames must satisfy (F - 1) % 8 == 0, got {args.num_frames}")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir, args.log_level)

    dtype = DTYPES[args.dtype]
    autocast_dtype = dtype if (args.fp32_master_weights and dtype != torch.float32) else None

    logger.info("=" * 64)
    logger.info("TRaM-VSR Stage-2 training (pixel-space refinement)")
    for key, value in vars(args).items():
        logger.info(f"  {key:<28}: {value}")
    logger.info(f"  {'autocast_dtype':<28}: {autocast_dtype}")
    if not (args.val_lr_dir and args.val_gt_dir):
        logger.info("  validation disabled (set --val_lr_dir and --val_gt_dir to enable)")
    logger.info("=" * 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available; training on CPU will be extremely slow.")

    logger.info(f"Loading CogVideoX pipeline from {args.model_path} ...")
    pipe = load_pipeline(args, dtype, device)
    
    trainable_params = setup_trainable_modules(pipe, args, dtype)
    if args.param_report != "none":
        log_parameter_summary(pipe, args.use_lora, args.fp32_master_weights and dtype != torch.float32,
                              detailed=args.param_report == "detailed")

    empty_prompt_embedding = load_empty_prompt_embedding(args.empty_prompt_embedding)
    prompt_cache: Dict[str, torch.Tensor] = {}
    batches = build_batch_iterator(args)
    perceptual_fn = PerceptualLoss(args, torch.device(device))

    train(pipe, batches, trainable_params, perceptual_fn, args, autocast_dtype, empty_prompt_embedding, prompt_cache)


if __name__ == "__main__":
    main()