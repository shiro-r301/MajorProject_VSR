"""
TRaM-VSR / DOVE Stage-1 training with on-the-fly real-world degradation.

Layout
------
1. Logging utilities
2. Video / tensor utilities
3. Stage-1 forward pass (VAE encode -> DiT -> x0 prediction)
4. Model setup (pipeline, TCG module, freezing, LoRA)
5. Data setup
6. Checkpointing
7. Training loop
8. CLI + entry point
"""

from __future__ import annotations

from MajorProject_VSR.patchTransformer import patch_CogVideoXTransformer3DModel
patch_CogVideoXTransformer3DModel()

import argparse
import logging
import os
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.training_utils import cast_training_params
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

import decord  # isort:skip
from MajorProject_VSR.dataloading.VideoFileLoader import VideoFileSRDataset
from MajorProject_VSR.evaluation.validation import validation_pred
from MajorProject_VSR.inference_func.inference import DOVEInferenceFn

decord.bridge.set_bridge("torch")  # dataset loader expects torch tensors from decord

logger = logging.getLogger("tram_vsr_stage1")

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
DEFAULT_EMPTY_PROMPT_PATH = (
    "pretrained_models/prompt_embeddings/"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
)
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


# --------------------------------------------------------------------------- #
# 1. Logging utilities
# --------------------------------------------------------------------------- #
def setup_logging(output_dir: str, level: str) -> None:
    """Console shows INFO+; the log file receives everything the logger level allows."""
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(os.path.join(output_dir, "train_debug.log"), mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.setLevel(getattr(logging, level))
    logger.propagate = False


def log_tensor_stats(tensor: Optional[torch.Tensor], name: str, level: int = logging.DEBUG) -> None:
    """Log shape/dtype/device and value distribution; flag NaN/Inf.

    Returns immediately if `level` is disabled, so no GPU sync is paid
    when running with --log_level INFO.
    """
    if not logger.isEnabledFor(level):
        return
    if tensor is None:
        logger.log(level, f"[{name}] is None")
        return
    if not torch.is_tensor(tensor):
        logger.log(level, f"[{name}] is not a tensor: {type(tensor)}")
        return

    meta = f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    t = tensor.detach().float()
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    if n_nan or n_inf:
        logger.log(level, f"[{name}] {meta} | !! NaN={n_nan}, Inf={n_inf} !!")
        return

    std = t.std().item() if t.numel() > 1 else 0.0
    logger.log(
        level,
        f"[{name}] {meta} | min={t.min().item():.4f}, max={t.max().item():.4f}, "
        f"mean={t.mean().item():.4f}, std={std:.4f}",
    )


def log_memory(tag: str) -> None:
    if not (torch.cuda.is_available() and logger.isEnabledFor(logging.DEBUG)):
        return
    gb = 1024**3
    logger.debug(
        f"[MEM] {tag}: alloc={torch.cuda.memory_allocated() / gb:.2f}GB, "
        f"reserved={torch.cuda.memory_reserved() / gb:.2f}GB, "
        f"max_alloc={torch.cuda.max_memory_allocated() / gb:.2f}GB"
    )


def dump_tensors(state: Dict[str, Any], prefix: str) -> None:
    """Log stats of every tensor in `state` at ERROR level (used on crashes)."""
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
# 2. Video / tensor utilities
# --------------------------------------------------------------------------- #
def spatial_upsample_video(video: torch.Tensor, target_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """Per-frame 2D resize of [B, C, T, H, W] -> [B, C, T, target_H, target_W]."""
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor [B,C,T,H,W], got shape {tuple(video.shape)}")

    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames = F.interpolate(
        frames,
        size=tuple(target_hw),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )
    frames = frames.reshape(b, t, c, *target_hw)
    return frames.permute(0, 2, 1, 3, 4).contiguous()


def prepare_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Dataset [B,T,C,H,W] in [0,1] -> model layout [B,C,T,H,W] in [-1,1], LR upsampled to HR size."""
    lr_video, hr_video, prompts = batch["lr"], batch["hr"], batch["prompt"]
    log_tensor_stats(lr_video, "Batch LR (CPU)")
    log_tensor_stats(hr_video, "Batch HR (CPU)")

    lr_video = lr_video.permute(0, 2, 1, 3, 4).contiguous() * 2.0 - 1.0
    hr_video = hr_video.permute(0, 2, 1, 3, 4).contiguous() * 2.0 - 1.0
    lr_video = spatial_upsample_video(lr_video, hr_video.shape[-2:], mode="bilinear")

    log_tensor_stats(lr_video, "LR Video (Upsampled)")
    log_tensor_stats(hr_video, "HR Video")

    # NOTE: only the first prompt is used for the whole batch.
    prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
    return lr_video, hr_video, prompt


# --------------------------------------------------------------------------- #
# 3. Stage-1 forward pass
# --------------------------------------------------------------------------- #
def vae_spatial_scale(vae) -> int:
    return 2 ** (len(vae.config.block_out_channels) - 1)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Any,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE for pixel-space (height, width) and *latent* num_frames."""
    patch = transformer_config.patch_size
    patch_t = transformer_config.patch_size_t

    grid_h = height // (vae_scale_factor_spatial * patch)
    grid_w = width // (vae_scale_factor_spatial * patch)
    base_h = transformer_config.sample_height // patch
    base_w = transformer_config.sample_width // patch
    base_frames = num_frames if patch_t is None else (num_frames + patch_t - 1) // patch_t

    return get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_h, grid_w),
        temporal_size=base_frames,
        grid_type="slice",
        max_size=(base_h, base_w),
        device=device,
    )


def encode_video(vae, video: torch.Tensor) -> torch.Tensor:
    """[B,C,T,H,W] pixels -> scaled latent sample [B,C',T',H',W']."""
    return vae.encode(video).latent_dist.sample() * vae.config.scaling_factor


def pad_latent_frames(latent: torch.Tensor, patch_size_t: Optional[int]) -> Tuple[torch.Tensor, int]:
    """Left-pad by repeating frame 0 so T is divisible by patch_size_t. Returns (latent, n_pad)."""
    if patch_size_t is None:
        return latent, 0
    n_pad = latent.shape[2] % patch_size_t
    if n_pad > 0:
        latent = torch.cat([latent[:, :, :1].repeat(1, 1, n_pad, 1, 1), latent], dim=2)
    return latent, n_pad


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
        embedding = empty_prompt_embedding
    elif getattr(pipe, "text_encoder", None) is not None:
        embedding = encode_text_cached(pipe, prompt, device, prompt_cache)
    else:
        raise RuntimeError(
            f"Non-empty prompt {prompt!r} but the text encoder isn't loaded (text_encoder=None) "
            "and there's no cached embedding for it. Use empty prompts or cache embeddings."
        )

    embedding = embedding.to(device, dtype=dtype)
    if embedding.ndim == 2:                      # [seq, dim] -> [1, seq, dim]
        embedding = embedding.unsqueeze(0)
    if embedding.shape[0] != batch_size:
        if embedding.shape[0] != 1:
            raise ValueError(f"Cannot broadcast prompt embedding {tuple(embedding.shape)} to batch {batch_size}")
        embedding = embedding.expand(batch_size, -1, -1)
    return embedding



def forward_stage1(
    pipe: CogVideoXPipeline,
    lr_video: torch.Tensor,
    hr_video: torch.Tensor,
    prompt: str,
    noise_step: int,
    sr_noise_step: int,
    empty_prompt_embedding: Optional[torch.Tensor],
    freeze_vae: bool,
    autocast_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (predicted_latent, target_latent), both [B, T', C', H', W']."""
    try:
        logger.debug("--- [FORWARD STAGE 1 START] ---")
        log_tensor_stats(lr_video, "Input LR Video")
        log_tensor_stats(hr_video, "Input HR Video")
        log_memory("Before VAE Encode")

        device, dtype = pipe.vae.device, pipe.vae.dtype
        lr_video = lr_video.to(device, dtype=dtype)
        hr_video = hr_video.to(device, dtype=dtype)

        # Target: HR latent (never needs grad)
        with torch.no_grad():
            target_latent = encode_video(pipe.vae, hr_video).permute(0, 2, 1, 3, 4)
        log_tensor_stats(target_latent, "Target Latent (HR)")

        # Input: LR latent (grad only if the VAE is being trained)
        with torch.no_grad() if freeze_vae else torch.enable_grad():
            lr_latent = encode_video(pipe.vae, lr_video)

        transformer_config = pipe.transformer.config
        lr_latent, n_pad = pad_latent_frames(lr_latent, transformer_config.patch_size_t)
        batch_size, _, num_frames, latent_h, latent_w = lr_latent.shape
        log_tensor_stats(lr_latent, "LR Latent (padded)")

        prompt_embedding = resolve_prompt_embedding(pipe, prompt, batch_size, empty_prompt_embedding, device, dtype, prompt_cache={})
        log_tensor_stats(prompt_embedding, "Prompt Embedding")

        lr_latent = lr_latent.permute(0, 2, 1, 3, 4)  # -> [B, T', C', H', W']

        if noise_step != 0:
            noise = torch.randn_like(lr_latent)
            add_timesteps = torch.full((batch_size,), noise_step, dtype=torch.long, device=device)
            lr_latent = pipe.scheduler.add_noise(lr_latent, noise, add_timesteps)

        timesteps = torch.full((batch_size,), sr_noise_step, dtype=torch.long, device=device)

        rotary_emb = None
        if transformer_config.use_rotary_positional_embeddings:
            scale = vae_spatial_scale(pipe.vae)
            rotary_emb = prepare_rotary_positional_embeddings(
                height=latent_h * scale,
                width=latent_w * scale,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=scale,
                device=device,
            )

        log_tensor_stats(lr_latent, "LR Latent (DiT Input)")
        log_memory("Before DiT Forward")

        with autocast_ctx(device=pipe.device, dtype=autocast_dtype):
            model_output = pipe.transformer(
                hidden_states=lr_latent,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]
        log_tensor_stats(model_output, "DiT Output")

        pred_latent = pipe.scheduler.get_velocity(model_output, lr_latent, timesteps)
        log_tensor_stats(pred_latent, "Predicted Latent (x0)")

        if n_pad > 0:
            pred_latent = pred_latent[:, n_pad:]

        if pred_latent.shape != target_latent.shape:
            raise RuntimeError(f"Shape mismatch: predicted {tuple(pred_latent.shape)} vs target {tuple(target_latent.shape)}")

        logger.debug("--- [FORWARD STAGE 1 END] ---")
        return pred_latent, target_latent

    except Exception as e:
        logger.error(f"!!! CRITICAL ERROR in forward_stage1: {e} !!!")
        dump_tensors(locals(), "CRASH_STATE")
        raise

def run_periodic_validation(pipe, args, global_step, empty_prompt_embedding):
    
    if not (args.val_lr_dir and args.val_gt_dir):
        return
    fn = DOVEInferenceFn(
        pipe=pipe, upscale=args.upscale, upscale_mode="bilinear",
        noise_step=args.noise_step, sr_noise_step=args.sr_noise_step,
        empty_prompt_embedding=empty_prompt_embedding,
        # tile_size_hw=(320, 640), overlap_hw=(32, 32),  # optional, lower memory
    )
    tcg_log = logging.getLogger("tcg_logger")  # use the real name from the transformer file
    prev_level = tcg_log.level
    tcg_log.setLevel(logging.WARNING)
    pipe.transformer.eval()
    torch.cuda.empty_cache()
    pred_dir = os.path.join(args.output_dir, "val_preds", f"step_{global_step}")
    try:
        with torch.no_grad():
            report = validation_pred(
                pred_path=pred_dir, gt_path=args.val_gt_dir,
                eval_metrics=[m.strip().lower() for m in args.val_metrics.split(",") if m.strip()],
                model=pipe.transformer, lr_path=args.val_lr_dir, inference_fn=fn,
                output_json=os.path.join(pred_dir, "validation_metrics.json"),
                fps=args.val_fps, overwrite_preds=True,
            )
        logger.info(f"[VALIDATION] step {global_step}: {report['average']}")
    except Exception as e:
        logger.error(f"[VALIDATION] step {global_step} failed: {e}")
    finally:
        pipe.transformer.train()
        tcg_log.setLevel(prev_level)
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
# 4. Model setup
# --------------------------------------------------------------------------- #
def load_pipeline(model_path: str, dtype: torch.dtype, device: str, args) -> CogVideoXPipeline:
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype, tokenizer=None, text_encoder=None)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    if args.enable_slicing:
        pipe.vae.enable_slicing()
        logger.info("VAE slicing enabled.")
    if args.enable_tiling:
        pipe.vae.enable_tiling()
        logger.info("VAE tiling enabled.")

    return pipe.to(device)


def configure_tcg_module(pipe: CogVideoXPipeline, drop_ratio: float, block_intervals: List[int]) -> None:
    """Set token-merge/unmerge layers on the transformer.

    Must run BEFORE the transformer is wrapped by PEFT so the attributes land
    on the base module whose forward reads them.
    """
    intervals = sorted(block_intervals)
    pipe.transformer.tcg.drop_ratio = drop_ratio
    pipe.transformer.merger_layers = intervals[::2]
    pipe.transformer.unmerger_layers = intervals[1::2]
    logger.info(f"TCG MODULE CONFIGURATIONS:- Drop Ratio: {pipe.transformer.tcg.drop_ratio}, \
                Merge Intervals: {pipe.transformer.merger_layers}, Unmerger Intervals: {pipe.transformer.unmerger_layers} \
                Mode: {'TEMPORAL' if pipe.transformer.tcg.use_temporal_grouping else 'SPATIAL'}")

def load_full_transformer_weights(pipe: CogVideoXPipeline, init_from: str) -> None:
    ckpt_path = os.path.join(init_from, "transformer.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"--init_from given but no transformer.pt found at {ckpt_path}")

    logger.info(f"Loading stage-1 transformer weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    
    # Load into the transformer
    missing, unexpected = pipe.transformer.load_state_dict(new_state_dict, strict=True)
    print("✅ Full custom transformer weights loaded successfully.")
    if missing:
        logger.warning(f"{len(missing)} missing key(s) loading stage-1 ckpt (up to 5): {missing[:5]}")
    if unexpected:
        logger.warning(f"{len(unexpected)} unexpected key(s) loading stage-1 ckpt (up to 5): {unexpected[:5]}")
    del state_dict, new_state_dict
    gc.collect()

def autocast_ctx(device, dtype: Optional[torch.dtype]):
    """bf16/fp16 autocast context; a no-op when dtype is None (fp32 / pure-bf16 run)."""
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)

def setup_trainable_modules(
    pipe: CogVideoXPipeline, args: argparse.Namespace, dtype: torch.dtype
) -> List[torch.nn.Parameter]:
    """Freeze text encoder + VAE; set up full-FT or LoRA on the transformer; return trainable params."""
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()

    if args.use_lora:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as e:
            raise ImportError("--use_lora requires `peft`: pip install peft") from e
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        if args.init_from and os.path.exists(os.path.join(args.init_from, "adapter_config.json")):
            logger.info("LOADING PRETRAINED LORA WEIGHTS")
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.init_from, is_trainable=True)
        else:
            if args.init_from:
                load_full_transformer_weights(pipe, args.init_from)  # Stage-1 full weights as the base
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
# 5. Data setup
# --------------------------------------------------------------------------- #
def build_dataloader(args: argparse.Namespace) -> DataLoader:
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
        logger=logger
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# --------------------------------------------------------------------------- #
# 6. Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(pipe: CogVideoXPipeline, path: str, use_lora: bool) -> None:
    os.makedirs(path, exist_ok=True)
    if use_lora:
        pipe.transformer.save_pretrained(path)
    else:
        torch.save(pipe.transformer.state_dict(), os.path.join(path, "transformer.pt"))
    # if train_vae:
        # torch.save(pipe.vae.state_dict(), os.path.join(path, "vae.pt"))
    logger.info(f"Saved checkpoint -> {path}")


# --------------------------------------------------------------------------- #
# 7. Training loop
# --------------------------------------------------------------------------- #
def compute_loss(
    pipe: CogVideoXPipeline,
    batch: Dict[str, Any],
    args: argparse.Namespace,
    empty_prompt_embedding: Optional[torch.Tensor],
    freeze_vae: bool,
    autocast_dtype: torch.dtype
) -> torch.Tensor:
    lr_video, hr_video, prompt = prepare_batch(batch)
    pred_latent, target_latent = forward_stage1(
        pipe=pipe,
        lr_video=lr_video,
        hr_video=hr_video,
        prompt=prompt,
        noise_step=args.noise_step,
        sr_noise_step=args.sr_noise_step,
        empty_prompt_embedding=empty_prompt_embedding,
        freeze_vae=freeze_vae,
        autocast_dtype=autocast_dtype
    )
    loss = F.mse_loss(pred_latent.float(), target_latent.float())
    if not torch.isfinite(loss):
        logger.error("!!! Loss is NaN/Inf !!!")
        log_tensor_stats(pred_latent, "NaN_Loss -> pred_latent", logging.ERROR)
        log_tensor_stats(target_latent, "NaN_Loss -> target_latent", logging.ERROR)
    log_tensor_stats(loss, "Loss")
    return loss

def grad_report(module):
    rows = {"lora_A": [], "lora_B": []}
    for n, p in module.named_parameters():
        if p.requires_grad and p.grad is not None:
            k = "lora_A" if "lora_A" in n else "lora_B" if "lora_B" in n else None
            if k:
                rows[k].append(p.grad)
    for k, gs in rows.items():
        if not gs:
            print(k, "no grads"); continue
        numel = sum(g.numel() for g in gs)
        zeros = sum((g == 0).sum().item() for g in gs) / numel
        norm = torch.sqrt(sum((g.float() ** 2).sum() for g in gs)).item()
        print(f"{k}: dtype={ {g.dtype for g in gs} } norm={norm:.3e} "
              f"rms/param={norm/numel**0.5:.2e} zero_frac={zeros:.4f} "
              f"nonfinite={sum((~torch.isfinite(g)).sum().item() for g in gs)}")

# after the final backward of the accumulation window, before optimizer.step():

def train(
    pipe: CogVideoXPipeline,
    dataloader: DataLoader,
    trainable_params: List[torch.nn.Parameter],
    args: argparse.Namespace,
    empty_prompt_embedding: Optional[torch.Tensor],
    freeze_vae: bool,
    autocast_dtype: torch.dtype
) -> None:
    accum = args.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    steps_per_epoch = max(len(dataloader) // accum, 1)
    total_steps = args.max_train_steps or steps_per_epoch * args.num_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        logger.info(f"---- Epoch {epoch + 1}/{args.num_epochs} ----")
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}")
        # Any partial accumulation left over from the previous epoch is discarded here.
        optimizer.zero_grad()
        t_iter_end = time.time()
        stop_training = False
        print(f"{'='*20} EPOCH START VALIDATION {'='*20}")
        # run_periodic_validation(pipe, args, global_step, empty_prompt_embedding)

        for micro_step, batch in enumerate(progress):
            logger.debug(f"[DATA] micro_step {micro_step}: waited {time.time() - t_iter_end:.3f}s for batch")
            try:
                loss = compute_loss(pipe, batch, args, empty_prompt_embedding, freeze_vae, autocast_dtype=autocast_dtype)
                (loss / accum).backward()
                log_memory("After Backward")
            except Exception as e:
                logger.error(f"!!! CRITICAL ERROR at micro_step {micro_step}: {e} !!!")
                logger.error(f"Prompt(s): {batch.get('prompt')}")
                raise

            loss_value = loss.item()
            running_loss += loss_value
            epoch_loss_sum += loss_value
            epoch_loss_count += 1

            if (micro_step + 1) % accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    logger.warning(f"non-finite grad_norm at step {global_step}; skipping")
                    optimizer.zero_grad(); continue
                optimizer.step()
                grad_report(pipe.transformer)
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                logger.debug(f"[OPT] optimizer step {global_step} (micro_step {micro_step})")

                if global_step % args.log_steps == 0:
                    avg_loss = running_loss / (args.log_steps * accum)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    elapsed_min = (time.time() - start_time) / 60
                    logger.info(
                        f"step {global_step:06d}/{total_steps} | loss {avg_loss:.6f} | "
                        f"grad_norm {float(grad_norm):.4f} | lr {current_lr:.2e} | elapsed {elapsed_min:.1f}min"
                    )
                    progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")
                    running_loss = 0.0

                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        pipe, os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                        args.use_lora
                    )
                    run_periodic_validation(pipe, args, global_step, empty_prompt_embedding)

                if args.max_train_steps and global_step >= args.max_train_steps:
                    stop_training = True
                    break

            t_iter_end = time.time()

        if epoch_loss_count:
            logger.info(f"Epoch {epoch + 1} mean loss: {epoch_loss_sum / epoch_loss_count:.6f}")
        if stop_training:
            break

    save_checkpoint(pipe, os.path.join(args.output_dir, "final"), args.use_lora)
    logger.info(f"Training complete in {(time.time() - start_time) / 60:.1f}min.")


# --------------------------------------------------------------------------- #
# 8. CLI + entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRaM-VSR / DOVE Stage-1 training (on-the-fly degradation)")

    # Paths
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--prompt_json", type=str, default=None)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./stage1_ckpt")
    p.add_argument("--empty_prompt_embedding", type=str, default=DEFAULT_EMPTY_PROMPT_PATH)
    p.add_argument("--init_from", type=str, default=None,
                   help="Directory containing a stage-1 transformer.pt to initialise the transformer from.")

    # Data
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--crop_size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--blur_prob", type=float, default=0.8)
    p.add_argument("--noise_prob", type=float, default=0.5)
    p.add_argument("--jpeg_prob", type=float, default=0.7)
    p.add_argument("--video_compress_prob", type=float, default=0.0)

    # Diffusion
    p.add_argument("--noise_step", type=int, default=0)
    p.add_argument("--sr_noise_step", type=int, default=399)

    # Optimisation
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=None)

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
    
    # TCG module
    p.add_argument("--drop_ratio", type=float, default=0.35)
    p.add_argument("--block_intervals", type=int, nargs="+", default=[20, 25, 32, 37])

    # Validation args
    p.add_argument("--val_lr_dir", type=str, default=None)
    p.add_argument("--val_gt_dir", type=str, default=None)
    p.add_argument("--val_metrics", type=str, default="psnr,ssim,lpips,dists")
    p.add_argument("--val_steps", type=int, default=0, help="0 disables")
    p.add_argument("--val_fps", type=int, default=8)
    
    # Logging / checkpointing
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING"],
                   help="DEBUG enables per-tensor stats (GPU syncs); use INFO for speed.")
    p.add_argument("--param_report", type=str, default="summary", choices=["none", "summary", "detailed"],
                   help="'detailed' writes a per-layer table to the log file.")

    args = p.parse_args()

    if len(args.block_intervals) < 2 or len(args.block_intervals) % 2 != 0:
        p.error("--block_intervals needs an even number of values (>= 2): merger/unmerger pairs")
    if (args.num_frames - 1) % 8 != 0:
        p.error(f"--num_frames must satisfy (F - 1) % 8 == 0, got {args.num_frames}")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir, args.log_level)

    freeze_vae = True
    dtype = DTYPES[args.dtype]

    logger.info("=" * 64)
    logger.info("TRaM-VSR Stage-1 training (on-the-fly real-world degradation)")
    for key, value in vars(args).items():
        logger.info(f"  {key:<28}: {value}")
    logger.info("=" * 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available — training on CPU will be extremely slow.")

    logger.info(f"Loading CogVideoX pipeline from {args.model_path} ...")
    pipe = load_pipeline(args.model_path, dtype, device, args)

    configure_tcg_module(pipe, args.drop_ratio, args.block_intervals)  # before PEFT wrapping
    autocast_dtype = dtype if (args.fp32_master_weights and dtype != torch.float32) else None    
    trainable_params = setup_trainable_modules(pipe, args, dtype=autocast_dtype)
    if args.param_report != "none":
        log_parameter_summary(pipe, args.use_lora, args.fp32_master_weights and dtype != torch.float32, detailed=args.param_report == "detailed")

    empty_prompt_embedding = load_empty_prompt_embedding(args.empty_prompt_embedding)
    dataloader = build_dataloader(args)

    train(pipe, dataloader, trainable_params, args, empty_prompt_embedding, freeze_vae, autocast_dtype=autocast_dtype)


if __name__ == "__main__":
    main()