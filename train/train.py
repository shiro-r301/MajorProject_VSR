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

from MajorProject_VSR.patchTransformer import patch_CogVideoXTransformer3DModel
patch_CogVideoXTransformer3DModel()

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

import decord  # isort:skip
from dataloading.VideoFileLoader import VideoFileSRDataset

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

    file_handler = logging.FileHandler(os.path.join(output_dir, "train_debug.log"))
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


def log_parameter_summary(pipe: CogVideoXPipeline, use_lora: bool, train_vae: bool, detailed: bool) -> None:
    """Count trainable/frozen params. Per-layer lines go to DEBUG (i.e. the log file)."""
    modules = [("", pipe.transformer)]
    if train_vae:
        modules.append(("VAE.", pipe.vae))

    total = trainable = lora_trainable = orig_transformer_trainable = 0
    for prefix, module in modules:
        for name, param in module.named_parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
                is_lora = "lora" in name.lower()
                if is_lora:
                    lora_trainable += n
                elif prefix == "":
                    orig_transformer_trainable += n
                tag = "LoRA" if is_lora else "TRAINABLE"
            else:
                tag = "FROZEN"
            if detailed:
                logger.debug(f"  [{tag:<9}] {prefix}{name:<65} | shape={tuple(param.shape)} | {n:,}")

    logger.info(f"Params (transformer{' + VAE' if train_vae else ''}): total={total:,}, "
                f"trainable={trainable:,} ({100 * trainable / max(total, 1):.4f}%)")
    if use_lora:
        logger.info(f"  LoRA trainable: {lora_trainable:,}")
        if orig_transformer_trainable > 0:
            logger.warning(f"  {orig_transformer_trainable:,} non-LoRA transformer params are trainable "
                           f"(expected 0 for pure LoRA)")


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


def get_prompt_embedding(
    pipe: CogVideoXPipeline,
    prompt: str,
    batch_size: int,
    empty_prompt_embedding: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Text embedding for `prompt`, broadcast across the batch."""
    if prompt == "" and empty_prompt_embedding is not None:
        embedding = empty_prompt_embedding.to(device, dtype=dtype)
    else:
        token_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            embedding = pipe.text_encoder(token_ids.to(device))[0].to(dtype=dtype)

    if embedding.shape[0] != batch_size:
        if embedding.shape[0] != 1:
            raise ValueError(f"Cannot broadcast prompt embedding {tuple(embedding.shape)} to batch {batch_size}")
        embedding = embedding.repeat(batch_size, 1, 1)
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

        prompt_embedding = get_prompt_embedding(pipe, prompt, batch_size, empty_prompt_embedding, device, dtype)
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


# --------------------------------------------------------------------------- #
# 4. Model setup
# --------------------------------------------------------------------------- #
def load_pipeline(model_path: str, dtype: torch.dtype, device: str) -> CogVideoXPipeline:
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe.to(device)


def configure_tcg_module(pipe: CogVideoXPipeline, drop_ratio: float, block_intervals: List[int]) -> None:
    """Set token-merge/unmerge layers on the transformer.

    Must run BEFORE the transformer is wrapped by PEFT so the attributes land
    on the base module whose forward reads them.
    """
    intervals = sorted(block_intervals)
    pipe.transformer.drop_ratio = drop_ratio
    pipe.transformer.merger_layers = intervals[::2]
    pipe.transformer.unmerger_layers = intervals[1::2]


def load_init_weights(pipe: CogVideoXPipeline, init_from: str) -> None:
    """Initialise the transformer from a previous stage's `transformer.pt`.

    Must run BEFORE PEFT wrapping: wrapping renames parameter keys
    (adds `base_model.model.` and splits LoRA layers), so a plain
    transformer state dict would no longer match.
    """
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


def setup_trainable_modules(pipe: CogVideoXPipeline, args: argparse.Namespace) -> List[torch.nn.Parameter]:
    """Freeze/unfreeze components, optionally wrap in LoRA, return trainable params."""
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()

    if args.train_vae:
        pipe.vae.requires_grad_(True)
        pipe.vae.train()
    else:
        pipe.vae.requires_grad_(False)
        pipe.vae.eval()

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            init_lora_weights="gaussian",
        )
        pipe.transformer = get_peft_model(pipe.transformer, lora_config)
        params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    else:
        pipe.transformer.requires_grad_(True)
        params = list(pipe.transformer.parameters())

    if args.train_vae:
        params += list(pipe.vae.parameters())

    pipe.transformer.train()
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    return params


def load_empty_prompt_embedding(path: str) -> Optional[torch.Tensor]:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Empty-prompt embedding not found at {p}; empty prompts will go through the text encoder.")
        return None
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
def save_checkpoint(pipe: CogVideoXPipeline, path: str, use_lora: bool, train_vae: bool) -> None:
    os.makedirs(path, exist_ok=True)
    if use_lora:
        pipe.transformer.save_pretrained(path)
    else:
        torch.save(pipe.transformer.state_dict(), os.path.join(path, "transformer.pt"))
    if train_vae:
        torch.save(pipe.vae.state_dict(), os.path.join(path, "vae.pt"))
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
    )
    loss = F.mse_loss(pred_latent.float(), target_latent.float())
    if not torch.isfinite(loss):
        logger.error("!!! Loss is NaN/Inf !!!")
        log_tensor_stats(pred_latent, "NaN_Loss -> pred_latent", logging.ERROR)
        log_tensor_stats(target_latent, "NaN_Loss -> target_latent", logging.ERROR)
    log_tensor_stats(loss, "Loss")
    return loss


def train(
    pipe: CogVideoXPipeline,
    dataloader: DataLoader,
    trainable_params: List[torch.nn.Parameter],
    args: argparse.Namespace,
    empty_prompt_embedding: Optional[torch.Tensor],
    freeze_vae: bool,
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

        for micro_step, batch in enumerate(progress):
            logger.debug(f"[DATA] micro_step {micro_step}: waited {time.time() - t_iter_end:.3f}s for batch")
            try:
                loss = compute_loss(pipe, batch, args, empty_prompt_embedding, freeze_vae)
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
                optimizer.step()
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
                        args.use_lora, args.train_vae,
                    )

                if args.max_train_steps and global_step >= args.max_train_steps:
                    stop_training = True
                    break

            t_iter_end = time.time()

        if epoch_loss_count:
            logger.info(f"Epoch {epoch + 1} mean loss: {epoch_loss_sum / epoch_loss_count:.6f}")
        if stop_training:
            break

    save_checkpoint(pipe, os.path.join(args.output_dir, "final"), args.use_lora, args.train_vae)
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
    p.add_argument("--train_vae", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)

    # TCG module
    p.add_argument("--drop_ratio", type=float, default=0.35)
    p.add_argument("--block_intervals", type=int, nargs="+", default=[20, 25, 32, 37])

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

    freeze_vae = not args.train_vae
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
    pipe = load_pipeline(args.model_path, dtype, device)

    configure_tcg_module(pipe, args.drop_ratio, args.block_intervals)  # before PEFT wrapping
    if args.init_from:
        load_init_weights(pipe, args.init_from)  # before PEFT wrapping

    trainable_params = setup_trainable_modules(pipe, args)
    if args.param_report != "none":
        log_parameter_summary(pipe, args.use_lora, args.train_vae, detailed=args.param_report == "detailed")

    empty_prompt_embedding = load_empty_prompt_embedding(args.empty_prompt_embedding)
    dataloader = build_dataloader(args)

    train(pipe, dataloader, trainable_params, args, empty_prompt_embedding, freeze_vae)


if __name__ == "__main__":
    main()