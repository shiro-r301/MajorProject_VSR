"""
validation.py
-------------
Video-SR validation with ONE fixed protocol:

  LR video -> inference_fn -> pred (float32, [F,C,H,W], [0,1], CPU)
           -> (optional) snap to the 8-bit grid
           -> metrics against GT, computed IN MEMORY

The model output is never re-encoded before scoring, so the numbers measure the
model and not a codec. Predictions are written to disk only for eyeballing, and
losslessly (libx264rgb, crf 0), so a file on disk reproduces the scored tensor.

Reproducibility:
  * every video is seeded from (seed, filename stem), so results do not depend on
    video order or on how many videos ran before it. This matters because DOVE's
    process_video draws latent_dist.sample() and nothing else seeds it.
  * the global RNG state is saved and restored, so calling this mid-training does
    not perturb the training run's randomness.
  * the model's train/eval mode is restored even if inference raises.

Usage:
    from validation import validation_pred, build_metric_models

    metric_models = build_metric_models(["psnr", "ssim", "lpips", "dists"], device)  # once

    report = validation_pred(
        model=pipe.transformer,           # only used for eval()/train() handling
        inference_fn=fn,                  # DOVEInferenceFn(...)
        lr_path="UDM10/LR_video",
        gt_path="UDM10/GT_video",
        pred_path="results/step_1000/pred",
        eval_metrics=["psnr", "ssim", "lpips", "dists"],
        metric_models=metric_models,      # optional; built per call if omitted
        output_json="results/step_1000/metrics.json",
    )
    print(report["average"])
"""

from __future__ import annotations
import os, sys
import json
import logging
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

import decord  # isort:skip  (import after torch)

decord.bridge.set_bridge("torch")

import imageio.v3 as iio
import pyiqa
from peft import LoraConfig, PeftModel

from MajorProject_VSR.evaluation.metric_utils import evaluate_video_metrics
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
import gc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validation")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #
def read_video_tensor(video_path: Union[str, Path]) -> torch.Tensor:
    """Video file -> [F, C, H, W] float32 in [0, 1]."""
    reader = decord.VideoReader(str(video_path))
    frames = reader.get_batch(list(range(len(reader))))  # [F, H, W, C] uint8
    return frames.permute(0, 3, 1, 2).float() / 255.0


def write_video_lossless(video: torch.Tensor, out_path: Union[str, Path], fps: int = 8) -> None:
    """[F, C, H, W] float in [0, 1] -> lossless RGB H.264 (for viewing only)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = (video.clamp(0, 1) * 255).round().to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [F, H, W, C]
    iio.imwrite(
        out_path,
        frames,
        fps=fps,
        codec="libx264rgb",
        pixelformat="rgb24",
        macro_block_size=None,
        ffmpeg_params=["-crf", "0"],
    )


def list_videos(dir_path: Union[str, Path]) -> Dict[str, Path]:
    """{stem: path} for every video file directly inside dir_path."""
    return {
        p.stem: p
        for p in sorted(Path(dir_path).iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_metric_models(eval_metrics: List[str], device: torch.device) -> Dict[str, torch.nn.Module]:
    models = {}
    for name in eval_metrics:
        name = name.strip().lower()
        try:
            models[name] = pyiqa.create_metric(name).to(device).eval()
        except Exception as e:
            logger.warning(f"Could not create metric '{name}', skipping. Reason: {e}")
    return models


def _video_seed(base_seed: int, stem: str) -> int:
    """Order-independent per-video seed."""
    return (base_seed + zlib.crc32(stem.encode("utf-8"))) % (2**31)


def _save_rng_state():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu, cuda


def _restore_rng_state(state) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validation_pred(
    inference_fn: Callable[[object, torch.Tensor, torch.device], torch.Tensor],
    lr_path: Union[str, Path],
    gt_path: Union[str, Path],
    eval_metrics: List[str],
    model: Optional[torch.nn.Module] = None,
    pred_path: Optional[Union[str, Path]] = None,
    metric_models: Optional[Dict[str, torch.nn.Module]] = None,
    device: Optional[torch.device] = None,
    output_json: Optional[Union[str, Path]] = None,
    crop: int = 0,
    test_y_channel: bool = False,
    batch_mode: bool = False,
    seed: int = 0,
    quantize_8bit: bool = True,
    save_preds: bool = True,
    fps: int = 8,
) -> dict:
    """
    Run `inference_fn` on every LR video, score the in-memory output against GT.

    Args:
        inference_fn: (model, lr[F,C,H,W] in [0,1], device) -> pred[F,C,H',W'] in [0,1].
        lr_path / gt_path: directories matched by filename stem.
        model: only used to switch to eval() and restore the previous mode.
        pred_path: where to write preview videos (lossless). None disables saving.
        metric_models: pre-built pyiqa models (reuse across calls to avoid reloading).
        seed: base seed; each video uses a seed derived from (seed, stem).
        quantize_8bit: snap pred to the 8-bit grid before scoring, so the number
            equals what a lossless file on disk would give. The effect is tiny
            (about 0.006 dB at 30 dB PSNR); set False to score raw floats.
        save_preds: write preview videos to pred_path (never read back for scoring).

    Returns:
        {"per_video": {...}, "average": {...}, "num_videos": int,
         "eval_metrics": [...], "protocol": {...}}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_videos, gt_videos = list_videos(lr_path), list_videos(gt_path)
    if not lr_videos:
        raise RuntimeError(f"No LR videos in {lr_path}")
    if not gt_videos:
        raise RuntimeError(f"No GT videos in {gt_path}")

    stems = [s for s in lr_videos if s in gt_videos]
    skipped = sorted(set(lr_videos) ^ set(gt_videos))
    if skipped:
        logger.warning(f"Skipping videos without an LR/GT pair: {skipped}")
    if not stems:
        raise RuntimeError("No LR/GT pairs matched by filename stem")
    
    built_here = metric_models is None
    models = metric_models if metric_models is not None else build_metric_models(eval_metrics, device)
    if not models:
        raise RuntimeError(f"None of the requested metrics could be created: {eval_metrics}")

    if pred_path is not None and save_preds:
        Path(pred_path).mkdir(parents=True, exist_ok=True)

    was_training = model.transformer.training if model is not None else None
    rng_state = _save_rng_state()
    if model is not None:
        model.transformer.eval()
    is_tiling_enabled = is_slicing_enabled = None
    per_video: Dict[str, Dict[str, float]] = {}
    try:
        is_tiling_enabled = model.vae.use_tiling
        is_slicing_enabled = model.vae.use_slicing
        model.vae.enable_tiling()
        model.vae.enable_slicing()

        with torch.no_grad():
            for stem in stems:
                logger.info(f"[SR] {lr_videos[stem].name}")
                torch.manual_seed(_video_seed(seed, stem))  # seeds CPU and all CUDA devices

                lr = read_video_tensor(lr_videos[stem]).to(device)
                pred = inference_fn(model, lr, device).detach().float().cpu()
                gt = read_video_tensor(gt_videos[stem])

                if pred.shape != gt.shape:
                    raise ValueError(f"{stem}: pred {tuple(pred.shape)} vs GT {tuple(gt.shape)}")

                pred = pred.clamp(0.0, 1.0)
                if quantize_8bit:
                    pred = (pred * 255.0).round() / 255.0

                per_video[stem] = evaluate_video_metrics(
                    pred_video=pred,
                    ref_video=gt,
                    models=models,
                    crop=crop,
                    test_y_channel=test_y_channel,
                    device=device,
                    batch_mode=batch_mode,
                    name=stem,
                )
                logger.info(f"[{stem}] {per_video[stem]}")

                if pred_path is not None and save_preds:
                    write_video_lossless(pred, Path(pred_path) / f"{stem}.mkv", fps=fps)
    finally:
        _restore_rng_state(rng_state)
        if not is_tiling_enabled:
            model.vae.disable_tiling()
        if not is_slicing_enabled:
            model.vae.disable_slicing() 

        if model is not None:
            model.transformer.train(was_training)
        if built_here:
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric_names = list(next(iter(per_video.values())).keys())
    average = {
        n: round(float(np.mean([r[n] for r in per_video.values()])), 4) for n in metric_names
    }
    report = {
        "per_video": per_video,
        "average": average,
        "num_videos": len(per_video),
        "eval_metrics": metric_names,
        "protocol": {
            "scored": "in-memory tensor (no codec)",
            "quantize_8bit": quantize_8bit,
            "seed": seed,
            "crop": crop,
            "test_y_channel": test_y_channel,
            "batch_mode": batch_mode,
        },
    }

    logger.info(f"Validation complete on {len(per_video)} videos.")
    logger.info(f"Average metrics: {json.dumps(average, indent=2)}")

    if output_json is None and pred_path is not None:
        output_json = Path(pred_path) / "validation_metrics.json"
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report to {output_json}")

    return report

def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Video-SR validation (DOVE / CogVideoX)")
    # data
    p.add_argument("--lr_path", required=True, help="Dir of LR videos")
    p.add_argument("--gt_path", required=True, help="Dir of GT videos")
    p.add_argument("--pred_path", default=None, help="Where to write preview videos")
    p.add_argument("--output_json", default=None)
    # model
    p.add_argument("--model_path", required=True, help="DOVE / CogVideoX checkpoint dir")
    p.add_argument("--load_s1", default=None, help="Dir with S1 weights (<dir>/S1/ or <dir>/)")
    p.add_argument("--load_s2", default=None, help="Dir with S2 weights (<dir>/S2/ or <dir>/)")
    p.add_argument("--empty_prompt_embedding", default=None, help="Precomputed .pt (skips T5)")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    # inference
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--noise_step", type=int, default=0)
    p.add_argument("--sr_noise_step", type=int, default=399)
    p.add_argument("--tile_size_hw", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32), metavar=("H", "W"))

    p.add_argument("--drop_ratio", type=float, default=0.35)
    p.add_argument("--block_intervals", type=int, nargs="+", default=[20, 25, 32, 37])

    # metrics / protocol
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips", "dists"])
    p.add_argument("--crop", type=int, default=0)
    p.add_argument("--test_y_channel", action="store_true")
    p.add_argument("--batch_mode", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_quantize_8bit", action="store_true")
    p.add_argument("--no_save_preds", action="store_true")
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def configure_tcg_module(pipe: CogVideoXPipeline, drop_ratio: float, block_intervals: List[int], logger) -> None:
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

def _find_in_stage(root: str, stage: str, filename: str) -> Optional[str]:
    """First existing <root>/<stage>/<filename> or <root>/<filename>, else None."""
    for cand in (os.path.join(root, stage, filename), os.path.join(root, filename)):
        if os.path.isfile(cand):
            return cand
    return None


def load_full_transformer_weights(pipe: CogVideoXPipeline, ckpt_path: str) -> None:
    logger.info(f"Loading full transformer weights from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}  # strip DDP prefix
    pipe.transformer.load_state_dict(state, strict=True)  # raises on any mismatch
    del state


def load_stage_weights(pipe: CogVideoXPipeline, path: str, stage: str) -> None:
    """Load a LoRA adapter (merged in) or a full transformer.pt for the given stage."""
    adapter_cfg = _find_in_stage(path, stage, "adapter_config.json")
    if adapter_cfg is not None:
        adapter_dir = os.path.dirname(adapter_cfg)
        logger.info(f"Loading {stage} LoRA from {adapter_dir}")
        peft_model = PeftModel.from_pretrained(pipe.transformer, adapter_dir, is_trainable=False)
        # Merge so the transformer is a plain module again: stages stack cleanly
        pipe.transformer = peft_model.merge_and_unload()
        return

    ckpt = _find_in_stage(path, stage, "transformer.pt")
    if ckpt is not None:
        load_full_transformer_weights(pipe, ckpt)
        return

    raise FileNotFoundError(
        f"No adapter_config.json or transformer.pt for {stage} under {path} "
        f"(looked in {path}/{stage}/ and {path}/)"
    )


def load_pipe(args, device: torch.device) -> CogVideoXPipeline:
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_path,
        tokenizer=None,
        text_encoder=None,
        torch_dtype=DTYPES[args.dtype],
    )
    pipe.to(device)

    for path, stage in ((args.load_s1, "S1"), (args.load_s2, "S2")):
        if path:
            load_stage_weights(pipe, path, stage)

    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()
    return pipe

@torch.no_grad()
def build_empty_prompt_embedding(args, device: torch.device) -> torch.Tensor:
    """Empty-prompt T5 embedding. T5 is loaded temporarily and freed afterwards."""
    dtype = DTYPES[args.dtype]
    if args.empty_prompt_embedding:
        return torch.load(args.empty_prompt_embedding, map_location=device, weights_only=True).to(dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = (
        T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=dtype)
        .to(device)
        .eval()
    )
    ids = tokenizer(
        "",
        padding="max_length",
        max_length=226,  # must match the value used when training
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)
    embeds = text_encoder(ids)[0].to(dtype)

    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeds

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from MajorProject_VSR.inference import DOVEInferenceFn
    pipe = load_pipe(args, device)
    empty_prompt_embedding = build_empty_prompt_embedding(args, device)
    configure_tcg_module(pipe=pipe, drop_ratio=args.drop_ratio, block_intervals=args.block_intervals, logger=logger)

    fn_kwargs = {}
    if args.tile_size_hw is not None:
        fn_kwargs.update(tile_size_hw=tuple(args.tile_size_hw), overlap_hw=tuple(args.overlap_hw))

    fn = DOVEInferenceFn(
        pipe=pipe,
        upscale=args.upscale,
        upscale_mode="bilinear",
        noise_step=args.noise_step,
        sr_noise_step=args.sr_noise_step,
        empty_prompt_embedding=empty_prompt_embedding,
        **fn_kwargs,
    )

    metric_models = build_metric_models(args.metrics, device)

    report = validation_pred(
        model=pipe,  # NOT pipe.transformer: validation_pred accesses model.transformer / model.vae
        inference_fn=fn,
        lr_path=args.lr_path,
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        eval_metrics=args.metrics,
        metric_models=metric_models,
        device=device,
        output_json=args.output_json,
        crop=args.crop,
        test_y_channel=args.test_y_channel,
        batch_mode=args.batch_mode,
        seed=args.seed,
        quantize_8bit=not args.no_quantize_8bit,
        save_preds=not args.no_save_preds,
        fps=args.fps,
    )

    print(json.dumps(report["average"], indent=2))

if __name__ == "__main__":
    main()