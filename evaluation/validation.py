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

from MajorProject_VSR.evaluation.metric_utils import evaluate_video_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validation")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


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