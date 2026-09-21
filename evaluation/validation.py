"""
validation.py
--------------
Benchmarks video-SR predictions against ground-truth videos.

Two responsibilities, kept decoupled:

1. (optional) `generate_sr_videos(...)` — runs the trained model on LR
   validation videos and writes the resulting SR videos into `pred_path`.
   The actual forward/sampling call is pipeline-specific (CogVideoX /
   DOVE / TRaM-VSR all differ), so you plug it in via `inference_fn`.

2. `validation_pred(...)` — the function you asked for. Given
   `pred_path`, `gt_path` and a list of metric names, it reads the
   matching video pairs, computes every metric with
   `metric_utils.evaluate_video_metrics`, averages across the
   validation set, and returns/saves a JSON report.

Typical use during training:

    from validation import validation_pred

    results = validation_pred(
        pred_path="results/step_1000/pred",
        gt_path="dataset/VideoSR/UDM10/GT_video",
        eval_metrics=["psnr", "ssim", "lpips", "dists"],
        output_json="results/step_1000/metrics.json",
    )
    print(results["average"])

Or, if you want this script to also run the model and populate
`pred_path` first:

    def my_inference_fn(model, lr_video, device):
        # lr_video: [F, C, H, W] float32 in [0, 1]
        # -> must return pred_video: [F, C, H, W] float32 in [0, 1]
        ...

    results = validation_pred(
        pred_path="results/step_1000/pred",
        gt_path="dataset/VideoSR/UDM10/GT_video",
        eval_metrics=["psnr", "ssim", "lpips", "dists"],
        model=my_model,
        lr_path="dataset/VideoSR/UDM10/LR_video",
        inference_fn=my_inference_fn,
    )
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

import decord  # isort:skip
decord.bridge.set_bridge("torch")

import pyiqa

from metric_utils import evaluate_video_metrics, fr_metrics

try:
    from torchvision.io import write_video
    _WRITE_VIDEO_AVAILABLE = True
except ImportError:
    _WRITE_VIDEO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validation")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# --------------------------------------------------------------------------- #
# Video I/O helpers
# --------------------------------------------------------------------------- #
def read_video_tensor(video_path: Union[str, Path]) -> torch.Tensor:
    """Read a video file into a [F, C, H, W] float32 tensor in [0, 1]."""
    video_reader = decord.VideoReader(str(video_path))
    frames = video_reader.get_batch(list(range(len(video_reader))))  # [F, H, W, C], uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
    return frames


def write_video_tensor(video: torch.Tensor, out_path: Union[str, Path], fps: int = 8) -> None:
    """Write a [F, C, H, W] float32 tensor in [0, 1] to an mp4 file."""
    if not _WRITE_VIDEO_AVAILABLE:
        raise RuntimeError("torchvision.io.write_video is unavailable in this environment.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = (video.clamp(0, 1) * 255).round().to(torch.uint8)  # [F, C, H, W]
    frames = frames.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
    write_video(str(out_path), frames, fps=fps, video_codec="libx264")


def list_videos(dir_path: Union[str, Path]) -> Dict[str, Path]:
    """Map {stem: path} for every video file directly inside dir_path."""
    dir_path = Path(dir_path)
    videos = {}
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos[p.stem] = p
    return videos


# --------------------------------------------------------------------------- #
# (Optional) Step 1: run the model on LR validation videos -> pred_path
# --------------------------------------------------------------------------- #
def default_inference_fn(model, lr_video: torch.Tensor, device: torch.device) -> torch.Tensor:
    raise NotImplementedError(
        "No inference_fn was provided. Pass an `inference_fn(model, lr_video, device) -> "
        "pred_video` callable that runs your SR model's forward/sampling pass — this is "
        "pipeline-specific (CogVideoX/DOVE/TRaM-VSR), so validation.py doesn't assume one."
    )


def generate_sr_videos(
    model,
    lr_path: Union[str, Path],
    pred_path: Union[str, Path],
    inference_fn: Optional[Callable[[object, torch.Tensor, torch.device], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    fps: int = 8,
    overwrite: bool = False,
) -> None:
    """
    Run `model` on every LR video in `lr_path` and save the resulting SR
    video into `pred_path` (same stem/filename, always .mp4).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_fn = inference_fn or default_inference_fn

    lr_videos = list_videos(lr_path)
    if len(lr_videos) == 0:
        logger.warning(f"No LR videos found in {lr_path}")
        return

    pred_path = Path(pred_path)
    pred_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    torch.set_grad_enabled(False)

    for stem, lr_file in lr_videos.items():
        out_file = pred_path / f"{stem}.mp4"
        if out_file.exists() and not overwrite:
            logger.info(f"[SKIP] {out_file} already exists")
            continue

        logger.info(f"[SR] {lr_file.name} -> {out_file.name}")
        lr_video = read_video_tensor(lr_file).to(device)
        pred_video = inference_fn(model, lr_video, device)
        pred_video = pred_video.detach().to(torch.float32).cpu()
        write_video_tensor(pred_video, out_file, fps=fps)

    torch.set_grad_enabled(True)


# --------------------------------------------------------------------------- #
# Step 2: benchmark pred_path against gt_path
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


def validation_pred(
    pred_path: Union[str, Path],
    gt_path: Union[str, Path],
    eval_metrics: List[str],
    model=None,
    lr_path: Optional[Union[str, Path]] = None,
    inference_fn: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    output_json: Optional[Union[str, Path]] = None,
    crop: int = 0,
    test_y_channel: bool = False,
    batch_mode: bool = False,
    fps: int = 8,
    overwrite_preds: bool = False,
) -> dict:
    """
    Benchmark SR predictions against ground truth and return/save a JSON report.

    Args:
        pred_path: directory of predicted (SR) videos. If `model` and
            `lr_path` are given, this directory is (re)populated first.
        gt_path: directory of ground-truth videos. Matched to predictions
            by filename stem.
        eval_metrics: e.g. ["psnr", "ssim", "lpips", "dists"] (full-reference)
            and/or ["clipiqa", "musiq", "niqe"] (no-reference).
        model / lr_path / inference_fn: optional — if provided, SR videos
            are generated from `model` on the LR videos in `lr_path` and
            written to `pred_path` before benchmarking. See
            `generate_sr_videos` for details.
        output_json: where to save the report. Defaults to
            `<pred_path>/validation_metrics.json`.
        crop / test_y_channel / batch_mode: forwarded to
            `metric_utils.evaluate_video_metrics`.

    Returns:
        {
          "per_video": {video_name: {metric: value, ...}, ...},
          "average":   {metric: value, ...},
          "num_videos": int,
          "eval_metrics": [...],
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1 (optional): populate pred_path from the model.
    if model is not None:
        if lr_path is None:
            raise ValueError("`lr_path` is required when `model` is provided.")
        generate_sr_videos(
            model=model,
            lr_path=lr_path,
            pred_path=pred_path,
            inference_fn=inference_fn,
            device=device,
            fps=fps,
            overwrite=overwrite_preds,
        )

    # Step 2: match pred/gt pairs.
    pred_videos = list_videos(pred_path)
    gt_videos = list_videos(gt_path)

    common_stems = sorted(set(pred_videos) & set(gt_videos))
    missing_gt = sorted(set(pred_videos) - set(gt_videos))
    missing_pred = sorted(set(gt_videos) - set(pred_videos))
    if missing_gt:
        logger.warning(f"{len(missing_gt)} predictions have no matching GT, skipped: {missing_gt}")
    if missing_pred:
        logger.warning(f"{len(missing_pred)} GT videos have no matching prediction, skipped: {missing_pred}")
    if len(common_stems) == 0:
        raise RuntimeError(f"No matching video pairs found between {pred_path} and {gt_path}")

    # Step 3: build metric models once, reuse across all videos.
    models = build_metric_models(eval_metrics, device)
    if len(models) == 0:
        raise RuntimeError(f"None of the requested metrics could be created: {eval_metrics}")

    per_video_results = {}
    for stem in common_stems:
        pred_video = read_video_tensor(pred_videos[stem])
        gt_video = read_video_tensor(gt_videos[stem])

        result = evaluate_video_metrics(
            pred_video=pred_video,
            ref_video=gt_video,
            models=models,
            crop=crop,
            test_y_channel=test_y_channel,
            device=device,
            batch_mode=batch_mode,
            name=stem,
        )
        logger.info(f"[{stem}] {result}")
        per_video_results[stem] = result

    # Step 4: average across the validation set.
    metric_names = list(models.keys())
    average_results = {
        name: round(float(np.mean([per_video_results[s][name] for s in common_stems])), 4)
        for name in metric_names
    }

    report = {
        "per_video": per_video_results,
        "average": average_results,
        "num_videos": len(common_stems),
        "eval_metrics": metric_names,
    }

    logger.info(f"Validation complete on {len(common_stems)} videos.")
    logger.info(f"Average metrics: {json.dumps(average_results, indent=2)}")

    if output_json is None:
        output_json = Path(pred_path) / "validation_metrics.json"
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report to {output_json}")

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SR predictions against GT videos.")
    parser.add_argument("--pred_path", type=str, required=True, help="Dir of predicted SR videos.")
    parser.add_argument("--gt_path", type=str, required=True, help="Dir of ground-truth videos.")
    parser.add_argument(
        "--eval_metrics", type=str, default="psnr,ssim,lpips,dists",
        help="Comma-separated metric names, e.g. 'psnr,ssim,lpips,dists,clipiqa,musiq,niqe'.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Where to save the JSON report.")
    parser.add_argument("--crop", type=int, default=0, help="Border pixels to crop before metrics.")
    parser.add_argument("--test_y_channel", action="store_true", help="Evaluate FR metrics on Y channel only.")
    parser.add_argument("--batch_mode", action="store_true", help="Compute metrics batched instead of per-frame.")
    parser.add_argument("--device", type=str, default=None, help="e.g. 'cuda' or 'cpu'.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    eval_metrics = [m.strip().lower() for m in args.eval_metrics.split(",") if m.strip()]

    validation_pred(
        pred_path=args.pred_path,
        gt_path=args.gt_path,
        eval_metrics=eval_metrics,
        device=device,
        output_json=args.output_json,
        crop=args.crop,
        test_y_channel=args.test_y_channel,
        batch_mode=args.batch_mode,
    )


if __name__ == "__main__":
    main()
