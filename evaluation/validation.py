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

Inputs (LR and GT) can be either:
  * video files (.mp4, .mov, .avi, .mkv, .webm), or
  * directories of frames (.png, .jpg, ...), one directory per video,
    frames ordered by natural sort (frame2.png < frame10.png).
Videos are matched between LR and GT by file stem / directory name.

Metrics:
  * frame-level / full-reference (psnr, ssim, lpips, dists, ...) go through
    evaluate_video_metrics against GT.
  * DOVER is NOT in pyiqa. It is run through the official repo
    (github.com/VQAssessment/DOVER) via DoverScorer on the whole predicted clip,
    no GT needed. Reports dover (fused, 0-1), dover_technical, dover_aesthetic.

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
    dover = DoverScorer("/path/to/DOVER", device)                                   # once, optional

    report = validation_pred(
        model=pipe,                       # CogVideoXPipeline-like: .transformer and .vae
        inference_fn=fn,                  # DOVEInferenceFn(...)
        lr_path="UDM10/LR_video",
        gt_path="UDM10/GT_video",
        pred_path="results/step_1000/pred",
        eval_metrics=["psnr", "ssim", "lpips", "dists"],
        metric_models=metric_models,      # optional; built per call if omitted
        dover_scorer=dover,               # optional
        output_json="results/step_1000/metrics.json",
    )
    print(report["average"])
"""

from __future__ import annotations
from MajorProject_VSR.patchTransformer import patch_CogVideoXTransformer3DModel
patch_CogVideoXTransformer3DModel()
import os, sys
import json
import logging
import re
import tempfile
import time
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

import decord  # isort:skip  (import after torch)
import math
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger("validation")
decord.bridge.set_bridge("torch")

import imageio.v3 as iio
import pyiqa
from PIL import Image
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
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #
def _natural_key(p: Path):
    """Natural sort key so frame2.png < frame10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _list_frames(dir_path: Union[str, Path]) -> List[Path]:
    """Image files directly inside dir_path, in natural order."""
    return sorted(
        (p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )


def read_video_tensor(video_path: Union[str, Path]) -> torch.Tensor:
    """Video file OR directory of frames -> [F, C, H, W] float32 in [0, 1]."""
    video_path = Path(video_path)
    if video_path.is_dir():
        frame_files = _list_frames(video_path)
        if not frame_files:
            raise RuntimeError(f"No image frames in {video_path}")
        frames = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in frame_files])  # [F, H, W, C]
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

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
    """{name: path} for every video file or frame-folder directly inside dir_path."""
    out: Dict[str, Path] = {}
    for p in sorted(Path(dir_path).iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            out[p.stem] = p
        elif p.is_dir() and _list_frames(p):
            out[p.name] = p  # p.stem would truncate names containing dots
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_metric_models(eval_metrics: List[str], device: torch.device) -> Dict[str, torch.nn.Module]:
    if not eval_metrics:
        return {}
    logger.info(f"Initializing pyiqa metric models for: {eval_metrics}")
    models = {}
    for name in eval_metrics:
        name = name.strip().lower()
        try:
            models[name] = pyiqa.create_metric(name).to(device).eval()
        except Exception as e:
            logger.warning(f"Could not create metric '{name}', skipping. Reason: {e}")
    return models


class DoverScorer:
    """Official DOVER (VQAssessment/DOVER) on an in-memory clip. Load once, call per video.

    Setup: git clone the DOVER repo, download DOVER.pth into <dover_root>/pretrained_weights/,
    then pass dover_root. The official frame sampling reads from a video path, so the clip is
    written to a temporary LOSSLESS file first (libx264rgb, crf 0): the pixels DOVER sees are
    identical to the scored tensor.

    Returns {"dover": fused 0-1, "dover_technical": raw, "dover_aesthetic": raw}.
    """

    MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
    STD = torch.FloatTensor([58.395, 57.12, 57.375])

    def __init__(self, dover_root: Union[str, Path], device: torch.device, opt_file: str = "dover.yml"):
        import yaml

        root = os.path.abspath(str(dover_root))
        if root not in sys.path:
            sys.path.insert(0, root)
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        from dover.models import DOVER

        with open(os.path.join(root, opt_file)) as f:
            opt = yaml.safe_load(f)
        weights = opt["test_load_path"]
        if not os.path.isabs(weights):
            weights = os.path.join(root, weights)

        self.device = device
        self.evaluator = DOVER(**opt["model"]["args"]).to(device).eval()
        self.evaluator.load_state_dict(torch.load(weights, map_location=device))

        self._decompose = spatial_temporal_view_decomposition
        self.sample_types = opt["data"]["val-l1080p"]["args"]["sample_types"]
        self.samplers = {}
        for stype, sopt in self.sample_types.items():
            if "t_frag" not in sopt:  # resized temporal sampling (technical branch)
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:  # fragment temporal sampling (aesthetic branch)
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

    @staticmethod
    def _fuse(technical: float, aesthetic: float) -> float:
        x = (technical - 0.1107) / 0.07355 * 0.6104 + (aesthetic + 0.08285) / 0.03774 * 0.3896
        return float(1 / (1 + np.exp(-x)))

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> Dict[str, float]:
        """video: [F, C, H, W] float in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mkv"
            t0 = time.time()
            write_video_lossless(video, path)
            logger.info(f"[DOVER] Wrote temp lossless video in {time.time() - t0:.2f}s")
            
            # DOVER's reader code expects decord's native NDArray, not our torch bridge
            decord.bridge.set_bridge("native")
            try:
                t1 = time.time()
                views, _ = self._decompose(str(path), self.sample_types, self.samplers)
                
                for k, v in views.items():
                    num_clips = self.sample_types[k].get("num_clips", 1)
                    views[k] = (
                        ((v.permute(1, 2, 3, 0) - self.MEAN) / self.STD)
                        .permute(3, 0, 1, 2)
                        .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                        .transpose(0, 1)
                        .to(self.device)
                    )
                technical, aesthetic = [r.mean().item() for r in self.evaluator(views)]
                logger.info(f"[DOVER] Model inference took {time.time() - t1:.2f}s")
            finally:
                decord.bridge.set_bridge("torch")

        return {
            "dover": self._fuse(technical, aesthetic),
            "dover_technical": technical,
            "dover_aesthetic": aesthetic,
        }


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
    dover_scorer: Optional[Callable[[torch.Tensor], Dict[str, float]]] = None,
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
        lr_path / gt_path: directories matched by name. Each entry is a video file
            or a sub-directory of image frames (png/jpg/...).
        model: pipeline exposing .transformer and .vae; used to switch to eval(),
            enable VAE tiling/slicing, and restore previous state. None skips all of that.
        pred_path: where to write preview videos (lossless). None disables saving.
        metric_models: pre-built pyiqa models (reuse across calls to avoid reloading).
            All go through evaluate_video_metrics vs GT.
        dover_scorer: DoverScorer (or any clip -> {name: score} callable) for no-reference
            whole-clip metrics; its scores are added to each video's results.
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

    val_start_time = time.time()
    lr_videos, gt_videos = list_videos(lr_path), list_videos(gt_path)
    if not lr_videos:
        raise RuntimeError(f"No LR videos or frame folders in {lr_path}")
    if not gt_videos:
        raise RuntimeError(f"No GT videos or frame folders in {gt_path}")

    stems = [s for s in lr_videos if s in gt_videos]
    skipped = sorted(set(lr_videos) ^ set(gt_videos))
    if skipped:
        logger.warning(f"Skipping videos without an LR/GT pair: {skipped}")
    if not stems:
        raise RuntimeError("No LR/GT pairs matched by name")

    logger.info(f"Starting validation on {len(stems)} matched videos.")

    built_here = metric_models is None
    models = metric_models if metric_models is not None else build_metric_models(eval_metrics, device)
    if not models and dover_scorer is None:
        raise RuntimeError(f"None of the requested metrics could be created: {eval_metrics}")

    if pred_path is not None and save_preds:
        Path(pred_path).mkdir(parents=True, exist_ok=True)

    was_training = model.transformer.training if model is not None else None
    rng_state = _save_rng_state()
    if model is not None:
        model.transformer.eval()
    is_tiling_enabled = is_slicing_enabled = None
    per_video: Dict[str, Dict[str, float]] = {}
    total_videos = len(stems)
    
    try:
        if model is not None:
            is_tiling_enabled = model.vae.use_tiling
            is_slicing_enabled = model.vae.use_slicing
            model.vae.enable_tiling()
            model.vae.enable_slicing()

        with torch.no_grad():
            for idx, stem in enumerate(tqdm(stems, desc="Processing videos")):
                vid_start_time = time.time()
                logger.info(f"[{idx+1}/{total_videos}] [SR] {lr_videos[stem].name}")
                torch.manual_seed(_video_seed(seed, stem))  # seeds CPU and all CUDA devices

                lr = read_video_tensor(lr_videos[stem]).to(device)
                
                inf_start = time.time()
                pred = inference_fn(model, lr, device).detach().float().cpu()
                logger.info(f"[{stem}] Inference took {time.time() - inf_start:.2f}s")
                
                gt = read_video_tensor(gt_videos[stem])

                if pred.shape != gt.shape:
                    raise ValueError(f"{stem}: pred {tuple(pred.shape)} vs GT {tuple(gt.shape)}")

                pred = pred.clamp(0.0, 1.0)
                if quantize_8bit:
                    pred = (pred * 255.0).round() / 255.0

                scores: Dict[str, float] = {}
                eval_start = time.time()
                if models:
                    scores.update(
                        evaluate_video_metrics(
                            pred_video=pred,
                            ref_video=gt,
                            models=models,
                            crop=crop,
                            test_y_channel=test_y_channel,
                            device=device,
                            batch_mode=batch_mode,
                            name=stem,
                        )
                    )
                if dover_scorer is not None:
                    dover_start = time.time()
                    scores.update(dover_scorer(pred))
                    logger.info(f"[{stem}] DOVER scoring took {time.time() - dover_start:.2f}s")
                
                logger.info(f"[{stem}] Evaluation took {time.time() - eval_start:.2f}s")
                per_video[stem] = scores
                
                elapsed = time.time() - vid_start_time
                logger.info(f"[{stem}] Metrics: {per_video[stem]} | Total time: {elapsed:.2f}s")

                if pred_path is not None and save_preds:
                    save_start = time.time()
                    write_video_lossless(pred, Path(pred_path) / f"{stem}.mkv", fps=fps)
                    logger.info(f"[{stem}] Saved prediction in {time.time() - save_start:.2f}s")
    finally:
        _restore_rng_state(rng_state)
        if model is not None:
            if is_tiling_enabled is False:
                model.vae.disable_tiling()
            if is_slicing_enabled is False:
                model.vae.disable_slicing()
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
            "dover": dover_scorer is not None,
        },
    }

    total_time = time.time() - val_start_time
    logger.info(f"Validation complete on {len(per_video)} videos in {total_time:.2f}s.")
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
    p.add_argument("--lr_path", required=True, help="Dir of LR videos or per-video frame folders (png/jpg)")
    p.add_argument("--gt_path", required=True, help="Dir of GT videos or per-video frame folders (png/jpg)")
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
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips", "dists"],
        help="pyiqa metric names; add 'dover' for the official DOVER (needs --dover_root)",
    )
    p.add_argument("--dover_root", default=None, help="Clone of github.com/VQAssessment/DOVER with weights")
    p.add_argument("--dover_opt", default="dover.yml", help="DOVER option file inside --dover_root")
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
    logger.info(f"Loading CogVideoXPipeline from {args.model_path}...")
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
        path = args.empty_prompt_embedding
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            tensors = load_file(path, device=str(device))
            if len(tensors) != 1:
                logger.warning(f"{path} has {len(tensors)} tensors, using the first: {list(tensors)}")
            emb = next(iter(tensors.values()))
        else:
            emb = torch.load(path, map_location=device, weights_only=True)
        return emb.to(dtype)

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
    
    main_start_time = time.time()
    logger.info("="*50)
    logger.info("Starting Video-SR Validation")
    logger.info(f"Device: {device}")
    logger.info("="*50)

    from MajorProject_VSR.inference_func.inference import DOVEInferenceFn
    
    pipe_start = time.time()
    pipe = load_pipe(args, device)
    logger.info(f"Pipeline loaded in {time.time() - pipe_start:.2f}s")
    
    emb_start = time.time()
    logger.info("Building empty prompt embedding...")
    empty_prompt_embedding = build_empty_prompt_embedding(args, device)
    logger.info(f"Empty prompt embedding built in {time.time() - emb_start:.2f}s")
    

    logger.info("Configuring TCG module...")
    configure_tcg_module(pipe=pipe, drop_ratio=args.drop_ratio, block_intervals=args.block_intervals, logger=logger)
    if not args.load_s1 and not args.load_s2:
        logger.info("Disabling TCG Module..... only baseline loaded")
        pipe.transformer.merger_layers = []
        pipe.transformer.unmerger_layers = []

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

    metric_start = time.time()
    pyiqa_metrics = [m for m in args.metrics if m.strip().lower() != "dover"]
    metric_models = build_metric_models(pyiqa_metrics, device)
    logger.info(f"Metric models built in {time.time() - metric_start:.2f}s")

    dover_scorer = None
    if len(pyiqa_metrics) != len(args.metrics):
        if not args.dover_root:
            raise SystemExit("'dover' requested but --dover_root was not given")
        dover_start = time.time()
        logger.info("Initializing DOVER scorer...")
        dover_scorer = DoverScorer(args.dover_root, device, opt_file=args.dover_opt)
        logger.info(f"DOVER scorer initialized in {time.time() - dover_start:.2f}s")

    logger.info("Starting inference and evaluation loop...")
    report = validation_pred(
        model=pipe,  # NOT pipe.transformer: validation_pred accesses model.transformer / model.vae
        inference_fn=fn,
        lr_path=args.lr_path,
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        eval_metrics=pyiqa_metrics,
        metric_models=metric_models,
        dover_scorer=dover_scorer,
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

    total_time = time.time() - main_start_time
    logger.info(f"Total script execution time: {total_time:.2f}s")
    logger.info("="*50)

    print(json.dumps(report["average"], indent=2))


if __name__ == "__main__":
    main()