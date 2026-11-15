"""
validation.py
-------------
Video-SR validation / memory profiling for DOVE (CogVideoX) + token merging (TCG).

Two modes (--mode):

  eval    (default) LR video -> inference_fn -> pred (float32, [F,C,H,W], [0,1], CPU)
                    -> (optional) snap to the 8-bit grid
                    -> metrics against GT, computed IN MEMORY
          The model output is never re-encoded before scoring, so the numbers measure
          the model and not a codec. Predictions are written to disk only for eyeballing,
          and losslessly (libx264rgb, crf 0).

  memory  Loads the model, runs synthetic clips through the inference function for a
          baseline (no merging) and each --mem_drop_ratios config, and reports peak
          GPU memory (total / per phase / per attention layer / SDPA workspace) plus
          latency. No data, metrics or DOVER needed.

Inputs (LR and GT) in eval mode can be either video files (.mp4, .mov, .avi, .mkv,
.webm) or directories of frames (one directory per video, natural-sorted frames).
Videos are matched between LR and GT by file stem / directory name.

Metrics (eval mode):
  * frame-level / full-reference (psnr, ssim, lpips, dists, ...) via pyiqa.
  * DOVER is NOT in pyiqa. It runs through the official repo
    (github.com/VQAssessment/DOVER) via DoverScorer on the whole predicted clip,
    no GT needed. Reports dover (fused, 0-1), dover_technical, dover_aesthetic.

Reproducibility:
  * every video is seeded from (seed, filename stem), so results do not depend on
    video order. DOVE's process_video draws latent_dist.sample() and nothing else seeds it.
  * the global RNG state is saved and restored, so calling validation_pred mid-training
    does not perturb the training run's randomness.
  * the model's train/eval mode is restored even if inference raises.

Library use:
    from validation import validation_pred, build_metric_models, DoverScorer

    metric_models = build_metric_models(["psnr", "ssim", "lpips", "dists"], device)  # once
    dover = DoverScorer("/path/to/DOVER", device)                                   # once, optional
    report = validation_pred(model=pipe, inference_fn=fn, lr_path=..., gt_path=...,
                             eval_metrics=[...], metric_models=metric_models,
                             dover_scorer=dover, pred_path=..., output_json=...)

CLI:
    python -m MajorProject_VSR.evaluation.validation --mode eval   --model_path ... --lr_path ... --gt_path ...
    python -m MajorProject_VSR.evaluation.validation --mode memory --model_path ... --mem_frames 17 25 33
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import logging
import math
import os
import re
import statistics
import sys
import tempfile
import time
import warnings
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

import decord  # isort:skip  (import after torch)

decord.bridge.set_bridge("torch")

import imageio.v3 as iio
import pyiqa
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import CogVideoXPipeline

from MajorProject_VSR.evaluation.metric_utils import evaluate_video_metrics
from MajorProject_VSR.patchTransformer import patch_CogVideoXTransformer3DModel
patch_CogVideoXTransformer3DModel()

logger = logging.getLogger("validation")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
GIB = 2**30


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO", log_file: Optional[Union[str, Path]] = None) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    for noisy in ("PIL", "matplotlib", "urllib3", "httpx", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        root.addHandler(fh)
        logger.info(f"Logging to {log_file}")


@contextlib.contextmanager
def log_step(msg: str):
    t0 = time.time()
    logger.info(f"{msg} ...")
    yield
    logger.info(f"{msg} done in {time.time() - t0:.2f}s")


@contextlib.contextmanager
def quiet_third_party(name: str):
    """Silence warnings and stdout prints from a noisy library; replay prints at DEBUG."""
    buf = io.StringIO()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
            warnings.simplefilter("ignore")
            yield
    finally:  # replay even if the block raised, so nothing useful is lost on a crash
        for line in buf.getvalue().splitlines():
            if line.strip():
                logger.debug(f"[{name}] {line}")


def log_environment(args) -> dict:
    import platform
    import subprocess

    import diffusers
    import peft
    import transformers

    info = {"torch": torch.__version__, "git_commit": None, "gpu": None}
    logger.info(
        f"python {platform.python_version()} | torch {torch.__version__} | diffusers {diffusers.__version__} | "
        f"transformers {transformers.__version__} | peft {peft.__version__} | pyiqa {getattr(pyiqa, '__version__', '?')}"
    )
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        info["gpu"] = p.name
        logger.info(
            f"GPU: {p.name} | {p.total_memory / GIB:.1f} GiB | CUDA {torch.version.cuda} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
    else:
        logger.warning("CUDA not available: this will be extremely slow on CPU")
    try:
        here = Path(__file__).resolve().parent
        run = lambda *c: subprocess.check_output(c, cwd=here, stderr=subprocess.DEVNULL, text=True).strip()
        commit = run("git", "rev-parse", "--short", "HEAD")
        dirty = bool(run("git", "status", "--porcelain"))
        info["git_commit"] = commit + ("+dirty" if dirty else "")
        logger.info(f"git commit: {info['git_commit']}")
    except Exception:
        logger.debug("git info unavailable")
    logger.info("Arguments: " + json.dumps(vars(args), sort_keys=True, default=str))
    return info


def log_gpu_mem(tag: str) -> None:
    if torch.cuda.is_available():
        logger.info(
            f"[{tag}] GPU mem allocated={torch.cuda.memory_allocated() / GIB:.2f} GiB "
            f"reserved={torch.cuda.memory_reserved() / GIB:.2f} GiB"
        )


@torch.no_grad()
def model_fingerprint(module: torch.nn.Module) -> dict:
    """One cheap pass over all params (milliseconds on GPU, one device sync in total)."""
    names, sums, dtypes, devices = [], [], set(), set()
    total = trainable = 0
    for name, p in module.named_parameters():
        names.append(name)
        sums.append(p.detach().abs().sum(dtype=torch.float32))
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
        dtypes.add(str(p.dtype).replace("torch.", ""))
        devices.add(str(p.device))
    stacked = torch.stack([s.to(sums[0].device) for s in sums])
    finite = torch.isfinite(stacked)
    return {
        "params": total,
        "trainable": trainable,
        "dtypes": sorted(dtypes),
        "devices": sorted(devices),
        "checksum": float(stacked[finite].double().sum()),
        "nonfinite": [names[i] for i in (~finite).nonzero().flatten().tolist()],
    }


def log_model_state(tag: str, fp: dict) -> None:
    logger.info(
        f"[{tag}] params={fp['params'] / 1e9:.3f}B trainable={fp['trainable']} "
        f"dtype={','.join(fp['dtypes'])} device={','.join(fp['devices'])} checksum={fp['checksum']:.6e}"
    )
    if fp["nonfinite"]:
        logger.error(f"[{tag}] NaN/Inf in {len(fp['nonfinite'])} tensor(s), e.g. {fp['nonfinite'][:3]}")


def log_pipe_summary(pipe: CogVideoXPipeline) -> None:
    t, v = pipe.transformer, pipe.vae
    logger.info(f"Transformer: {type(t).__name__} | training={t.training} | dtype={t.dtype} | device={t.device}")
    logger.info(f"VAE: {type(v).__name__} | dtype={v.dtype} | device={v.device}")
    logger.info(f"Scheduler: {type(pipe.scheduler).__name__}")


def _fmt_scores(scores: Dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in scores.items())


def log_results_table(per_video: Dict[str, Dict[str, float]], average: Dict[str, float]) -> None:
    names = list(average)
    w = max(len(s) for s in per_video) + 2
    lines = ["video".ljust(w) + "".join(n.rjust(16) for n in names)]
    for stem, r in per_video.items():
        lines.append(stem.ljust(w) + "".join(f"{r[n]:16.4f}" for n in names))
    lines.append("-" * len(lines[0]))
    lines.append("AVERAGE".ljust(w) + "".join(f"{average[n]:16.4f}" for n in names))
    logger.info("Results:\n" + "\n".join(lines))


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


def write_video_lossless(
    video: torch.Tensor, out_path: Union[str, Path], fps: int = 8, preset: str = "medium"
) -> None:
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
        ffmpeg_params=["-crf", "0", "-preset", preset],
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
# Metrics
# --------------------------------------------------------------------------- #
def build_metric_models(eval_metrics: List[str], device: torch.device) -> Dict[str, torch.nn.Module]:
    if not eval_metrics:
        return {}
    models, failed = {}, []
    for name in eval_metrics:
        name = name.strip().lower()
        try:
            models[name] = pyiqa.create_metric(name).to(device).eval()
        except Exception as e:
            failed.append(name)
            logger.warning(f"Could not create metric '{name}'. Reason: {e}")
    ready = " ".join(
        f"{n}({'lower' if getattr(m, 'lower_better', False) else 'higher'}=better)" for n, m in models.items()
    )
    logger.info(f"pyiqa metrics ready: {ready}")
    if failed:
        logger.error(f"Requested metrics NOT available, missing from results: {failed}")
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

        with quiet_third_party("DOVER"):
            from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
            from dover.models import DOVER

            with open(os.path.join(root, opt_file)) as f:
                opt = yaml.safe_load(f)
            weights = opt["test_load_path"]
            if not os.path.isabs(weights):
                weights = os.path.join(root, weights)

            self.device = device
            self.evaluator = DOVER(**opt["model"]["args"]).to(device).eval()
            self.evaluator.load_state_dict(torch.load(weights, map_location=device))  # strict

        n_params = sum(p.numel() for p in self.evaluator.parameters())
        logger.info(f"DOVER ready | weights={weights} | params={n_params / 1e6:.1f}M (strict load)")

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
            write_video_lossless(video, path, preset="ultrafast")  # lossless either way, just faster
            logger.debug(f"[DOVER] Wrote temp lossless video in {time.time() - t0:.2f}s")

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
                logger.debug(f"[DOVER] Model inference took {time.time() - t1:.2f}s")
            finally:
                decord.bridge.set_bridge("torch")

        return {
            "dover": self._fuse(technical, aesthetic),
            "dover_technical": technical,
            "dover_aesthetic": aesthetic,
        }


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
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
# Validation (eval mode)
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
    run_info: Optional[dict] = None,
) -> dict:
    """
    Run `inference_fn` on every LR video, score the in-memory output against GT.

    Args:
        inference_fn: (model, lr[F,C,H,W] in [0,1], device) -> pred[F,C,H',W'] in [0,1].
        lr_path / gt_path: directories matched by name. Each entry is a video file
            or a sub-directory of image frames (png/jpg/...).
        model: pipeline exposing .transformer and .vae (the whole pipe, NOT pipe.transformer);
            used to switch to eval(), enable VAE tiling/slicing, and restore previous state.
            None skips all of that.
        pred_path: where to write preview videos (lossless). None disables saving.
        metric_models: pre-built pyiqa models (reuse across calls to avoid reloading).
            All go through evaluate_video_metrics vs GT.
        dover_scorer: DoverScorer (or any clip -> {name: score} callable) for no-reference
            whole-clip metrics; its scores are added to each video's results.
        seed: base seed; each video uses a seed derived from (seed, stem).
        quantize_8bit: snap pred to the 8-bit grid before scoring, so the number
            equals what a lossless file on disk would give (effect ~0.006 dB at 30 dB PSNR).
        save_preds: write preview videos to pred_path (never read back for scoring).
        run_info: free-form dict (args, commit, checksums) stored in the report.

    Returns:
        {"per_video", "average", "num_videos", "eval_metrics", "protocol", "timing", "run_info"}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_start = time.time()
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

    built_here = metric_models is None
    models = metric_models if metric_models is not None else build_metric_models(eval_metrics, device)
    if not models and dover_scorer is None:
        raise RuntimeError(f"None of the requested metrics could be created: {eval_metrics}")

    logger.info(
        f"Validation | {len(stems)} videos | metrics={list(models)}{'+dover' if dover_scorer else ''} | "
        f"in-memory scoring, quantize_8bit={quantize_8bit} seed={seed} crop={crop} "
        f"y_channel={test_y_channel} batch_mode={batch_mode}"
    )

    if pred_path is not None and save_preds:
        Path(pred_path).mkdir(parents=True, exist_ok=True)

    was_training = model.transformer.training if model is not None else None
    rng_state = _save_rng_state()
    if model is not None:
        model.transformer.eval()
    is_tiling_enabled = is_slicing_enabled = None
    per_video: Dict[str, Dict[str, float]] = {}
    timings: Dict[str, float] = defaultdict(float)
    total_frames = 0
    total_videos = len(stems)
    on_cuda = torch.cuda.is_available()

    try:
        if model is not None:
            is_tiling_enabled = model.vae.use_tiling
            is_slicing_enabled = model.vae.use_slicing
            model.vae.enable_tiling()
            model.vae.enable_slicing()

        with torch.no_grad():
            # tqdm only when attached to a terminal; piped/tee'd runs get clean log lines instead
            for idx, stem in enumerate(tqdm(stems, desc="Videos", disable=not sys.stderr.isatty())):
                tag = f"[{idx + 1}/{total_videos}] {stem}"
                t_vid = time.time()
                video_seed = _video_seed(seed, stem)
                torch.manual_seed(video_seed)  # seeds CPU and all CUDA devices
                logger.debug(f"{tag} seed={video_seed}")
                if on_cuda:
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                lr = read_video_tensor(lr_videos[stem]).to(device)
                gt = read_video_tensor(gt_videos[stem])
                timings["read"] += time.time() - t0

                t0 = time.time()
                pred = inference_fn(model, lr, device).detach().float().cpu()  # .cpu() syncs: timing is exact
                t_inf = time.time() - t0
                timings["inference"] += t_inf

                if pred.shape != gt.shape:
                    raise ValueError(f"{stem}: pred {tuple(pred.shape)} vs GT {tuple(gt.shape)}")
                total_frames += pred.shape[0]

                pmin, pmax = pred.min().item(), pred.max().item()  # also catches NaN/Inf
                logger.info(
                    f"{tag} | {pred.shape[0]} frames {tuple(lr.shape[-2:])} -> {tuple(pred.shape[-2:])} "
                    f"| pred range [{pmin:.3f}, {pmax:.3f}]"
                )
                if not (math.isfinite(pmin) and math.isfinite(pmax)):
                    logger.warning(f"{tag} prediction contains NaN/Inf: its metrics will be invalid")

                pred = pred.clamp(0.0, 1.0)
                if quantize_8bit:
                    pred = (pred * 255.0).round() / 255.0

                scores: Dict[str, float] = {}
                t0 = time.time()
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
                t_eval = time.time() - t0
                timings["eval"] += t_eval

                t_dover = 0.0
                if dover_scorer is not None:
                    t0 = time.time()
                    scores.update(dover_scorer(pred))
                    t_dover = time.time() - t0
                    timings["dover"] += t_dover
                per_video[stem] = scores

                t_save = 0.0
                if pred_path is not None and save_preds:
                    t0 = time.time()
                    write_video_lossless(pred, Path(pred_path) / f"{stem}.mkv", fps=fps)
                    t_save = time.time() - t0
                    timings["save"] += t_save

                elapsed = time.time() - val_start
                eta = elapsed / (idx + 1) * (total_videos - idx - 1)
                peak = f" | peak GPU {torch.cuda.max_memory_allocated() / GIB:.1f} GiB" if on_cuda else ""
                logger.info(f"{tag} | {_fmt_scores(scores)}")
                logger.info(
                    f"{tag} | infer {t_inf:.1f}s eval {t_eval:.1f}s dover {t_dover:.1f}s save {t_save:.1f}s "
                    f"| total {time.time() - t_vid:.1f}s{peak} | ETA {eta / 60:.1f} min"
                )
                if total_videos > 1 and idx + 1 < total_videos:
                    running = {n: float(np.mean([r[n] for r in per_video.values()])) for n in scores}
                    logger.info(f"running avg over {idx + 1}: {_fmt_scores(running)}")
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
    average = {n: round(float(np.mean([r[n] for r in per_video.values()])), 4) for n in metric_names}
    bad = [s for s, r in per_video.items() if any(not math.isfinite(v) for v in r.values())]
    if bad:
        logger.warning(f"Non-finite metrics in {bad}: the averages involving them are invalid")

    total_time = time.time() - val_start
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
        "timing": {**{k: round(v, 2) for k, v in timings.items()}, "total": round(total_time, 2)},
        "run_info": run_info or {},
    }

    log_results_table(per_video, average)
    inf_fps = total_frames / timings["inference"] if timings["inference"] else 0.0
    logger.info(
        "Time breakdown: " + " | ".join(f"{k} {v:.1f}s" for k, v in timings.items())
        + f" | total {total_time:.1f}s | {total_frames} frames, {inf_fps:.2f} frames/s inference"
    )

    if output_json is None and pred_path is not None:
        output_json = Path(pred_path) / "validation_metrics.json"
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved report to {output_json}")

    return report


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def configure_tcg_module(pipe: CogVideoXPipeline, drop_ratio: float, block_intervals: List[int], logger) -> None:
    """Set token-merge/unmerge layers on the transformer."""
    t = pipe.transformer
    if not hasattr(t, "tcg"):
        raise RuntimeError(
            "transformer has no .tcg: patch_CogVideoXTransformer3DModel() must run BEFORE from_pretrained"
        )
    intervals = sorted(block_intervals)
    t.tcg.drop_ratio = drop_ratio
    t.merger_layers = intervals[::2]
    t.unmerger_layers = intervals[1::2]
    if len(t.merger_layers) != len(t.unmerger_layers):
        logger.warning(f"Odd number of block_intervals {intervals}: merge/unmerge layers are unbalanced")
    logger.info(
        f"TCG | drop_ratio={t.tcg.drop_ratio} merge_layers={t.merger_layers} "
        f"unmerge_layers={t.unmerger_layers} mode={'TEMPORAL' if t.tcg.use_temporal_grouping else 'SPATIAL'}"
    )


def _find_in_stage(root: str, stage: str, filename: str) -> Optional[str]:
    """First existing <root>/<stage>/<filename> or <root>/<filename>, else None."""
    for cand in (os.path.join(root, stage, filename), os.path.join(root, filename)):
        if os.path.isfile(cand):
            return cand
    return None


def load_full_transformer_weights(pipe: CogVideoXPipeline, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}  # strip DDP prefix
    pipe.transformer.load_state_dict(state, strict=True)  # raises on any mismatch
    logger.info(f"Full transformer weights: {ckpt_path} ({len(state)} tensors, strict load OK)")
    del state


def load_stage_weights(pipe: CogVideoXPipeline, path: str, stage: str) -> None:
    """Load a LoRA adapter (merged in) or a full transformer.pt for the given stage."""
    adapter_cfg = _find_in_stage(path, stage, "adapter_config.json")
    if adapter_cfg is not None:
        adapter_dir = os.path.dirname(adapter_cfg)
        peft_model = PeftModel.from_pretrained(pipe.transformer, adapter_dir, is_trainable=False)
        cfg = next(iter(peft_model.peft_config.values()))
        n_adapted = sum(hasattr(m, "lora_A") for m in peft_model.modules())
        targets = cfg.target_modules
        targets = sorted(targets) if isinstance(targets, (set, list)) else targets
        logger.info(
            f"[{stage}] LoRA from {adapter_dir} | r={cfg.r} alpha={cfg.lora_alpha} "
            f"adapted_layers={n_adapted} targets={targets}"
        )
        if n_adapted == 0:
            logger.error(f"[{stage}] adapter attached to 0 layers: target_modules do not match this model")
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
    with log_step(f"Loading CogVideoXPipeline from {args.model_path}"):
        pipe = CogVideoXPipeline.from_pretrained(
            args.model_path, tokenizer=None, text_encoder=None, torch_dtype=DTYPES[args.dtype]
        )
        pipe.to(device)

    fp = model_fingerprint(pipe.transformer)
    base_params = fp["params"]
    log_model_state("base", fp)

    for path, stage in ((args.load_s1, "S1"), (args.load_s2, "S2")):
        if not path:
            continue
        with log_step(f"Loading {stage} weights"):
            load_stage_weights(pipe, path, stage)
        new = model_fingerprint(pipe.transformer)
        log_model_state(stage, new)
        if new["params"] != base_params:
            logger.error(f"[{stage}] param count changed {base_params} -> {new['params']} (PEFT wrapper left in?)")
        if new["checksum"] == fp["checksum"]:
            logger.warning(f"[{stage}] weights IDENTICAL to before loading: this stage had no effect, check the path")
        fp = new

    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()
    log_pipe_summary(pipe)
    log_gpu_mem("after pipeline load")
    return pipe


@torch.no_grad()
def build_empty_prompt_embedding(args, device: torch.device) -> torch.Tensor:
    """Empty-prompt T5 embedding (from file if given, else T5 is loaded temporarily and freed)."""
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
        source = path
    else:
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
        emb = text_encoder(ids)[0]
        source = "computed with T5 (max_length=226)"
        del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    emb = emb.to(dtype)
    stats = emb.float()
    logger.info(
        f"Empty prompt embedding | {source} | shape={tuple(emb.shape)} dtype={emb.dtype} "
        f"mean={stats.mean().item():.4f} std={stats.std().item():.4f}"
    )
    if not torch.isfinite(stats).all():
        raise ValueError("Empty prompt embedding contains NaN/Inf")
    return emb


# --------------------------------------------------------------------------- #
# Memory mode
# --------------------------------------------------------------------------- #
class PhaseMemoryTracker:
    """Per-module peak CUDA memory via forward hooks (instrumented run only).

    A stack of active modules lets nested hooks and the SDPA wrapper reset the allocator's
    peak counter without losing the enclosing modules' peaks: every fold credits all active modules.
    """

    def __init__(self):
        self.alloc: Dict[str, int] = defaultdict(int)  # peak allocated while module active
        self.local: Dict[str, int] = defaultdict(int)  # peak minus allocated at entry (module's own memory)
        self.sdpa: Dict[str, dict] = {}                # per-attention SDPA kernel stats
        self.g_alloc = 0
        self.g_reserved = 0
        self._stack: List[list] = []                   # [name, alloc_at_entry, peak_so_far]
        self._handles: list = []
        self._orig_sdpa = None

    def _fold(self) -> None:
        a, r = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        self.g_alloc, self.g_reserved = max(self.g_alloc, a), max(self.g_reserved, r)
        for e in self._stack:
            e[2] = max(e[2], a)

    def attach(self, name: str, module: torch.nn.Module) -> None:
        def pre(_m, _args):
            self._fold()
            torch.cuda.reset_peak_memory_stats()
            self._stack.append([name, torch.cuda.memory_allocated(), 0])

        def post(_m, _args, _out):
            self._fold()
            torch.cuda.reset_peak_memory_stats()
            n, entry, peak = self._stack.pop()
            self.alloc[n] = max(self.alloc[n], peak)
            self.local[n] = max(self.local[n], peak - entry)

        self._handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]

    def patch_sdpa(self) -> None:
        """Wrap F.scaled_dot_product_attention to measure kernel workspace per attention layer."""
        import torch.nn.functional as F

        orig = self._orig_sdpa = F.scaled_dot_product_attention

        def wrapped(q, k, v, *args, **kwargs):
            self._fold()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = orig(q, k, v, *args, **kwargs)
            self._fold()
            peak = torch.cuda.max_memory_allocated()
            ws = max(0, peak - before - out.numel() * out.element_size())  # extra beyond the output tensor
            key = next((e[0] for e in reversed(self._stack) if e[0].startswith("attn_")), "sdpa")
            if q.dim() == 4:  # [B, heads, N_q, d]
                scores = q.shape[0] * q.shape[1] * q.shape[-2] * k.shape[-2] * q.element_size()
            else:
                scores = 0
            rec = self.sdpa.setdefault(key, {"workspace": 0, "scores_if_materialized": 0})
            rec["workspace"] = max(rec["workspace"], ws)
            rec["scores_if_materialized"] = max(rec["scores_if_materialized"], scores)
            rec["q_shape"] = tuple(q.shape)
            rec["dtype"] = str(q.dtype).replace("torch.", "")
            return out

        F.scaled_dot_product_attention = wrapped

    def finish(self) -> None:
        self._fold()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._stack.clear()
        if self._orig_sdpa is not None:
            import torch.nn.functional as F

            F.scaled_dot_product_attention = self._orig_sdpa
            self._orig_sdpa = None


def _cuda_cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def set_merging(pipe: CogVideoXPipeline, drop_ratio: Optional[float], block_intervals: List[int]) -> None:
    """drop_ratio=None -> baseline with token merging switched off."""
    if drop_ratio is None:
        pipe.transformer.merger_layers, pipe.transformer.unmerger_layers = [], []
    else:
        configure_tcg_module(pipe, drop_ratio, block_intervals, logger)


def _attn_recs(row: dict) -> List[dict]:
    """SDPA records belonging to transformer attention layers (excludes e.g. VAE attention)."""
    return [v for k, v in row.get("sdpa", {}).items() if k.startswith("attn_")]

def latent_geometry(pipe, frames: int, pix_h: int, pix_w: int):
    """Pixel-space clip size -> latent size and transformer token count. Fails loudly on bad sizes."""
    cfg = pipe.transformer.config
    s, t = pipe.vae_scale_factor_spatial, pipe.vae_scale_factor_temporal
    print(f"S: {s}, T: {t}")
    p, p_t = cfg.patch_size, getattr(cfg, "patch_size_t", None)
    if (frames - 1) % t:
        raise ValueError(f"frames={frames}: need 1 + {t}k (e.g. 17, 25, 33)")
    if pix_h % s or pix_w % s:
        raise ValueError(f"pixel size {pix_h}x{pix_w} not divisible by VAE spatial factor {s}")
    lat_f, lat_h, lat_w = (frames - 1) // t, pix_h // s, pix_w // s
    if lat_h % p or lat_w % p:
        raise ValueError(f"latent {lat_h}x{lat_w} not divisible by patch_size {p}")
    if p_t and lat_f % p_t:
        raise ValueError(f"latent frames {lat_f} not divisible by patch_size_t {p_t}")
    tokens = (lat_f // (p_t or 1)) * (lat_h // p) * (lat_w // p)
    return lat_f, lat_h, lat_w, tokens


def make_transformer_fwd(pipe, prompt_emb: torch.Tensor, timestep: int, rotary):
    """x: [B, F_lat, C, h, w] latent -> transformer output. Nothing else is in the timed path."""
    t = pipe.transformer

    def fwd(x: torch.Tensor):
        B = x.shape[0]
        ts = torch.full((B,), timestep, device=x.device, dtype=torch.long)
        return t(
            hidden_states=x,
            encoder_hidden_states=prompt_emb.expand(B, -1, -1),
            timestep=ts,
            image_rotary_emb=rotary,
            return_dict=False,
        )[0]

    return fwd

@torch.no_grad()
def profile_cell(fwd, pipe, shape, dtype, device, reps: int, warmup: int, per_block: bool, seed: int) -> dict:
    """Warmup -> one instrumented run (memory by attention layer) -> `reps` clean timed runs. Transformer only."""
    _cuda_cleanup()
    row: dict = {"status": "ok", "base_alloc": torch.cuda.memory_allocated()}  # transformer weights + embeddings
    tracker, x = None, None
    try:
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, device=device, dtype=dtype, generator=g)

        for _ in range(warmup):
            fwd(x)
        torch.cuda.synchronize()

        tracker = PhaseMemoryTracker()
        tracker.attach("transformer", pipe.transformer)
        blocks = list(getattr(pipe.transformer, "transformer_blocks", []))
        n_attn = 0
        for i, blk in enumerate(blocks):
            attn = getattr(blk, "attn1", None)
            if attn is not None:
                tracker.attach(f"attn_{i:02d}", attn)
                n_attn += 1
            if per_block:
                tracker.attach(f"block_{i:02d}", blk)
        if n_attn == 0:
            logger.warning("No transformer_blocks[*].attn1 found: attention memory will not be tracked")
        tracker.patch_sdpa()
        _cuda_cleanup()
        out = fwd(x)
        torch.cuda.synchronize()
        tracker.finish()
        tracker.detach()
        del out

        times, peaks_a, peaks_r = [], [], []
        for _ in range(reps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fwd(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            peaks_a.append(torch.cuda.max_memory_allocated())
            peaks_r.append(torch.cuda.max_memory_reserved())
            del out

        row.update(
            peak_alloc=max([tracker.g_alloc] + peaks_a),
            peak_reserved=max([tracker.g_reserved] + peaks_r),
            phases={k: v for k, v in tracker.alloc.items() if not k.startswith(("block_", "attn_"))},
            blocks=[tracker.alloc[f"block_{i:02d}"] for i in range(len(blocks))] if per_block else None,
            time=statistics.median(times),
            times=[round(t, 3) for t in times],
        )
        row["act"] = row["peak_alloc"] - row["base_alloc"]

        attn_ids = [i for i in range(len(blocks)) if f"attn_{i:02d}" in tracker.alloc]
        if attn_ids:
            row["attn_layers_peak"] = [tracker.alloc[f"attn_{i:02d}"] for i in attn_ids]
            row["attn_layers_own"] = [tracker.local[f"attn_{i:02d}"] for i in attn_ids]
            row["attn_peak"] = max(row["attn_layers_peak"])
            row["attn_own"] = max(row["attn_layers_own"])
            row["attn_worst_layer"] = attn_ids[row["attn_layers_own"].index(row["attn_own"])]
            row["sdpa"] = dict(tracker.sdpa)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
        logger.error(f"OOM at input shape {tuple(shape)}")
    finally:
        if tracker is not None:
            tracker.detach()
        del x
        _cuda_cleanup()
    return row

def _block_ranges(block_bytes: List[int]) -> str:
    """Collapse per-block peaks into ranges: '0-19: 14.20 | 20-24: 11.80 | ...' (GiB)."""
    vals = [b / GIB for b in block_bytes]
    out, start = [], 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or round(vals[i] * 20) != round(vals[start] * 20):  # 0.05 GiB buckets
            span = f"{start}" if i - 1 == start else f"{start}-{i - 1}"
            out.append(f"{span}: {max(vals[start:i]):.2f}")
            start = i
    return " | ".join(out)

def _find_base(rows: List[dict], r: dict) -> Optional[dict]:
    return next((b for b in rows if b["frames"] == r["frames"] and b["is_baseline"]
                 and b["status"] == "ok"), None)


def _pct(x, base) -> str:
    return f"{(x - base) / base * 100:+8.1f}%" if x is not None and base else "        -"


def _avg_stats(r: dict, merged_ids: set) -> dict:
    """Averages over blocks. 'merged' = only the blocks that run on the merged sequence.
    The same ids are used for the baseline row, so the comparison is layer-for-layer."""
    out = {}
    own = r.get("attn_layers_own")                     # per-layer attention own memory (bytes)
    if own:
        out["attn_own_mean"] = statistics.fmean(own)
        m = [v for i, v in enumerate(own) if i in merged_ids]
        out["attn_own_merged_mean"] = statistics.fmean(m) if m else None
    if r.get("blocks"):                                # needs --mem_per_block
        act = [b - r["base_alloc"] for b in r["blocks"]]   # per-block peak minus weights
        out["blk_act_mean"] = statistics.fmean(act)
        m = [v for i, v in enumerate(act) if i in merged_ids]
        out["blk_act_merged_mean"] = statistics.fmean(m) if m else None
    return out

def log_memory_table(rows: List[dict], base_gib: float, total_gib: float) -> None:
    g = lambda x: f"{x / GIB:8.2f}" if x is not None else "     n/a"
    head = (f"{'config':<22}{'frames':>7}{'tokens':>9}{'peak':>8}{'act':>8}{'d_act':>9}{'reserved':>9}"
            f"{'attn_pk':>8}{'attn_own':>9}{'sdpa_ws':>9}{'time_s':>8}{'speedup':>9}")
    lines = [f"weights/static = {base_gib:.2f} GiB | GPU total = {total_gib:.1f} GiB "
             f"(GiB, except sdpa_ws in MiB) | TRANSFORMER FORWARD ONLY", head]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"{r['config']:<22}{r['frames']:>7}  {r['status']}")
            continue
        base = _find_base(rows, r)
        d_act = _pct(r["act"], base["act"]) if base else "        -"
        spd = f"{base['time'] / r['time']:8.2f}x" if base else "        -"
        ws = max((v["workspace"] for v in _attn_recs(r)), default=None)
        ws_s = f"{ws / 2**20:9.0f}" if ws is not None else "      n/a"
        own = f"{r['attn_own'] / GIB:9.3f}" if r.get("attn_own") is not None else "      n/a"
        lines.append(
            f"{r['config']:<22}{r['frames']:>7}{r['tokens']:>9}{g(r['peak_alloc'])}{g(r['act'])}{d_act}"
            f"{g(r['peak_reserved'])}{g(r.get('attn_peak'))}{own}{ws_s}{r['time']:8.2f}{spd}"
        )
    logger.info("Memory profile:\n" + "\n".join(lines))

    for r in rows:
        base = _find_base(rows, r)
        if r["status"] != "ok" or r["is_baseline"] or base is None:
            continue
        parts = [f"peak {_pct(r['peak_alloc'], base['peak_alloc']).strip()}"]
        for key, name in (("attn_own_merged_mean", "merged-layer attention"),
                          ("blk_act_merged_mean", "merged-block activations"),
                          ("blk_act_mean", "mean block activations")):
            if r.get(key) is not None and base.get(key):
                parts.append(f"{name} {_pct(r[key], base[key]).strip()}")
        logger.info(f"{r['config']} @ {r['frames']}f: " + " | ".join(parts))
    
def log_avg_table(rows: List[dict]) -> None:
    gi = lambda x: f"{x / GIB:10.3f}" if x is not None else "       n/a"
    head = (f"{'config':<22}{'frames':>7}{'attn_mean':>10}{'d':>9}{'attn_mrg':>10}{'d':>9}"
            f"{'blk_mean':>10}{'d':>9}{'blk_mrg':>10}{'d':>9}")
    lines = ["Average memory over blocks (GiB). attn = attention own memory, blk = per-block peak minus weights; "
             "mrg = blocks that run merged only", head]
    for r in rows:
        if r["status"] != "ok":
            continue
        b = _find_base(rows, r) or {}
        cells = [f"{r['config']:<22}{r['frames']:>7}"]
        for key in ("attn_own_mean", "attn_own_merged_mean", "blk_act_mean", "blk_act_merged_mean"):
            cells.append(gi(r.get(key)) + _pct(r.get(key), b.get(key)))
        lines.append("".join(cells))
    logger.info("\n".join(lines))


def run_memory_mode(args, pipe, empty_prompt_embedding, device: torch.device, env: dict) -> dict:
    if device.type != "cuda":
        raise SystemExit("--mode memory needs a CUDA device")
    H, W = args.mem_lr_size
    pix_h, pix_w = H * args.upscale, W * args.upscale  # what the VAE encoder would have received
    configs = ([] if args.mem_no_baseline else [("baseline (no merge)", None)]) + [
        (f"drop={r:g}", r) for r in args.mem_drop_ratios
    ]

    mi = sorted(args.block_intervals)
    merged_ids = {i for a, b in zip(mi[::2], mi[1::2]) for i in range(a + 1, b + 1)}  # e.g. 15-21, 33-39

    # The VAE is never run here. Moving it off the GPU keeps its weights out of the reported peak.
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.eval()
    t = pipe.transformer
    dtype, C = t.dtype, t.config.in_channels

    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / GIB
    logger.info(
        f"Memory mode (TRANSFORMER ONLY) | pixel {pix_h}x{pix_w} (LR {H}x{W} x{args.upscale}) | "
        f"frames={args.mem_frames} | configs={[c[0] for c in configs]} | warmup={args.mem_warmup} "
        f"reps={args.mem_reps} | blocks={args.block_intervals} | latent C={C} dtype={dtype} "
        f"timestep={args.sr_noise_step} | per_block={args.mem_per_block}"
    )

    rows: List[dict] = []
    for frames in args.mem_frames:
        lat_f, lat_h, lat_w, tokens = latent_geometry(pipe, frames, pix_h, pix_w)
        shape = (1, lat_f, C, lat_h, lat_w)  # transformer layout: [B, F, C, h, w]
        rotary = None
        if t.config.use_rotary_positional_embeddings:  # built once, outside the timed region
            rotary = pipe._prepare_rotary_positional_embeddings(pix_h, pix_w, lat_f, device)
        fwd = make_transformer_fwd(pipe, empty_prompt_embedding, args.sr_noise_step, rotary)
        logger.info(f"=== {frames} frames -> latent {lat_f}x{lat_h}x{lat_w} (F x h x w) -> {tokens:,} video tokens ===")

        for label, ratio in configs:
            set_merging(pipe, ratio, args.block_intervals)
            with log_step(f"Profiling {label} @ {frames}f"):
                row = profile_cell(fwd, pipe, shape, dtype, device, args.mem_reps, args.mem_warmup,
                                   args.mem_per_block, args.seed)
            row.update(config=label, drop_ratio=ratio, frames=frames, tokens=tokens,
                       latent_shape=shape, is_baseline=ratio is None)
            if row["status"] == "ok":
                row.update(_avg_stats(row, merged_ids))
            
            rows.append(row)
            if row["status"] != "ok":
                continue
            logger.info(
                f"{label} @ {frames}f: peak {row['peak_alloc'] / GIB:.2f} GiB "
                f"(activations {row['act'] / GIB:.2f}) | reserved {row['peak_reserved'] / GIB:.2f} GiB "
                f"| {row['time']:.3f}s/forward"
            )
            if row["blocks"]:
                logger.info(f"  per-block peak (GiB): {_block_ranges(row['blocks'])}")
            if row.get("attn_own") is not None:
                recs = _attn_recs(row)
                ws = max((v["workspace"] for v in recs), default=0)
                sc = max((v["scores_if_materialized"] for v in recs), default=0)
                kind = ("n/a (no SDPA calls seen)" if not sc else
                        "likely fused kernel (flash / mem-efficient)" if ws < 0.25 * sc else
                        "LIKELY MATH FALLBACK (N^2 scores materialized)")
                logger.info(
                    f"  attention: peak {row['attn_peak'] / GIB:.2f} GiB | worst layer {row['attn_worst_layer']} "
                    f"own memory {row['attn_own'] / GIB:.3f} GiB | sdpa workspace {ws / 2**20:.0f} MiB vs "
                    f"{sc / 2**20:.0f} MiB if scores were materialized -> {kind}"
                )
                if args.mem_per_block:
                    logger.info(f"  per-layer attention own memory (GiB): {_block_ranges(row['attn_layers_own'])}")

    ok = [r for r in rows if r["status"] == "ok"]
    base_gib = ok[0]["base_alloc"] / GIB if ok else 0.0
    log_memory_table(rows, base_gib, total_gib)
    log_avg_table(rows)
    report = {"rows": rows, "gpu": props.name, "gpu_total_gib": round(total_gib, 2),
              "env": env, "args": vars(args)}
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved memory report to {args.output_json}")
    return report

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Video-SR validation / memory profiling (DOVE / CogVideoX)")
    p.add_argument("--mode", default="eval", choices=["eval", "memory"])
    # data (eval mode)
    p.add_argument("--lr_path", default=None, help="Dir of LR videos or per-video frame folders (png/jpg)")
    p.add_argument("--gt_path", default=None, help="Dir of GT videos or per-video frame folders (png/jpg)")
    p.add_argument("--pred_path", default=None, help="Where to write preview videos")
    p.add_argument("--output_json", default=None)
    # model
    p.add_argument("--model_path", required=True, help="DOVE / CogVideoX checkpoint dir")
    p.add_argument("--load_s1", default=None, help="Dir with S1 weights (<dir>/S1/ or <dir>/)")
    p.add_argument("--load_s2", default=None, help="Dir with S2 weights (<dir>/S2/ or <dir>/)")
    p.add_argument("--empty_prompt_embedding", default=None, help="Precomputed .pt/.safetensors (skips T5)")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    # inference
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--noise_step", type=int, default=0)
    p.add_argument("--sr_noise_step", type=int, default=399)
    p.add_argument("--tile_size_hw", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32), metavar=("H", "W"))
    # token merging (TCG)
    p.add_argument("--drop_ratio", type=float, default=0.35)
    p.add_argument("--block_intervals", type=int, nargs="+", default=[20, 25, 32, 37])
    # metrics / protocol (eval mode)
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
    # memory mode
    p.add_argument("--mem_frames", type=int, nargs="+", default=[25], help="Clip lengths to profile")
    p.add_argument("--mem_lr_size", type=int, nargs=2, default=[180, 320], metavar=("H", "W"), help="LR resolution")
    p.add_argument("--mem_drop_ratios", type=float, nargs="+", default=[0.2, 0.35, 0.5])
    p.add_argument("--mem_reps", type=int, default=3, help="Timed runs per config")
    p.add_argument("--mem_warmup", type=int, default=1)
    p.add_argument("--mem_per_block", action="store_true", help="Per-transformer-block / per-attention-layer peaks")
    p.add_argument("--mem_no_baseline", action="store_true")
    p.add_argument("--mem_no_vae_tiling", action="store_true")
    # logging
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    p.add_argument("--log_file", default=None, help="Defaults to <output_json>.log")

    args = p.parse_args()
    if args.mode == "eval" and not (args.lr_path and args.gt_path):
        p.error("--lr_path and --gt_path are required in eval mode")
    return args


def preflight(args) -> None:
    """Fail in seconds, not after a multi-minute model load."""
    paths = [("--model_path", args.model_path)]
    if args.mode == "eval":
        paths += [("--lr_path", args.lr_path), ("--gt_path", args.gt_path)]
    for flag, p in (("--load_s1", args.load_s1), ("--load_s2", args.load_s2),
                    ("--empty_prompt_embedding", args.empty_prompt_embedding)):
        if p:
            paths.append((flag, p))
    for flag, p in paths:
        if not Path(p).exists():
            raise SystemExit(f"{flag} does not exist: {p}")
    if not (args.load_s1 or args.load_s2):
        logger.warning("No --load_s1/--load_s2 given: using the BASE model")
    if args.mode != "eval":
        return

    if any(m.strip().lower() == "dover" for m in args.metrics):
        if not args.dover_root or not Path(args.dover_root).exists():
            raise SystemExit("'dover' requested but --dover_root is missing or does not exist")
    lr, gt = list_videos(args.lr_path), list_videos(args.gt_path)
    matched = sorted(set(lr) & set(gt))
    logger.info(
        f"Data: {len(lr)} LR, {len(gt)} GT, {len(matched)} matched: "
        f"{matched[:5]}{' ...' if len(matched) > 5 else ''}"
    )
    if not matched:
        raise SystemExit("No LR/GT pairs matched by name")


def main():
    args = parse_args()
    log_file = args.log_file or (str(Path(args.output_json).with_suffix(".log")) if args.output_json else None)
    setup_logging(args.log_level, log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_main = time.time()

    logger.info("=" * 60)
    logger.info(f"Video-SR validation | mode={args.mode}")
    logger.info("=" * 60)
    env = log_environment(args)
    preflight(args)

    from MajorProject_VSR.inference_func.inference import DOVEInferenceFn

    pipe = load_pipe(args, device)

    with log_step("Building empty prompt embedding"):
        empty_prompt_embedding = build_empty_prompt_embedding(args, device)

    configure_tcg_module(pipe=pipe, drop_ratio=args.drop_ratio, block_intervals=args.block_intervals, logger=logger)

    final_fp = model_fingerprint(pipe.transformer)
    log_model_state("final", final_fp)
    log_gpu_mem("model ready")

    fn_kwargs = {}
    if args.tile_size_hw is not None:
        fn_kwargs.update(tile_size_hw=tuple(args.tile_size_hw), overlap_hw=tuple(args.overlap_hw))
    logger.info(
        f"Inference | upscale={args.upscale} noise_step={args.noise_step} sr_noise_step={args.sr_noise_step} "
        f"tiling={fn_kwargs.get('tile_size_hw', 'off')}"
    )
    log_gpu_mem("model ready")

    if args.mode == "memory":
        run_memory_mode(args, pipe, empty_prompt_embedding, device, env)
        logger.info(f"Total script time: {time.time() - t_main:.1f}s")
        return

    fn = DOVEInferenceFn(
        pipe=pipe,
        upscale=args.upscale,
        upscale_mode="bilinear",
        noise_step=args.noise_step,
        sr_noise_step=args.sr_noise_step,
        empty_prompt_embedding=empty_prompt_embedding,
        **fn_kwargs,
    )

    pyiqa_metrics = [m for m in args.metrics if m.strip().lower() != "dover"]
    with log_step("Building metric models"):
        metric_models = build_metric_models(pyiqa_metrics, device)

    dover_scorer = None
    if len(pyiqa_metrics) != len(args.metrics):
        with log_step("Initializing DOVER"):
            dover_scorer = DoverScorer(args.dover_root, device, opt_file=args.dover_opt)
    log_gpu_mem("all models ready")

    run_info = {**env, "args": vars(args), "model_checksum": final_fp["checksum"], "model_params": final_fp["params"]}
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
        run_info=run_info,
    )

    logger.info(f"Total script time: {time.time() - t_main:.1f}s")
    print(json.dumps(report["average"], indent=2))


if __name__ == "__main__":
    main()