import random
import logging
import tempfile
from typing import Dict, Tuple, List, Optional
import os 
import json
import numpy as np
from torchvision import transforms
from pathlib import Path
import math
import decord
try:
    from torchvision.io import write_video, read_video
    _VIDEO_IO_AVAILABLE = True
except ImportError:
    _VIDEO_IO_AVAILABLE = False

from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import functional as F

import cv2


class RealWorldDegradation:
    def __init__(self, scale: int = 4, blur_prob: float = 0.8, blur_kernel_range: Tuple[int, int] = (7, 21),
                 blur_sigma_range: Tuple[float, float] = (0.2, 3.0), resize_modes: Tuple[str, ...] = ("bilinear", "bicubic", "area"),
                 noise_prob: float = 0.5, noise_sigma_range: Tuple[float, float] = (0.0, 15.0),
                 jpeg_prob: float = 0.7, jpeg_quality_range: Tuple[int, int] = (30, 95),
                 video_compress_prob: float = 0.0, video_compress_crf_range: Tuple[int, int] = (23, 35),
                 fps: int = 24, shuffle_order: bool = True, logger: logging.Logger | None = None):
        self.scale = scale
        self.blur_prob, self.blur_kernel_range, self.blur_sigma_range = blur_prob, blur_kernel_range, blur_sigma_range
        self.resize_modes = resize_modes
        self.noise_prob, self.noise_sigma_range = noise_prob, noise_sigma_range
        self.jpeg_prob, self.jpeg_quality_range = jpeg_prob, jpeg_quality_range
        self.video_compress_prob, self.video_compress_crf_range = video_compress_prob, video_compress_crf_range
        self.fps, self.shuffle_order = fps, shuffle_order
        self.logger = logger
        
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
        if self.logger is not None:
            self.logger.debug(f"[DEGRADE] Pipeline: {' -> '.join(applied_ops)} | Input HR: {tuple(hr_frames.shape)}")

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
                 logger=logging.Logger, scale: int = 4, is_train: bool = True, video_extensions: Optional[Tuple[str, ...]] = None,
                 degradation_kwargs: Optional[dict] = None, prompt_json: Optional[str] = None, ):

        super().__init__()
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.hr_crop_size = hr_crop_size
        self.scale = scale
        self.is_train = is_train
        self.video_extensions = tuple(e.lower() for e in (video_extensions or self.DEFAULT_VIDEO_EXTS))
        self.logger = logger
        self.video_paths = sorted([
            os.path.join(video_dir, f) for f in os.listdir(video_dir)
            if f.lower().endswith(self.video_extensions)
        ])
        if not self.video_paths:
            raise ValueError(f"No video files found in {video_dir}")

        degradation_kwargs = dict(degradation_kwargs or {})
        degradation_kwargs.setdefault('scale', scale)
        degradation_kwargs.setdefault('logger', logger)
        self.degrade = RealWorldDegradation(**degradation_kwargs)
        
        self.prompts = {}
        if prompt_json is not None and os.path.exists(prompt_json):
            with open(prompt_json, "r") as f:
                self.prompts = json.load(f)
            if self.logger is not None:
                self.logger.info(f"Loaded {len(self.prompts)} prompts from {prompt_json}")

        if self.logger is not None:
            self.logger.info(f"VideoFileSRDataset initialized: {len(self.video_paths)} clips found in {video_dir}")

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
