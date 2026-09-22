"""
dove_inference_fn.py
--------------------
Plugs the standalone DOVE inference (`inference.py`) into `validation.py`.

`validation.generate_sr_videos` expects

    inference_fn(model, lr_video, device) -> pred_video

    lr_video   : [F, C, H, W] float32 in [0, 1]   (LR, native resolution)
    pred_video : [F, C, H_lr*upscale, W_lr*upscale] float32 in [0, 1]

`DOVEInferenceFn` implements that contract with exactly the same steps as the
`__main__` loop of inference.py:

    pad frames to (F-1) % 8 == 0 (repeat last) and H, W to multiples of 16
    -> bilinear upscale -> [-1, 1] -> temporal chunks x spatial tiles
    -> process_video() per chunk -> blend valid regions -> crop padding

Usage
-----
    from dove_inference_fn import DOVEInferenceFn, load_dove_pipeline, load_empty_prompt_embedding
    from validation import validation_pred

    pipe = load_dove_pipeline("THUDM/CogVideoX1.5-5B", lora_path=None)
    fn = DOVEInferenceFn(pipe, empty_prompt_embedding=load_empty_prompt_embedding())

    report = validation_pred(
        pred_path="results/pred", gt_path="UDM10/GT", lr_path="UDM10/LR",
        eval_metrics=["psnr", "ssim", "lpips", "dists"],
        model=pipe.transformer,      # only used for .eval()/.train() handling
        inference_fn=fn,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from safetensors.torch import load_file

from MajorProject_VSR.inference_func.dove_inference import (
    get_valid_tile_region,
    make_spatial_tiles,
    make_temporal_chunks,
    process_video,
)
from MajorProject_VSR.patchTransformer import patch_CogVideoXTransformer3DModel

logger = logging.getLogger("dove_inference_fn")

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
DEFAULT_EMPTY_PROMPT_PATH = (
    "pretrained_models/prompt_embeddings/"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
)

FRAME_MULTIPLE = 8     # the VAE needs (F - 1) % 8 == 0
SPATIAL_MULTIPLE = 16  # LR-resolution H and W are padded to a multiple of this


# --------------------------------------------------------------------------- #
# Pipeline / embedding loading (mirrors the setup in inference.py)
# --------------------------------------------------------------------------- #
def load_empty_prompt_embedding(path: str = DEFAULT_EMPTY_PROMPT_PATH) -> Optional[torch.Tensor]:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Empty prompt embedding not found at {p}; empty prompts will be encoded on the fly.")
        return None
    try:
        return load_file(str(p))["prompt_embedding"]
    except Exception as e:
        logger.warning(f"Failed to load empty prompt embedding from {p}: {e}")
        return None


def load_dove_pipeline(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    lora_path: Optional[str] = None,
    cpu_offload: bool = False,
    vae_slicing_tiling: bool = False,
) -> CogVideoXPipeline:
    """Same setup order as inference.py. Call ONCE per process: the transformer patch is global."""
    patch_CogVideoXTransformer3DModel()  # must run before from_pretrained
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)

    if lora_path:
        logger.info(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)  # lora_scale = lora_alpha / rank

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    if vae_slicing_tiling:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    return pipe

def pad_lr_video(video: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
    """[F, C, H, W] -> padded video and (pad_f, pad_h, pad_w).

    Frames: repeat the last frame until (F - 1) % 8 == 0.
    Space : zero-pad right/bottom to multiples of 16 (same as inference.py).
    """
    pad_f = (-(video.shape[0] - 1)) % FRAME_MULTIPLE
    if pad_f:
        video = torch.cat([video, video[-1:].expand(pad_f, -1, -1, -1)], dim=0)

    h, w = video.shape[-2:]
    pad_h = (-h) % SPATIAL_MULTIPLE
    pad_w = (-w) % SPATIAL_MULTIPLE
    if pad_h or pad_w:
        video = F.pad(video, (0, pad_w, 0, pad_h))  # (w_left, w_right, h_top, h_bottom)
    return video, pad_f, pad_h, pad_w


# --------------------------------------------------------------------------- #
# The inference function
# --------------------------------------------------------------------------- #
class DOVEInferenceFn:
    """Callable `inference_fn(model, lr_video, device)` for validation.generate_sr_videos.

    The `model` argument is ignored: inference runs through the captured `pipe`.
    Validation has no per-video prompts, so `prompt` is used for every video
    ("" selects the pre-computed empty-prompt embedding when one is given).
    """

    def __init__(
        self,
        pipe: CogVideoXPipeline,
        upscale: int = 4,
        upscale_mode: str = "bilinear",
        noise_step: int = 0,
        sr_noise_step: int = 399,
        empty_prompt_embedding: Optional[torch.Tensor] = None,
        prompt: str = "",
        chunk_len: int = 0,
        overlap_t: int = 8,
        tile_size_hw: Sequence[int] = (0, 0),
        overlap_hw: Sequence[int] = (32, 32),
    ):
        if chunk_len > 0 and overlap_t >= chunk_len:
            raise ValueError("chunk_len must be greater than overlap_t")
        self.pipe = pipe
        self.upscale = upscale
        self.upscale_mode = upscale_mode
        self.noise_step = noise_step
        self.sr_noise_step = sr_noise_step
        self.empty_prompt_embedding = empty_prompt_embedding
        self.prompt = prompt
        self.chunk_len = chunk_len
        self.tile_size_hw = tuple(tile_size_hw)
        # Overlaps only apply when chunking/tiling is on (same rule as inference.py).
        self.overlap_t = overlap_t if chunk_len > 0 else 0
        self.overlap_hw = tuple(overlap_hw) if any(self.tile_size_hw) else (0, 0)

    def _upscale_to_model_range(self, video: torch.Tensor) -> torch.Tensor:
        """[F, C, H, W] in [0, 1] -> [1, C, F, sH, sW] in [-1, 1]."""
        h, w = video.shape[-2:]
        video = F.interpolate(
            video,
            size=(h * self.upscale, w * self.upscale),
            mode=self.upscale_mode,
            align_corners=False if self.upscale_mode in ("bilinear", "bicubic") else None,
        )
        return (video * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    @torch.no_grad()
    def __call__(self, model, lr_video: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Assemble on CPU so long videos don't hold the full-res output on the GPU;
        # process_video moves each chunk to the pipeline's device itself.
        lr_video = lr_video.detach().float().cpu()
        orig_f, _, orig_h, orig_w = lr_video.shape

        padded, pad_f, pad_h, pad_w = pad_lr_video(lr_video)
        video = self._upscale_to_model_range(padded)  # [1, C, F, H, W]
        _, _, total_f, total_h, total_w = video.shape

        time_chunks = make_temporal_chunks(total_f, self.chunk_len, self.overlap_t)
        spatial_tiles = make_spatial_tiles(total_h, total_w, self.tile_size_hw, self.overlap_hw)
        logger.info(
            f"SR: {orig_f}f {orig_h}x{orig_w} -> {total_f}f {total_h}x{total_w} "
            f"(pad f/h/w = {pad_f}/{pad_h}/{pad_w}) | {len(time_chunks)} chunk(s) x {len(spatial_tiles)} tile(s)"
        )

        output = torch.zeros_like(video)
        coverage = torch.zeros(1, 1, total_f, total_h, total_w, dtype=torch.uint8)

        for t0, t1 in time_chunks:
            for h0, h1, w0, w1 in spatial_tiles:
                chunk = video[:, :, t0:t1, h0:h1, w0:w1]
                sr_chunk = process_video(
                    pipe=self.pipe,
                    video=chunk,
                    prompt=self.prompt,
                    noise_step=self.noise_step,
                    sr_noise_step=self.sr_noise_step,
                    empty_prompt_embedding=self.empty_prompt_embedding,
                )  # [1, C, f, h, w] in [0, 1]

                r = get_valid_tile_region(
                    t0, t1, h0, h1, w0, w1,
                    video_shape=video.shape,
                    overlap_t=self.overlap_t,
                    overlap_h=self.overlap_hw[0],
                    overlap_w=self.overlap_hw[1],
                )

                dst = (slice(None), slice(None),
                       slice(r["out_t_start"], r["out_t_end"]),
                       slice(r["out_h_start"], r["out_h_end"]),
                       slice(r["out_w_start"], r["out_w_end"]))
                src = (slice(None), slice(None),
                       slice(r["valid_t_start"], r["valid_t_end"]),
                       slice(r["valid_h_start"], r["valid_h_end"]),
                       slice(r["valid_w_start"], r["valid_w_end"]))
                output[dst] = sr_chunk[src].to("cpu", torch.float32)
                coverage[dst] += 1

        if (coverage == 0).any():
            raise RuntimeError("Tiling left part of the output unwritten (check chunk/tile/overlap settings).")
        if (coverage > 1).any():
            raise RuntimeError("Tiling wrote part of the output more than once (check chunk/tile/overlap settings).")

        # [1, C, F, H, W] -> [F, C, H, W], drop padding using the ORIGINAL sizes
        pred = output[0].permute(1, 0, 2, 3)
        pred = pred[:orig_f, :, : orig_h * self.upscale, : orig_w * self.upscale].contiguous()
        return pred