import logging
import sys
import time
from math import floor
from typing import Any

import torch
from torch import nn

# --- CHANGED: Absolute imports from diffusers instead of relative imports ---
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import apply_lora_scale
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, AttentionMixin, FeedForward
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils.logging import get_logger

logger = get_logger(__name__)

from math import floor

activations = []
import logging
import sys

def setup_tcg_logging(log_file="tcg_execution.log"):
    logger = logging.getLogger("TCG_Logger")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Prevent messages from propagating to the root logger
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [TCG] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Overwrite log file on every run
    fh = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger

# Initialize the global logger for this file
tcg_logger = setup_tcg_logging()


class TemporalCurvatureGuidance(torch.nn.Module):
    def __init__(self, patch_size_t: int = 2, use_temporal_grouping: bool = True):
        super().__init__()
        self.p_t = patch_size_t
        self.Fg = None  
        self.Hg = None  
        self.Wg = None  
        self.ep = 1e-9
        self.g_min = 100
        self.g_max = 200
        self.group_ratio = 0.2
        self.drop_ratio = 0.35
        self.use_temporal_grouping = use_temporal_grouping
 
    def set_grid_dimensions(self, Fg: int, Hg: int, Wg: int):
            """Called by the transformer to set the exact 3D grid size."""
            self.Fg = Fg
            self.Hg = Hg
            self.Wg = Wg
            tcg_logger.info(f"[Grid] Set TCG Grid -> Fg: {self.Fg}, Hg: {self.Hg}, Wg: {self.Wg} (Total: {Fg*Hg*Wg})")
 
    def _check_grid(self, N: int, context: str):
        """Cheap but meaningful sanity check: catches BOTH non-divisibility
        (loud crash, safe) and — for anyone auditing logs — surfaces the
        Fg/P actually being used so a wrong-but-divisible Fg is at least
        visible in the log stream even though it can't be auto-detected
        from N alone."""
        if self.Fg is None:
            raise RuntimeError(
                f"[{context}] self.Fg is None — set_frame_count() was not called "
                f"before this forward pass. Refusing to guess a grid."
            )
        if N % self.Fg != 0:
            raise ValueError(
                f"[{context}] Token count {N} is not divisible by Fg={self.Fg}. "
                f"This usually means set_frame_count() was called with the wrong "
                f"num_frames, or patch_embed's actual token layout doesn't match "
                f"the floor/ceil division assumed here."
            )
        P = N // self.Fg
        tcg_logger.info(f"[{context}] Grid check OK: N={N}, Fg={self.Fg}, P={P}")
        return P

    def temporal_errors(self, X):
        B, P, C = X.shape
        tcg_logger.info(f"[Scoring] Input shape: {X.shape} | Frames: {self.Fg} | Patch_t: {self.p_t}")

        Xf = X.reshape([B, self.Fg, self.Wg * self.Hg, C]).to(dtype=torch.float32)
        velocity = Xf[:, 1:, :, :] - Xf[:, :-1, :, :]
        v_norms = velocity.norm(dim=-1)
        vf_n = velocity[:, 1:, :, :]
        vf = velocity[:, :-1, :, :]
        num = (vf_n * vf).sum(dim=-1)

        cos_theta = num / (v_norms[:, 1:, :] * v_norms[:, :-1, :] + self.ep)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        curv = torch.arccos(cos_theta)
        pad_curv = self.fill_edges(curv)

        s_min = pad_curv.amin(dim=-1, keepdim=True)
        s_max = pad_curv.amax(dim=-1, keepdim=True)
        s_tcg = (pad_curv - s_min) / (s_max - s_min + self.ep)

        tcg_logger.info(f"[Scoring] Curvature Min: {s_min.shape}, {s_min.dtype}, {Xf.shape}, {pad_curv.shape} | Max: {s_max.mean().item():.4f}")
        return s_tcg

    def fill_edges(self, curvature):
        start = curvature[:, 0:1, :]
        end = curvature[:, -1:, :]
        padded_curv = torch.cat([start, curvature, end], dim=1)
        return padded_curv.reshape([curvature.shape[0], -1])

    @staticmethod
    def _extract_indices(bool_mask: torch.Tensor, count: int) -> torch.Tensor:
        """
        bool_mask: [B, N], with exactly `count` True values in every row
        (guaranteed here since k/Mr are fixed by N and drop_ratio, not by
        per-sample data). Returns [B, count] long tensor: the indices of the
        True entries per row, in ascending order. Batch-vectorized; reduces
        to plain top-k/bottom-k indexing when B==1.
        """
        order = torch.argsort(bool_mask.long(), dim=1, descending=True, stable=True)
        idx = order[:, :count]
        idx, _ = torch.sort(idx, dim=1)
        return idx

    def _pack_positions(self, mask: torch.Tensor):
        """
        Shared packing structure for the temporal-grouping merge, used by
        both `temporal_group_tokens` and `get_rope_for_merged_tokens_temporal`
        so their outputs line up token-for-token. Batch elements can end up
        with different numbers of "touched" spatial positions (P), since
        which tokens get dropped depends on each video's own content — this
        packs each row's valid positions first (padding the rest with
        zero-valued slots) instead of assuming a single shared count.

        mask: [B, N] bool, True = dropped.
        Returns a dict: mask_grid [B,Fg,P], mask_sum [B,P], order [B,P]
        (positions sorted valid-first, stable so ties keep spatial order),
        counts [B] (num valid per row), max_valid (int), P (int),
        pad_mask [B, max_valid] bool (True = real, not padding).
        """
        B, N = mask.shape
        P = self._check_grid(N, "temporal_group_tokens")
        mask_grid = mask.view(B, self.Fg, P)
        mask_sum = mask_grid.sum(dim=1)              # [B, P]
        valid_p = mask_sum > 0                        # [B, P]
        counts = valid_p.sum(dim=1)                   # [B]
        max_valid = int(counts.max().item()) if counts.numel() > 0 else 0

        order = torch.argsort(valid_p.long(), dim=1, descending=True, stable=True)  # [B, P]

        if max_valid == 0:
            pad_mask = torch.zeros(B, 0, dtype=torch.bool, device=mask.device)
        else:
            ar = torch.arange(max_valid, device=mask.device).unsqueeze(0)
            pad_mask = ar < counts.unsqueeze(1)

        return {
            "mask_grid": mask_grid,
            "mask_sum": mask_sum,
            "order": order,
            "counts": counts,
            "max_valid": max_valid,
            "pad_mask": pad_mask,
            "P": P,
        }

    def get_token_set(self, tokens, s):
        """
        Batch-vectorized: tokens [B,N,C], s [B,N]. k (number kept) and
        Mr=N-k (number dropped) are constants that only depend on N and
        drop_ratio, so every row in the batch keeps/drops the same COUNT of
        tokens — just not the same ones, since s (the curvature score)
        differs per sample. That's what makes this batchable without any
        padding at this stage (padding only becomes necessary later, in
        temporal_group_tokens, where the *number of touched spatial
        positions* genuinely varies per sample).
        """
        B, N, C = tokens.shape
        k = max(1, N - floor(self.drop_ratio * N))
        topk = torch.topk(s, k=k, dim=-1)
        kept_idx, _ = torch.sort(topk.indices, dim=-1)  # [B, k] ascending

        keep_mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        keep_mask.scatter_(1, kept_idx, True)
        mask = ~keep_mask  # True = dropped

        Hr = torch.gather(tokens, 1, kept_idx.unsqueeze(-1).expand(-1, -1, C))

        Mr = N - k
        dropped_idx = self._extract_indices(mask, Mr)
        Dr = torch.gather(tokens, 1, dropped_idx.unsqueeze(-1).expand(-1, -1, C))

        tcg_logger.info(f"[Split] Total: {N} | Kept (Local): {k} | Dropped: {N-k} | Batch: {B}")
        return Hr, k, Dr, mask, kept_idx  # kept, how many kept, dropped, drop_mask, kept original positions

    def get_even_splits(self, Mr, Gr):
        base = Mr // Gr
        remainder = Mr % Gr
        sizes = [base + 1] * remainder + [base] * (Gr - remainder)
        return sizes

    def mean_global_tokens(self, Dr):
        """Reference merge: split the flat dropped-token set into Gr
        contiguous chunks (in whatever order Dr happens to be in) and mean
        each chunk. Kept for A/B comparison against temporal_group_tokens.

        NOT batch-vectorized (unlike temporal_group_tokens) — only the
        `use_temporal_grouping=True` path was extended to batch>1. This is
        the `use_temporal_grouping=False` fallback; guard it rather than
        silently mixing tokens across samples.
        """
        if Dr.shape[0] != 1:
            raise NotImplementedError(
                "mean_global_tokens (use_temporal_grouping=False) only supports "
                "batch size 1. Use use_temporal_grouping=True for batch>1 training, "
                "or fall back to batch_size=1 with gradient accumulation."
            )
        Mr = Dr.shape[1]
        Gr = min(Mr, self.g_max, max(self.g_min, int(self.group_ratio * Mr)))
        mean_splits = self.get_even_splits(Mr, Gr)
        groups = torch.split(Dr, mean_splits, dim=1)
        H_global = torch.stack([g.mean(dim=1) for g in groups], dim=1)

        tcg_logger.info(f"[Merge] Dropped Tokens: {Mr} | Groups: {Gr} | Final Global Tokens: {H_global.shape[1]}")
        return H_global

    def temporal_group_tokens(self, tokens: torch.Tensor, mask: torch.Tensor, pack: dict = None):
        """
        Batch-vectorized. Which spatial positions have >=1 dropped frame
        (and therefore produce a "global" summary token) depends on each
        sample's own curvature scores, so the COUNT of global tokens can
        differ per batch element. We pack each row's real global tokens
        first and pad the rest with exact-zero tokens out to the batch max
        (`pack["max_valid"]`), and return `pad_mask` so the caller can feed
        an attention mask that ignores the padding — the padding never
        participates in attention and is discarded again by
        `identity_restore` in any case.

        tokens: [B, N, C]   mask: [B, N] bool (True = dropped)
        pack: optional pre-computed `_pack_positions(mask)` result — pass
        the SAME pack into `get_rope_for_merged_tokens_temporal` so the
        global-token ordering/padding for tokens and RoPE stay aligned.
        Returns (H_global [B, max_valid, C], pad_mask [B, max_valid] bool, pack).
        """
        B, N, C = tokens.shape
        if pack is None:
            pack = self._pack_positions(mask)

        Mr = int(mask.sum(dim=1).max().item())
        if pack["max_valid"] == 0:
            tcg_logger.info("[TemporalGroup] No dropped tokens in any sample; returning empty global set.")
            return tokens.new_zeros(B, 0, C), pack["pad_mask"], pack

        P = pack["P"]
        tok_grid = tokens.view(B, self.Fg, P, C)

        # 1. Zero out un-dropped tokens
        mask_expanded = pack["mask_grid"].unsqueeze(-1)
        masked_tok_grid = tok_grid.masked_fill(~mask_expanded, 0.0)

        # 2. Sum over temporal dimension and divide by per-position count
        sum_tokens = masked_tok_grid.sum(dim=1)  # [B, P, C]
        mean_tokens = sum_tokens / pack["mask_sum"].unsqueeze(-1).clamp(min=1).to(tokens.dtype)

        # 3. Pack valid positions first (per row), pad the rest with zeros
        gathered = torch.gather(mean_tokens, 1, pack["order"].unsqueeze(-1).expand(-1, -1, C))
        H_global = gathered[:, : pack["max_valid"], :]

        # ---------------------------------------------------------
        # Logging & Invariant Checks (Vectorized, per-sample)
        # ---------------------------------------------------------
        total_grouped_per_row = pack["mask_sum"].sum(dim=1)  # [B]
        dropped_per_row = mask.sum(dim=1)                    # [B]
        if not torch.equal(total_grouped_per_row, dropped_per_row):
            raise RuntimeError(
                f"[TemporalGroup] Grouped token counts {total_grouped_per_row.tolist()} != "
                f"dropped token counts {dropped_per_row.tolist()}. Tokens are being lost or "
                f"double-counted during grouping."
            )

        fully_dropped_positions = int((pack["mask_sum"] == self.Fg).sum().item())
        if fully_dropped_positions > 0:
            tcg_logger.warning(
                f"[TemporalGroup] {fully_dropped_positions} (sample, spatial-position) pair(s) "
                f"had ALL {self.Fg} frames dropped — those pixels have zero individually "
                f"preserved frames this interval, fully collapsed to one summary token."
            )

        tcg_logger.info(
            f"[TemporalGroup] Batch: {B} | Dropped Tokens/sample (max): {Mr} | "
            f"Spatial positions touched per sample: {pack['counts'].tolist()}/{P} | "
            f"Final Global Tokens (padded): {H_global.shape[1]}"
        )

        return H_global, pack["pad_mask"], pack

    def get_rope_for_merged_tokens_temporal(self, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor, k: int, pack: dict):
        """
        Batch-vectorized. `cos`/`sin` are the batch-independent rotary
        table (shape [N, D], or [1, N, D] — RoPE only depends on the
        (frame, spatial) grid, not on video content, so it is never
        batched itself). What *is* per-sample is which absolute positions
        got kept vs. dropped, so `local_cos`/`local_sin` (RoPE for the kept
        tokens) and the merged "global" RoPE both need a per-row gather.

        mask: [B, N] bool (True=dropped). k: number of kept tokens (constant
        across the batch). pack: the SAME dict returned by
        `_pack_positions(mask)` / used in `temporal_group_tokens`, so the
        global RoPE rows line up with H_global's rows (including padding).

        Returns final_cos, final_sin: [B, k + max_valid, D] (or with a
        leading dim of 1 preserved, matching the input's dimensionality).
        Padding rows are exact-zero — harmless, since those positions are
        also excluded from attention via pack["pad_mask"].
        """
        squeeze_needed = False
        if cos.dim() == 3:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            squeeze_needed = True

        B, N = mask.shape
        D = cos.shape[-1]
        P = pack["P"]

        keep_mask = ~mask
        kept_idx = self._extract_indices(keep_mask, k)  # [B, k] ascending

        cos_b = cos.unsqueeze(0).expand(B, N, D)
        sin_b = sin.unsqueeze(0).expand(B, N, D)
        local_cos = torch.gather(cos_b, 1, kept_idx.unsqueeze(-1).expand(-1, -1, D))
        local_sin = torch.gather(sin_b, 1, kept_idx.unsqueeze(-1).expand(-1, -1, D))

        if pack["max_valid"] == 0:
            final_cos, final_sin = local_cos, local_sin
        else:
            cos_grid = cos.view(self.Fg, P, D).unsqueeze(0).expand(B, self.Fg, P, D)
            sin_grid = sin.view(self.Fg, P, D).unsqueeze(0).expand(B, self.Fg, P, D)
            mask_expanded = pack["mask_grid"].unsqueeze(-1)
            masked_cos = cos_grid.masked_fill(~mask_expanded, 0.0)
            masked_sin = sin_grid.masked_fill(~mask_expanded, 0.0)

            mask_sum_clamped = pack["mask_sum"].unsqueeze(-1).clamp(min=1).to(cos.dtype)
            g_cos_all = masked_cos.sum(dim=1) / mask_sum_clamped  # [B, P, D]
            g_sin_all = masked_sin.sum(dim=1) / mask_sum_clamped

            norm = torch.sqrt(g_cos_all**2 + g_sin_all**2 + self.ep)
            g_cos_all = g_cos_all / norm
            g_sin_all = g_sin_all / norm

            gathered_cos = torch.gather(g_cos_all, 1, pack["order"].unsqueeze(-1).expand(-1, -1, D))
            gathered_sin = torch.gather(g_sin_all, 1, pack["order"].unsqueeze(-1).expand(-1, -1, D))
            global_cos = gathered_cos[:, : pack["max_valid"], :]
            global_sin = gathered_sin[:, : pack["max_valid"], :]

            # Zero out padding rows explicitly (dtype/precision-safe; they're
            # already ~0 from the masked mean at untouched positions).
            pad = pack["pad_mask"].unsqueeze(-1)
            global_cos = global_cos * pad.to(global_cos.dtype)
            global_sin = global_sin * pad.to(global_sin.dtype)

            final_cos = torch.cat([local_cos, global_cos], dim=1)
            final_sin = torch.cat([local_sin, global_sin], dim=1)

        # NOTE: unlike the old batch=1 version, final_cos/final_sin are
        # already rank-3 [B, seq, D] at this point (that's the whole point
        # of batching them) — `squeeze_needed` only tells us the *input*
        # table had a redundant leading dim of 1, there's nothing left to
        # "restore" it to.
        del squeeze_needed
        tcg_logger.info(f"[RoPE] Mapped RoPE to temporally-grouped sequence (batched). New shape: {final_cos.shape}")

        return final_cos, final_sin
    
    def get_rope_for_merged_tokens(self, cos: torch.Tensor, sin: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor):
        """Reference RoPE merge (contiguous splits). Kept for A/B comparison
        against get_rope_for_temporal_groups.

        NOT batch-vectorized — see mean_global_tokens. `mask` here is
        expected 1D (batch size 1).
        """
        if tokens.shape[0] != 1:
            raise NotImplementedError(
                "get_rope_for_merged_tokens (use_temporal_grouping=False) only supports "
                "batch size 1. Use use_temporal_grouping=True for batch>1 training."
            )
        squeeze_needed = False
        if cos.dim() == 3:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            squeeze_needed = True
        if mask.dim() == 2:
            mask = mask[0]  # [1, N] -> [N]; safe since tokens.shape[0]==1 was checked above

        N = tokens.shape[1]
        all_indices = torch.arange(N, device=tokens.device)
        local_indices = all_indices[~mask]
        dropped_indices = all_indices[mask]

        local_cos = cos[local_indices]
        local_sin = sin[local_indices]

        Mr = dropped_indices.shape[0]
        Gr = min(Mr, self.g_max, max(self.g_min, int(self.group_ratio * Mr)))
        mean_splits = self.get_even_splits(Mr, Gr)

        dropped_cos = cos[dropped_indices]
        dropped_sin = sin[dropped_indices]

        groups_cos = torch.split(dropped_cos, mean_splits, dim=0)
        groups_sin = torch.split(dropped_sin, mean_splits, dim=0)

        global_cos = torch.stack([g.mean(dim=0) for g in groups_cos], dim=0)
        global_sin = torch.stack([g.mean(dim=0) for g in groups_sin], dim=0)

        norm = torch.sqrt(global_cos**2 + global_sin**2 + self.ep)
        global_cos = global_cos / norm
        global_sin = global_sin / norm

        final_cos = torch.cat([local_cos, global_cos], dim=0)
        final_sin = torch.cat([local_sin, global_sin], dim=0)

        if squeeze_needed:
            final_cos = final_cos.unsqueeze(0)
            final_sin = final_sin.unsqueeze(0)

        tcg_logger.info(f"[RoPE] Mapped RoPE to reduced sequence. New shape: {final_cos.shape}")
        return final_cos, final_sin

    def identity_restore(self, original_tokens: torch.Tensor, merged_tokens: torch.Tensor, k: int, kept_idx: torch.Tensor) -> torch.Tensor:
        """
        Batch-vectorized. Which absolute positions were "kept" (and
        therefore need their updated values scattered back) differs per
        sample, so this takes `kept_idx` ([B, k] ascending original
        positions, as returned by `get_token_set`) rather than a single
        boolean mask shared across the batch — a plain boolean-mask
        assignment (`restored[:, keep_mask, :] = ...`) only works when the
        mask is identical for every row, which isn't true once batch>1
        samples can each drop different tokens. The padded "global" tokens
        in `merged_tokens[:, k:, :]` are simply not used here — they were
        already folded into the kept tokens' updates by attention, and are
        discarded once the local tokens are restored to their original
        positions.
        """
        B, N, D = original_tokens.shape
        idx = kept_idx.unsqueeze(-1).expand(-1, -1, D)
        src = merged_tokens[:, :k, :].to(dtype=original_tokens.dtype)

        restored = torch.scatter(original_tokens, 1, idx, src)   # new tensor, autograd-safe
        tcg_logger.info(f"[Restore] Restored sequence length to {N} for batch {B}. Kept tokens updated.")
        return restored

    # ------------------------------------------------------------------
    # TRaM-VSR CHANGE: temporal grouping
    # ------------------------------------------------------------------
    # def temporal_group_tokens(self, tokens, mask):
    #     assert tokens.shape[0] == 1, "temporal_group_tokens assumes batch size 1"
    #     N = tokens.shape[1]
    #     C = tokens.shape[-1]
    #     P = self._check_grid(N, "temporal_group_tokens")
 
    #     Mr = int(mask.sum().item())
    #     if Mr == 0:
    #         tcg_logger.info("[TemporalGroup] No dropped tokens; returning empty global set.")
    #         return tokens.new_zeros(1, 0, C), []
 
    #     mask_grid = mask.view(self.Fg, P)
    #     tok_grid = tokens[0].view(self.Fg, P, C)
 
    #     pos_tokens = []
    #     pos_indices = []
    #     fully_dropped_positions = 0
    #     group_sizes = []
 
    #     for p in range(P):
    #         dropped_f = mask_grid[:, p].nonzero(as_tuple=True)[0]
    #         if dropped_f.numel() == 0:
    #             continue
    #         if dropped_f.numel() == self.Fg:
    #             fully_dropped_positions += 1   # NEW: track the degenerate "whole pixel timeline dropped" case
    #         group_sizes.append(dropped_f.numel())
    #         pos_tokens.append(tok_grid[dropped_f, p, :].mean(dim=0))
    #         pos_indices.append((dropped_f * P + p).tolist())
 
    #     H_global = torch.stack(pos_tokens, dim=0).unsqueeze(0)
 
    #     # NEW: invariant check — every dropped token must land in exactly one group
    #     total_grouped = sum(len(g) for g in pos_indices)
    #     if total_grouped != Mr:
    #         raise RuntimeError(
    #             f"[TemporalGroup] Grouped token count ({total_grouped}) != "
    #             f"dropped token count (Mr={Mr}). Tokens are being lost or "
    #             f"double-counted during grouping."
    #         )
 
    #     if fully_dropped_positions > 0:
    #         tcg_logger.warning(
    #             f"[TemporalGroup] {fully_dropped_positions} spatial position(s) had "
    #             f"ALL {self.Fg} frames dropped — those pixels have zero individually "
    #             f"preserved frames this interval, fully collapsed to one summary token."
    #         )
 
    #     if group_sizes:
    #         tcg_logger.info(
    #             f"[TemporalGroup] Dropped Tokens: {Mr} | Spatial positions touched: "
    #             f"{len(pos_indices)}/{P} | Final Global Tokens: {H_global.shape[1]} | "
    #             f"Frames-merged-per-position min/mean/max: "
    #             f"{min(group_sizes)}/{sum(group_sizes)/len(group_sizes):.2f}/{max(group_sizes)}"
    #         )
    #     return H_global, pos_indices


    # def get_rope_for_temporal_groups(self, cos, sin, mask, pos_indices):
    #     squeeze_needed = False
    #     if cos.dim() == 3:
    #         cos = cos.squeeze(0)
    #         sin = sin.squeeze(0)
    #         squeeze_needed = True
 
    #     N = mask.shape[0]
    #     all_indices = torch.arange(N, device=cos.device)
    #     local_indices = all_indices[~mask]
    #     local_cos = cos[local_indices]
    #     local_sin = sin[local_indices]
 
    #     if len(pos_indices) == 0:
    #         final_cos, final_sin = local_cos, local_sin
    #     else:
    #         group_cos, group_sin = [], []
    #         for idxs in pos_indices:
    #             idx_tensor = torch.as_tensor(idxs, device=cos.device, dtype=torch.long)
    #             group_cos.append(cos[idx_tensor].mean(dim=0))
    #             group_sin.append(sin[idx_tensor].mean(dim=0))
    #         global_cos = torch.stack(group_cos, dim=0)
    #         global_sin = torch.stack(group_sin, dim=0)
 
    #         norm = torch.sqrt(global_cos**2 + global_sin**2 + self.ep)
    #         global_cos = global_cos / norm
    #         global_sin = global_sin / norm
 
    #         final_cos = torch.cat([local_cos, global_cos], dim=0)
    #         final_sin = torch.cat([local_sin, global_sin], dim=0)
 
    #     if squeeze_needed:
    #         final_cos = final_cos.unsqueeze(0)
    #         final_sin = final_sin.unsqueeze(0)
 
    #     tcg_logger.info(f"[RoPE] Mapped RoPE to temporally-grouped sequence. New shape: {final_cos.shape}")
    #     return final_cos, final_sin

@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
        use_temporal_grouping (`bool`, defaults to `True`):
            TRaM-VSR: if True, the merger uses temporal_group_tokens / get_rope_for_temporal_groups
            (position-aware temporal merge). If False, falls back to the reference
            mean_global_tokens / get_rope_for_merged_tokens (arbitrary contiguous split).
    """

    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int | None = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: int | None = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: int | None = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        merger_layers: list[int] | None = None,
        unmerger_layers: list[int] | None = None,
        use_temporal_grouping: bool = True,
    ):
        super().__init__()

        self.merger_layers = [20, 32]
        self.unmerger_layers = [25, 37]
        inner_dim = num_attention_heads * attention_head_dim

        p_t = patch_size_t if patch_size_t is not None else 1
        self.tcg = TemporalCurvatureGuidance(patch_size_t=p_t, use_temporal_grouping=use_temporal_grouping)
        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        > [!WARNING] > This API is 🧪 experimental.
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        > [!WARNING] > This API is 🧪 experimental.

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: int | float | torch.LongTensor,
        timestep_cond: torch.Tensor | None = None,
        ofs: int | float | torch.LongTensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        """
        The [`CogVideoXTransformer3DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_frames, channels, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            timestep_cond (`torch.Tensor`, *optional*):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the final timestep embeddings.
            ofs (`torch.Tensor`, *optional*):
                Offset embeddings used in CogVideoX-5b-I2V.
            image_rotary_emb (`tuple` of `torch.Tensor`, *optional*):
                Pre-computed rotary positional embeddings.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        batch_size, num_frames, channels, height, width = hidden_states.shape
        p = self.config.patch_size if self.config.patch_size is not None else 2
        p_t = self.config.patch_size_t if self.config.patch_size_t is not None else 1

        Fg = (num_frames + p_t - 1) // p_t
        Hg = height // p
        Wg = width // p
        self.tcg.set_grid_dimensions(Fg, Hg, Wg)

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        original_rope = None
        original_tokens = None
        mask = None
        kept_idx = None
        k = 0
        base_attention_kwargs = attention_kwargs
        active_attention_kwargs = attention_kwargs

        if hidden_states.is_cuda:
            torch.cuda.synchronize() # Ensure previous GPU ops are finished
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter() # Fallback for CPU inference

        for i, block in enumerate(self.transformer_blocks):

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, emb, image_rotary_emb, active_attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=emb, image_rotary_emb=image_rotary_emb, attention_kwargs=active_attention_kwargs,
                )

            # --- MERGER INJECTION ---
            if self.merger_layers is not None and i in self.merger_layers:
                try:
                    tcg_logger.info(f"--- MERGE TRIGGERED AT LAYER {i} (batch={hidden_states.shape[0]}) ---")
                    original_tokens = hidden_states
                    original_rope = image_rotary_emb

                    s_tcg = self.tcg.temporal_errors(hidden_states)
                    H_local, k, Dr, mask, kept_idx = self.tcg.get_token_set(hidden_states, s_tcg)
                    cos, sin = image_rotary_emb

                    if self.tcg.use_temporal_grouping:
                        tcg_logger.info(f"[LOG] MERGING IN TEMPORAL DIMENSION")
                        # TRaM-VSR CHANGE: position-aware temporal grouping
                        # instead of an arbitrary contiguous split of Dr.
                        # Batched: which spatial positions get a "global"
                        # token can differ per sample, so H_global is padded
                        # to the batch max and `global_pad_mask` marks which
                        # slots are real.
                        pack = self.tcg._pack_positions(mask)
                        H_global, global_pad_mask, pack = self.tcg.temporal_group_tokens(hidden_states, mask, pack=pack)
                        hidden_states = torch.cat([H_local, H_global], dim=1)
                        image_rotary_emb = self.tcg.get_rope_for_merged_tokens_temporal(
                            cos=cos, sin=sin, mask=mask, k=k, pack=pack
                        )

                        # Build the attention mask for the merged region: text
                        # tokens (unchanged, all real) + k local tokens (all
                        # real) + padded global tokens (real per global_pad_mask).
                        B = hidden_states.shape[0]
                        text_ok = torch.ones(B, text_seq_length, dtype=torch.bool, device=hidden_states.device)
                        local_ok = torch.ones(B, k, dtype=torch.bool, device=hidden_states.device)
                        merge_attn_mask = torch.cat([text_ok, local_ok, global_pad_mask], dim=1)
                        merged_kwargs = dict(base_attention_kwargs or {})
                        merged_kwargs["attention_mask"] = merge_attn_mask
                        active_attention_kwargs = merged_kwargs

                        if global_pad_mask.numel() > 0 and not bool(global_pad_mask.all()):
                            tcg_logger.info(
                                f"[Merge] Padded global tokens to batch max "
                                f"({global_pad_mask.shape[1]}); per-sample real counts: "
                                f"{global_pad_mask.sum(dim=1).tolist()}"
                            )

                    else:
                        tcg_logger.info(f"[LOG] MERGING IN SPATIAL DIMENSION")
                        H_global = self.tcg.mean_global_tokens(Dr)
                        hidden_states = torch.cat([H_local, H_global], dim=1)
                        image_rotary_emb = self.tcg.get_rope_for_merged_tokens(
                            cos=cos, sin=sin, tokens=original_tokens, mask=mask
                        )
                        # No padding in this (batch=1-only) fallback path — no attention mask needed.
                    expected_len = H_local.shape[1] + H_global.shape[1]
                    if hidden_states.shape[1] != expected_len:
                        raise RuntimeError(f"[Merge] H_work length {hidden_states.shape[1]} != "
                                            f"H_local+H_global ({expected_len})")
                    cos_chk, sin_chk = image_rotary_emb
                    rope_len = cos_chk.shape[-2] if cos_chk.dim() == 3 else cos_chk.shape[0]
                    if rope_len != hidden_states.shape[1]:
                        raise RuntimeError(f"[Merge] RoPE length {rope_len} != token length "
                                            f"{hidden_states.shape[1]} — attention will silently "
                                            f"misapply rotary embeddings if this ever passes unchecked.")
                    tcg_logger.info(f"[Merge] Post-merge check OK: {hidden_states.shape[1]} tokens, "
                                    f"RoPE length {rope_len} matches.")

                    tcg_logger.info(f"--- MERGE COMPLETE. New Hidden States Shape: {hidden_states.shape} ---")
                except Exception as e:
                    # Logs the exact error and traceback to the text file!
                    tcg_logger.exception(f"CRITICAL ERROR during MERGE at layer {i}: {str(e)}")
                    raise e  # Re-raise so the program still halts

            # --- UNMERGER INJECTION ---
            if self.unmerger_layers is not None and i in self.unmerger_layers:
                try:
                    tcg_logger.info(f"--- UNMERGE TRIGGERED AT LAYER {i} ---")
                    if self.tcg.use_temporal_grouping:
                        hidden_states = self.tcg.identity_restore(
                            original_tokens=original_tokens,
                            merged_tokens=hidden_states,
                            k=k,
                            kept_idx=kept_idx,
                        )
                    else:
                        # Legacy batch=1-only path still keyed off the boolean mask.
                        hidden_states = self.tcg.identity_restore(
                            original_tokens=original_tokens,
                            merged_tokens=hidden_states,
                            k=k,
                            kept_idx=self.tcg._extract_indices(~mask, k),
                        )
                    image_rotary_emb = original_rope
                    active_attention_kwargs = base_attention_kwargs

                    original_rope = None
                    original_tokens = None
                    mask = None
                    kept_idx = None
                    k = 0
                    tcg_logger.info(f"--- UNMERGE COMPLETE. Restored Hidden States Shape: {hidden_states.shape} ---")
                except Exception as e:
                    tcg_logger.exception(f"CRITICAL ERROR during UNMERGE at layer {i}: {str(e)}")
                    raise e
        if hidden_states.is_cuda:
            end_event.record()
            torch.cuda.synchronize() # Wait for all 42 blocks to finish
            block_inference_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        else:
            block_inference_time = time.perf_counter() - start_time
            
        tcg_logger.info(
            f"[Timing] Total inference time for all {len(self.transformer_blocks)} transformer blocks (incl. TCG overhead): "
            f"{block_inference_time:.4f} seconds"
        )
        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)