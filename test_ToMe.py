import torch
from typing import Any
import time
import torch
from torch import nn
from math import floor

activations = []
import logging
import sys

# Shared test grid dimensions — change these in one place to test a
# different Fg/Hg/Wg configuration across TestMatrix, test_batch_independence,
# and test_reference_paths_guard_batch, instead of editing each separately.
TEST_FG = 10
TEST_HG = 80
TEST_WG = 80

def setup_tcg_logging(log_file="tcg_execution.log"):
    logger = logging.getLogger("TCG_Logger")
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times if re-run in Jupyter/interactive shells
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] [TCG] %(message)s', datefmt='%H:%M:%S')

        # 1. Console Handler (Standard Output)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # 2. File Handler (Text File)
        # Use mode='w' to overwrite on every new run, or 'a' to append
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


class TemporalCurvatureGuidance(torch.nn.Module):
    def __init__(self, patch_size_t: int = 2, use_temporal_grouping: bool = True, drop_ratio: int | None = 0.35):
        super().__init__()
        self.p_t = patch_size_t
        self.Fg = None  
        self.Hg = None  
        self.Wg = None  
        self.ep = 1e-9
        self.g_min = 100
        self.g_max = 200
        self.group_ratio = 0.2
        self.drop_ratio = drop_ratio
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

        Xf = X.reshape([B, self.Fg, self.Wg * self.Hg, C])
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

        tcg_logger.info(f"[Scoring] Curvature Min: {s_min.mean().item():.4f} | Max: {s_max.mean().item():.4f}")
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
        no_drop_positions = int((pack["mask_sum"] == 0).sum().item())

        if fully_dropped_positions > 0:
            tcg_logger.warning(
                f"[TemporalGroup] {fully_dropped_positions} (sample, spatial-position) pair(s) "
                f"had ALL {self.Fg} frames dropped — those pixels have zero individually "
                f"preserved frames this interval, fully collapsed to one summary token."
            )

        if no_drop_positions > 0:
            tcg_logger.warning(
                f"[NoDropTemporalGroup] {no_drop_positions} (sample, spatial-position) pair(s) "
                f"had NOT A SINGLE frame dropped — those pixels have EXTREMELY UNSTABLE "
                f"INFORMATION over the frames this interval."
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
        restored = original_tokens.clone()
        important_updated = merged_tokens[:, :k, :]
        restored.scatter_(1, kept_idx.unsqueeze(-1).expand(-1, -1, D), important_updated)

        tcg_logger.info(f"[Restore] Restored sequence length to {N} for batch {B}. Kept tokens updated.")
        tcg_logger.info("="*60)
        return restored


class TestMatrix():
    """
    Builds synthetic token-space test data for TemporalCurvatureGuidance.

    TCG operates on already-patchified tokens (shape [B, N, C] with
    N = Fg*Hg*Wg), not raw pixels, so this generates directly in token
    space: a batch of B samples, each a Fg x Hg x Wg grid of C-dim tokens,
    plus a matching RoPE table of shape [N, D_rope]. F/H/W are kept as the
    *pre-patchify* video shape for documentation/bookkeeping (e.g. Fg would
    typically be F // patch_size_t in a real pipeline); Fg/Hg/Wg are the
    actual post-patchify grid TCG consumes and default to F/H/W unless
    overridden via `_change_grouping_factors`.
    """
    def __init__(
        self, 
        batches: int, 
        F: int, H: int, W: int, 
        dtype: torch.dtype | None = torch.float32,
        seed: int = 30000,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = dtype
        self.seed = seed
        self.B = batches
        self.F, self.H, self.W = F, H, W
        self.rng = torch.manual_seed(seed=seed)
        self.tensor = torch.rand([F, H, W], generator=self.rng, device=self.device, dtype=dtype)
        # Fg, Hg, Wg defines how many frames need to be combined when converting them to embeddings. 
        # Fg of 2 combines info across 2 frames into one block.
        self.Fg = F
        self.Hg = H
        self.Wg = W

    def _change_grouping_factors(self, Fg: int | None = None, Hg: int | None = None, Wg: int | None = None):
        self.Fg = Fg if Fg is not None else self.Fg
        self.Hg = Hg if Hg is not None else self.Hg
        self.Wg = Wg if Wg is not None else self.Wg

    def build_tokens(self, C: int, D_rope: int):
        """
        Returns (tokens [B,N,C], cos [N,D_rope], sin [N,D_rope]) where
        N = Fg*Hg*Wg, using this instance's current Fg/Hg/Wg. cos/sin are
        built as actual unit-circle values (not plain randn) so the
        normalize-then-average step inside get_rope_for_merged_tokens_temporal
        is exercised the way it would be on real rotary tables.
        """
        N = self.Fg * self.Hg * self.Wg
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        tokens = torch.randn(self.B, N, C, generator=g, device=self.device, dtype=self.dtype)
        angles = torch.rand(N, D_rope, generator=g, device=self.device, dtype=self.dtype) * 2 * 3.14159265
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return tokens, cos, sin


# ---------------------------------------------------------------------------
# Shape / invariant tests
# ---------------------------------------------------------------------------

class TCGTestError(AssertionError):
    pass


def _check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        raise TCGTestError(f"{name}: {detail}")


def test_scoring_shape(tcg: TemporalCurvatureGuidance, tokens: torch.Tensor):
    print("\n[Test] temporal_errors shape")
    s = tcg.temporal_errors(tokens)
    B, N, C = tokens.shape
    _check(f"s_tcg has shape [B,N] = [{B},{N}]", tuple(s.shape) == (B, N), f"got {tuple(s.shape)}, expected {(B, N)}")
    _check("s_tcg is in [0, 1]", bool((s >= 0).all() and (s <= 1).all()), "values escaped normalized range")
    return s


def test_token_split_shapes(tcg: TemporalCurvatureGuidance, tokens: torch.Tensor, s: torch.Tensor):
    print("\n[Test] get_token_set shapes")
    B, N, C = tokens.shape
    Hr, k, Dr, mask, kept_idx = tcg.get_token_set(tokens, s)
    _check(f"Hr shape = {tuple(Hr.shape)}", tuple(Hr.shape) == (B, k, C), f"expected {tuple((B, k, C))}")
    _check(f"Dr shape = {tuple(Dr.shape)}", tuple(Dr.shape) == (B, N - k, C), f"expected {tuple((B, N - k, C))}")
    _check(f"mask shape = {tuple(mask.shape)}", tuple(mask.shape) == (B, N), f"expected {tuple((B, N))}")
    _check(f"kept_idx shape = {tuple(kept_idx.shape)}", tuple(kept_idx.shape) == (B, k), f"expected {tuple((B, k))}")
    _check(
        f"drop counts = {mask.sum(dim=1).tolist()}",
        bool((mask.sum(dim=1) == (N - k)).all()),
        f"expected {[N - k] * B}",
    )
    _check(
        f"kept counts = {(~mask).sum(dim=1).tolist()}",
        bool(torch.all((~mask).sum(dim=1) == k)),
        f"expected {[k] * B}",
    )
    return Hr, k, Dr, mask, kept_idx


def test_temporal_group_shapes(tcg: TemporalCurvatureGuidance, tokens: torch.Tensor, mask: torch.Tensor):
    print("\n[Test] temporal_group_tokens shapes")
    B, N, C = tokens.shape
    H_global, pad_mask, pack = tcg.temporal_group_tokens(tokens, mask)
    max_valid = pack["max_valid"]
    _check(f"H_global shape = {tuple(H_global.shape)}", tuple(H_global.shape) == (B, max_valid, C), f"expected {tuple((B, max_valid, C))}")
    _check(f"pad_mask shape = {tuple(pad_mask.shape)}", tuple(pad_mask.shape) == (B, max_valid), f"expected {tuple((B, max_valid))}")
    _check(
        f"pad counts = {pad_mask.sum(dim=1).tolist()}",
        bool(torch.equal(pad_mask.sum(dim=1), pack["counts"])),
        f"expected {pack['counts'].tolist()}",
    )
    _check(
        f"grouped mass = {pack['mask_sum'].sum(dim=1).tolist()}",
        bool(torch.equal(pack["mask_sum"].sum(dim=1), mask.sum(dim=1))),
        f"expected {mask.sum(dim=1).tolist()}",
    )
    _check(
        f"padding max abs = {H_global[~pad_mask].abs().max().item() if (~pad_mask).any() else 0.0:.4f}",
        bool(torch.all(H_global[~pad_mask] == 0)) if (~pad_mask).any() else True,
        "expected 0.0000",
    )
    return H_global, pad_mask, pack


def test_rope_merge_shapes(
    tcg: TemporalCurvatureGuidance, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor, k: int, pack: dict
):
    print("\n[Test] get_rope_for_merged_tokens_temporal shapes")
    B, N = mask.shape
    D = cos.shape[-1]
    max_valid = pack["max_valid"]
    final_cos, final_sin = tcg.get_rope_for_merged_tokens_temporal(cos, sin, mask, k, pack)
    expected = (B, k + max_valid, D)
    _check(f"final_cos has shape [B,k+max_valid,D] = [{B},{k + max_valid},{D}]", tuple(final_cos.shape) == expected, f"got {tuple(final_cos.shape)}")
    _check(f"final_sin has shape [B,k+max_valid,D] = [{B},{k + max_valid},{D}]", tuple(final_sin.shape) == expected, f"got {tuple(final_sin.shape)}")
    pad = pack["pad_mask"]
    if max_valid > 0 and (~pad).any():
        global_cos = final_cos[:, k:, :]
        global_sin = final_sin[:, k:, :]
        _check(
            "global RoPE padding rows are exact zero",
            bool(torch.all(global_cos[~pad] == 0) and torch.all(global_sin[~pad] == 0)),
            "non-zero values found in padded RoPE rows",
        )
    return final_cos, final_sin


def test_identity_restore(
    tcg: TemporalCurvatureGuidance,
    original_tokens: torch.Tensor,
    merged_tokens: torch.Tensor,
    k: int,
    kept_idx: torch.Tensor,
    mask: torch.Tensor,
):
    print("\n[Test] identity_restore shape + invariants")
    B, N, C = original_tokens.shape
    restored = tcg.identity_restore(original_tokens, merged_tokens, k, kept_idx)
    _check(f"restored has original shape [B,N,C] = [{B},{N},{C}]", tuple(restored.shape) == (B, N, C), f"got {tuple(restored.shape)}")

    # Kept positions must equal the (possibly-updated) local tokens.
    restored_kept = torch.gather(restored, 1, kept_idx.unsqueeze(-1).expand(-1, -1, C))
    expected_kept = merged_tokens[:, :k, :]
    _check(
        "kept positions equal merged_tokens[:, :k, :]",
        bool(torch.allclose(restored_kept, expected_kept)),
        "scatter did not place updated local tokens at kept_idx",
    )

    # Dropped positions must be untouched (identity).
    _check(
        "dropped positions are untouched (identity)",
        bool(torch.allclose(restored[mask], original_tokens[mask])),
        "values at dropped positions changed unexpectedly",
    )
    return restored


def test_batch_independence(tcg: TemporalCurvatureGuidance, C: int, D_rope: int, seed: int = 12345):
    """
    Runs two samples individually (B=1 each) and together (B=2), and checks
    the batched path produces identical per-sample results either way. This
    is the invariant that actually matters for the batch-vectorization work:
    a bug here would mean samples are leaking into each other's scores,
    drops, or merges.
    """
    print("\n[Test] batch independence (B=1+1 vs B=2)")
    Fg, Hg, Wg = TEST_FG, TEST_HG, TEST_WG
    N = Fg * Hg * Wg
    g = torch.Generator().manual_seed(seed)
    t0 = torch.randn(1, N, C, generator=g)
    t1 = torch.randn(1, N, C, generator=g)
    tokens_batched = torch.cat([t0, t1], dim=0)

    angles = torch.rand(N, D_rope, generator=g) * 2 * 3.14159265
    cos, sin = torch.cos(angles), torch.sin(angles)

    tcg.set_grid_dimensions(Fg, Hg, Wg)

    def run(tokens):
        s = tcg.temporal_errors(tokens)
        Hr, k, Dr, mask, kept_idx = tcg.get_token_set(tokens, s)
        H_global, pad_mask, pack = tcg.temporal_group_tokens(tokens, mask)
        final_cos, final_sin = tcg.get_rope_for_merged_tokens_temporal(cos, sin, mask, k, pack)
        merged = torch.cat([Hr, H_global], dim=1)
        restored = tcg.identity_restore(tokens, merged, k, kept_idx)
        return s, mask, H_global, pad_mask, restored

    s0, mask0, g0, pad0, r0 = run(t0)
    s1, mask1, g1, pad1, r1 = run(t1)
    sB, maskB, gB, padB, rB = run(tokens_batched)

    _check("scores match for sample 0", bool(torch.allclose(s0[0], sB[0])), "score mismatch")
    _check("scores match for sample 1", bool(torch.allclose(s1[0], sB[1])), "score mismatch")
    _check("drop masks match for sample 0", bool(torch.equal(mask0[0], maskB[0])), "mask mismatch")
    _check("drop masks match for sample 1", bool(torch.equal(mask1[0], maskB[1])), "mask mismatch")
    _check("restored tokens match for sample 0", bool(torch.allclose(r0[0], rB[0])), "restore mismatch")
    _check("restored tokens match for sample 1", bool(torch.allclose(r1[0], rB[1])), "restore mismatch")


def test_reference_paths_guard_batch(tcg: TemporalCurvatureGuidance, C: int, D_rope: int):
    """mean_global_tokens / get_rope_for_merged_tokens are batch=1-only;
    verify they raise on batch>1 and succeed on batch=1."""
    print("\n[Test] reference (non-temporal) paths reject batch>1")
    Fg, Hg, Wg = TEST_FG, TEST_HG, TEST_WG
    N = Fg * Hg * Wg
    tcg.set_grid_dimensions(Fg, Hg, Wg)

    tokens_b2 = torch.randn(2, N, C)
    s_b2 = tcg.temporal_errors(tokens_b2)
    _, k, Dr_b2, mask_b2, _ = tcg.get_token_set(tokens_b2, s_b2)

    raised = False
    try:
        tcg.mean_global_tokens(Dr_b2)
    except NotImplementedError:
        raised = True
    _check("mean_global_tokens raises NotImplementedError for B=2", raised)

    tokens_b1 = torch.randn(1, N, C)
    s_b1 = tcg.temporal_errors(tokens_b1)
    _, k1, Dr_b1, mask_b1, _ = tcg.get_token_set(tokens_b1, s_b1)
    H_global_b1 = tcg.mean_global_tokens(Dr_b1)
    _check(
        "mean_global_tokens succeeds for B=1",
        H_global_b1.shape[0] == 1 and H_global_b1.dim() == 3,
        f"got shape {tuple(H_global_b1.shape)}",
    )

    angles = torch.rand(N, D_rope) * 2 * 3.14159265
    cos, sin = torch.cos(angles), torch.sin(angles)
    raised = False
    try:
        tcg.get_rope_for_merged_tokens(cos, sin, tokens_b2, mask_b2)
    except NotImplementedError:
        raised = True
    _check("get_rope_for_merged_tokens raises NotImplementedError for B=2", raised)


def main():
    global tcg_logger
    tcg_logger = setup_tcg_logging()

    print("=" * 60)
    print("STARTING TCG PIPELINE VERIFICATION")
    print("=" * 60)

    C = 32
    D_rope = 16
    B = 10

    # 1. Setup: synthetic token-space data via TestMatrix
    print("\n[1] Building synthetic tokens + RoPE via TestMatrix...")
    tm = TestMatrix(batches=B, F=TEST_FG, H=TEST_HG, W=TEST_WG, dtype=torch.float32, seed=310805)
    tm._change_grouping_factors(Fg=TEST_FG, Hg=TEST_HG, Wg=TEST_WG)  # exercise the setter explicitly
    tokens, cos, sin = tm.build_tokens(C=C, D_rope=D_rope)
    print(f"Tokens shape: {tuple(tokens.shape)} | cos/sin shape: {tuple(cos.shape)}")

    tcg = TemporalCurvatureGuidance(patch_size_t=2, use_temporal_grouping=True, drop_ratio=0.5)
    tcg.set_grid_dimensions(tm.Fg, tm.Hg, tm.Wg)

    failures = 0
    try:
        # 2. Scoring
        s = test_scoring_shape(tcg, tokens)

        # 3. Token split
        Hr, k, Dr, mask, kept_idx = test_token_split_shapes(tcg, tokens, s)

        # 4. Temporal grouping (global tokens)
        H_global, pad_mask, pack = test_temporal_group_shapes(tcg, tokens, mask)

        # 5. RoPE merge
        final_cos, final_sin = test_rope_merge_shapes(tcg, cos, sin, mask, k, pack)

        # 6. Simulate downstream processing on the merged sequence, then restore
        print("\n[Simulate] Doubling the kept-token values to prove restoration picks up updates...")
        merged_tokens = torch.cat([Hr, H_global], dim=1)
        processed = merged_tokens.clone()
        processed[:, :k, :] = processed[:, :k, :] * 2.0

        # 7. Identity restore
        restored = test_identity_restore(tcg, tokens, processed, k, kept_idx, mask)

        # 8. Batch independence (the main risk surface of the recent vectorization work)
        test_batch_independence(tcg, C=C, D_rope=D_rope)

        # 9. Reference (batch=1-only) paths correctly guard against batch>1
        test_reference_paths_guard_batch(tcg, C=C, D_rope=D_rope)

    except TCGTestError as e:
        failures += 1
        print(f"\n  !! TEST FAILURE: {e}")

    print("\n" + "=" * 60)
    if failures == 0:
        print("SUCCESS: All pipeline steps verified successfully!")
    else:
        print(f"FAILURE: {failures} test group(s) failed.")
    print("=" * 60)
    return failures == 0


if __name__ == "__main__":
    tcg_logger = setup_tcg_logging()
    ok = main()
    sys.exit(0 if ok else 1)