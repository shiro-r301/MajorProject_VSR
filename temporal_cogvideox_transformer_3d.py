# Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# TRaM-VSR CHANGES (on top of the reference TCG merger/unmerger):
#
#   1. TemporalCurvatureGuidance.temporal_group_tokens()
#      New method implementing "pure temporal grouping": instead of taking
#      the flat set of dropped tokens (Dr) and splitting it into arbitrary
#      contiguous chunks (mean_global_tokens), we group dropped tokens by
#      their *spatial position* and average only across the temporal
#      (frame) axis at that position. Positions with no dropped frames
#      contribute nothing (they stay represented individually in H_local).
#      This keeps each global/summary token semantically tied to a single
#      spatial location instead of mixing unrelated spatial content.
#
#   2. TemporalCurvatureGuidance.get_rope_for_temporal_groups()
#      RoPE companion to (1): since temporal groups are irregular
#      (variable-size, non-contiguous index sets defined by pos_indices),
#      we can't reuse the reference's contiguous torch.split-based RoPE
#      averaging (get_rope_for_merged_tokens). This gathers cos/sin at the
#      exact original indices belonging to each group before averaging
#      and re-normalizing.
#
#   3. TemporalCurvatureGuidance.__init__ gains `use_temporal_grouping`
#      (default True) so the reference path (mean_global_tokens +
#      get_rope_for_merged_tokens) is still selectable for A/B comparison
#      against the baseline. Same flag threaded through
#      CogVideoXTransformer3DModel.__init__.
#
#   4. Bug fix: forward() previously did `self.tcg = TemporalCurvatureGuidance()`
#      unconditionally on every call, silently discarding the `patch_size_t`
#      (and now `use_temporal_grouping`) configured in __init__ and resetting
#      it to defaults on every forward pass. Removed; self.tcg from __init__
#      is reused as-is.
#
#   identity_restore() is unchanged: it only ever restores kept (H_local)
#   tokens into their original positions and leaves dropped positions equal
#   to the original, pre-merge tokens, regardless of how the discarded
#   global/summary tokens were computed.
# ============================================================================

from typing import Any
import time
import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import apply_lora_scale, logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, AttentionMixin, FeedForward
from ..attention_processor import CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from ..cache_utils import CacheMixin
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from math import floor

activations = []
import logging
import sys


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

    def get_token_set(self, tokens, s):
        N = tokens.shape[1]
        k = max(1, N - floor(self.drop_ratio * N))
        TopK = torch.topk(s, k=k)
        all_indices = torch.arange(s.shape[1], device=s.device)
        local_indices, _ = torch.sort(TopK.indices.squeeze(0))

        Hr = tokens[:, local_indices, :]
        mask = ~torch.isin(all_indices, local_indices)
        Dr = tokens[:, mask, :]

        tcg_logger.info(f"[Split] Total: {N} | Kept (Local): {k} | Dropped: {N-k}")
        return Hr, k, Dr, mask #kept, how many kept, dropped, drop_mask

    def get_even_splits(self, Mr, Gr):
        base = Mr // Gr
        remainder = Mr % Gr
        sizes = [base + 1] * remainder + [base] * (Gr - remainder)
        return sizes

    def mean_global_tokens(self, Dr):
        """Reference merge: split the flat dropped-token set into Gr
        contiguous chunks (in whatever order Dr happens to be in) and mean
        each chunk. Kept for A/B comparison against temporal_group_tokens."""
        Mr = Dr.shape[1]
        Gr = min(Mr, self.g_max, max(self.g_min, int(self.group_ratio * Mr)))
        mean_splits = self.get_even_splits(Mr, Gr)
        groups = torch.split(Dr, mean_splits, dim=1)
        H_global = torch.stack([g.mean(dim=1) for g in groups], dim=1)

        tcg_logger.info(f"[Merge] Dropped Tokens: {Mr} | Groups: {Gr} | Final Global Tokens: {H_global.shape[1]}")
        return H_global

    def temporal_group_tokens(self, tokens: torch.Tensor, mask: torch.Tensor):
        assert tokens.shape[0] == 1, "temporal_group_tokens assumes batch size 1"
        N = tokens.shape[1]
        C = tokens.shape[-1]
        P = self._check_grid(N, "temporal_group_tokens")
        Mr = int(mask.sum().item())
        if Mr == 0:
            tcg_logger.info("[TemporalGroup] No dropped tokens; returning empty global set.")
            return tokens.new_zeros(1, 0, C)

        mask_grid = mask.view(self.Fg, P)            
        tok_grid = tokens[0].view(self.Fg, P, C)      

        # 1. Zero out un-dropped tokens
        mask_expanded = mask_grid.unsqueeze(-1)  
        masked_tok_grid = tok_grid.masked_fill(~mask_expanded, 0.0)
        
        # 2. Find valid spatial positions with dropped tokens
        mask_sum = mask_grid.sum(dim=0)          
        valid_p = mask_sum > 0                   
        
        # 3. Sum over temporal dimension and divide by count
        sum_tokens = masked_tok_grid.sum(dim=0)  
        mean_tokens = sum_tokens / mask_sum.unsqueeze(-1).clamp(min=1).to(tokens.dtype)
        
        # 4. Filter and return
        H_global = mean_tokens[valid_p].unsqueeze(0) 

        # ---------------------------------------------------------
        # Logging & Invariant Checks (Vectorized)
        # ---------------------------------------------------------
        # mask_sum[valid_p] gives exactly the number of dropped tokens for each active position
        group_sizes = mask_sum[valid_p] 
        total_grouped = int(group_sizes.sum().item())

        if total_grouped != Mr:
            raise RuntimeError(
                f"[TemporalGroup] Grouped token count ({total_grouped}) != "
                f"dropped token count (Mr={Mr}). Tokens are being lost or "
                f"double-counted during grouping."
            )

        # Count how many positions had every single frame dropped
        fully_dropped_positions = int((mask_sum == self.Fg).sum().item())
        if fully_dropped_positions > 0:
            tcg_logger.warning(
                f"[TemporalGroup] {fully_dropped_positions} spatial position(s) had "
                f"ALL {self.Fg} frames dropped — those pixels have zero individually "
                f"preserved frames this interval, fully collapsed to one summary token."
            )

        tcg_logger.info(
            f"[TemporalGroup] Dropped Tokens: {Mr} | Spatial positions touched: "
            f"{group_sizes.numel()}/{P} | Final Global Tokens: {H_global.shape[1]} | "
            f"Frames-merged-per-position min/mean/max: "
            f"{group_sizes.min().item()}/{group_sizes.float().mean().item():.2f}/{group_sizes.max().item()}"
        )

        return H_global

    def get_rope_for_merged_tokens_temporal(self, cos: torch.Tensor, sin: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor):
        # Handle 3D inputs if necessary
        squeeze_needed = False
        if cos.dim() == 3:  
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            squeeze_needed = True

        N = tokens.shape[1]
        all_indices = torch.arange(N, device=tokens.device)
        local_indices = all_indices[~mask]

        local_cos = cos[local_indices]
        local_sin = sin[local_indices]

        Mr = int(mask.sum().item())
        if Mr == 0:
            final_cos, final_sin = local_cos, local_sin
        else:
            # --- Vectorized RoPE merging ---
            P = self._check_grid(N, "temporal_group_tokens")
            
            mask_grid = mask.view(self.Fg, P)
            mask_expanded = mask_grid.unsqueeze(-1)
            
            cos_grid = cos.view(self.Fg, P, -1)
            sin_grid = sin.view(self.Fg, P, -1)
            
            masked_cos = cos_grid.masked_fill(~mask_expanded, 0.0)
            masked_sin = sin_grid.masked_fill(~mask_expanded, 0.0)
            
            mask_sum = mask_grid.sum(dim=0)
            valid_p = mask_sum > 0
            
            if not valid_p.any():
                final_cos, final_sin = local_cos, local_sin
            else:
                mask_sum_clamped = mask_sum.unsqueeze(-1).clamp(min=1).to(cos.dtype)
                
                g_cos_all = masked_cos.sum(dim=0) / mask_sum_clamped
                g_sin_all = masked_sin.sum(dim=0) / mask_sum_clamped
                
                global_cos_unnorm = g_cos_all[valid_p]
                global_sin_unnorm = g_sin_all[valid_p]
                
                norm = torch.sqrt(global_cos_unnorm**2 + global_sin_unnorm**2 + self.ep)
                global_cos = global_cos_unnorm / norm
                global_sin = global_sin_unnorm / norm

                final_cos = torch.cat([local_cos, global_cos], dim=0)
                final_sin = torch.cat([local_sin, global_sin], dim=0)

        # Restore 3D shape if necessary
        if squeeze_needed:
            final_cos = final_cos.unsqueeze(0)
            final_sin = final_sin.unsqueeze(0)

        tcg_logger.info(f"[RoPE] Mapped RoPE to temporally-grouped sequence. New shape: {final_cos.shape}")
        
        return final_cos, final_sin
    def get_rope_for_merged_tokens(self, cos: torch.Tensor, sin: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor):
        """Reference RoPE merge (contiguous splits). Kept for A/B comparison
        against get_rope_for_temporal_groups."""
        squeeze_needed = False
        if cos.dim() == 3:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            squeeze_needed = True

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

    def identity_restore(self, original_tokens: torch.Tensor, merged_tokens: torch.Tensor, k: int, mask: torch.Tensor) -> torch.Tensor:
        keep_mask = ~mask
        B, N, D = original_tokens.shape
        restored = original_tokens.clone()
        important_updated = merged_tokens[:, :k, :]
        restored[:, keep_mask, :] = important_updated

        tcg_logger.info(f"[Restore] Restored sequence length to {N}. Kept tokens updated.")
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
        use_temporal_grouping: bool = False,
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
        k = 0

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
                    block, hidden_states, encoder_hidden_states, emb, image_rotary_emb, attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=emb, image_rotary_emb=image_rotary_emb, attention_kwargs=attention_kwargs,
                )

            # --- MERGER INJECTION ---
            if self.merger_layers is not None and i in self.merger_layers:
                try:
                    tcg_logger.info(f"--- MERGE TRIGGERED AT LAYER {i} ---")
                    original_tokens = hidden_states
                    original_rope = image_rotary_emb

                    s_tcg = self.tcg.temporal_errors(hidden_states)
                    H_local, k, Dr, mask = self.tcg.get_token_set(hidden_states, s_tcg)
                    cos, sin = image_rotary_emb

                    if self.tcg.use_temporal_grouping:
                        tcg_logger.info(f"[LOG] MERGING IN TEMPORAL DIMENSION")
                        # TRaM-VSR CHANGE: position-aware temporal grouping
                        # instead of an arbitrary contiguous split of Dr.
                        H_global = self.tcg.temporal_group_tokens(hidden_states, mask)
                        hidden_states = torch.cat([H_local, H_global], dim=1)
                        image_rotary_emb = self.tcg.get_rope_for_merged_tokens_temporal(
                            cos=cos, sin=sin, tokens=original_tokens, mask=mask
                        )
                        
                    else:
                        tcg_logger.info(f"[LOG] MERGING IN SPATIAL DIMENSION")
                        H_global = self.tcg.mean_global_tokens(Dr)
                        hidden_states = torch.cat([H_local, H_global], dim=1)
                        image_rotary_emb = self.tcg.get_rope_for_merged_tokens(
                            cos=cos, sin=sin, tokens=original_tokens, mask=mask
                        )
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
                    hidden_states = self.tcg.identity_restore(
                        original_tokens=original_tokens,
                        merged_tokens=hidden_states,
                        k=k,
                        mask=mask
                    )
                    image_rotary_emb = original_rope

                    original_rope = None
                    original_tokens = None
                    mask = None
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