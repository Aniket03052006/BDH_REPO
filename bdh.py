# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        targets=None,
        ablation_indices=None,
        return_activations=False,
    ):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        activations = None

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            # Hook for ablation and visualization at the last layer
            if level == C.n_layer - 1:
                if getattr(self, "accumulate_stats", False):
                     # Accumulate stats for Option-B consolidation
                     # shape: [B, nh, T, N] -> Sum over B and T -> [nh, N]
                     current_stats = xy_sparse.detach().sum(dim=(0, 2))
                     if not hasattr(self, "xy_sparse_stats"):
                         self.xy_sparse_stats = torch.zeros_like(current_stats)
                         self.stats_count = 0
                     self.xy_sparse_stats += current_stats
                     self.stats_count += (B * T)

                if ablation_indices is not None and len(ablation_indices) > 0:
                    # Flatten to [B, T, nh * N] for easier indexing or just mask based on global index
                    # Here we assume ablation_indices refer to the flattened "neuron" index (0 to nh*N - 1)
                    # We need to act on the last dimension of size N, per head.
                    # Total neurons = nh * N.
                    # Global index g = h * N + n
                    # We can construct a binary mask.
                    mask = torch.ones(
                        (nh * N), device=xy_sparse.device, dtype=xy_sparse.dtype
                    )
                    mask[ablation_indices] = 0.0
                    mask = mask.view(nh, N)
                    # Broadcast to [B, nh, T, N]
                    xy_sparse = xy_sparse * mask.unsqueeze(0).unsqueeze(2)

                if return_activations:
                    # Capture the activations after ablation
                    # We want to return the activations for the *last token* typically,
                    # but let's return the full sequence or just the last step depending on caller need.
                    # For visualization we usually just want the last step's activations [B, nh, 1, N]
                    # Let's return the full tensor for now, user can slice.
                    activations = xy_sparse

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_activations:
            return logits, loss, activations
        return logits, loss

    @torch.no_grad()
    def consolidate(self, alpha=0.01, quantile=0.9):
        """
        Option-B Consolidation:
        1. Select top-quantile features per head based on accumulated stats.
        2. Reinforce weights associated with these features.
        3. Return diagnostics.
        """
        if not hasattr(self, "xy_sparse_stats") or self.stats_count == 0:
            return {"status": "skipped", "reason": "No stats accumulated"}
        
        # Normalize stats
        avg_stats = self.xy_sparse_stats / self.stats_count # [nh, N]
        
        # 1. Determine thresholds per head
        # Flatten N dimension to find quantile
        # avg_stats shape: [nh, N]
        nh, N = avg_stats.shape
        thresholds = []
        mask = torch.zeros_like(avg_stats, dtype=torch.bool)
        
        for h in range(nh):
            vals = avg_stats[h]
            # quantile returns the value at the q-th percentile
            k = int((1 - quantile) * N)
            if k < 1: k = 1
            top_vals, _ = torch.topk(vals, k)
            thresh = top_vals[-1]
            thresholds.append(thresh.item())
            mask[h] = vals >= thresh
        
        # 2. Reinforce Weights
        # Encoder: [nh, D, N]. We reinforce indices where mask[h, n] is True.
        # Decoder: [nh * N, D]. We need to map [nh, N] mask to [nh * N] flat mask.
        
        # Diagnostics: Weight norms before
        norm_enc_before = self.encoder.norm().item()
        norm_dec_before = self.decoder.norm().item()
        
        # Encoder Reinforcement
        # mask is [nh, N]. encoder is [nh, D, N]. Broadcast over D.
        # W_new = W_old * (1 + alpha) where mask is True
        reinforce_factor = 1.0 + alpha
        
        # We can use advanced indexing or simplified loop
        for h in range(nh):
            active_indices = torch.where(mask[h])[0]
            if len(active_indices) > 0:
                self.encoder.data[h, :, active_indices] *= reinforce_factor
                self.encoder_v.data[h, :, active_indices] *= reinforce_factor
        
        # Decoder Reinforcement
        # decoder is [nh * N, D].
        # Flatten mask to [nh * N]
        flat_mask = mask.view(-1)
        active_flat = torch.where(flat_mask)[0]
        if len(active_flat) > 0:
            self.decoder.data[active_flat, :] *= reinforce_factor
        
        # Diagnostics: Weight norms after
        norm_enc_after = self.encoder.norm().item()
        norm_dec_after = self.decoder.norm().item()
        
        result = {
            "status": "success",
            "thresholds": thresholds,
            "active_features_count": mask.sum().item(),
            "active_fraction": mask.float().mean().item(),
            "weight_norm_change": {
                "encoder": norm_enc_after - norm_enc_before,
                "decoder": norm_dec_after - norm_dec_before
            },
            "accumulated_tokens": self.stats_count
        }
        
        # Reset stats
        self.xy_sparse_stats.zero_()
        self.stats_count = 0
        
        return result

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
