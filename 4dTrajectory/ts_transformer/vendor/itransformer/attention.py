"""Vendored from iTransformer ``layers/SelfAttention_Family.py`` (MIT, see LICENSE).

Only ``FullAttention`` and ``AttentionLayer`` are carried over — the classes the
iTransformer ``Model`` actually constructs. The upstream file additionally defines
``FlowAttention`` / ``FlashAttention`` / ``ProbAttention`` / ``ReformerLayer`` for the
iFlowformer / iFlashformer / iInformer / iReformer variants, and imports
``reformer_pytorch`` at module scope to do it. Carrying the whole file would make an
unused pip dependency mandatory, so the variants are dropped.

``TriangularCausalMask`` is inlined from upstream ``utils/masking.py`` for the same
reason (its sibling ``ProbMask`` only serves ProbAttention). Note that iTransformer
constructs ``FullAttention(False, ...)`` — ``mask_flag`` is off, so the mask is never
built on this path; it is kept so the class stays a faithful copy.

Class bodies below are verbatim upstream. See PROVENANCE.md for the pinned commit.
"""

from math import sqrt

import numpy as np
import torch
import torch.nn as nn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag # Causal masking is not used in the iTransformer model, but the class is kept for fidelity to the upstream source.
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # B: batch_size;    L: token length, but here, inverted, so the channel dimension
        B, L, _ = queries.shape
        # S: key token length
        _, S, _ = keys.shape
        # H: number of heads
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
