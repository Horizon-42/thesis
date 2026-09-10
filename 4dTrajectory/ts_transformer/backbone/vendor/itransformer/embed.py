"""Vendored from iTransformer ``layers/Embed.py`` (MIT, see LICENSE).

Only ``DataEmbedding_inverted`` is carried over — the one embedding the iTransformer
``Model`` constructs. Upstream's other six classes (``PositionalEmbedding``,
``TokenEmbedding``, ``FixedEmbedding``, ``TemporalEmbedding``, ``TimeFeatureEmbedding``,
``DataEmbedding``) serve the vanilla Transformer / Informer baselines in that repo and
are unreachable from here.

Note what the inverted embedding does NOT do: it takes ``embed_type`` and ``freq``
arguments and ignores both — there is no calendar-feature embedding on this path. The
"variate tokens" are whole series (one token per channel, length ``seq_len``), which is
the entire point of the inverted design. Covariates, if passed as ``x_mark``, are
concatenated as ADDITIONAL tokens along the variate axis, not summed into the values.

Class body below is verbatim upstream. See PROVENANCE.md for the pinned commit.
"""

import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)
