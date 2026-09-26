"""Fixed vocabulary boundaries, style definitions and model dimensions of YuE2-3B."""
from __future__ import annotations

# Special tokens of the YuE2 protocol (see upstream yue2/protocol.py).
EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768

# Autoregressive stages that read the style tag. "abc" is the score stage,
# "semantic" the music-token stage. Acoustic rendering (NAR) is never intervened on.
STAGES = ("abc", "semantic")

# Five styles of the paper. Keys are used in arrays and JSON files; labels are
# the exact style texts placed in the [Tags] field of the prompt.
STYLES = ("pop_ballad", "metal_punk", "funk", "jazz_blues", "country_folk")
LABELS = {"pop_ballad": "contemporary pop", "metal_punk": "metal rock", "funk": "classic funk",
          "jazz_blues": "electric blues", "country_folk": "country"}
NONPOP = ("metal_punk", "funk", "jazz_blues", "country_folk")

# YuE2-3B layout used by the released circuits.
NUM_LAYERS = 28
HIDDEN = 2048
MLP_WIDTH = 6144
NUM_HEADS = 16
HEAD_DIM = 128
WIDTHS = {"mlp": MLP_WIDTH, "attention": NUM_HEADS * HEAD_DIM}

# Circuit budget per stage and style: ten components per sublayer, split 7:3.
BUDGETS = {"mlp": 392, "attention": 168}

# Residual sites where directions are fitted: after attention (index 0, where
# heads write) and after the MLP (index 1, where neurons write).
SITES = ("attention", "mlp")
SITE_OF_KIND = {"attention": 0, "mlp": 1}


def channels(kind, units, head_dim=HEAD_DIM):
    """Input channels of the write projection for a list of component ids.

    MLP neurons are single channels of ``down_proj``; an attention head is the
    contiguous block of ``head_dim`` channels of ``o_proj``.
    """
    if kind == "mlp":
        return [int(u) for u in units]
    return [int(h) * head_dim + j for h in units for j in range(head_dim)]


def canonical_selection(selection):
    """``{kind: {int layer: sorted unique ids}}``; empty layers dropped; JSON keys accepted."""
    if not isinstance(selection, dict) or set(selection) - {"mlp", "attention"}:
        raise ValueError("A selection maps 'mlp' and/or 'attention' to {layer: [ids]}")
    result = {"mlp": {}, "attention": {}}
    for kind in result:
        for layer, units in selection.get(kind, {}).items():
            ids = sorted({int(u) for u in units})
            if any(u < 0 for u in ids):
                raise ValueError("Component ids must be nonnegative")
            if ids:
                result[kind][int(layer)] = ids
    return result


def union_selection(selections):
    result = {"mlp": {}, "attention": {}}
    for selection in selections:
        for kind, layers in canonical_selection(selection).items():
            for layer, units in layers.items():
                result[kind][layer] = sorted(set(result[kind].get(layer, [])) | set(units))
    return result


def masked_delta(raw, selection, head_dim=HEAD_DIM):
    """Copy of ``raw = {kind: [layer, width]}`` that is zero outside the selected channels."""
    import numpy as np
    selection = canonical_selection(selection)
    out = {}
    for kind in ("mlp", "attention"):
        value = np.asarray(raw[kind], dtype=np.float64)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("Delta must be a finite [layer, width] array")
        masked = np.zeros_like(value)
        for layer, units in selection[kind].items():
            idx = channels(kind, units, head_dim)
            if layer >= value.shape[0] or max(idx) >= value.shape[1]:
                raise ValueError("Selection exceeds delta shape")
            masked[layer, idx] = value[layer, idx]
        out[kind] = masked
    return out
