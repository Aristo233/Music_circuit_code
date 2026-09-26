"""Reading YuE2 internals: prefixes, residual/component captures and next-token distributions.

Everything here is a plain teacher-forced forward pass of the autoregressive
model (``use_cache=False``, batch one, no sampling). Four sites are read in
every decoder layer:

* ``res_attention``  residual stream after the attention sublayer (input of
                     ``post_attention_layernorm``), where attention heads write;
* ``res_mlp``        residual stream after the MLP sublayer (layer output),
                     where MLP neurons write;
* ``gated_mlp``      gated MLP activation entering ``down_proj`` (component
                     activation of neurons);
* ``head_concat``    concatenated head outputs entering ``o_proj`` (component
                     activation of heads; head ``h`` owns channels
                     ``h*head_dim:(h+1)*head_dim``).
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from .constants import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END

CAPTURE_KEYS = ("res_attention", "res_mlp", "gated_mlp", "head_concat")


# ----------------------------------------------------------------------------- prefixes

def song_request(lyrics, style_text, *, seed=0, cot="melody"):
    """A YuE2 request. ``style_text=""`` is the empty tag :math:`c_\\varnothing`."""
    from yue2.protocol import SongRequest
    return SongRequest(style=style_text, lyrics=lyrics, cot=cot, seed=int(seed), cfg_scale=1.0)


def matched_prefixes(tokenizer, lyrics, style_text, stage, abc_ids=None, *, seed=0, cot="melody"):
    """Token prefixes of the empty tag and of the style tag for one stage.

    Score stage: ``[EOD] + text(tag, lyrics) + [ABC_START]``.
    Music-token stage: the same followed by the score ``abc_ids`` and
    ``[ABC_END, MUSIC_START]``. Only the tag text differs between the two
    prefixes; the tags have different token lengths, which is kept as is.
    """
    from yue2.protocol import token_prefixes
    out = {}
    for name, text in (("blank", ""), ("target", style_text)):
        request = song_request(lyrics, text, seed=seed, cot=cot)
        out[name] = token_prefixes(request, tokenizer, abc_ids if stage == "semantic" else None)
    return out


def stage_logits(logits, stage):
    """Restrict full-vocabulary logits to the tokens legal in a stage."""
    if stage == "abc":
        return torch.cat((logits[..., :EOD], logits[..., ABC_END:ABC_END + 1]), dim=-1)
    return logits[..., MUSIC_END:CODEC_OFFSET + CODEC_SIZE]


def kl(logp_p, logp_q):
    """KL(p || q) from log-probabilities over the last axis, clamped at zero."""
    return (logp_p.exp() * (logp_p - logp_q)).sum(-1).clamp_min(0)


# ----------------------------------------------------------------------------- hooks

def hook_inventory(model):
    return {name: (len(m._forward_pre_hooks), len(m._forward_hooks))
            for name, m in model.named_modules() if m._forward_pre_hooks or m._forward_hooks}


def require_clean(model):
    if model.training:
        raise ValueError("Capture requires eval mode")
    if hook_inventory(model):
        raise ValueError("Close every intervention/capture hook before a new capture")


def _layer_output(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def capture_hooks(model, save):
    """Register the four capture sites in every layer; ``save(key, layer, tensor)``."""
    handles = []
    try:
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, a, l=index: save("res_attention", l, a[0])))
            handles.append(layer.register_forward_hook(
                lambda m, a, o, l=index: save("res_mlp", l, _layer_output(o))))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda m, a, l=index: save("gated_mlp", l, a[0])))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, l=index: save("head_concat", l, a[0])))
        yield
    finally:
        for handle in handles:
            handle.remove()


# ----------------------------------------------------------------------------- captures

@torch.inference_mode()
def capture_positions(model, ids, positions):
    """FP32 arrays ``{key: [layer, width]}``: the mean over ``positions`` of one forward.

    A single position gives that position's activations; a window (e.g. the
    central 64 music tokens) gives the window mean.
    """
    require_clean(model)
    device = next(model.parameters()).device
    index = torch.as_tensor(list(positions), device=device, dtype=torch.long)
    store = {key: [None] * len(model.model.layers) for key in CAPTURE_KEYS}

    def save(key, layer, value):
        store[key][layer] = value[0].index_select(0, index).float().mean(0).cpu().numpy()

    with capture_hooks(model, save):
        model(torch.tensor([list(ids)], device=device), use_cache=False, logits_to_keep=1)
    out = {key: np.stack(v) for key, v in store.items()}
    if any(not np.isfinite(v).all() for v in out.values()):
        raise FloatingPointError("Nonfinite capture")
    return out


@torch.inference_mode()
def capture_sequence(model, ids, start, stage, *, keys=CAPTURE_KEYS):
    """Activations at every prediction position ``start..len(ids)-1`` plus stage log-probs.

    Returns ``(acts, logp)`` with ``acts[key]`` of shape ``[rows, layer, width]``
    (kept in the model dtype on the model device) and ``logp`` of shape
    ``[rows, vocab_stage]`` in FP32. Row ``r`` is the position that predicts
    token ``start + r + 1``; the final row predicts the token after the last
    input token.
    """
    require_clean(model)
    device = next(model.parameters()).device
    rows = len(ids) - start
    store = {key: [None] * len(model.model.layers) for key in keys}

    def save(key, layer, value):
        if key in store:
            store[key][layer] = value[0, start:]

    with capture_hooks(model, save):
        logits = model(torch.tensor([list(ids)], device=device), use_cache=False,
                       logits_to_keep=rows).logits[0]
    acts = {key: torch.stack(v, dim=1) for key, v in store.items()}
    logp = stage_logits(logits, stage).float().log_softmax(-1)
    return acts, logp


@torch.inference_mode()
def stepwise_kl(model, blank_ids, target_ids, content_length, stage):
    """KL(p(.|c_s, x_<p) || p(.|c_0, x_<p)) at every content position of a shared history.

    ``blank_ids``/``target_ids`` are ``prefix + content``; the two prefixes may
    differ in length, the content is identical. Returns ``[content_length]``.
    """
    device = next(model.parameters()).device
    out = []
    for ids in (target_ids, blank_ids):
        logits = model(torch.tensor([list(ids)], device=device), use_cache=False,
                       logits_to_keep=content_length + 1).logits[0, :-1]
        out.append(stage_logits(logits, stage).float().log_softmax(-1))
    return kl(out[0], out[1])
