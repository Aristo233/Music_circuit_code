"""Automatic style judgement of generated audio with CLAP.

The whole song is cut into 10 s windows (480,000 samples at 48 kHz) that cover
every sample; window scores are averaged with coverage weights so that each
second of audio counts once. Each window is scored against the five style
descriptions ``"This audio is a {style} song."``. The *margin* is the target
style's song score minus the best other style's; a complete generation with a
positive margin is a *hit*. Truncated generations are misses.

The paper's CLAP checkpoint is a music-trained CLAP; any encoder exposing
``audio(windows, batch_size) -> [n, d]`` and ``text(prompts) -> [k, d]`` with
L2-normalized embeddings can be plugged in (see :class:`LaionClapEncoder`).
"""
from __future__ import annotations

import numpy as np

from .constants import LABELS, STYLES

SR, WIDTH = 48000, 480000
TEMPLATE = "This audio is a {style} song."


def planned_intervals_and_weights(frames):
    """Windows of ``WIDTH`` samples covering ``frames`` samples, with coverage weights summing to one."""
    if frames < WIDTH:
        return [(0, frames)], np.ones(1)
    starts = list(range(0, frames - WIDTH + 1, WIDTH))
    if starts[-1] + WIDTH != frames:
        starts.append(frames - WIDTH)
    intervals = [(s, s + WIDTH) for s in starts]
    bounds = sorted({0, frames, *(x for pair in intervals for x in pair)})
    mass = np.zeros(len(intervals))
    for lo, hi in zip(bounds, bounds[1:]):
        covered = [i for i, (a, b) in enumerate(intervals) if a <= lo and hi <= b]
        mass[covered] += (hi - lo) / len(covered)
    return intervals, mass / frames


def windows(mono):
    mono = np.asarray(mono, dtype=np.float32)
    intervals, weights = planned_intervals_and_weights(len(mono))
    out = []
    for lo, hi in intervals:
        part = mono[lo:hi]
        if len(part) < WIDTH:
            part = np.pad(part, (0, WIDTH - len(part)))
        out.append(part)
    return out, weights


def outcome(scores, target_index, *, complete=True, styles=STYLES):
    scores = np.asarray(scores, dtype=np.float64)
    other = float(np.delete(scores, target_index).max())
    margin = float(scores[target_index] - other)
    return dict(target_score=float(scores[target_index]), max_other_score=other, target_margin=margin,
                predicted_style=styles[int(np.argmax(scores))], hit=bool(complete and margin > 0))


def score_song(encoder, stereo_or_mono, target_style, *, complete=True, batch_size=8, styles=STYLES):
    """Song-level CLAP scores for the five styles, margin and hit of one generation."""
    audio = np.asarray(stereo_or_mono, dtype=np.float32)
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    parts, weights = windows(mono)
    text = encoder.text([TEMPLATE.format(style=LABELS[s]) for s in styles])
    embeddings = encoder.audio(parts, batch_size)
    window_scores = embeddings @ text.T
    song_scores = weights @ window_scores
    return dict(outcome(song_scores, styles.index(target_style), complete=complete, styles=styles),
                song_scores=song_scores.tolist(), windows=len(parts))


class LaionClapEncoder:
    """Example encoder on top of ``laion_clap`` (music checkpoint); windows are resampled to 48 kHz already."""

    def __init__(self, checkpoint=None, device="cuda", enable_fusion=False, amodel="HTSAT-base"):
        import laion_clap
        import torch
        self.torch = torch
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel, device=device)
        if checkpoint:
            self.model.load_ckpt(checkpoint)
        else:
            self.model.load_ckpt()

    @staticmethod
    def _norm(x):
        x = np.asarray(x, dtype=np.float64)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def text(self, prompts):
        return self._norm(self.model.get_text_embedding(list(prompts), use_tensor=False))

    def audio(self, parts, batch_size=8):
        out = []
        for i in range(0, len(parts), batch_size):
            batch = np.stack(parts[i:i + batch_size])
            out.append(self.model.get_audio_embedding_from_data(x=batch, use_tensor=False))
        return self._norm(np.concatenate(out))
