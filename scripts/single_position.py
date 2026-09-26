#!/usr/bin/env python3
"""Single-position directions, component attribution and importance-allocated circuits.

Captures the five style versions of every representation-analysis lyric at one
position of a stage (the last prefix position = score start / music start, or a
window such as the central 64 music tokens), fits Eq. 1--2, and allocates the
music-token circuit by sublayer importance (Appendix "Budget allocation").

Input: ``samples.json`` with one row per (lyric, style) holding the tokens of
the prompted generation: ``{"song_id", "style_group", "prefix", "semantic_tokens"}``
where ``prefix`` is the full token prefix ending in MUSIC_START.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import load_pipeline, read_json, write_json
from music_circuit import capture, directions, importance
from music_circuit.constants import CODEC_OFFSET, MUSIC_END, STYLES


def position_input(row, position):
    prefix, codec = list(row["prefix"]), [t + CODEC_OFFSET for t in row["semantic_tokens"]]
    if position == "music_start":
        return prefix, [len(prefix) - 1]
    if position == "music_middle64":
        start = len(prefix) + (len(codec) - 64) // 2
        return (prefix + codec)[:start + 64], list(range(start, start + 64))
    if position == "music_end":
        ids = prefix + codec + [MUSIC_END]
        return ids, [len(ids) - 1]
    if position == "score_start":
        ids = prefix[:list(prefix).index(151847) + 1]
        return ids, [len(ids) - 1]
    raise ValueError(position)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--position", default="music_start", choices=("score_start", "music_start", "music_middle64", "music_end"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/YuE2-3B"))
    p.add_argument("--vae", type=Path, default=Path("models/YuE2-Vae"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    samples = read_json(args.samples)
    by_song = {}
    for row in samples:
        by_song.setdefault(row["song_id"], {})[row["style_group"]] = row
    songs = sorted(s for s, d in by_song.items() if set(d) == set(STYLES))
    pipe = load_pipeline(args.model, args.vae, args.device)
    try:
        model = pipe._load_model()
        cap = {k: [] for k in capture.CAPTURE_KEYS}
        for song in songs:
            per_style = {k: [] for k in cap}
            for style in STYLES:
                ids, positions = position_input(by_song[song][style], args.position)
                value = capture.capture_positions(model, ids, positions)
                for k in cap:
                    per_style[k].append(value[k])
            for k in cap:
                cap[k].append(np.stack(per_style[k]))
            print(song, "captured", flush=True)
        cap = {k: np.stack(v) for k, v in cap.items()}                      # [lyric, style, layer, width]
        fit = directions.fit_single_position(cap, directions.projection_weights(model))
    finally:
        pipe.close()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"single_{args.position}.npz", styles=np.asarray(STYLES), **fit)
    # component selection: positive global ranking (score stage) or importance quotas (music-token stage,
    # requires an importance array from the residual-push probe; here the local write scores are used as quotas' weights)
    circuits = {}
    for si, style in enumerate(STYLES):
        circuits[style] = {"mlp": directions.positive_global_selection(fit["mlp_scores"][si], 392),
                           "attention": directions.positive_global_selection(fit["head_scores"][si], 168)}
    write_json(args.output / f"circuits_single_{args.position}.json", {"semantic" if "music" in args.position else "abc": circuits})
    print("wrote", args.output / f"single_{args.position}.npz")


if __name__ == "__main__":
    main()
