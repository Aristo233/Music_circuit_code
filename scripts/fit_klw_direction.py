#!/usr/bin/env python3
"""Fit the KL-weighted score-stage directions and select the dynamic circuits.

Writes ``klw_abc.npz`` (component differences, residual directions) and
``circuits_abc_klw.json`` (392 neurons + 168 heads per style). With
``--uniform-window 128`` the same script fits the equal-weight first-128-step
mean instead (used for the music-token stage on music-token histories with
``--stage semantic``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import load_pipeline, read_json, write_json, style_texts
from music_circuit import directions
from music_circuit.constants import CODEC_OFFSET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--histories", type=Path, required=True, help="histories.json from prepare_histories.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("abc", "semantic"), default="abc")
    p.add_argument("--uniform-window", type=int, default=None, help="equal weights over the first N steps instead of KL weights")
    p.add_argument("--model", type=Path, default=Path("models/YuE2-3B"))
    p.add_argument("--vae", type=Path, default=Path("models/YuE2-Vae"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dataset = read_json(args.dataset)
    texts = style_texts(dataset)
    histories = read_json(args.histories)["records"]
    if args.stage == "semantic":
        # music-token stage: the history is the score plus the music tokens
        for row in histories:
            row["abc_ids"] = list(row["abc_ids"])
        if args.uniform_window is None:
            raise SystemExit("The music-token stage uses --uniform-window 128 in the paper")
    pipe = load_pipeline(args.model, args.vae, args.device)
    try:
        model = pipe._load_model()
        if args.stage == "abc":
            bank = directions.fit_klw_directions(model, pipe.tokenizer, histories, texts,
                                                 uniform_window=args.uniform_window,
                                                 on_progress=lambda v: print(v, flush=True))
        else:
            bank = fit_music_stage(model, pipe.tokenizer, histories, texts, args.uniform_window)
        weights = directions.projection_weights(model)
    finally:
        pipe.close()
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"klw_{args.stage}" if args.uniform_window is None else f"mean{args.uniform_window}_{args.stage}"
    np.savez_compressed(args.output / f"{name}.npz", **{k: v for k, v in bank.items() if k != "kl_summary"})
    scores = directions.klw_component_scores(bank, weights)
    circuits = directions.select_klw_circuits(scores, list(bank["styles"]))
    write_json(args.output / f"circuits_{name}.json", {args.stage: circuits})
    write_json(args.output / f"{name}_manifest.json", dict(stage=args.stage, uniform_window=args.uniform_window,
               lyrics=bank["lyrics"], kl_summary=bank["kl_summary"],
               selected={s: {k: sum(len(v) for v in c[k].values()) for k in c} for s, c in circuits.items()}))
    print("wrote", args.output / f"{name}.npz")


def fit_music_stage(model, tokenizer, histories, texts, window):
    """Equal-weight mean over the first ``window`` music-token steps (teacher forced on the shared history)."""
    import torch
    from music_circuit import capture
    styles = list(texts)
    sums, count = None, 0
    for row in histories:
        codec = [t + CODEC_OFFSET for t in row["semantic_tokens"]][:window]
        if len(codec) < window:
            continue
        prefixes = capture.matched_prefixes(tokenizer, row["lyrics"], texts[styles[0]], "semantic", row["abc_ids"], seed=row.get("music_seed", 0))
        blank, _ = capture.capture_sequence(model, prefixes["blank"] + codec, len(prefixes["blank"]) - 1, "semantic")
        per_style = {}
        for s in styles:
            target = capture.matched_prefixes(tokenizer, row["lyrics"], texts[s], "semantic", row["abc_ids"], seed=row.get("music_seed", 0))["target"]
            acts, _ = capture.capture_sequence(model, target + codec, len(target) - 1, "semantic")
            per_style[s] = {k: (acts[k][:window].float() - blank[k][:window].float()).mean(0).double().cpu().numpy() for k in acts}
            del acts
            torch.cuda.empty_cache()
        if sums is None:
            sums = {s: {k: np.zeros_like(v) for k, v in per_style[s].items()} for s in styles}
        for s in styles:
            for k in per_style[s]:
                sums[s][k] += per_style[s][k]
        count += 1
        print(dict(phase="mean_window", done=count), flush=True)
    mean = {s: {k: v / count for k, v in sums[s].items()} for s in styles}
    residual = np.stack([np.stack([mean[s]["res_attention"], mean[s]["res_mlp"]]) for s in styles])
    return {"styles": np.asarray(styles), "lyrics": count, "kl_summary": [],
            "component_delta_mlp": np.stack([mean[s]["gated_mlp"] for s in styles]),
            "component_delta_attention": np.stack([mean[s]["head_concat"] for s in styles]),
            "residual_delta": residual, "residual_directions": directions.normalized(residual)}


if __name__ == "__main__":
    main()
