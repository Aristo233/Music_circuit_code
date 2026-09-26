#!/usr/bin/env python3
"""Stage-wise tag ablation: 2x2x2 factorial over the three YuE2 stages, Shapley decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_pipeline, read_json, write_json, split_records
from music_circuit import evaluation, stage_ablation, steering
from music_circuit.constants import STYLES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", default="representation")
    p.add_argument("--styles", nargs="+", default=STYLES)
    p.add_argument("--seeds", type=int, nargs=2, default=(1, 2), metavar=("ABC_SEED", "MUSIC_SEED"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clap-checkpoint", type=Path, default=None)
    p.add_argument("--model", type=Path, default=Path("models/YuE2-3B"))
    p.add_argument("--vae", type=Path, default=Path("models/YuE2-Vae"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    import soundfile as sf
    dataset = read_json(args.dataset)
    pipe = load_pipeline(args.model, args.vae, args.device)
    encoder, rows = None, []
    try:
        for record in split_records(dataset, args.split):
            for style in args.styles:
                for arm, r in stage_ablation.factorial_recipes(style).items():
                    case = f"{record['song_id']}__{style}__{arm}"
                    folder = args.output / case
                    if (folder / "result.json").exists():
                        rows.append(read_json(folder / "result.json"))
                        continue
                    # NOTE: the paper reuses earlier-stage outputs across arms through a
                    # content-addressed cache; with fixed seeds the stages are deterministic,
                    # so regenerating them here yields the same tokens.
                    song = steering.generate_song(pipe, record["lyrics"], r, abc_seed=args.seeds[0],
                                                  music_seed=args.seeds[1], song_id=record["song_id"])
                    folder.mkdir(parents=True, exist_ok=True)
                    sf.write(folder / "audio.flac", song["waveform"], song["sample_rate"], subtype="PCM_24")
                    if encoder is None:
                        encoder = evaluation.LaionClapEncoder(args.clap_checkpoint, device=args.device)
                    clap = evaluation.score_song(encoder, song["waveform"], style, complete=song["complete"])
                    result = dict(song_id=record["song_id"], style=style, arm=arm, complete=song["complete"],
                                  margin=clap["target_margin"] if song["complete"] else None, hit=clap["hit"])
                    write_json(folder / "result.json", result)
                    rows.append(result)
                    print(json.dumps(result), flush=True)
    finally:
        pipe.close()
    summary = stage_ablation.decompose(rows)
    write_json(args.output / "shapley.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
