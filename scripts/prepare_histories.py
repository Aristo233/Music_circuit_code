#!/usr/bin/env python3
"""Write one empty-tag score (and music-token history) per lyric of a split.

These histories are the shared teacher-forcing inputs of every direction fit
and causal test: the score written under the empty tag is fed to the model
under the empty tag and under each style tag, so the two forward passes differ
only in the tag. Music-token histories are capped at 256 tokens (they only
serve the first-128-step mean of the music-token stage).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from common import load_pipeline, read_json, write_json, split_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True, help="dataset.json (see README for the format)")
    p.add_argument("--split", default="representation", help="split whose lyrics get histories")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/YuE2-3B"))
    p.add_argument("--vae", type=Path, default=Path("models/YuE2-Vae"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--music-cap", type=int, default=256)
    args = p.parse_args()
    from yue2.protocol import SongRequest
    dataset = read_json(args.dataset)
    pipe = load_pipeline(args.model, args.vae, args.device)
    rows = []
    try:
        for record in split_records(dataset, args.split):
            path = args.output / "histories" / f"{record['song_id']}.json"
            if path.exists():
                rows.append(read_json(path))
                continue
            request = SongRequest(style="", lyrics=record["lyrics"], cot="melody", seed=int(record["score_seed"]),
                                  cfg_scale=1.0, id=record["song_id"])
            plan = pipe.plan(request=request)
            music_request = replace(request, seed=int(record["music_seed"]))
            music_plan = replace(plan, request=music_request)
            from yue2.protocol import token_prefixes
            music_plan = replace(music_plan, prefix=token_prefixes(music_request, pipe.tokenizer, plan.abc_ids))
            sampling = replace(pipe.generation_config.semantic, max_tokens=args.music_cap)
            semantic = pipe.generate_semantic(music_plan, sampling=sampling)
            row = dict(song_id=record["song_id"], family_id=record.get("family_id"), lyrics=record["lyrics"],
                       seed=int(record["score_seed"]), music_seed=int(record["music_seed"]), abc=plan.abc,
                       abc_ids=list(plan.abc_ids), abc_truncated=bool(plan.truncated),
                       semantic_tokens=list(semantic.tokens), semantic_cap=args.music_cap)
            write_json(path, row)
            rows.append(row)
            print(record["song_id"], len(row["abc_ids"]), "score tokens", flush=True)
    finally:
        pipe.close()
    write_json(args.output / "histories.json", dict(split=args.split, records=rows))


if __name__ == "__main__":
    main()
