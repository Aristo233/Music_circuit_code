# Music Circuit

Reference implementation of **Music Circuit: Tracing and Controlling Style Dynamics in Large Music Generation Model** (anonymous submission). The code traces where a music language model (YuE2) forms musical style during generation and steers style without any style text by injecting a fixed, KL-weighted activation difference into a sparse set of MLP neurons and attention heads at every step of the score stage.

The package is organized by the three steps of the method. Each module's docstring states the equation it implements.

| Paper section | Module | What it does |
|---|---|---|
| Problem setup, notation | `music_circuit/capture.py` | Prefixes for the empty tag and the style tag, teacher-forced capture of the residual stream after each sublayer and of component activations, stage-restricted next-token distributions, stepwise KL |
| Stage localization | `music_circuit/stage_ablation.py` | 2×2×2 stage-wise tag ablation, Shapley decomposition of the full-tag CLAP effect |
| Static observation (Eq. 1–2) | `music_circuit/directions.py` (`fit_single_position`, `cross_position_agreement`) | Within-lyric centered single-position directions, component attribution by write projection |
| Sublayer importance, budget allocation | `music_circuit/importance.py` | Residual push probe, sign-aligned reference direction, largest-remainder quotas (min 4 neurons / 2 heads per layer, 40 % cap) |
| Dynamic circuit (KL weights, Eq. KLW) | `music_circuit/directions.py` (`fit_klw_directions`, `klw_component_scores`, `select_klw_circuits`) | KL-weighted activation differences and unit residual directions, steering scores, positive global selection of 392 neurons + 168 heads |
| Causal validation | `music_circuit/causal.py` | Ablation (style tag, zero) and enhancement (empty tag, add) at every score position; ΔS with KL and uniform weights; output-KL effect; disjoint and norm-matched random controls |
| Steering, baselines | `music_circuit/hooks.py`, `music_circuit/steering.py` | Every-step component injection wrapped around one AR call, coverage check; recipes for the circuit, empty tag, prompt, random circuit and CAA (residual vector expressed in `down_proj` input space) |
| Evaluation | `music_circuit/evaluation.py` | Coverage-weighted 10 s CLAP windows, five-style scores, margin and hit |

## Installation

```bash
git clone --recurse-submodules https://github.com/multimodal-art-projection/YuE   # or your local YuE2 checkout
pip install -e YuE/upstream            # provides the `yue2` package (pipeline, protocol, modeling)
pip install -r requirements.txt
python -c "import music_circuit"       # from this directory
python -m pytest tests                 # CPU checks on a tiny YuE2 architecture
```

Model weights (`YuE2-3B`, `YuE2-Vae`) are downloaded with the upstream tools and passed to every script with `--model` / `--vae`. All experiments use the native `torch-eager` backend without quantization, because the hooks edit the inputs of `down_proj` and `o_proj` inside the eager backbone.

## Data format

`dataset.json`:

```json
{
  "styles": [{"style_group": "metal_punk", "style_text": "metal rock"}, ...],
  "records": [{"song_id": "s001", "family_id": "f01", "split": "representation",
               "lyrics": "...", "score_seed": 11, "music_seed": 12}, ...]
}
```

`split` is `representation` (the set directions are fitted on) or `test`; the two splits share no lyric family. Style keys and texts of the paper are in `music_circuit/constants.py` (contemporary pop, metal rock, classic funk, electric blues, country). The lyrics used in the paper are released separately with the dataset.

## Reproducing the pipeline

1. **Shared histories.** One empty-tag score (and a 256-token music-token continuation) per representation lyric:
   `python scripts/prepare_histories.py --dataset dataset.json --split representation --output runs/histories`
2. **KL-weighted score-stage direction and circuit** (Eq. KLW, 392 neurons + 168 heads per style):
   `python scripts/fit_klw_direction.py --dataset dataset.json --histories runs/histories/histories.json --output runs/klw`
3. **Music-token-stage direction** (equal-weight mean over the first 128 steps):
   `python scripts/fit_klw_direction.py --stage semantic --uniform-window 128 --dataset dataset.json --histories runs/histories/histories.json --output runs/klw`
4. **Single-position directions and music-token circuit** (Music Start; needs the prompted generations as `samples.json`):
   `python scripts/single_position.py --samples samples.json --position music_start --output runs/single`
5. **Causal validation** (ablation/enhancement curves, random controls):
   `python scripts/causal_validation.py --dataset dataset.json --histories runs/histories/histories.json --bank runs/klw/klw_abc.npz --output runs/causal`
6. **Sublayer importance**: `python scripts/sublayer_importance.py --dataset dataset.json --histories runs/histories/histories.json --bank runs/klw/klw_abc.npz --output runs/importance`
7. **Stage-wise tag ablation**: `python scripts/stage_factorial.py --dataset dataset.json --output runs/factorial`
8. **Steering on the test set** with baselines and CLAP scoring:
   ```bash
   python scripts/steer.py --dataset dataset.json --split test \
     --klw-bank runs/klw/klw_abc.npz --klw-circuits runs/klw/circuits_klw_abc.json \
     --music-bank runs/klw/mean128_semantic.npz --music-circuits runs/single/circuits_single_music_start.json \
     --music-reference runs/single/single_music_start.npz --arms baseline prompt circuit random caa --output runs/steer
   ```

The final recipe (`steering.circuit_recipe`): all style texts empty; score stage adds `0.5 × δ_s` to the selected components at every step; music-token stage adds the first-128-step mean, per-layer norm matched to the single-position direction, at strength 1.0 (1.5 for classic funk and electric blues); acoustic rendering untouched.

## Conventions

* Layer, neuron and head indices are zero-based. Attention head `h` owns input channels `h*128:(h+1)*128` of `o_proj`.
* A "prediction step" is the last position of an AR forward: step 0 is the final prefill position, later steps are single-token cached decodes. Interventions edit only that position, so earlier tokens are never changed.
* Directions are fitted after the sublayer where the component writes: neurons are scored against the direction after the MLP, heads against the direction after attention.
* Random controls are drawn only from components the target does not select and, for enhancement, are rescaled to the target's per-layer injection norm.
* CLAP scoring: 48 kHz mono, 480,000-sample windows, coverage weights summing to one, text template `"This audio is a {style} song."`, margin = target score − best other score, hit = complete generation with positive margin.

## Citation

```bibtex
@inproceedings{musiccircuit2027,
  title     = {Music Circuit: Tracing and Controlling Style Dynamics in Large Music Generation Model},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```
