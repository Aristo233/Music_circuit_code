"""Music Circuit: tracing and controlling style dynamics in YuE2.

Reference implementation of the paper's three steps:

1. stage localization      -> :mod:`music_circuit.stage_ablation`
2. static observation      -> :mod:`music_circuit.directions` (single-position part),
                              :mod:`music_circuit.importance`
3. dynamic circuit steering-> :mod:`music_circuit.directions` (KL-weighted part),
                              :mod:`music_circuit.causal`, :mod:`music_circuit.steering`

Model access (activation capture and interventions) lives in
:mod:`music_circuit.capture` and :mod:`music_circuit.hooks`; audio style
judgement in :mod:`music_circuit.evaluation`.
"""
from . import constants, capture, hooks, directions, causal, importance, steering, stage_ablation, evaluation

__all__ = ["constants", "capture", "hooks", "directions", "causal", "importance",
           "steering", "stage_ablation", "evaluation"]
