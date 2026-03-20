"""Parameter injection strategies for the optimization loop.

Bridges between Optuna's parameter suggestions and the Surge XT plugin state,
handling normalization, type conversion, and patch mutation.
"""

from __future__ import annotations

from synth2surge.surge.parameter_space import ParameterDef, ParameterSpace, ParamType, Tier
from synth2surge.surge.patch import SurgePatch


def apply_trial_to_patch(
    patch: SurgePatch,
    suggestions: dict[str, float],
    param_space: ParameterSpace,
) -> SurgePatch:
    """Apply Optuna trial suggestions to a Surge XT patch.

    Args:
        patch: The base patch to mutate (will be cloned).
        suggestions: Dict of {param_name: normalized_value} from Optuna.
        param_space: Parameter definitions for denormalization.

    Returns:
        A new SurgePatch with the suggested parameter values applied.
    """
    mutated = patch.clone()
    for name, normalized_val in suggestions.items():
        pdef = param_space.by_name(name)
        if pdef is None:
            continue
        raw_val = pdef.denormalize(normalized_val)
        try:
            mutated.set_parameter(name, raw_val)
        except KeyError:
            pass  # Parameter may not exist in this patch variant
    return mutated


def suggest_parameters(
    trial,  # optuna.Trial
    active_params: list[ParameterDef],
) -> dict[str, float]:
    """Generate parameter suggestions from an Optuna trial.

    Returns normalized [0, 1] values for each active parameter.
    """
    suggestions = {}
    for pdef in active_params:
        if pdef.param_type == ParamType.FLOAT:
            val = trial.suggest_float(pdef.name, 0.0, 1.0)
        elif pdef.param_type == ParamType.INT:
            val = trial.suggest_float(pdef.name, 0.0, 1.0)
        elif pdef.param_type == ParamType.BOOL:
            val = trial.suggest_float(pdef.name, 0.0, 1.0)
        else:
            val = trial.suggest_float(pdef.name, 0.0, 1.0)
        suggestions[pdef.name] = val
    return suggestions


def get_stage_config(stage: int, param_space: ParameterSpace) -> list[ParameterDef]:
    """Get the active parameters for a given optimization stage.

    Stage 1: Tier 1 (structural) parameters only
    Stage 2: Tier 2 (shaping) parameters (tier 1 frozen at best)
    Stage 3: Tier 3 (detail) parameters (tiers 1+2 frozen at best)
    """
    tier_map = {1: Tier.STRUCTURAL, 2: Tier.SHAPING, 3: Tier.DETAIL}
    tier = tier_map.get(stage)
    if tier is None:
        raise ValueError(f"Invalid stage: {stage}")
    return param_space.by_tier(tier)
