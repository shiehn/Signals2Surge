"""Parameter space definition for Surge XT optimization.

Defines which parameters to optimize, their bounds, types, and tier assignments.
CMA-ES works well up to ~100-200 dimensions, so we use staged optimization
across 3 tiers of decreasing importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from synth2surge.surge.patch import SurgePatch


class ParamType(IntEnum):
    INT = 0
    BOOL = 1
    FLOAT = 2


class Tier(IntEnum):
    """Optimization tier — lower tiers are optimized first."""

    STRUCTURAL = 1  # Oscillator/filter types, ADSR, mixer — most impactful
    SHAPING = 2  # LFO, fine-tuning, FX primary params
    DETAIL = 3  # Modulation depths, secondary FX, detuning


@dataclass(frozen=True)
class ParameterDef:
    """Definition of a single optimizable parameter."""

    name: str
    param_type: ParamType
    min_val: float
    max_val: float
    tier: Tier
    default: float = 0.0

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        if self.max_val == self.min_val:
            return 0.0
        return (value - self.min_val) / (self.max_val - self.min_val)

    def denormalize(self, normalized: float) -> float:
        """Denormalize from [0, 1] back to parameter range."""
        value = self.min_val + normalized * (self.max_val - self.min_val)
        if self.param_type in (ParamType.INT, ParamType.BOOL):
            return round(value)
        return value


class ParameterSpace:
    """Collection of parameter definitions for optimization."""

    def __init__(self, params: list[ParameterDef]) -> None:
        self._params = {p.name: p for p in params}

    @property
    def all_params(self) -> list[ParameterDef]:
        return list(self._params.values())

    def by_tier(self, tier: Tier) -> list[ParameterDef]:
        return [p for p in self._params.values() if p.tier == tier]

    def by_name(self, name: str) -> ParameterDef | None:
        return self._params.get(name)

    def tier_names(self, tier: Tier) -> list[str]:
        return [p.name for p in self.by_tier(tier)]

    @property
    def float_params(self) -> list[ParameterDef]:
        return [p for p in self._params.values() if p.param_type == ParamType.FLOAT]

    @property
    def int_params(self) -> list[ParameterDef]:
        return [
            p
            for p in self._params.values()
            if p.param_type in (ParamType.INT, ParamType.BOOL)
        ]

    def __len__(self) -> int:
        return len(self._params)


def build_parameter_space_from_patch(patch: SurgePatch, scene: str = "a") -> ParameterSpace:
    """Auto-discover parameters from a patch and assign tiers based on naming conventions.

    Args:
        patch: A parsed Surge XT patch to extract parameter info from.
        scene: Which scene to optimize ('a' or 'b').

    Returns:
        ParameterSpace with all scene parameters classified by tier.
    """
    all_params = patch.get_all_parameters()
    param_types = patch.get_parameter_types()
    prefix = f"{scene}_"

    defs = []
    for name, value in all_params.items():
        if not name.startswith(prefix):
            # Also include global params (volume, scene_active, etc.)
            if name.startswith(("a_", "b_")):
                continue
            # Skip non-optimizable globals
            if name in ("scene_active", "scenemode", "splitkey", "fx_disable", "fx_bypass"):
                continue

        ptype_str = param_types.get(name, "2")
        ptype = ParamType(int(ptype_str))
        tier = _classify_tier(name, ptype)
        min_val, max_val = _estimate_bounds(name, ptype, value)

        defs.append(
            ParameterDef(
                name=name,
                param_type=ptype,
                min_val=min_val,
                max_val=max_val,
                tier=tier,
                default=float(value),
            )
        )

    return ParameterSpace(defs)


def _classify_tier(name: str, ptype: ParamType) -> Tier:
    """Assign a parameter to an optimization tier based on naming."""
    # Tier 1: Structural — most impactful parameters
    tier1_patterns = [
        "_osc1_type", "_osc2_type", "_osc3_type",
        "_filtertype", "_f2_type",
        "_f1_cutoff", "_f2_cutoff", "_f1_resonance", "_f2_resonance",
        "_env1_attack", "_env1_decay", "_env1_sustain", "_env1_release",
        "_env2_attack", "_env2_decay", "_env2_sustain", "_env2_release",
        "_level_o1", "_level_o2", "_level_o3", "_level_noise", "_level_ring",
        "_volume", "_pan",
        "_fm_depth", "_fm_switch",
        "volume",
    ]
    for pattern in tier1_patterns:
        if pattern in name or name.endswith(pattern):
            return Tier.STRUCTURAL

    # Tier 2: Shaping — secondary tonal params
    tier2_patterns = [
        "_osc1_param", "_osc2_param", "_osc3_param",
        "_osc1_pitch", "_osc2_pitch", "_osc3_pitch",
        "_osc1_octave", "_osc2_octave", "_osc3_octave",
        "_f1_subtype", "_f2_subtype",
        "_f1_balance", "_f2_balance",
        "_config",
        "_feedback",
        "_lowcut",
        "_drift",
        "_noisecol",
        "_width",
    ]
    for pattern in tier2_patterns:
        if pattern in name:
            return Tier.SHAPING

    # LFO params in tier 2
    if "_lfo" in name:
        return Tier.SHAPING

    # Everything else is Tier 3: Detail
    return Tier.DETAIL


def _estimate_bounds(name: str, ptype: ParamType, current_value: float) -> tuple[float, float]:
    """Estimate reasonable min/max bounds for a parameter."""
    if ptype == ParamType.BOOL:
        return (0.0, 1.0)

    if ptype == ParamType.INT:
        # Oscillator type: 0-11 (12 osc algorithms)
        if "_type" in name and "osc" in name:
            return (0.0, 11.0)
        # Filter type
        if "filtertype" in name or "f2_type" in name:
            return (0.0, 23.0)
        # Octave: -3 to 3
        if "_octave" in name:
            return (-3.0, 3.0)
        # FM switch
        if "_fm_switch" in name:
            return (0.0, 3.0)
        # Polymode, config, etc.
        if "_polymode" in name:
            return (0.0, 5.0)
        # Generic int
        return (0.0, max(12.0, abs(current_value) * 2 + 1))

    # Float parameters — use generous ranges based on Surge defaults
    if "_cutoff" in name:
        return (-72.0, 72.0)
    if "_resonance" in name:
        return (0.0, 1.0)
    if "_attack" in name or "_decay" in name or "_release" in name:
        return (-8.0, 5.0)  # Surge uses log-scale for time
    if "_sustain" in name:
        return (0.0, 1.0)
    if "volume" in name or "_level" in name:
        return (-48.0, 12.0)
    if "_pitch" in name:
        return (-12.0, 12.0)
    if "_pan" in name or "_balance" in name:
        return (-1.0, 1.0)
    if "_depth" in name:
        return (-48.0, 48.0)
    if "_portamento" in name:
        return (-8.0, 2.0)
    if "_width" in name:
        return (0.0, 1.0)
    if "_drift" in name:
        return (0.0, 1.0)
    if "_feedback" in name:
        return (-1.0, 1.0)

    # Generic float: use a wide symmetric range
    return (-12.0, 12.0)
