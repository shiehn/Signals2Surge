"""Generate Surge XT patch variations for building the FAISS prior index."""

from __future__ import annotations

import numpy as np

from synth2surge.surge.patch import PARAM_TYPE_FLOAT, PARAM_TYPE_INT, SurgePatch


def generate_variations(
    patch: SurgePatch,
    n: int,
    sigma: float = 0.1,
    discrete_flip_prob: float = 0.1,
    seed: int | None = None,
) -> list[SurgePatch]:
    """Generate n mutated variations of a Surge XT patch.

    Applies Gaussian noise to continuous (float) parameters and randomly
    flips discrete (int) parameters. Structural parameters like oscillator
    type and filter type are kept fixed to preserve the basic character.

    Args:
        patch: Base patch to mutate.
        n: Number of variations to generate.
        sigma: Standard deviation of Gaussian noise (as fraction of parameter range).
        discrete_flip_prob: Probability of flipping each discrete parameter.
        seed: Random seed for reproducibility.

    Returns:
        List of n mutated SurgePatch instances.
    """
    rng = np.random.default_rng(seed)
    all_params = patch.get_all_parameters()
    param_types = patch.get_parameter_types()

    # Identify mutable parameters (skip structural ones)
    structural_patterns = ("_type", "filtertype", "f2_type", "_polymode", "_config")
    float_params = {
        k: v
        for k, v in all_params.items()
        if param_types.get(k) == PARAM_TYPE_FLOAT
        and not any(p in k for p in structural_patterns)
    }
    int_params = {
        k: v
        for k, v in all_params.items()
        if param_types.get(k) == PARAM_TYPE_INT
        and not any(p in k for p in structural_patterns)
    }

    variations = []
    for _ in range(n):
        clone = patch.clone()

        # Mutate float params with Gaussian noise
        for name, value in float_params.items():
            noise = rng.normal(0, sigma) * 10.0  # Scale relative to typical param ranges
            new_val = value + noise
            clone.set_parameter(name, new_val)

        # Randomly flip some discrete params
        for name, value in int_params.items():
            if rng.random() < discrete_flip_prob:
                # Simple flip: toggle between 0 and 1 for booleans, or small perturbation
                if value in (0, 1):
                    clone.set_parameter(name, 1 - int(value))
                else:
                    delta = rng.choice([-1, 1])
                    clone.set_parameter(name, max(0, int(value) + delta))

        variations.append(clone)

    return variations
