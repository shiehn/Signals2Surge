"""Unit tests for the patch variation generator."""

from pathlib import Path

import pytest

from synth2surge.prior.generator import generate_variations
from synth2surge.surge.patch import SurgePatch

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "patches"


@pytest.fixture
def init_patch() -> SurgePatch:
    return SurgePatch.from_file(FIXTURES_DIR / "init_fm2.fxp")


class TestGenerateVariations:
    def test_correct_count(self, init_patch: SurgePatch):
        variations = generate_variations(init_patch, n=5, seed=42)
        assert len(variations) == 5

    def test_variations_differ_from_original(self, init_patch: SurgePatch):
        variations = generate_variations(init_patch, n=3, seed=42)
        orig_params = init_patch.get_all_parameters()
        for var in variations:
            var_params = var.get_all_parameters()
            diffs = sum(
                1
                for k in orig_params
                if k in var_params and orig_params[k] != var_params[k]
            )
            assert diffs > 0, "Variation should differ from original"

    def test_variations_differ_from_each_other(self, init_patch: SurgePatch):
        variations = generate_variations(init_patch, n=3, seed=42)
        params_list = [v.get_all_parameters() for v in variations]
        # Check first two variations differ
        common = set(params_list[0].keys()) & set(params_list[1].keys())
        diffs = sum(1 for k in common if params_list[0][k] != params_list[1][k])
        assert diffs > 0

    def test_deterministic_with_seed(self, init_patch: SurgePatch):
        v1 = generate_variations(init_patch, n=2, seed=123)
        v2 = generate_variations(init_patch, n=2, seed=123)
        p1 = v1[0].get_all_parameters()
        p2 = v2[0].get_all_parameters()
        common = set(p1.keys()) & set(p2.keys())
        for k in common:
            assert p1[k] == pytest.approx(p2[k], abs=1e-10)

    def test_structural_params_preserved(self, init_patch: SurgePatch):
        """Oscillator types should not change."""
        orig_type = init_patch.get_parameter("a_osc1_type")
        variations = generate_variations(init_patch, n=5, seed=42)
        for var in variations:
            assert var.get_parameter("a_osc1_type") == orig_type

    def test_zero_variations(self, init_patch: SurgePatch):
        variations = generate_variations(init_patch, n=0, seed=42)
        assert len(variations) == 0
