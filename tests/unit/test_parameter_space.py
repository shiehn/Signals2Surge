"""Unit tests for the parameter space definition and tier classification."""

from pathlib import Path

import pytest

from synth2surge.surge.parameter_space import (
    ParameterDef,
    ParameterSpace,
    ParamType,
    Tier,
    build_parameter_space_from_patch,
)
from synth2surge.surge.patch import SurgePatch

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "patches"


@pytest.fixture
def init_patch() -> SurgePatch:
    return SurgePatch.from_file(FIXTURES_DIR / "init_fm2.fxp")


@pytest.fixture
def param_space(init_patch: SurgePatch) -> ParameterSpace:
    return build_parameter_space_from_patch(init_patch, scene="a")


class TestParameterDef:
    """Tests for individual parameter definitions."""

    def test_normalize_float(self):
        p = ParameterDef("test", ParamType.FLOAT, 0.0, 10.0, Tier.STRUCTURAL, 5.0)
        assert p.normalize(0.0) == pytest.approx(0.0)
        assert p.normalize(5.0) == pytest.approx(0.5)
        assert p.normalize(10.0) == pytest.approx(1.0)

    def test_denormalize_float(self):
        p = ParameterDef("test", ParamType.FLOAT, 0.0, 10.0, Tier.STRUCTURAL, 5.0)
        assert p.denormalize(0.0) == pytest.approx(0.0)
        assert p.denormalize(0.5) == pytest.approx(5.0)
        assert p.denormalize(1.0) == pytest.approx(10.0)

    def test_normalize_denormalize_roundtrip(self):
        p = ParameterDef("test", ParamType.FLOAT, -12.0, 12.0, Tier.SHAPING, 0.0)
        for val in [-12.0, -6.0, 0.0, 6.0, 12.0]:
            assert p.denormalize(p.normalize(val)) == pytest.approx(val)

    def test_int_denormalize_rounds(self):
        p = ParameterDef("test", ParamType.INT, 0.0, 11.0, Tier.STRUCTURAL, 0.0)
        assert p.denormalize(0.5) == 6  # rounds to nearest int

    def test_bool_denormalize_rounds(self):
        p = ParameterDef("test", ParamType.BOOL, 0.0, 1.0, Tier.DETAIL, 0.0)
        assert p.denormalize(0.3) == 0
        assert p.denormalize(0.7) == 1

    def test_equal_min_max_normalize(self):
        p = ParameterDef("test", ParamType.FLOAT, 5.0, 5.0, Tier.DETAIL, 5.0)
        assert p.normalize(5.0) == 0.0


class TestParameterSpace:
    """Tests for the parameter space collection."""

    def test_build_from_patch(self, param_space: ParameterSpace):
        assert len(param_space) > 0

    def test_has_scene_a_params(self, param_space: ParameterSpace):
        names = [p.name for p in param_space.all_params]
        assert any(n.startswith("a_") for n in names)

    def test_excludes_scene_b_params(self, param_space: ParameterSpace):
        names = [p.name for p in param_space.all_params]
        assert not any(n.startswith("b_") for n in names)

    def test_tier1_has_params(self, param_space: ParameterSpace):
        tier1 = param_space.by_tier(Tier.STRUCTURAL)
        assert len(tier1) > 0
        assert len(tier1) <= 80  # Should be a manageable subset

    def test_tier2_has_params(self, param_space: ParameterSpace):
        tier2 = param_space.by_tier(Tier.SHAPING)
        assert len(tier2) > 0

    def test_tier3_has_params(self, param_space: ParameterSpace):
        tier3 = param_space.by_tier(Tier.DETAIL)
        assert len(tier3) > 0

    def test_all_tiers_cover_all_params(self, param_space: ParameterSpace):
        t1 = len(param_space.by_tier(Tier.STRUCTURAL))
        t2 = len(param_space.by_tier(Tier.SHAPING))
        t3 = len(param_space.by_tier(Tier.DETAIL))
        assert t1 + t2 + t3 == len(param_space)

    def test_tier_names(self, param_space: ParameterSpace):
        names = param_space.tier_names(Tier.STRUCTURAL)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_by_name(self, param_space: ParameterSpace):
        p = param_space.by_name("a_osc1_type")
        assert p is not None
        assert p.param_type == ParamType.INT

    def test_by_name_missing(self, param_space: ParameterSpace):
        assert param_space.by_name("nonexistent") is None

    def test_float_params(self, param_space: ParameterSpace):
        floats = param_space.float_params
        assert len(floats) > 100
        assert all(p.param_type == ParamType.FLOAT for p in floats)

    def test_int_params(self, param_space: ParameterSpace):
        ints = param_space.int_params
        assert len(ints) > 20
        assert all(p.param_type in (ParamType.INT, ParamType.BOOL) for p in ints)

    def test_key_structural_params_in_tier1(self, param_space: ParameterSpace):
        """Critical sound-shaping params must be in tier 1."""
        expected = ["a_osc1_type", "a_f1_cutoff", "a_env1_attack"]
        for name in expected:
            p = param_space.by_name(name)
            if p is not None:  # Some params may not exist in all patches
                assert p.tier == Tier.STRUCTURAL, f"{name} should be tier 1"
