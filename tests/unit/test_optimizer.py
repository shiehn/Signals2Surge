"""Unit tests for the optimizer — no plugin required."""

from synth2surge.optimizer.loop import classify_parameter_tier


class TestClassifyParameterTier:
    def test_osc_type_is_tier1(self):
        assert classify_parameter_tier("a_osc_1_type") == 1

    def test_filter_cutoff_is_tier1(self):
        assert classify_parameter_tier("a_filter_1_cutoff") == 1

    def test_amp_eg_attack_is_tier1(self):
        assert classify_parameter_tier("a_amp_eg_attack") == 1

    def test_volume_is_tier1(self):
        assert classify_parameter_tier("volume") == 1

    def test_osc_param_is_tier2(self):
        assert classify_parameter_tier("a_osc_1_param_1") == 2

    def test_lfo_is_tier2(self):
        assert classify_parameter_tier("a_lfo_1_rate") == 2

    def test_drift_is_tier2(self):
        assert classify_parameter_tier("a_drift") == 2

    def test_unknown_param_is_tier3(self):
        assert classify_parameter_tier("a_some_obscure_param") == 3

    def test_fx_is_tier3(self):
        assert classify_parameter_tier("fx_1_param_0") == 3
