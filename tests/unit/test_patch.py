"""Unit tests for Surge XT patch parsing, mutation, and serialization."""

import struct
from pathlib import Path

import pytest

from synth2surge.surge.patch import SurgePatch, extract_xml_from_bytes

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "patches"


@pytest.fixture
def init_patch() -> SurgePatch:
    return SurgePatch.from_file(FIXTURES_DIR / "init_fm2.fxp")


@pytest.fixture
def lead_patch() -> SurgePatch:
    return SurgePatch.from_file(FIXTURES_DIR / "lead_acidofil.fxp")


class TestParsing:
    """Tests for loading and parsing Surge patches."""

    def test_load_fxp_file(self, init_patch: SurgePatch):
        assert init_patch is not None

    def test_metadata_extracted(self, init_patch: SurgePatch):
        meta = init_patch.metadata
        assert meta.name == "Init FM2"
        assert meta.category == "Templates"
        assert meta.author == "Surge Synth Team"

    def test_revision_number(self, init_patch: SurgePatch):
        assert init_patch.revision >= 1

    def test_parameter_count(self, init_patch: SurgePatch):
        params = init_patch.get_all_parameters()
        # Surge patches have ~558 typed parameters
        assert len(params) >= 500
        assert len(params) <= 600

    def test_parameter_names_list(self, init_patch: SurgePatch):
        names = init_patch.parameter_names()
        assert len(names) >= 500
        assert "a_osc1_type" in names
        assert "b_volume" in names

    def test_float_parameter_names(self, init_patch: SurgePatch):
        float_names = init_patch.float_parameter_names()
        assert len(float_names) > 200
        assert "a_pitch" in float_names

    def test_int_parameter_names(self, init_patch: SurgePatch):
        int_names = init_patch.int_parameter_names()
        assert len(int_names) > 50
        assert "a_osc1_type" in int_names

    def test_different_patches_differ(self, init_patch: SurgePatch, lead_patch: SurgePatch):
        init_params = init_patch.get_all_parameters()
        lead_params = lead_patch.get_all_parameters()
        # Should have substantial overlap in parameter names
        common = set(init_params.keys()) & set(lead_params.keys())
        assert len(common) > 400  # Core params shared across all patches
        # But different values on common params
        diffs = sum(1 for k in common if init_params[k] != lead_params[k])
        assert diffs > 10  # Patches should differ in many parameters


class TestGetSetParameters:
    """Tests for reading and writing individual parameters."""

    def test_get_parameter(self, init_patch: SurgePatch):
        val = init_patch.get_parameter("a_osc1_type")
        assert val is not None
        assert isinstance(val, int)

    def test_get_float_parameter(self, init_patch: SurgePatch):
        val = init_patch.get_parameter("a_pitch")
        assert val is not None
        assert isinstance(val, float)

    def test_get_nonexistent_returns_none(self, init_patch: SurgePatch):
        val = init_patch.get_parameter("nonexistent_param_xyz")
        assert val is None

    def test_set_float_parameter(self, init_patch: SurgePatch):
        init_patch.set_parameter("a_pitch", 5.0)
        assert init_patch.get_parameter("a_pitch") == pytest.approx(5.0)

    def test_set_int_parameter(self, init_patch: SurgePatch):
        init_patch.set_parameter("a_osc1_type", 3)
        assert init_patch.get_parameter("a_osc1_type") == 3

    def test_set_nonexistent_raises(self, init_patch: SurgePatch):
        with pytest.raises(KeyError, match="nonexistent"):
            init_patch.set_parameter("nonexistent_param_xyz", 1.0)

    def test_set_all_parameters(self, init_patch: SurgePatch):
        updates = {"a_pitch": 7.5, "a_osc1_type": 2}
        init_patch.set_all_parameters(updates)
        assert init_patch.get_parameter("a_pitch") == pytest.approx(7.5)
        assert init_patch.get_parameter("a_osc1_type") == 2


class TestRoundTrip:
    """Tests for serialize/deserialize round-trip fidelity."""

    def test_xml_round_trip(self, init_patch: SurgePatch):
        """Parse -> serialize -> re-parse should produce identical parameters."""
        original_params = init_patch.get_all_parameters()
        xml = init_patch.to_xml_string()
        reparsed = SurgePatch.from_xml_string(xml)
        reparsed_params = reparsed.get_all_parameters()

        assert set(original_params.keys()) == set(reparsed_params.keys())
        for key in original_params:
            assert original_params[key] == pytest.approx(
                reparsed_params[key], abs=1e-6
            ), f"Mismatch on {key}: {original_params[key]} != {reparsed_params[key]}"

    def test_modification_persists_through_round_trip(self, init_patch: SurgePatch):
        """Modify a param, serialize, re-parse — modification should persist."""
        init_patch.set_parameter("a_pitch", 3.14159)
        xml = init_patch.to_xml_string()
        reparsed = SurgePatch.from_xml_string(xml)
        assert reparsed.get_parameter("a_pitch") == pytest.approx(3.14159, abs=1e-5)

    def test_file_round_trip(self, init_patch: SurgePatch, tmp_path: Path):
        """Write to file, read back — should preserve parameters."""
        out_path = tmp_path / "test_patch.xml"
        init_patch.to_file(out_path)
        reloaded = SurgePatch.from_file(out_path)
        original = init_patch.get_all_parameters()
        reloaded_params = reloaded.get_all_parameters()
        for key in original:
            assert original[key] == pytest.approx(reloaded_params[key], abs=1e-6)


class TestClone:
    """Tests for deep-copy functionality."""

    def test_clone_independent(self, init_patch: SurgePatch):
        """Cloned patch should be independent of original."""
        clone = init_patch.clone()
        clone.set_parameter("a_pitch", 99.0)
        assert init_patch.get_parameter("a_pitch") != pytest.approx(99.0)

    def test_clone_preserves_values(self, init_patch: SurgePatch):
        clone = init_patch.clone()
        orig_params = init_patch.get_all_parameters()
        clone_params = clone.get_all_parameters()
        for key in orig_params:
            assert orig_params[key] == pytest.approx(clone_params[key], abs=1e-6)


class TestParameterTypes:
    """Tests for parameter type introspection."""

    def test_parameter_types_returned(self, init_patch: SurgePatch):
        types = init_patch.get_parameter_types()
        assert len(types) >= 500
        assert types["a_osc1_type"] == "0"  # int
        assert types["a_pitch"] == "2"  # float


class TestVST3StateBytes:
    """Tests for from_state_bytes with synthetic VST3 format input."""

    def test_from_state_bytes_vst3_envelope(self):
        """from_state_bytes should extract XML from a VST3 envelope."""
        xml = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<patch revision="1"><parameters>'
            b'<a_pitch type="2" value="0.00000000000000"/>'
            b'</parameters></patch>'
        )
        # Build a synthetic VST3 envelope:
        # bytes 0-3: "VST3"
        # bytes 4-39: padding
        # bytes 40-47: chunk_list_offset (little-endian u64)
        # bytes 48+: component data containing XML
        component_data = b"\x00" * 10 + xml + b"\x00" * 10
        chunk_list_offset = 48 + len(component_data)
        header = b"VST3" + b"\x00" * 36 + struct.pack("<Q", chunk_list_offset)
        vst3_data = header + component_data

        patch = SurgePatch.from_state_bytes(vst3_data)
        params = patch.get_all_parameters()
        assert "a_pitch" in params

    def test_extract_xml_from_vst3_bounds_check(self):
        """extract_xml_from_bytes should handle truncated VST3 safely."""
        xml = (
            b'<?xml version="1.0"?>'
            b'<patch><parameters>'
            b'<x type="2" value="1.0"/>'
            b'</parameters></patch>'
        )
        component_data = xml
        # Set chunk_list_offset beyond actual data length
        chunk_list_offset = 48 + len(component_data) + 9999
        header = b"VST3" + b"\x00" * 36 + struct.pack("<Q", chunk_list_offset)
        vst3_data = header + component_data

        result = extract_xml_from_bytes(vst3_data)
        assert b"<patch>" in result
