"""Unit tests for preset_loader (no Surge XT required)."""

import struct

import pytest

from synth2surge.surge.patch import extract_xml_from_bytes
from synth2surge.surge.preset_loader import (
    LoadResult,
    LoadStrategy,
    reset_mapping_cache,
)


class TestResetMappingCache:
    """Tests for the reset_mapping_cache() function."""

    def test_reset_clears_caches(self):
        """reset_mapping_cache should set both module-level caches to None."""
        import synth2surge.surge.preset_loader as pl

        # Populate caches with dummy data
        pl._cached_mapping = {"foo": "bar"}
        pl._cached_calibration = {"baz": (0.0, 1.0)}

        reset_mapping_cache()

        assert pl._cached_mapping is None
        assert pl._cached_calibration is None

    def test_reset_idempotent(self):
        """Calling reset_mapping_cache twice should not raise."""
        reset_mapping_cache()
        reset_mapping_cache()


class TestLoadResultDefaults:
    """Tests for LoadResult dataclass defaults."""

    def test_default_values(self):
        result = LoadResult(success=False, strategy=None)
        assert result.matched_params == 0
        assert result.total_params == 0
        assert result.error == ""

    def test_with_strategy(self):
        result = LoadResult(
            success=True,
            strategy=LoadStrategy.PARAMETER_MAP,
            matched_params=10,
            total_params=20,
        )
        assert result.success is True
        assert result.strategy == LoadStrategy.PARAMETER_MAP
        assert result.matched_params == 10
        assert result.total_params == 20


class TestExtractXmlFromBytesVST3:
    """Tests for extract_xml_from_bytes with synthetic VST3 envelope."""

    def _make_vst3_envelope(self, xml: bytes) -> bytes:
        """Build a minimal synthetic VST3 envelope wrapping the given XML."""
        component_data = b"\x00" * 4 + xml + b"\x00" * 4
        chunk_list_offset = 48 + len(component_data)
        header = b"VST3" + b"\x00" * 36 + struct.pack("<Q", chunk_list_offset)
        return header + component_data

    def test_extracts_xml_from_vst3(self):
        xml = (
            b'<?xml version="1.0"?>'
            b"<patch><parameters>"
            b'<test type="2" value="0.5"/>'
            b"</parameters></patch>"
        )
        vst3_data = self._make_vst3_envelope(xml)
        result = extract_xml_from_bytes(vst3_data)
        assert result.startswith(b"<?xml")
        assert b"</patch>" in result

    def test_vst3_without_xml_decl(self):
        """VST3 envelope where XML starts with <patch directly."""
        xml = (
            b"<patch><parameters>"
            b'<test type="2" value="0.5"/>'
            b"</parameters></patch>"
        )
        vst3_data = self._make_vst3_envelope(xml)
        result = extract_xml_from_bytes(vst3_data)
        assert result.startswith(b"<patch")
        assert b"</patch>" in result

    def test_vst3_no_xml_falls_through(self):
        """VST3 envelope with no XML should raise ValueError."""
        component_data = b"\x00" * 100
        chunk_list_offset = 48 + len(component_data)
        header = b"VST3" + b"\x00" * 36 + struct.pack("<Q", chunk_list_offset)
        vst3_data = header + component_data
        with pytest.raises(ValueError, match="No XML content"):
            extract_xml_from_bytes(vst3_data)
