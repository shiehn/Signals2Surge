"""Unit tests for FXP export (no Surge XT required)."""

import struct
from pathlib import Path

import pytest

from synth2surge.surge.fxp_export import (
    FXP_CHUNK_TYPE,
    FXP_MAGIC,
    FXP_NAME_SIZE,
    SURGE_PLUGIN_ID,
    build_fxp_header,
    patch_to_fxp,
    state_to_fxp,
)
from synth2surge.surge.patch import SurgePatch

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "patches"


@pytest.fixture
def init_patch() -> SurgePatch:
    return SurgePatch.from_file(FIXTURES_DIR / "init_fm2.fxp")


class TestBuildFxpHeader:
    """Tests for build_fxp_header()."""

    def test_header_starts_with_magic(self):
        header = build_fxp_header("Test", xml_size=100)
        assert header[:4] == FXP_MAGIC

    def test_header_contains_chunk_type(self):
        header = build_fxp_header("Test", xml_size=100)
        assert FXP_CHUNK_TYPE in header

    def test_header_contains_plugin_id(self):
        header = build_fxp_header("Test", xml_size=100)
        assert SURGE_PLUGIN_ID in header

    def test_header_size_is_60_bytes(self):
        header = build_fxp_header("Test", xml_size=100)
        assert len(header) == 60

    def test_name_embedded_in_header(self):
        header = build_fxp_header("My Preset", xml_size=100)
        assert b"My Preset" in header

    def test_name_truncated_to_28_bytes(self):
        long_name = "A" * 50
        header = build_fxp_header(long_name, xml_size=100)
        assert len(header) == 60
        # Name field is 28 bytes starting at offset 28
        name_field = header[28:56]
        assert len(name_field) == FXP_NAME_SIZE
        assert name_field == (b"A" * 28)

    def test_name_null_padded(self):
        header = build_fxp_header("Hi", xml_size=100)
        name_field = header[28:56]
        assert name_field[:2] == b"Hi"
        assert name_field[2:] == b"\x00" * 26

    def test_big_endian_chunk_size(self):
        """Chunk size field at offset 4 should be big-endian."""
        xml_size = 1000
        header = build_fxp_header("Test", xml_size=xml_size)
        chunk_size = struct.unpack_from(">I", header, 4)[0]
        # chunk_size = 52 (header fields after magic+size) + xml_size
        assert chunk_size == 52 + xml_size

    def test_big_endian_xml_size(self):
        """XML size field at offset 56 should be big-endian."""
        xml_size = 2048
        header = build_fxp_header("Test", xml_size=xml_size)
        stored_size = struct.unpack_from(">I", header, 56)[0]
        assert stored_size == xml_size

    def test_malformed_fxp_no_xml_raises(self):
        """FXP magic followed by no XML should raise ValueError via extract."""
        from synth2surge.surge.patch import extract_xml_from_bytes

        # CcnK magic + garbage, no XML inside
        data = b"CcnK" + b"\x00" * 56
        with pytest.raises(ValueError, match="No XML content"):
            extract_xml_from_bytes(data)


class TestStateToFxp:
    """Tests for state_to_fxp()."""

    def test_fxp_passthrough(self, tmp_path: Path):
        """If state is already FXP, write it directly."""
        fxp_data = (FIXTURES_DIR / "init_fm2.fxp").read_bytes()
        out = state_to_fxp(fxp_data, tmp_path / "out.fxp")
        assert out.exists()
        result = out.read_bytes()
        assert result[:4] == FXP_MAGIC
        assert result == fxp_data

    def test_xml_gets_header_prepended(self, tmp_path: Path):
        """Raw XML should get an FXP header prepended."""
        xml_data = b'<?xml version="1.0"?><patch><parameters/></patch>'
        out = state_to_fxp(xml_data, tmp_path / "out.fxp", preset_name="Test")
        result = out.read_bytes()
        assert result[:4] == FXP_MAGIC
        # XML should follow header
        xml_offset = result.find(b"<?xml")
        assert xml_offset > 0
        assert result[xml_offset:] == xml_data

    def test_patch_tag_detected(self, tmp_path: Path):
        """State starting with <patch should also work."""
        xml_data = b"<patch><parameters/></patch>"
        out = state_to_fxp(xml_data, tmp_path / "out.fxp")
        result = out.read_bytes()
        assert result[:4] == FXP_MAGIC
        assert xml_data in result

    def test_embedded_xml_extracted(self, tmp_path: Path):
        """XML embedded after binary junk should be found."""
        junk = b"\x00" * 20
        xml_data = b'<?xml version="1.0"?><patch/>'
        out = state_to_fxp(junk + xml_data, tmp_path / "out.fxp")
        result = out.read_bytes()
        assert result[:4] == FXP_MAGIC
        assert xml_data in result

    def test_no_xml_raises(self, tmp_path: Path):
        """State with no XML should raise ValueError."""
        with pytest.raises(ValueError, match="No XML content"):
            state_to_fxp(b"\x00" * 50, tmp_path / "out.fxp")


class TestPatchToFxp:
    """Tests for patch_to_fxp() and SurgePatch FXP methods."""

    def test_patch_to_fxp_creates_valid_file(
        self, init_patch: SurgePatch, tmp_path: Path
    ):
        out = patch_to_fxp(init_patch, tmp_path / "exported.fxp")
        assert out.exists()
        data = out.read_bytes()
        assert data[:4] == FXP_MAGIC
        # Should contain parseable XML
        xml_start = data.find(b"<?xml")
        assert xml_start > 0

    def test_patch_to_fxp_uses_metadata_name(
        self, init_patch: SurgePatch, tmp_path: Path
    ):
        out = patch_to_fxp(init_patch, tmp_path / "exported.fxp")
        data = out.read_bytes()
        assert b"Init FM2" in data

    def test_patch_to_fxp_custom_name(
        self, init_patch: SurgePatch, tmp_path: Path
    ):
        out = patch_to_fxp(
            init_patch, tmp_path / "exported.fxp", preset_name="Custom"
        )
        data = out.read_bytes()
        assert b"Custom" in data

    def test_roundtrip_params_preserved(
        self, init_patch: SurgePatch, tmp_path: Path
    ):
        """Write as FXP, reload, compare params."""
        original_params = init_patch.get_all_parameters()
        out = patch_to_fxp(init_patch, tmp_path / "rt.fxp")
        reloaded = SurgePatch.from_file(out)
        reloaded_params = reloaded.get_all_parameters()

        assert set(original_params.keys()) == set(reloaded_params.keys())
        for key in original_params:
            assert original_params[key] == pytest.approx(
                reloaded_params[key], abs=1e-6
            ), f"Mismatch on {key}"


class TestSurgePatchFxpMethods:
    """Tests for the new SurgePatch.to_fxp_bytes/to_fxp_file/from_state_bytes methods."""

    def test_to_fxp_bytes(self, init_patch: SurgePatch):
        fxp = init_patch.to_fxp_bytes()
        assert fxp[:4] == FXP_MAGIC
        assert b"<?xml" in fxp

    def test_to_fxp_file(self, init_patch: SurgePatch, tmp_path: Path):
        out_path = tmp_path / "patch.fxp"
        init_patch.to_fxp_file(out_path)
        assert out_path.exists()
        data = out_path.read_bytes()
        assert data[:4] == FXP_MAGIC

    def test_from_state_bytes_fxp(self):
        """from_state_bytes should handle FXP-format input."""
        fxp_data = (FIXTURES_DIR / "init_fm2.fxp").read_bytes()
        patch = SurgePatch.from_state_bytes(fxp_data)
        assert patch.metadata.name == "Init FM2"
        assert len(patch.get_all_parameters()) > 500

    def test_from_state_bytes_xml(self, init_patch: SurgePatch):
        """from_state_bytes should handle raw XML input."""
        xml_bytes = init_patch.to_xml_bytes()
        patch = SurgePatch.from_state_bytes(xml_bytes)
        assert len(patch.get_all_parameters()) > 500

    def test_from_state_bytes_no_xml_raises(self):
        with pytest.raises(ValueError, match="No XML content"):
            SurgePatch.from_state_bytes(b"\x00" * 50)

    def test_to_fxp_bytes_custom_name(self, init_patch: SurgePatch):
        fxp = init_patch.to_fxp_bytes(preset_name="CustomName")
        assert b"CustomName" in fxp
