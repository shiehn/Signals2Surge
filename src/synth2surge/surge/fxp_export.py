"""FXP header construction and state-to-fxp conversion.

The FXP format for Surge XT:
- 60-byte binary header (CcnK/FPCh chunk with cjs3 plugin ID)
- Followed by raw XML patch data

Header layout:
  Offset  Size  Field
  0       4     Magic "CcnK"
  4       4     Chunk size (big-endian, total_size - 8)
  8       4     Chunk type "FPCh" (opaque chunk preset)
  12      4     Format version (1)
  16      4     Plugin ID "cjs3"
  20      4     Plugin version (1)
  24      4     Num programs (1)
  28      28    Preset name (null-padded)
  56      4     Chunk data size (big-endian, XML length)
  60      N     Chunk data (XML bytes)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from synth2surge.surge.patch import extract_xml_from_bytes

if TYPE_CHECKING:
    from synth2surge.surge.patch import SurgePatch

# FXP constants
FXP_MAGIC = b"CcnK"
FXP_CHUNK_TYPE = b"FPCh"  # opaque chunk preset
SURGE_PLUGIN_ID = b"cjs3"
SURGE_SUB_ID = b"sub3"
FXP_FORMAT_VERSION = 1
FXP_PLUGIN_VERSION = 1
FXP_NUM_PROGRAMS = 1
FXP_NAME_SIZE = 28


def build_fxp_header(preset_name: str, xml_size: int) -> bytes:
    """Construct the FXP header bytes for a Surge XT preset.

    Args:
        preset_name: Name to embed in the header (truncated to 28 bytes).
        xml_size: Size of the XML chunk data that follows the header.

    Returns:
        Complete FXP header bytes (60 bytes).
    """
    # Encode and truncate name to 28 bytes, null-padded
    name_bytes = preset_name.encode("ascii", errors="replace")[:FXP_NAME_SIZE]
    name_bytes = name_bytes.ljust(FXP_NAME_SIZE, b"\x00")

    # Total chunk size = everything after the first 8 bytes (magic + size)
    # Header fields after size: chunk_type(4) + version(4) + plugin_id(4) +
    #   plugin_version(4) + num_programs(4) + name(28) + chunk_size(4) = 52
    # Plus the XML data
    chunk_size = 52 + xml_size

    header = struct.pack(
        ">4sI4sI4sII",
        FXP_MAGIC,           # 4: magic
        chunk_size,          # 4: chunk size (big-endian)
        FXP_CHUNK_TYPE,      # 4: chunk type
        FXP_FORMAT_VERSION,  # 4: format version
        SURGE_PLUGIN_ID,     # 4: plugin ID
        FXP_PLUGIN_VERSION,  # 4: plugin version
        FXP_NUM_PROGRAMS,    # 4: num programs
    )
    # Append name (28 bytes) and chunk data size
    header += name_bytes
    header += struct.pack(">I", xml_size)

    return header


def state_to_fxp(
    state_bytes: bytes,
    output_path: str | Path,
    preset_name: str = "Synth2Surge",
) -> Path:
    """Convert plugin state bytes (from get_state()) to an FXP file.

    Auto-detects the format of state_bytes:
    - If starts with CcnK: already FXP, write directly (optionally update name)
    - If starts with <?xml or <patch: raw XML, prepend header
    - Otherwise: search for XML within bytes, prepend header

    Args:
        state_bytes: Raw bytes from PluginHost.get_state().
        output_path: Path to write the .fxp file.
        preset_name: Name to embed in the FXP header.

    Returns:
        Path to the written .fxp file.

    Raises:
        ValueError: If no XML content can be found in state_bytes.
    """
    output_path = Path(output_path)

    if state_bytes[:4] == FXP_MAGIC:
        # Already FXP format — write as-is
        output_path.write_bytes(state_bytes)
        return output_path

    # Try to find XML content
    xml_bytes = extract_xml_from_bytes(state_bytes)

    # Build header and write
    header = build_fxp_header(preset_name, len(xml_bytes))
    output_path.write_bytes(header + xml_bytes)
    return output_path


def patch_to_fxp(
    patch: SurgePatch,
    output_path: str | Path,
    preset_name: str | None = None,
) -> Path:
    """Write a SurgePatch as an FXP file.

    Args:
        patch: The SurgePatch to export.
        output_path: Path to write the .fxp file.
        preset_name: Name for the header. Defaults to patch metadata name.

    Returns:
        Path to the written .fxp file.
    """
    output_path = Path(output_path)

    if preset_name is None:
        preset_name = patch.metadata.name or "Synth2Surge"

    xml_bytes = patch.to_xml_bytes()
    header = build_fxp_header(preset_name, len(xml_bytes))
    output_path.write_bytes(header + xml_bytes)
    return output_path


