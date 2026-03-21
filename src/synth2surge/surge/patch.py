"""Surge XT XML patch parser, writer, and mutator.

Handles both raw XML and .fxp files (which have a binary header before XML).
The FXP format: 60 bytes binary header (CcnK/FPCh chunk) followed by XML.
"""

from __future__ import annotations

import struct
from pathlib import Path

from lxml import etree

from synth2surge.types import PatchMetadata

# FXP binary header: magic "CcnK" at offset 0
FXP_MAGIC = b"CcnK"

# Surge parameter type codes
PARAM_TYPE_INT = "0"
PARAM_TYPE_BOOL = "1"
PARAM_TYPE_FLOAT = "2"


class SurgePatch:
    """Parse, mutate, and serialize Surge XT XML patches."""

    def __init__(self, tree: etree._ElementTree) -> None:
        self._tree = tree
        self._root = tree.getroot()

    @classmethod
    def from_file(cls, path: str | Path) -> SurgePatch:
        """Load a Surge XT patch from .fxp or raw XML file."""
        path = Path(path)
        data = path.read_bytes()
        xml_bytes = extract_xml_from_bytes(data)
        return cls._from_xml_bytes(xml_bytes)

    @classmethod
    def from_xml_string(cls, xml: str) -> SurgePatch:
        """Parse a Surge XT patch from an XML string."""
        return cls._from_xml_bytes(xml.encode("utf-8"))

    @classmethod
    def _from_xml_bytes(cls, xml_bytes: bytes) -> SurgePatch:
        parser = etree.XMLParser(recover=True)
        tree = etree.ElementTree(etree.fromstring(xml_bytes, parser=parser))
        return cls(tree)

    @property
    def metadata(self) -> PatchMetadata:
        """Extract patch metadata."""
        meta = self._root.find(".//meta")
        if meta is None:
            return PatchMetadata()
        return PatchMetadata(
            name=meta.get("name", ""),
            category=meta.get("category", ""),
            author=meta.get("author", ""),
            comment=meta.get("comment", ""),
        )

    @property
    def revision(self) -> int:
        """Patch format revision number."""
        return int(self._root.get("revision", "0"))

    def get_parameter(self, name: str) -> float | int | None:
        """Get a single parameter value by element name."""
        elem = self._root.find(f".//parameters/{name}")
        if elem is None:
            return None
        return _parse_param_value(elem)

    def set_parameter(self, name: str, value: float | int) -> None:
        """Set a single parameter value by element name."""
        elem = self._root.find(f".//parameters/{name}")
        if elem is None:
            raise KeyError(f"Parameter not found: {name}")
        elem.set("value", _format_value(value, elem.get("type", PARAM_TYPE_FLOAT)))

    def get_all_parameters(self) -> dict[str, float | int]:
        """Extract all typed parameters as a flat dictionary."""
        params = {}
        parameters_elem = self._root.find(".//parameters")
        if parameters_elem is None:
            return params
        for elem in parameters_elem:
            if elem.get("type") is not None and elem.get("value") is not None:
                params[elem.tag] = _parse_param_value(elem)
        return params

    def set_all_parameters(self, params: dict[str, float | int]) -> None:
        """Set multiple parameter values at once."""
        for name, value in params.items():
            self.set_parameter(name, value)

    def get_parameter_types(self) -> dict[str, str]:
        """Get the type code for each parameter."""
        types = {}
        parameters_elem = self._root.find(".//parameters")
        if parameters_elem is None:
            return types
        for elem in parameters_elem:
            ptype = elem.get("type")
            if ptype is not None:
                types[elem.tag] = ptype
        return types

    def to_xml_string(self) -> str:
        """Serialize the patch back to XML."""
        return etree.tostring(
            self._root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        ).decode("utf-8")

    def to_xml_bytes(self) -> bytes:
        """Serialize the patch to XML bytes."""
        return etree.tostring(
            self._root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )

    def to_file(self, path: str | Path) -> None:
        """Write the patch as raw XML to a file."""
        path = Path(path)
        path.write_bytes(self.to_xml_bytes())

    def to_fxp_bytes(self, preset_name: str | None = None) -> bytes:
        """Serialize as complete FXP bytes (header + XML).

        Args:
            preset_name: Name for the FXP header. Defaults to patch metadata name.

        Returns:
            Complete FXP file contents as bytes.
        """
        from synth2surge.surge.fxp_export import build_fxp_header

        if preset_name is None:
            preset_name = self.metadata.name or "Synth2Surge"

        xml_bytes = self.to_xml_bytes()
        header = build_fxp_header(preset_name, len(xml_bytes))
        return header + xml_bytes

    def to_fxp_file(self, path: str | Path, preset_name: str | None = None) -> None:
        """Write this patch as an FXP file.

        Args:
            path: Output file path.
            preset_name: Name for the FXP header. Defaults to patch metadata name.
        """
        path = Path(path)
        path.write_bytes(self.to_fxp_bytes(preset_name))

    @classmethod
    def from_state_bytes(cls, state: bytes) -> SurgePatch:
        """Parse a SurgePatch from PluginHost.get_state() output.

        Handles FXP format (strips header), VST3 envelope, raw XML, or
        binary with embedded XML.

        Args:
            state: Raw bytes from PluginHost.get_state().

        Returns:
            Parsed SurgePatch.

        Raises:
            ValueError: If no XML content can be found.
        """
        xml_bytes = extract_xml_from_bytes(state)
        return cls._from_xml_bytes(xml_bytes)

    def clone(self) -> SurgePatch:
        """Create a deep copy of this patch."""
        import copy

        new_tree = copy.deepcopy(self._tree)
        return SurgePatch(new_tree)

    def parameter_names(self) -> list[str]:
        """List all parameter element names."""
        parameters_elem = self._root.find(".//parameters")
        if parameters_elem is None:
            return []
        return [
            elem.tag
            for elem in parameters_elem
            if elem.get("type") is not None
        ]

    def float_parameter_names(self) -> list[str]:
        """List names of all float (type=2) parameters."""
        parameters_elem = self._root.find(".//parameters")
        if parameters_elem is None:
            return []
        return [
            elem.tag
            for elem in parameters_elem
            if elem.get("type") == PARAM_TYPE_FLOAT
        ]

    def int_parameter_names(self) -> list[str]:
        """List names of all int (type=0) parameters."""
        parameters_elem = self._root.find(".//parameters")
        if parameters_elem is None:
            return []
        return [
            elem.tag
            for elem in parameters_elem
            if elem.get("type") == PARAM_TYPE_INT
        ]


def extract_xml_from_bytes(data: bytes) -> bytes:
    """Extract XML content from arbitrary state bytes.

    Handles three formats:
    - FXP: CcnK magic header, scan for XML after header
    - VST3: VST3 envelope with chunk list offset at byte 40, XML in component chunk
    - Raw XML: starts with <?xml or <patch

    Args:
        data: Raw bytes (FXP, VST3 envelope, or raw XML).

    Returns:
        The XML portion as bytes.

    Raises:
        ValueError: If no XML content can be found.
    """
    if data[:4] == FXP_MAGIC:
        xml_start = data.find(b"<?xml")
        if xml_start < 0:
            xml_start = data.find(b"<patch")
        if xml_start < 0:
            raise ValueError("No XML content found in FXP data")
        return data[xml_start:]

    if data[:4] == b"VST3" and len(data) >= 48:
        chunk_list_offset = struct.unpack_from("<Q", data, 40)[0]
        if chunk_list_offset > len(data):
            chunk_list_offset = len(data)
        comp = data[48:chunk_list_offset]
        xml_start = comp.find(b"<?xml")
        if xml_start < 0:
            xml_start = comp.find(b"<patch")
        if xml_start >= 0:
            xml_end = comp.find(b"</patch>", xml_start)
            if xml_end >= 0:
                xml_end += len(b"</patch>")
                return comp[xml_start:xml_end]

    # Raw XML or XML embedded in unknown binary
    xml_start = data.find(b"<?xml")
    if xml_start >= 0:
        return data[xml_start:]
    xml_start = data.find(b"<patch")
    if xml_start >= 0:
        return data[xml_start:]

    raise ValueError("No XML content found in state bytes")


def _parse_param_value(elem: etree._Element) -> float | int:
    """Parse a parameter element's value based on its type."""
    ptype = elem.get("type", PARAM_TYPE_FLOAT)
    raw = elem.get("value", "0")
    if ptype == PARAM_TYPE_FLOAT:
        return float(raw)
    else:
        return int(float(raw))  # int() via float() handles "1.00000" strings


def _format_value(value: float | int, ptype: str) -> str:
    """Format a value for writing back to XML."""
    if ptype == PARAM_TYPE_FLOAT:
        return f"{float(value):.14f}"
    else:
        return str(int(value))
