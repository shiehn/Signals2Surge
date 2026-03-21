"""Load .fxp preset files into a live PluginHost.

Pedalboard's set_state() does not reliably transfer state for Surge XT.
Instead, we use a parameter mapping approach: parse the FXP's XML, map
each XML parameter name to its pedalboard counterpart, convert values
to raw [0,1] range, and apply via set_raw_values().

The mapping is auto-discovered by probing each pedalboard parameter and
observing which XML parameter changes. This is cached per session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from synth2surge.audio.engine import PluginHost
from synth2surge.surge.patch import SurgePatch, extract_xml_from_bytes

logger = logging.getLogger(__name__)

# Module-level cache for the parameter mapping and calibration
_cached_mapping: dict[str, str] | None = None  # pb_name -> xml_name
# xml_name -> (val_at_0, val_at_1)
_cached_calibration: dict[str, tuple[float, float]] | None = None


class LoadStrategy(Enum):
    """Strategy for loading preset data into a plugin host."""

    PARAMETER_MAP = auto()


@dataclass
class LoadResult:
    """Result of attempting to load a preset into a host."""

    success: bool
    strategy: LoadStrategy | None
    matched_params: int = 0
    total_params: int = 0
    error: str = ""


def detect_state_format(host: PluginHost) -> str:
    """Detect the format of a host's get_state() output.

    Returns:
        "vst3" if output starts with VST3 magic,
        "fxp" if output starts with CcnK magic,
        "xml" if output starts with or contains XML,
        "unknown" otherwise.
    """
    state = host.get_state()

    if state[:4] == b"VST3":
        return "vst3"
    if state[:4] == b"CcnK":
        return "fxp"
    if state[:5] == b"<?xml" or state[:6] == b"<patch":
        return "xml"
    if b"<?xml" in state or b"<patch" in state:
        return "xml"
    return "unknown"


def load_fxp_into_host(
    fxp_path: Path,
    host: PluginHost,
) -> LoadResult:
    """Load an .fxp file into a live PluginHost via parameter mapping.

    Parses the FXP's XML parameters, maps them to pedalboard parameter
    names using an auto-discovered mapping, converts values to raw [0,1]
    range, and applies via set_raw_values().

    The mapping and calibration are auto-built on first call and cached.

    Args:
        fxp_path: Path to the .fxp file.
        host: PluginHost to load the preset into.

    Returns:
        LoadResult with match statistics.
    """
    fxp_path = Path(fxp_path)

    # Parse the FXP file
    fxp_patch = SurgePatch.from_file(fxp_path)
    fxp_params = fxp_patch.get_all_parameters()

    # Ensure mapping and calibration are available
    mapping, calibration = _ensure_mapping(host)

    # Reverse mapping: xml_name -> pb_name
    xml_to_pb = {v: k for k, v in mapping.items()}

    # Convert FXP XML values to raw_values
    target_raw_values: dict[str, float] = {}
    matched = 0

    for xml_name, target_val in fxp_params.items():
        if xml_name not in xml_to_pb:
            continue
        if xml_name not in calibration:
            continue

        pb_name = xml_to_pb[xml_name]
        val_at_0, val_at_1 = calibration[xml_name]

        range_size = val_at_1 - val_at_0
        if abs(range_size) < 1e-10:
            target_raw_values[pb_name] = 0.5
        else:
            raw = (target_val - val_at_0) / range_size
            raw = max(0.0, min(1.0, raw))
            target_raw_values[pb_name] = raw
        matched += 1

    if not target_raw_values:
        return LoadResult(
            success=False,
            strategy=LoadStrategy.PARAMETER_MAP,
            error=f"No mappable parameters found in {fxp_path.name}",
        )

    # Apply raw values
    host.set_raw_values(target_raw_values)

    logger.info(
        f"Loaded {fxp_path.name} via PARAMETER_MAP: "
        f"{matched}/{len(fxp_params)} params mapped"
    )

    return LoadResult(
        success=True,
        strategy=LoadStrategy.PARAMETER_MAP,
        matched_params=matched,
        total_params=len(fxp_params),
    )


def _ensure_mapping(
    host: PluginHost,
) -> tuple[dict[str, str], dict[str, tuple[float, float]]]:
    """Ensure the parameter mapping and calibration are available.

    Auto-builds them on first call and caches for the session.
    """
    global _cached_mapping, _cached_calibration

    if _cached_mapping is not None and _cached_calibration is not None:
        return _cached_mapping, _cached_calibration

    logger.info("Building parameter mapping (one-time, ~6s)...")
    _cached_mapping = _build_parameter_mapping(host)
    _cached_calibration = _calibrate_ranges(host)
    logger.info(
        f"Mapping complete: {len(_cached_mapping)} params mapped, "
        f"{len(_cached_calibration)} calibrated"
    )

    return _cached_mapping, _cached_calibration


def _get_xml_params(host: PluginHost) -> dict[str, float | int]:
    """Extract XML parameters from the host's current state."""
    state = host.get_state()
    try:
        xml_bytes = extract_xml_from_bytes(state)
    except ValueError:
        return {}
    patch = SurgePatch.from_state_bytes(xml_bytes)
    return patch.get_all_parameters()


def _build_parameter_mapping(host: PluginHost) -> dict[str, str]:
    """Auto-discover the mapping from pedalboard names to XML names.

    For each pedalboard parameter, changes its raw_value and observes
    which XML parameter changes in response.

    Returns:
        Dict mapping pedalboard_name -> xml_name.
    """
    pb_names = host.parameter_names()
    default_raws = host.get_raw_values()
    baseline_xml = _get_xml_params(host)

    mapping: dict[str, str] = {}

    for pb_name in pb_names:
        default_val = default_raws.get(pb_name, 0.5)
        test_val = 0.95 if default_val < 0.5 else 0.05

        host.set_raw_values({pb_name: test_val})
        new_xml = _get_xml_params(host)

        # Find which XML param changed
        changed = []
        for xml_name in baseline_xml:
            if xml_name in new_xml:
                if abs(baseline_xml[xml_name] - new_xml[xml_name]) > 0.001:
                    changed.append(
                        (xml_name, abs(baseline_xml[xml_name] - new_xml[xml_name]))
                    )

        # Restore default
        host.set_raw_values({pb_name: default_val})

        if len(changed) == 1:
            mapping[pb_name] = changed[0][0]
        elif len(changed) > 1:
            # Take the one with the largest change
            best = max(changed, key=lambda x: x[1])
            mapping[pb_name] = best[0]

    # Safety: bulk restore all params to defaults in case of accumulated drift
    host.set_raw_values(default_raws)

    return mapping


def _calibrate_ranges(host: PluginHost) -> dict[str, tuple[float, float]]:
    """Calibrate the value range for each XML parameter.

    Sets all pedalboard params to 0 and records XML values,
    then sets all to 1 and records again. This gives the
    (min, max) range for linear interpolation.

    Saves and restores the original raw values so the host state
    is not destroyed by calibration.

    Returns:
        Dict mapping xml_name -> (value_at_raw_0, value_at_raw_1).
    """
    pb_names = host.parameter_names()
    saved_raws = host.get_raw_values()

    # Set all to 0
    host.set_raw_values({name: 0.0 for name in pb_names})
    xml_at_0 = _get_xml_params(host)

    # Set all to 1
    host.set_raw_values({name: 1.0 for name in pb_names})
    xml_at_1 = _get_xml_params(host)

    # Restore original state
    host.set_raw_values(saved_raws)

    calibration: dict[str, tuple[float, float]] = {}
    for xml_name in xml_at_0:
        if xml_name in xml_at_1:
            calibration[xml_name] = (xml_at_0[xml_name], xml_at_1[xml_name])

    return calibration


def reset_mapping_cache() -> None:
    """Clear the cached parameter mapping and calibration.

    Useful for testing or when switching plugins.
    """
    global _cached_mapping, _cached_calibration
    _cached_mapping = None
    _cached_calibration = None
