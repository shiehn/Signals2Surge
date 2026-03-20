"""Pedalboard-based plugin hosting engine.

Wraps Spotify's pedalboard library to provide a clean interface for
loading VST3/AU instrument plugins, rendering MIDI, and managing state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synth2surge.audio.midi import MultiProbeResult, create_probe, probe_duration
from synth2surge.config import AudioConfig, MidiProbeConfig


class PluginHost:
    """Manages a pedalboard instrument plugin instance."""

    def __init__(
        self,
        plugin_path: str | Path,
        sample_rate: int | None = None,
    ) -> None:
        from pedalboard import load_plugin

        self._sample_rate = sample_rate or AudioConfig().sample_rate
        self._plugin_path = str(plugin_path)
        self._plugin = load_plugin(self._plugin_path, parameter_values={})

        if not getattr(self._plugin, "is_instrument", False):
            raise ValueError(f"Plugin at {plugin_path} is not an instrument")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def plugin_name(self) -> str:
        return getattr(self._plugin, "name", "Unknown")

    def render_midi(
        self,
        midi_messages: list[tuple[float, int, int, int]] | None = None,
        duration: float | None = None,
        midi_config: MidiProbeConfig | None = None,
    ) -> np.ndarray:
        """Render MIDI through the plugin, returning audio as a numpy array.

        Returns:
            Stereo float32 numpy array of shape (2, n_samples).
        """
        if midi_messages is None:
            midi_messages = create_probe(midi_config)
        if duration is None:
            duration = probe_duration(midi_config)

        audio = self._plugin(
            midi_messages,
            duration=duration,
            sample_rate=float(self._sample_rate),
        )

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        return audio.astype(np.float32)

    def render_midi_mono(
        self,
        midi_messages: list[tuple[float, int, int, int]] | None = None,
        duration: float | None = None,
        midi_config: MidiProbeConfig | None = None,
    ) -> np.ndarray:
        """Render MIDI and return mono audio (mean of channels)."""
        stereo = self.render_midi(midi_messages, duration, midi_config)
        return np.mean(stereo, axis=0).astype(np.float32)

    def get_state(self) -> bytes:
        """Get the plugin's current state as binary data."""
        data = self._plugin.preset_data
        if data is None:
            raise RuntimeError("Plugin does not support preset_data")
        return bytes(data)

    def set_state(self, state: bytes) -> None:
        """Restore the plugin to a previously saved state."""
        self._plugin.preset_data = state

    def get_parameters(self) -> dict[str, float | str]:
        """Read all exposed plugin parameters (display values)."""
        params: dict[str, float | str] = {}
        for name in self._plugin.parameters.keys():
            try:
                val = getattr(self._plugin, name)
                if isinstance(val, (int, float)):
                    params[name] = float(val)
                elif isinstance(val, str):
                    params[name] = val
                else:
                    params[name] = float(val)
            except (AttributeError, TypeError, ValueError):
                pass
        return params

    def get_raw_values(self) -> dict[str, float]:
        """Read all parameters as normalized [0, 1] raw values."""
        raw = {}
        for name, param in self._plugin.parameters.items():
            try:
                raw[name] = float(param.raw_value)
            except (AttributeError, TypeError, ValueError):
                pass
        return raw

    def set_raw_values(self, values: dict[str, float]) -> None:
        """Set parameters via normalized [0, 1] raw values."""
        for name, val in values.items():
            param = self._plugin.parameters.get(name)
            if param is not None:
                try:
                    param.raw_value = float(val)
                except (AttributeError, TypeError, ValueError):
                    pass

    def set_parameters(self, params: dict[str, float]) -> None:
        """Set multiple plugin parameters (display values)."""
        for name, value in params.items():
            try:
                setattr(self._plugin, name, value)
            except (AttributeError, TypeError) as e:
                raise ValueError(f"Cannot set parameter '{name}': {e}") from e

    def parameter_names(self) -> list[str]:
        """List all exposed parameter names."""
        return list(self._plugin.parameters.keys())

    def get_parameter_info(self) -> list[dict]:
        """Get detailed info for each parameter (name, range, raw_value)."""
        info = []
        for name, param in self._plugin.parameters.items():
            try:
                info.append({
                    "name": name,
                    "raw_value": float(param.raw_value),
                    "range": param.range,
                    "label": getattr(param, "label", ""),
                })
            except (AttributeError, TypeError, ValueError):
                pass
        return info

    def render_multi_probe(
        self,
        multi_probe: MultiProbeResult,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Render a multi-probe MIDI sequence and slice into segments.

        Args:
            multi_probe: Composed multi-probe result with MIDI messages and segment info.

        Returns:
            Tuple of (full_mono_audio, [segment1_audio, segment2_audio, ...]).
        """
        full_audio = self.render_midi_mono(
            midi_messages=multi_probe.midi_messages,
            duration=multi_probe.total_duration,
        )

        segments = []
        for seg in multi_probe.segments:
            start = seg.start_sample
            end = min(seg.end_sample, len(full_audio))
            segments.append(full_audio[start:end])

        return full_audio, segments

    def reset(self) -> None:
        """Reset plugin internal state (buffers, LFOs, etc.)."""
        self._plugin.reset()
