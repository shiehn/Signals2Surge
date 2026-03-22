"""End-to-end tests for the batch preset porting workflow.

Uses synthetic audio (sine waves) for most tests. Mocks PluginHost where
needed to avoid requiring real VST plugins.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from synth2surge.batch.manifest import (
    QueueItem,
    QueueManifest,
    add_item,
    build_manifest_from_wav_folder,
    compute_state_hash,
    load_manifest,
    save_manifest,
)


def _make_sine_wav(path: Path, freq: float = 440.0, duration: float = 0.5) -> None:
    """Create a short sine wave WAV file."""
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def _make_sine_audio(freq: float = 440.0, duration: float = 0.5) -> np.ndarray:
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Manifest + WAV Folder Integration
# ---------------------------------------------------------------------------


class TestManifestWavFolderIntegration:
    """Test building manifests from WAV folder structures."""

    def test_build_manifest_from_nested_wav_folder(self, tmp_path):
        """Multi-category folder with WAVs produces correct manifest."""
        _make_sine_wav(tmp_path / "bass" / "deep_sub.wav", freq=60)
        _make_sine_wav(tmp_path / "bass" / "acid.wav", freq=100)
        _make_sine_wav(tmp_path / "lead" / "screamer.wav", freq=2000)
        _make_sine_wav(tmp_path / "pad" / "evolving.wav", freq=300)

        manifest = build_manifest_from_wav_folder(tmp_path)

        assert len(manifest.items) == 4
        cats = {item.category for item in manifest.items}
        assert cats == {"Bass", "Lead", "Pad"}

        names = {item.preset_name for item in manifest.items}
        assert names == {"deep_sub", "acid", "screamer", "evolving"}

        for item in manifest.items:
            assert item.status == "pending"
            assert item.audio_hash is not None
            assert item.audio_path.endswith(".wav")

    def test_build_manifest_skips_non_wav_files(self, tmp_path):
        """Only .wav files are included in the manifest."""
        _make_sine_wav(tmp_path / "bass" / "good.wav", freq=80)
        (tmp_path / "bass").mkdir(parents=True, exist_ok=True)
        (tmp_path / "bass" / "readme.txt").write_text("notes")
        (tmp_path / "bass" / "preset.fxp").write_bytes(b"\x00" * 100)

        manifest = build_manifest_from_wav_folder(tmp_path)
        assert len(manifest.items) == 1
        assert manifest.items[0].preset_name == "good"

    def test_build_manifest_dedup_identical_wavs(self, tmp_path):
        """Identical audio in different categories is detected as duplicate."""
        _make_sine_wav(tmp_path / "bass" / "sub_v1.wav", freq=80)
        _make_sine_wav(tmp_path / "lead" / "sub_v2.wav", freq=80)

        manifest = build_manifest_from_wav_folder(tmp_path)
        assert len(manifest.items) == 1


# ---------------------------------------------------------------------------
# Queue Workflow (mocked plugin)
# ---------------------------------------------------------------------------


class TestQueueWorkflow:
    """Test the queue capture-and-save workflow with mocked plugins."""

    def test_queue_captures_and_saves_to_manifest(self, tmp_path):
        """Simulate adding 3 presets to queue, verify manifest and files."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        manifest_path = queue_dir / "manifest.json"
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")

        presets = [
            ("Deep Sub", "Bass", b"state_bytes_1", 80.0),
            ("Acid Lead", "Lead", b"state_bytes_2", 1000.0),
            ("Warm Pad", "Pad", b"state_bytes_3", 300.0),
        ]

        for name, category, state_bytes, freq in presets:
            from synth2surge.batch.manifest import new_item_id

            item_id = new_item_id()
            item_dir = queue_dir / item_id
            item_dir.mkdir()

            # Simulate capture: save audio and state
            audio = _make_sine_audio(freq=freq)
            sf.write(str(item_dir / "target_audio.wav"), audio, 44100)
            (item_dir / "target_state.bin").write_bytes(state_bytes)

            state_hash = compute_state_hash(state_bytes)
            item = QueueItem(
                id=item_id,
                preset_name=name,
                category=category,
                audio_path=f"{item_id}/target_audio.wav",
                state_path=f"{item_id}/target_state.bin",
                state_hash=state_hash,
            )
            result = add_item(manifest, item)
            assert result is True

        save_manifest(manifest, manifest_path)

        # Verify manifest
        loaded = load_manifest(manifest_path)
        assert len(loaded.items) == 3
        assert {item.preset_name for item in loaded.items} == {"Deep Sub", "Acid Lead", "Warm Pad"}

        # Verify files exist on disk
        for item in loaded.items:
            audio_file = queue_dir / item.audio_path
            state_file = queue_dir / item.state_path
            assert audio_file.exists()
            assert state_file.exists()

    def test_queue_dedup_rejects_duplicate_state(self, tmp_path):
        """Adding the same state bytes twice should be rejected."""
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")
        state_bytes = b"identical_plugin_state"
        state_hash = compute_state_hash(state_bytes)

        item1 = QueueItem(
            id="id1", preset_name="First", category="Bass",
            audio_path="id1/audio.wav", state_hash=state_hash,
        )
        item2 = QueueItem(
            id="id2", preset_name="Second", category="Lead",
            audio_path="id2/audio.wav", state_hash=state_hash,
        )

        assert add_item(manifest, item1) is True
        assert add_item(manifest, item2) is False
        assert len(manifest.items) == 1

    def test_queue_preserves_existing_items(self, tmp_path):
        """Adding items to an existing manifest preserves old items."""
        manifest_path = tmp_path / "manifest.json"

        # First session: add 2 items
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")
        add_item(manifest, QueueItem(
            id="a1", preset_name="First", category="Bass",
            audio_path="a1/audio.wav", state_hash="hash1",
        ))
        add_item(manifest, QueueItem(
            id="a2", preset_name="Second", category="Lead",
            audio_path="a2/audio.wav", state_hash="hash2",
        ))
        save_manifest(manifest, manifest_path)

        # Second session: reload and add 1 more
        manifest = load_manifest(manifest_path)
        assert len(manifest.items) == 2

        add_item(manifest, QueueItem(
            id="a3", preset_name="Third", category="Pad",
            audio_path="a3/audio.wav", state_hash="hash3",
        ))
        save_manifest(manifest, manifest_path)

        # Verify all 3 present
        final = load_manifest(manifest_path)
        assert len(final.items) == 3
        assert {item.preset_name for item in final.items} == {"First", "Second", "Third"}


# ---------------------------------------------------------------------------
# Batch Optimize Workflow (mocked optimizer)
# ---------------------------------------------------------------------------


class TestBatchOptimizeWorkflow:
    """Test batch optimization with mocked optimizer to avoid needing Surge XT."""

    def _create_manifest_with_wavs(self, queue_dir: Path, items_data: list) -> Path:
        """Helper to create a manifest with real WAV files."""
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")

        for item_id, name, category, freq in items_data:
            item_dir = queue_dir / item_id
            item_dir.mkdir(parents=True)
            _make_sine_wav(item_dir / "target_audio.wav", freq=freq)
            (item_dir / "target_state.bin").write_bytes(f"state_{item_id}".encode())

            item = QueueItem(
                id=item_id,
                preset_name=name,
                category=category,
                audio_path=f"{item_id}/target_audio.wav",
                state_path=f"{item_id}/target_state.bin",
                state_hash=compute_state_hash(f"state_{item_id}".encode()),
            )
            manifest.items.append(item)

        manifest_path = queue_dir / "manifest.json"
        save_manifest(manifest, manifest_path)
        return manifest_path

    def _make_mock_optimize_result(self, output_dir: Path, preset_name: str):
        """Create mock optimization output files."""
        from synth2surge.types import OptimizationResult

        fxp_path = output_dir / "best_patch.fxp"
        audio_path = output_dir / "best_audio.wav"
        patch_path = output_dir / "best_patch.bin"

        fxp_path.write_bytes(b"fake_fxp")
        _make_sine_wav(audio_path, freq=440)
        patch_path.write_bytes(b"fake_state")

        return OptimizationResult(
            best_patch_path=patch_path,
            best_loss=0.05,
            best_audio_path=audio_path,
            total_trials=50,
            stages_completed=1,
            fxp_path=fxp_path,
        )

    def test_batch_optimize_from_manifest_produces_library(self, tmp_path):
        """Process 3 queued items, verify output folder structure."""
        queue_dir = tmp_path / "queue"
        output_dir = tmp_path / "library"
        queue_dir.mkdir()

        manifest_path = self._create_manifest_with_wavs(queue_dir, [
            ("id1", "Deep Sub", "Bass", 80),
            ("id2", "Screamer", "Lead", 2000),
            ("id3", "Warm Pad", "Pad", 300),
        ])

        # Mock the optimize function
        call_count = 0

        def mock_optimize(target_audio, surge_host, **kwargs):
            nonlocal call_count
            call_count += 1
            output = kwargs.get("output_dir", tmp_path / "work")
            output.mkdir(parents=True, exist_ok=True)
            return self._make_mock_optimize_result(output, kwargs.get("preset_name", "test"))

        with patch("synth2surge.optimizer.loop.optimize", side_effect=mock_optimize), \
             patch("synth2surge.audio.engine.PluginHost"):
            from typer.testing import CliRunner

            from synth2surge.cli.main import app

            runner = CliRunner()
            runner.invoke(app, [
                "batch-optimize",
                "--queue-dir", str(queue_dir),
                "--output-dir", str(output_dir),
                "--stages", "1",
                "--trials-t1", "10",
            ])

        assert call_count == 3

        # Verify manifest updated
        manifest = load_manifest(manifest_path)
        completed = [i for i in manifest.items if i.status == "completed"]
        assert len(completed) == 3

    def test_batch_optimize_resume_skips_completed(self, tmp_path):
        """Already-completed items should not be re-optimized."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        manifest = QueueManifest(created_at="2026-01-01T00:00:00")

        # 2 completed, 1 pending
        items_data = [
            ("id1", "Done1", "completed"),
            ("id2", "Done2", "completed"),
            ("id3", "Pending", "pending"),
        ]
        for item_id, name, status in items_data:
            item_dir = queue_dir / item_id
            item_dir.mkdir(parents=True)
            _make_sine_wav(item_dir / "target_audio.wav", freq=440)

            item = QueueItem(
                id=item_id, preset_name=name, category="Bass",
                audio_path=f"{item_id}/target_audio.wav",
                status=status,
            )
            manifest.items.append(item)

        manifest_path = queue_dir / "manifest.json"
        save_manifest(manifest, manifest_path)

        call_count = 0

        def mock_optimize(target_audio, surge_host, **kwargs):
            nonlocal call_count
            call_count += 1
            output = kwargs.get("output_dir", tmp_path / "work")
            output.mkdir(parents=True, exist_ok=True)
            return self._make_mock_optimize_result(output, kwargs.get("preset_name", "test"))

        with patch("synth2surge.optimizer.loop.optimize", side_effect=mock_optimize), \
             patch("synth2surge.audio.engine.PluginHost"):
            from typer.testing import CliRunner

            from synth2surge.cli.main import app

            runner = CliRunner()
            runner.invoke(app, [
                "batch-optimize",
                "--queue-dir", str(queue_dir),
                "--output-dir", str(tmp_path / "library"),
                "--stages", "1",
            ])

        # Only 1 call (the pending item)
        assert call_count == 1

    def test_batch_optimize_marks_failed_and_continues(self, tmp_path):
        """If optimization fails for one item, others still complete."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        self._create_manifest_with_wavs(queue_dir, [
            ("id1", "Good1", "Bass", 80),
            ("id2", "Bad", "Lead", 2000),
            ("id3", "Good2", "Pad", 300),
        ])

        call_count = 0

        def mock_optimize(target_audio, surge_host, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated optimization failure")
            output = kwargs.get("output_dir", tmp_path / "work")
            output.mkdir(parents=True, exist_ok=True)
            return self._make_mock_optimize_result(output, kwargs.get("preset_name", "test"))

        with patch("synth2surge.optimizer.loop.optimize", side_effect=mock_optimize), \
             patch("synth2surge.audio.engine.PluginHost"):
            from typer.testing import CliRunner

            from synth2surge.cli.main import app

            runner = CliRunner()
            runner.invoke(app, [
                "batch-optimize",
                "--queue-dir", str(queue_dir),
                "--output-dir", str(tmp_path / "library"),
                "--stages", "1",
            ])

        assert call_count == 3

        manifest = load_manifest(queue_dir / "manifest.json")
        statuses = {item.id: item.status for item in manifest.items}
        assert statuses["id1"] == "completed"
        assert statuses["id2"] == "failed"
        assert statuses["id3"] == "completed"

        # Failed item should have error message
        failed_item = next(i for i in manifest.items if i.id == "id2")
        assert "Simulated optimization failure" in failed_item.error

    def test_batch_optimize_preset_name_flows_to_optimize(self, tmp_path):
        """Preset name should be passed through to optimize()."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        self._create_manifest_with_wavs(queue_dir, [
            ("id1", "My Cool Preset", "Bass", 80),
        ])

        captured_preset_name = None

        def mock_optimize(target_audio, surge_host, **kwargs):
            nonlocal captured_preset_name
            captured_preset_name = kwargs.get("preset_name")
            output = kwargs.get("output_dir", tmp_path / "work")
            output.mkdir(parents=True, exist_ok=True)
            return self._make_mock_optimize_result(output, captured_preset_name or "test")

        with patch("synth2surge.optimizer.loop.optimize", side_effect=mock_optimize), \
             patch("synth2surge.audio.engine.PluginHost"):
            from typer.testing import CliRunner

            from synth2surge.cli.main import app

            runner = CliRunner()
            runner.invoke(app, [
                "batch-optimize",
                "--queue-dir", str(queue_dir),
                "--output-dir", str(tmp_path / "library"),
                "--stages", "1",
            ])

        assert captured_preset_name == "My Cool Preset"

    def test_batch_optimize_wav_folder_mode(self, tmp_path):
        """WAV folder mode builds manifest on-the-fly and optimizes."""
        wav_dir = tmp_path / "wavs"
        _make_sine_wav(wav_dir / "bass" / "sub.wav", freq=60)
        _make_sine_wav(wav_dir / "lead" / "bright.wav", freq=2000)

        call_count = 0

        def mock_optimize(target_audio, surge_host, **kwargs):
            nonlocal call_count
            call_count += 1
            output = kwargs.get("output_dir", tmp_path / "work")
            output.mkdir(parents=True, exist_ok=True)
            return self._make_mock_optimize_result(output, kwargs.get("preset_name", "test"))

        with patch("synth2surge.optimizer.loop.optimize", side_effect=mock_optimize), \
             patch("synth2surge.audio.engine.PluginHost"):
            from typer.testing import CliRunner

            from synth2surge.cli.main import app

            runner = CliRunner()
            runner.invoke(app, [
                "batch-optimize",
                "--input", str(wav_dir),
                "--output-dir", str(tmp_path / "library"),
                "--stages", "1",
            ])

        assert call_count == 2


# ---------------------------------------------------------------------------
# Full Pipeline E2E (requires Surge XT)
# ---------------------------------------------------------------------------


@pytest.mark.requires_surge
class TestBatchOptimizeRealSurge:
    """Real Surge XT integration tests for batch optimization.

    Uses subprocess to run CLI commands because pedalboard requires
    the main thread for plugin loading.
    """

    def test_batch_optimize_wav_folder_real_surge(self, tmp_path):
        """Full pipeline: WAV folder -> batch optimize -> valid FXP."""
        import subprocess

        wav_dir = tmp_path / "wavs"
        _make_sine_wav(
            wav_dir / "bass" / "low_tone.wav", freq=80, duration=2.0,
        )
        _make_sine_wav(
            wav_dir / "lead" / "high_tone.wav", freq=1000, duration=2.0,
        )

        output_dir = tmp_path / "library"

        result = subprocess.run(
            [
                "synth2surge", "batch-optimize",
                "--input", str(wav_dir),
                "--output-dir", str(output_dir),
                "--stages", "1",
                "--trials-t1", "10",
            ],
            capture_output=True, text=True, timeout=300,
        )

        assert result.returncode == 0, result.stderr

        bass_dir = output_dir / "bass" / "low_tone"
        lead_dir = output_dir / "lead" / "high_tone"

        assert bass_dir.exists()
        assert lead_dir.exists()
        assert (bass_dir / "low_tone.fxp").exists()
        assert (lead_dir / "high_tone.fxp").exists()
        assert (bass_dir / "low_tone.wav").exists()
        assert (lead_dir / "high_tone.wav").exists()

    def test_batch_optimize_resume_real_surge(self, tmp_path):
        """Resume after partial completion with real Surge XT."""
        import subprocess

        wav_dir = tmp_path / "wavs"
        _make_sine_wav(
            wav_dir / "bass" / "tone1.wav", freq=100, duration=2.0,
        )
        _make_sine_wav(
            wav_dir / "lead" / "tone2.wav", freq=800, duration=2.0,
        )

        output_dir = tmp_path / "library"
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        # Build manifest and mark first item as completed
        manifest = build_manifest_from_wav_folder(wav_dir)
        assert len(manifest.items) == 2
        manifest.items[0].status = "completed"
        manifest.items[0].result_dir = str(
            output_dir / manifest.items[0].category.lower()
        )
        manifest_path = queue_dir / "manifest.json"
        save_manifest(manifest, manifest_path)

        result = subprocess.run(
            [
                "synth2surge", "batch-optimize",
                "--queue-dir", str(queue_dir),
                "--output-dir", str(output_dir),
                "--stages", "1",
                "--trials-t1", "10",
            ],
            capture_output=True, text=True, timeout=300,
        )

        assert result.returncode == 0, result.stderr

        loaded = load_manifest(manifest_path)
        completed = [
            i for i in loaded.items if i.status == "completed"
        ]
        assert len(completed) == 2
