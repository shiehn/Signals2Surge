"""Tests for batch manifest module."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synth2surge.batch.manifest import (
    CATEGORIES,
    QueueItem,
    QueueManifest,
    add_item,
    build_manifest_from_wav_folder,
    compute_audio_hash,
    compute_state_hash,
    load_manifest,
    mark_completed,
    mark_failed,
    new_item_id,
    pending_items,
    sanitize_filename,
    save_manifest,
)


@pytest.mark.unit
class TestManifestRoundtrip:
    def test_empty_manifest_roundtrip(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded.version == 1
        assert loaded.items == []

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        manifest = load_manifest(path)
        assert manifest.version == 1
        assert manifest.items == []

    def test_manifest_with_items_roundtrip(self, tmp_path):
        path = tmp_path / "manifest.json"
        item = QueueItem(
            id="abc123",
            preset_name="Deep Sub",
            category="Bass",
            audio_path="abc123/target_audio.wav",
            state_path="abc123/target_state.bin",
            state_hash="deadbeef",
            added_at="2026-01-01T00:00:00",
        )
        manifest = QueueManifest(created_at="2026-01-01T00:00:00", items=[item])
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert len(loaded.items) == 1
        assert loaded.items[0].preset_name == "Deep Sub"
        assert loaded.items[0].category == "Bass"
        assert loaded.items[0].state_hash == "deadbeef"


@pytest.mark.unit
class TestAddItem:
    def test_add_item_success(self):
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")
        item = QueueItem(
            id="a1", preset_name="Test", category="Bass",
            audio_path="a1/audio.wav", state_hash="hash1",
        )
        assert add_item(manifest, item) is True
        assert len(manifest.items) == 1

    def test_dedup_by_state_hash(self):
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")
        item1 = QueueItem(
            id="a1", preset_name="Test1", category="Bass",
            audio_path="a1/audio.wav", state_hash="same_hash",
        )
        item2 = QueueItem(
            id="a2", preset_name="Test2", category="Lead",
            audio_path="a2/audio.wav", state_hash="same_hash",
        )
        assert add_item(manifest, item1) is True
        assert add_item(manifest, item2) is False
        assert len(manifest.items) == 1

    def test_dedup_by_audio_hash(self):
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")
        item1 = QueueItem(
            id="a1", preset_name="Test1", category="Bass",
            audio_path="a1/audio.wav", audio_hash="same_hash",
        )
        item2 = QueueItem(
            id="a2", preset_name="Test2", category="Lead",
            audio_path="a2/audio.wav", audio_hash="same_hash",
        )
        assert add_item(manifest, item1) is True
        assert add_item(manifest, item2) is False

    def test_add_sets_timestamp(self):
        manifest = QueueManifest(created_at="2026-01-01T00:00:00")
        item = QueueItem(id="a1", preset_name="Test", category="Bass", audio_path="a1/audio.wav")
        add_item(manifest, item)
        assert item.added_at != ""


@pytest.mark.unit
class TestHashing:
    def test_state_hash_deterministic(self):
        data = b"some plugin state bytes"
        assert compute_state_hash(data) == compute_state_hash(data)

    def test_state_hash_different_for_different_data(self):
        assert compute_state_hash(b"state_a") != compute_state_hash(b"state_b")

    def test_audio_hash_deterministic(self):
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
        assert compute_audio_hash(audio) == compute_audio_hash(audio)

    def test_audio_hash_different_for_different_audio(self):
        audio1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
        audio2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100)).astype(np.float32)
        assert compute_audio_hash(audio1) != compute_audio_hash(audio2)


@pytest.mark.unit
class TestStatusOperations:
    def test_pending_items_filter(self):
        manifest = QueueManifest(
            created_at="2026-01-01T00:00:00",
            items=[
                QueueItem(
                    id="a1", preset_name="A", category="Bass",
                    audio_path="a1/a.wav", status="completed",
                ),
                QueueItem(
                    id="a2", preset_name="B", category="Lead",
                    audio_path="a2/a.wav", status="pending",
                ),
                QueueItem(
                    id="a3", preset_name="C", category="Pad",
                    audio_path="a3/a.wav", status="failed",
                ),
                QueueItem(
                    id="a4", preset_name="D", category="Keys",
                    audio_path="a4/a.wav", status="pending",
                ),
            ],
        )
        pending = pending_items(manifest)
        assert len(pending) == 2
        assert {p.id for p in pending} == {"a2", "a4"}

    def test_mark_completed(self):
        manifest = QueueManifest(
            created_at="2026-01-01T00:00:00",
            items=[QueueItem(id="a1", preset_name="A", category="Bass", audio_path="a1/a.wav")],
        )
        mark_completed(manifest, "a1", "/output/bass/a")
        assert manifest.items[0].status == "completed"
        assert manifest.items[0].result_dir == "/output/bass/a"

    def test_mark_failed(self):
        manifest = QueueManifest(
            created_at="2026-01-01T00:00:00",
            items=[QueueItem(id="a1", preset_name="A", category="Bass", audio_path="a1/a.wav")],
        )
        mark_failed(manifest, "a1", "optimization crashed")
        assert manifest.items[0].status == "failed"
        assert manifest.items[0].error == "optimization crashed"


@pytest.mark.unit
class TestSanitizeFilename:
    def test_basic_sanitization(self):
        assert sanitize_filename("my/preset:name") == "my_preset_name"

    def test_preserves_safe_names(self):
        assert sanitize_filename("Deep Sub Bass") == "Deep Sub Bass"

    def test_empty_becomes_unnamed(self):
        assert sanitize_filename("") == "unnamed"

    def test_dots_only_becomes_unnamed(self):
        assert sanitize_filename("...") == "unnamed"

    def test_special_chars(self):
        assert sanitize_filename('a<b>c"d') == "a_b_c_d"


@pytest.mark.unit
class TestBuildManifestFromWavFolder:
    def _make_wav(self, path: Path, freq: float = 440.0):
        """Create a short sine wave WAV file."""
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sr)

    def test_basic_folder_structure(self, tmp_path):
        self._make_wav(tmp_path / "bass" / "deep_sub.wav", freq=80)
        self._make_wav(tmp_path / "lead" / "screamer.wav", freq=1000)
        self._make_wav(tmp_path / "pad" / "warm_pad.wav", freq=300)

        manifest = build_manifest_from_wav_folder(tmp_path)
        assert len(manifest.items) == 3

        categories = {item.category for item in manifest.items}
        assert categories == {"Bass", "Lead", "Pad"}

        names = {item.preset_name for item in manifest.items}
        assert names == {"deep_sub", "screamer", "warm_pad"}

        for item in manifest.items:
            assert item.status == "pending"
            assert item.audio_hash is not None

    def test_skips_non_wav_files(self, tmp_path):
        self._make_wav(tmp_path / "bass" / "good.wav", freq=80)
        (tmp_path / "bass" / "notes.txt").write_text("not a wav")
        (tmp_path / "bass" / "preset.mp3").write_bytes(b"fake mp3")

        manifest = build_manifest_from_wav_folder(tmp_path)
        assert len(manifest.items) == 1
        assert manifest.items[0].preset_name == "good"

    def test_dedup_identical_wavs(self, tmp_path):
        """Two identical WAVs in different folders should be deduped."""
        self._make_wav(tmp_path / "bass" / "sub1.wav", freq=80)
        self._make_wav(tmp_path / "lead" / "sub2.wav", freq=80)  # Same freq = same features

        manifest = build_manifest_from_wav_folder(tmp_path)
        assert len(manifest.items) == 1

    def test_skips_root_level_wavs(self, tmp_path):
        """WAVs directly in root (no category subdir) should be skipped."""
        self._make_wav(tmp_path / "bass" / "good.wav", freq=80)
        self._make_wav(tmp_path / "orphan.wav", freq=440)  # No subdir

        manifest = build_manifest_from_wav_folder(tmp_path)
        assert len(manifest.items) == 1


@pytest.mark.unit
class TestNewItemId:
    def test_ids_are_unique(self):
        ids = {new_item_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_is_12_chars(self):
        assert len(new_item_id()) == 12


@pytest.mark.unit
class TestCategories:
    def test_expected_categories(self):
        assert "Bass" in CATEGORIES
        assert "Lead" in CATEGORIES
        assert "Pad" in CATEGORIES
        assert "Other" in CATEGORIES
        assert len(CATEGORIES) == 8
