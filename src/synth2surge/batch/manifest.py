"""Queue manifest for batch preset porting.

Manages a JSON manifest of presets queued for optimization, with support
for deduplication, resume, and building manifests from WAV folder structures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CATEGORIES = ["Bass", "Lead", "Pad", "Keys", "Pluck", "FX", "Drums", "Other"]


@dataclass
class QueueItem:
    """A single preset queued for optimization."""

    id: str
    preset_name: str
    category: str
    audio_path: str  # Relative to manifest directory
    state_path: str | None = None  # Relative (None for WAV-folder mode)
    state_hash: str | None = None  # SHA-256 of binary plugin state
    audio_hash: str | None = None  # SHA-256 of audio feature vector
    added_at: str = ""
    status: str = "pending"  # pending | completed | failed
    error: str | None = None
    result_dir: str | None = None


@dataclass
class QueueManifest:
    """Manifest tracking all queued presets and their processing status."""

    version: int = 1
    plugin_path: str | None = None
    created_at: str = ""
    items: list[QueueItem] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_item_id() -> str:
    """Generate a short unique ID for a queue item."""
    return uuid.uuid4().hex[:12]


def sanitize_filename(name: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    sanitized = sanitized.strip(". ")
    return sanitized or "unnamed"


def compute_state_hash(state_bytes: bytes) -> str:
    """SHA-256 hex digest of binary plugin state bytes."""
    return hashlib.sha256(state_bytes).hexdigest()


def compute_audio_hash(audio: np.ndarray, sr: int = 44100) -> str:
    """SHA-256 of the audio feature vector (perceptual dedup for WAV mode)."""
    from synth2surge.loss.features import extract_features

    features = extract_features(audio, sr=sr)
    return hashlib.sha256(features.tobytes()).hexdigest()


def load_manifest(path: Path) -> QueueManifest:
    """Load a manifest from JSON. Returns empty manifest if file doesn't exist."""
    if not path.exists():
        return QueueManifest(created_at=_now_iso())

    data = json.loads(path.read_text())
    items = [QueueItem(**item) for item in data.get("items", [])]
    return QueueManifest(
        version=data.get("version", 1),
        plugin_path=data.get("plugin_path"),
        created_at=data.get("created_at", ""),
        items=items,
    )


def save_manifest(manifest: QueueManifest, path: Path) -> None:
    """Atomically save manifest to JSON (write to temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(asdict(manifest), indent=2))
    os.replace(str(tmp_path), str(path))


def add_item(manifest: QueueManifest, item: QueueItem) -> bool:
    """Add an item to the manifest. Returns False if a duplicate hash exists."""
    if item.state_hash:
        for existing in manifest.items:
            if existing.state_hash and existing.state_hash == item.state_hash:
                return False
    if item.audio_hash:
        for existing in manifest.items:
            if existing.audio_hash and existing.audio_hash == item.audio_hash:
                return False

    if not item.added_at:
        item.added_at = _now_iso()

    manifest.items.append(item)
    return True


def pending_items(manifest: QueueManifest) -> list[QueueItem]:
    """Return items with status 'pending'."""
    return [item for item in manifest.items if item.status == "pending"]


def mark_completed(manifest: QueueManifest, item_id: str, result_dir: str) -> None:
    """Mark an item as completed with its output directory."""
    for item in manifest.items:
        if item.id == item_id:
            item.status = "completed"
            item.result_dir = result_dir
            return


def mark_failed(manifest: QueueManifest, item_id: str, error: str) -> None:
    """Mark an item as failed with an error message."""
    for item in manifest.items:
        if item.id == item_id:
            item.status = "failed"
            item.error = error
            return


def build_manifest_from_wav_folder(wav_dir: Path) -> QueueManifest:
    """Build a manifest from a WAV folder where subdirectory names are categories.

    Expected structure:
        wav_dir/
        ├── bass/
        │   ├── deep_sub.wav
        │   └── acid.wav
        ├── lead/
        │   └── screamer.wav
        ...

    Files directly in wav_dir (no subdirectory) are skipped with a warning.
    """
    import logging

    import soundfile as sf

    logger = logging.getLogger(__name__)
    wav_dir = Path(wav_dir)
    manifest = QueueManifest(created_at=_now_iso())
    seen_hashes: set[str] = set()

    for category_dir in sorted(wav_dir.iterdir()):
        if not category_dir.is_dir():
            if category_dir.suffix.lower() == ".wav":
                logger.warning(
                    f"Skipping {category_dir.name} — WAV files must be in category subdirectories"
                )
            continue

        category = category_dir.name.title()

        for wav_file in sorted(category_dir.iterdir()):
            if wav_file.suffix.lower() != ".wav":
                continue

            audio, sr = sf.read(str(wav_file), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            audio_hash = compute_audio_hash(audio, sr=sr)
            if audio_hash in seen_hashes:
                logger.warning(f"Skipping duplicate: {wav_file} (same audio hash)")
                continue
            seen_hashes.add(audio_hash)

            item = QueueItem(
                id=new_item_id(),
                preset_name=wav_file.stem,
                category=category,
                audio_path=str(wav_file),
                audio_hash=audio_hash,
                added_at=_now_iso(),
            )
            manifest.items.append(item)

    return manifest
