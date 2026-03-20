"""Discover and load Surge XT factory patches."""

from __future__ import annotations

from pathlib import Path

from synth2surge.config import SurgeConfig
from synth2surge.surge.patch import SurgePatch


def discover_factory_patches(
    factory_dir: Path | None = None,
) -> list[Path]:
    """Find all .fxp factory patch files in the Surge XT installation."""
    if factory_dir is None:
        factory_dir = SurgeConfig().factory_patches_dir

    if not factory_dir.exists():
        return []

    return sorted(factory_dir.rglob("*.fxp"))


def discover_factory_categories(
    factory_dir: Path | None = None,
) -> dict[str, list[Path]]:
    """Find factory patches organized by category (subdirectory)."""
    if factory_dir is None:
        factory_dir = SurgeConfig().factory_patches_dir

    if not factory_dir.exists():
        return {}

    categories: dict[str, list[Path]] = {}
    for subdir in sorted(factory_dir.iterdir()):
        if subdir.is_dir():
            patches = sorted(subdir.glob("*.fxp"))
            if patches:
                categories[subdir.name] = patches

    return categories


def load_factory_patch(path: Path) -> SurgePatch:
    """Load a single factory patch file."""
    return SurgePatch.from_file(path)
