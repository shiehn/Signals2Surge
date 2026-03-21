"""Download and manage pretrained model weights from GitHub Releases."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# Default GitHub release URL pattern
GITHUB_REPO = "shiehn/signals-to-surge"
RELEASE_ASSET_NAME = "pretrained-model.zip"

# Well-known directory name for pretrained models
PRETRAINED_DIR_NAME = "predictor_pretrained"


def get_latest_release_url(repo: str = GITHUB_REPO) -> str | None:
    """Fetch the download URL for the pretrained model from the latest GitHub release."""
    import json as json_mod
    from urllib.request import Request, urlopen

    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        req = Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urlopen(req, timeout=15) as resp:
            release = json_mod.loads(resp.read())

        for asset in release.get("assets", []):
            if asset["name"] == RELEASE_ASSET_NAME:
                return asset["browser_download_url"]

        logger.warning(f"No asset named '{RELEASE_ASSET_NAME}' in latest release")
        return None

    except Exception:
        logger.exception("Failed to query GitHub releases API")
        return None


def download_pretrained(
    models_dir: Path,
    *,
    url: str | None = None,
    repo: str = GITHUB_REPO,
) -> Path | None:
    """Download a pretrained model checkpoint from GitHub Releases.

    Downloads the zip, extracts model.pt + config.json into
    models_dir/predictor_pretrained/.

    Args:
        models_dir: Parent directory for model checkpoints.
        url: Direct download URL. If None, queries the latest GitHub release.
        repo: GitHub repo (owner/name) to query if url is not provided.

    Returns:
        Path to the extracted checkpoint directory, or None on failure.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Resolve download URL
    if url is None:
        url = get_latest_release_url(repo)
        if url is None:
            logger.error("Could not find pretrained model in GitHub releases")
            return None

    # Download to temp file
    logger.info(f"Downloading pretrained model from {url}")
    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        urlretrieve(url, str(tmp_path))
        logger.info(f"Downloaded {tmp_path.stat().st_size / 1024:.0f} KB")

    except Exception:
        logger.exception("Failed to download pretrained model")
        return None

    # Extract
    checkpoint_dir = models_dir / PRETRAINED_DIR_NAME
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Validate contents
            names = zf.namelist()
            has_model = any(n.endswith("model.pt") for n in names)
            has_config = any(n.endswith("config.json") for n in names)

            if not has_model or not has_config:
                logger.error(
                    f"Invalid checkpoint zip: missing model.pt or config.json. "
                    f"Contents: {names}"
                )
                return None

            # Clear existing pretrained dir
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True)

            # Extract files, flattening any subdirectory structure
            for name in names:
                if name.endswith("/"):
                    continue
                basename = Path(name).name
                with zf.open(name) as src:
                    with open(checkpoint_dir / basename, "wb") as dst:
                        dst.write(src.read())

        logger.info(f"Extracted pretrained model to {checkpoint_dir}")

    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid zip archive")
        return None
    except Exception:
        logger.exception("Failed to extract pretrained model")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)

    # Validate the extracted checkpoint
    model_path = checkpoint_dir / "model.pt"
    config_path = checkpoint_dir / "config.json"

    if not model_path.exists() or not config_path.exists():
        logger.error("Extracted checkpoint is missing model.pt or config.json")
        return None

    try:
        config = json.loads(config_path.read_text())
        logger.info(
            f"Pretrained model: {config.get('architecture', '?')}, "
            f"{config.get('n_params', '?')} params, "
            f"trained on {config.get('n_training_samples', '?')} samples"
        )
    except Exception:
        logger.warning("Could not read config.json, but model.pt exists")

    return checkpoint_dir


def find_pretrained(models_dir: Path) -> Path | None:
    """Check if a pretrained model exists in the models directory."""
    checkpoint_dir = Path(models_dir) / PRETRAINED_DIR_NAME
    model_path = checkpoint_dir / "model.pt"
    config_path = checkpoint_dir / "config.json"

    if model_path.exists() and config_path.exists():
        return checkpoint_dir
    return None


def package_model_for_release(
    checkpoint_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Package a trained model checkpoint into a zip for GitHub release.

    Args:
        checkpoint_dir: Directory containing model.pt and config.json.
        output_path: Where to save the zip. Defaults to pretrained-model.zip
                     in the current directory.

    Returns:
        Path to the created zip file.
    """
    if output_path is None:
        output_path = Path(RELEASE_ASSET_NAME)

    model_path = checkpoint_dir / "model.pt"
    config_path = checkpoint_dir / "config.json"

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} must contain model.pt and config.json"
        )

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "model.pt")
        zf.write(config_path, "config.json")

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Packaged model to {output_path} ({size_kb:.0f} KB)")
    return output_path
