"""CLAP-based audio feature extraction for perceptual audio similarity.

Uses LAION-CLAP pretrained model to produce 512-dim audio embeddings
that encode perceptual timbre similarity. Drop-in replacement for
mel-spectrogram statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Singleton CLAP extractor — loaded once, reused
_clap_instance: CLAPExtractor | None = None


class CLAPExtractor:
    """Lazy-loading wrapper around LAION-CLAP model.

    Loads the pretrained checkpoint on first use and caches it.
    Produces 512-dim L2-normalized audio embeddings.
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._model = None
        self._checkpoint_dir = checkpoint_dir or Path("workspace/models/clap")

    def _ensure_loaded(self) -> None:
        """Lazy-load the CLAP model on first use."""
        if self._model is not None:
            return

        try:
            import laion_clap

            self._model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            self._model.load_ckpt()  # Downloads to default cache if needed
            logger.info("CLAP model loaded (HTSAT-base)")
        except ImportError:
            logger.warning(
                "laion-clap not installed. Falling back to transformers ClapModel."
            )
            self._load_transformers_clap()

    def _load_transformers_clap(self) -> None:
        """Fallback: load CLAP via HuggingFace transformers."""
        from transformers import ClapModel, ClapProcessor

        model_id = "laion/larger_clap_music_and_speech"
        self._processor = ClapProcessor.from_pretrained(model_id)
        self._hf_model = ClapModel.from_pretrained(model_id)
        self._hf_model.eval()
        self._model = self._hf_model  # Mark as loaded so _ensure_loaded won't re-run
        self._use_hf = True
        logger.info(f"CLAP model loaded via transformers ({model_id})")

    def extract(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """Extract 512-dim CLAP embedding from audio.

        Args:
            audio: Mono audio signal (1-D float32 array).
            sr: Sample rate of the input audio.

        Returns:
            L2-normalized 512-dim float32 feature vector.
        """
        if len(audio) == 0 or np.sqrt(np.mean(audio**2)) < 1e-8:
            return np.zeros(512, dtype=np.float32)

        self._ensure_loaded()

        # Resample to 48kHz if needed (CLAP expects 48kHz)
        if sr != 48000:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=48000)

        # Ensure float32
        audio = audio.astype(np.float32)

        if hasattr(self, "_use_hf") and self._use_hf:
            return self._extract_hf(audio)
        else:
            return self._extract_laion(audio)

    def _extract_laion(self, audio_48k: np.ndarray) -> np.ndarray:
        """Extract using laion-clap."""
        import torch

        # laion-clap expects [batch, samples] at 48kHz
        audio_tensor = torch.from_numpy(audio_48k).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model.get_audio_embedding_from_data(
                x=audio_tensor, use_tensor=True
            )

        result = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm
        return result

    def _extract_hf(self, audio_48k: np.ndarray) -> np.ndarray:
        """Extract using HuggingFace transformers CLAP."""
        import torch

        inputs = self._processor(
            audio=[audio_48k],
            sampling_rate=48000,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._hf_model.get_audio_features(**inputs)

        # get_audio_features returns a tensor or BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            result = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
        elif hasattr(outputs, "squeeze"):
            result = outputs.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            result = outputs[0].squeeze(0).cpu().numpy().astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm
        return result


def get_clap_extractor(checkpoint_dir: Path | None = None) -> CLAPExtractor:
    """Get or create the singleton CLAP extractor."""
    global _clap_instance
    if _clap_instance is None:
        _clap_instance = CLAPExtractor(checkpoint_dir)
    return _clap_instance


def extract_clap_features(
    audio: np.ndarray,
    sr: int = 44100,
) -> np.ndarray:
    """Extract 512-dim CLAP audio embedding.

    Drop-in replacement for extract_features(). Returns L2-normalized
    512-dim vector encoding perceptual audio similarity.

    Args:
        audio: Mono audio signal (1-D float array).
        sr: Sample rate.

    Returns:
        L2-normalized feature vector of shape (512,).
    """
    extractor = get_clap_extractor()
    return extractor.extract(audio, sr=sr)
