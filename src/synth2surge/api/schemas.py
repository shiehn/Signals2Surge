"""Pydantic schemas for FastAPI request/response models."""

from __future__ import annotations

from pydantic import BaseModel


class CaptureRequest(BaseModel):
    plugin_path: str
    output_dir: str = "./workspace"
    no_gui: bool = True
    state_file: str | None = None
    probe_mode: str = "single"


class OptimizeRequest(BaseModel):
    target_audio_path: str
    output_dir: str = "./workspace"
    surge_plugin_path: str = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    trials_t1: int = 300
    trials_t2: int = 300
    trials_t3: int = 200
    stages: list[int] = [1, 2, 3]
    probe_mode: str = "single"


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: dict | None = None
    result: dict | None = None
    error: str | None = None


class PriorStatus(BaseModel):
    built: bool
    n_entries: int
    index_path: str | None = None
