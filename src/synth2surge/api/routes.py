"""FastAPI route definitions."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from synth2surge.api.schemas import (
    CaptureRequest,
    JobStatus,
    OptimizeRequest,
    PriorStatus,
)

router = APIRouter()

# In-memory job storage (for MVP; swap to Redis/SQLite for production)
_jobs: dict[str, dict[str, Any]] = {}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/capture")
async def start_capture(req: CaptureRequest) -> JobStatus:
    """Start a capture job (headless only via API)."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "pending", "request": req.model_dump()}

    asyncio.get_event_loop().run_in_executor(None, _run_capture, job_id, req)

    return JobStatus(job_id=job_id, status="pending")


@router.post("/optimize")
async def start_optimize(req: OptimizeRequest) -> JobStatus:
    """Start an optimization job."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "pending", "request": req.model_dump()}

    asyncio.get_event_loop().run_in_executor(None, _run_optimize, job_id, req)

    return JobStatus(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatus:
    """Get the status of a job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> JobStatus:
    """Cancel a running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    _jobs[job_id]["status"] = "cancelled"
    return JobStatus(job_id=job_id, status="cancelled")


@router.get("/prior/status")
async def prior_status() -> PriorStatus:
    """Check if the FAISS prior index is built."""
    index_path = Path("./workspace/prior_index")
    built = (index_path / "index.faiss").exists()
    n_entries = 0
    if built:
        from synth2surge.prior.index import PriorIndex

        idx = PriorIndex.load(index_path)
        n_entries = idx.size

    return PriorStatus(
        built=built,
        n_entries=n_entries,
        index_path=str(index_path) if built else None,
    )


def _run_capture(job_id: str, req: CaptureRequest) -> None:
    """Run capture in a background thread."""
    try:
        _jobs[job_id]["status"] = "running"

        from synth2surge.capture.workflow import capture_from_state_file, capture_headless
        from synth2surge.config import MultiProbeConfig

        multi_probe_config = None
        if req.probe_mode == "thorough":
            multi_probe_config = MultiProbeConfig.thorough()
        elif req.probe_mode == "full":
            multi_probe_config = MultiProbeConfig.full()

        if req.state_file:
            result = capture_from_state_file(
                plugin_path=req.plugin_path,
                state_file=req.state_file,
                output_dir=req.output_dir,
                multi_probe_config=multi_probe_config,
            )
        else:
            result = capture_headless(
                plugin_path=req.plugin_path,
                output_dir=req.output_dir,
                multi_probe_config=multi_probe_config,
            )

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = {
            "audio_path": str(result.audio_path),
            "state_path": str(result.state_path),
            "n_parameters": len(result.parameters),
        }
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


def _run_optimize(job_id: str, req: OptimizeRequest) -> None:
    """Run optimization in a background thread."""
    try:
        _jobs[job_id]["status"] = "running"

        import numpy as np
        import soundfile as sf

        from synth2surge.audio.engine import PluginHost
        from synth2surge.config import MultiProbeConfig, OptimizationConfig
        from synth2surge.optimizer.loop import optimize
        from synth2surge.types import OptimizationProgress

        target_audio, sr = sf.read(req.target_audio_path, dtype="float32")
        if target_audio.ndim > 1:
            target_audio = np.mean(target_audio, axis=1)

        host = PluginHost(req.surge_plugin_path, sample_rate=sr)
        config = OptimizationConfig(
            n_trials_tier1=req.trials_t1,
            n_trials_tier2=req.trials_t2,
            n_trials_tier3=req.trials_t3,
        )

        multi_probe_config = None
        target_segments = None
        if req.probe_mode == "thorough":
            multi_probe_config = MultiProbeConfig.thorough()
        elif req.probe_mode == "full":
            multi_probe_config = MultiProbeConfig.full()

        if multi_probe_config is not None and multi_probe_config.mode != "single":
            segments_path = Path(req.output_dir) / "target_segments.npz"
            if segments_path.exists():
                data = np.load(str(segments_path))
                target_segments = [data[k] for k in sorted(data.files)]

        def on_progress(p: OptimizationProgress) -> None:
            _jobs[job_id]["progress"] = {
                "current_trial": p.current_trial,
                "total_trials": p.total_trials,
                "best_loss": p.best_loss,
                "current_loss": p.current_loss,
                "stage": p.stage,
            }

        result = optimize(
            target_audio=target_audio,
            surge_host=host,
            config=config,
            progress_callback=on_progress,
            stages=req.stages,
            output_dir=Path(req.output_dir),
            multi_probe_config=multi_probe_config,
            target_segments=target_segments,
        )

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = {
            "best_patch_path": str(result.best_patch_path),
            "best_loss": result.best_loss,
            "best_audio_path": str(result.best_audio_path),
            "total_trials": result.total_trials,
            "fxp_path": str(result.fxp_path) if result.fxp_path else None,
        }
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
