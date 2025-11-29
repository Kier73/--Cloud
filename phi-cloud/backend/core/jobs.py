from typing import Dict, Any, Literal
from uuid import uuid4
import asyncio
from pydantic import BaseModel, Field
import logging

from .physics import HolographicCore

logger = logging.getLogger(__name__)

class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    result: Any = None
    error: str = None
    timings: Dict[str, float] = None

# In-memory store for jobs
job_store: Dict[str, Job] = {}
# A lock to manage concurrent access to the job store
job_store_lock = asyncio.Lock()

async def create_job() -> Job:
    """Creates a new job and stores it."""
    async with job_store_lock:
        new_job = Job()
        job_store[new_job.job_id] = new_job
        logger.info(f"Created new job: {new_job.job_id}")
        return new_job

async def get_job(job_id: str) -> Job:
    """Retrieves a job from the store."""
    async with job_store_lock:
        return job_store.get(job_id)

async def update_job_status(job_id: str, status: str, result: Any = None, error: str = None, timings: dict = None):
    """Updates the status of a job."""
    async with job_store_lock:
        if job_id in job_store:
            job = job_store[job_id]
            job.status = status
            job.result = result
            job.error = error
            job.timings = timings
            logger.info(f"Updated job {job_id} status to {status}")
        else:
            logger.warning(f"Attempted to update non-existent job: {job_id}")

import json
from pathlib import Path

def run_computation(loop, job_id: str, n_size: int, a_path: str, b_path: str):
    """The actual computation logic to be run in the background."""
    logger.info(f"[{job_id}] Starting computation task...")
    try:
        core = HolographicCore(n_size=n_size, job_id=job_id)
        result_path = core.multiply(a_path, b_path)

        # Read timings
        timings_path = Path(result_path).parent / "timings.json"
        with open(timings_path, "r") as f:
            timings = json.load(f)

        # Schedule the async status update on the main event loop
        asyncio.run_coroutine_threadsafe(
            update_job_status(job_id, "completed", str(result_path), timings=timings),
            loop
        )

    except Exception as e:
        logger.error(f"[{job_id}] Computation failed: {e}")
        asyncio.run_coroutine_threadsafe(
            update_job_status(job_id, "failed", error=str(e)),
            loop
        )
    finally:
        # Clean up input files
        import os
        try:
            os.remove(a_path)
            os.remove(b_path)
        except OSError as e:
            logger.error(f"[{job_id}] Error cleaning up input files: {e}")
