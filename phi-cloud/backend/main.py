from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .core.jobs import create_job, get_job, run_computation, update_job_status
from .core.physics import get_preview_slice, HolographicCore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
API_KEY = os.environ.get("PHI_CLOUD_API_KEY", "my-secret-key")
STORAGE_PATH = "backend/storage"
MAX_DISK_USAGE_PERCENT = 90.0

app = FastAPI(
    title="ΦΦ-Cloud API",
    description="API for Holographic Tensor Processing Unit",
    version="1.0.0"
)

executor = ThreadPoolExecutor()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Helper Functions ---
def check_disk_pressure():
    """Checks if disk usage exceeds the maximum allowed percentage."""
    total, used, free = shutil.disk_usage("/")
    usage_percent = (used / total) * 100
    if usage_percent > MAX_DISK_USAGE_PERCENT:
        logger.warning(f"Disk pressure high: {usage_percent:.2f}% used.")
        raise HTTPException(status_code=503, detail="Vacuum Saturation: Service temporarily unavailable.")

def verify_api_key(x_api_key: str = Header(...)):
    """Dependency to verify the API key."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

import numpy as np
# --- API Endpoints ---

async def run_computation_in_executor(job_id: str, n_size: int, a_path: str, b_path: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, run_computation, loop, job_id, n_size, a_path, b_path)

@app.post("/genesis/random", status_code=202)
async def genesis_random(
    n_size: int = Form(...),
    x_api_key: str = Header(...)
):
    """
    Initiates a new holographic computation job with random matrices.
    """
    verify_api_key(x_api_key)
    check_disk_pressure()

    job = await create_job()

    storage = Path(STORAGE_PATH) / job.job_id
    storage.mkdir(exist_ok=True, parents=True)
    a_path = storage / "a.csv"
    b_path = storage / "b.csv"

    # Generate random matrices and save them
    matrix_a = np.random.rand(n_size, n_size).astype(np.float32)
    matrix_b = np.random.rand(n_size, n_size).astype(np.float32)
    np.savetxt(a_path, matrix_a, delimiter=',')
    np.savetxt(b_path, matrix_b, delimiter=',')

    await update_job_status(job.job_id, "running")
    asyncio.create_task(run_computation_in_executor(job.job_id, n_size, str(a_path), str(b_path)))

    return {"job_id": job.job_id, "status": "running"}


@app.post("/genesis", status_code=202)
async def genesis(
    n_size: int = Form(...),
    matrix_a: UploadFile = File(...),
    matrix_b: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    """
    Initiates a new holographic computation job.
    """
    verify_api_key(x_api_key)
    check_disk_pressure()

    job = await create_job()

    # Save uploaded files temporarily
    storage = Path(STORAGE_PATH) / job.job_id
    storage.mkdir(exist_ok=True, parents=True)
    a_path = storage / "a.csv"
    b_path = storage / "b.csv"

    with open(a_path, "wb") as buffer:
        shutil.copyfileobj(matrix_a.file, buffer)
    with open(b_path, "wb") as buffer:
        shutil.copyfileobj(matrix_b.file, buffer)

    await update_job_status(job.job_id, "running")
    asyncio.create_task(run_computation_in_executor(job.job_id, n_size, str(a_path), str(b_path)))

    return {"job_id": job.job_id, "status": "running"}

from typing import Optional

@app.get("/collapse/{job_id}")
async def collapse(job_id: str, download: Optional[bool] = None, x_api_key: str = Header(...)):
    """
    Retrieves the status and result of a computation job.
    """
    verify_api_key(x_api_key)

    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "completed":
        if download:
             return FileResponse(job.result, media_type='text/csv', filename='result.csv')
        return JSONResponse(content={
            "job_id": job.job_id,
            "status": job.status,
            "timings": job.timings
        })

    return JSONResponse(content={"job_id": job.job_id, "status": job.status, "error": job.error, "timings": job.timings})

@app.get("/preview/{job_id}")
async def preview(job_id: str, x_api_key: str = Header(...)):
    """
    Retrieves a downsampled heatmap of the spectral rupture.
    """
    verify_api_key(x_api_key)

    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    slice_data = get_preview_slice(job_id)
    if slice_data is None:
        raise HTTPException(status_code=404, detail="Preview not available yet.")

    return JSONResponse(content={"heatmap": slice_data.flatten().tolist()})

@app.on_event("startup")
async def on_startup():
    """Create storage directory on startup."""
    Path(STORAGE_PATH).mkdir(exist_ok=True)

@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup storage directory on shutdown."""
    # In a real production scenario, you might want to persist job data
    # and not clean up on every shutdown.
    storage_path = Path(STORAGE_PATH)
    for item in storage_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
    logger.info("Cleaned up storage directory.")
