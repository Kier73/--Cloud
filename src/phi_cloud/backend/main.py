from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from phi_cloud.backend.core.jobs import create_job, get_job, run_computation, update_job_status
from phi_cloud.backend.core.physics import get_preview_slice, HolographicCore
from phi_cloud_core.vpu import VirtualHolographicPU

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

# --- Pydantic Models ---
class TruthTableRequest(BaseModel):
    truth_table: List[Tuple[int, int]]

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

# --- VPU Compilation ---

def compile_and_visualize_sync(truth_table: List[Tuple[int, int]]) -> str:
    """
    Runs the VPU compilation and generates a visualization.
    Returns the path to the saved image.
    """
    try:
        logger.info("Initializing Virtual Holographic PU...")
        vpu = VirtualHolographicPU(size=128) # Reduced size for faster API response

        logger.info(f"Learning function from truth table: {truth_table}")
        vpu.learn_function(truth_table)

        logger.info("Executing kernel for visualization...")
        val_high, flow_high = vpu.execute(1)
        val_low, flow_low = vpu.execute(0)
        logger.info(f"Execution results: High={val_high:.2f}, Low={val_low:.2f}")

        # --- Visualization ---
        logger.info("Generating hardware schematic visualization...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#050505')

        hardware = vpu.substrate.n_base + vpu.substrate.n_learned
        axes[0].imshow(hardware, cmap='gray')
        axes[0].set_title("1. Compiled Geometry", color='white')
        axes[0].scatter(*vpu.PORT_IN_A, c='lime', label='In')
        axes[0].scatter(*vpu.PORT_OUT_Y, c='red', label='Out')
        axes[0].legend()

        axes[1].imshow(flow_high, cmap='inferno')
        axes[1].set_title(f"2. Execution (Input=1)", color='#00ff00')

        vmax = np.max(flow_high) if np.max(flow_high) > 0 else 1.0
        axes[2].imshow(flow_low, cmap='inferno', vmin=0, vmax=vmax)
        axes[2].set_title(f"3. Execution (Input=0)", color='gray')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()

        temp_dir = Path(STORAGE_PATH) / "vpu_temp"
        temp_dir.mkdir(exist_ok=True, parents=True)
        output_path = temp_dir / f"{os.urandom(16).hex()}.png"

        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

        logger.info(f"Visualization saved to {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error during VPU compilation: {e}", exc_info=True)
        return None

@app.post("/compile-hologram")
async def compile_hologram(
    request: TruthTableRequest,
    x_api_key: str = Header(...)
):
    """
    Compiles a function from a truth table and returns a visualization.
    """
    verify_api_key(x_api_key)
    check_disk_pressure()

    loop = asyncio.get_running_loop()
    image_path = await loop.run_in_executor(
        executor,
        compile_and_visualize_sync,
        request.truth_table
    )

    if image_path and os.path.exists(image_path):
        return FileResponse(image_path, media_type='image/png')
    else:
        raise HTTPException(status_code=500, detail="Failed to generate hologram visualization.")

@app.on_event("startup")
async def on_startup():
    """Create storage directory on startup."""
    Path(STORAGE_PATH).mkdir(exist_ok=True)

@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup storage directory on shutdown."""
    storage_path = Path(STORAGE_PATH)
    for item in storage_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
    logger.info("Cleaned up storage directory.")
