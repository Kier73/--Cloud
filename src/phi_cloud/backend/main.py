from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from phi_cloud_core.vpu import VirtualHolographicPU, PhiSubstrate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
API_KEY = os.environ.get("PHI_CLOUD_API_KEY", "my-secret-key")
STORAGE_PATH = "src/phi_cloud/backend/storage"
MAX_DISK_USAGE_PERCENT = 90.0

app = FastAPI(
    title="ΦΦ-Cloud API",
    description="API for General-Purpose Holographic Processing Unit",
    version="2.0.0"
)
executor = ThreadPoolExecutor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- Pydantic Models for API ---
class VPUConfigRequest(BaseModel):
    input_ports: List[int]
    output_ports: List[int]
    truth_table: List[Tuple[List[int], List[int]]]

# --- VPU Compilation Logic ---
def compile_and_visualize_sync(config: VPUConfigRequest) -> str:
    try:
        logger.info("Initializing VPU based on API request...")
        substrate = PhiSubstrate(shape=(256, 1024), dx=0.2e-6)
        vpu = VirtualHolographicPU(
            substrate=substrate,
            input_ports=config.input_ports,
            output_ports=config.output_ports,
            port_width=10
        )

        logger.info(f"Learning function from truth table: {config.truth_table}")
        delta_n, loss_history = vpu.learn_function(config.truth_table)

        is_verified = vpu.verify(config.truth_table)
        logger.info(f"Verification successful: {is_verified}")

        # --- Visualization ---
        logger.info("Generating hardware schematic visualization...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.patch.set_facecolor('#050505')

        im = ax.imshow(delta_n.T, cmap='viridis', aspect='auto', origin='lower')
        ax.set_title(f"Compiled Geometry (Verified: {is_verified})", color='white')

        # Plot ports
        for port_idx in vpu.input_ports:
            ax.axvline(x=port_idx, color='lime', linestyle='--', label='Input Port')
        for port_idx in vpu.output_ports:
            ax.axvline(x=port_idx, color='red', linestyle='--', label='Output Port')

        # Create a single legend entry for ports
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        fig.colorbar(im, ax=ax, label="Δn (Refractive Index Change)")
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
    request: VPUConfigRequest,
    x_api_key: str = Header(...)
):
    """
    Compiles a function from a truth table and returns a visualization.
    """
    verify_api_key(x_api_key)
    loop = asyncio.get_running_loop()
    image_path = await loop.run_in_executor(executor, compile_and_visualize_sync, request)

    if image_path and os.path.exists(image_path):
        return FileResponse(image_path, media_type='image/png')
    else:
        raise HTTPException(status_code=500, detail="Failed to generate hologram visualization.")

@app.on_event("startup")
async def on_startup():
    Path(STORAGE_PATH).mkdir(exist_ok=True, parents=True)

@app.on_event("shutdown")
async def on_shutdown():
    storage_path = Path(STORAGE_PATH)
    if storage_path.exists():
        shutil.rmtree(storage_path)
    logger.info("Cleaned up storage directory.")
