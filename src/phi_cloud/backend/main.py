from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from phi_cloud_core.vpu import VirtualHolographicPU, PhiSubstrate, NeuroOpticalCompiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("PHI_CLOUD_API_KEY", "my-secret-key")
STORAGE_PATH = Path("src/phi_cloud/backend/storage")
STATIC_PATH = STORAGE_PATH / "compiled_images"

# Ensure the static directory exists on application startup
STATIC_PATH.mkdir(exist_ok=True, parents=True)

app = FastAPI(
    title="ΦΦ-Cloud API",
    description="API for General-Purpose Holographic Processing Unit",
    version="2.1.0"
)
executor = ThreadPoolExecutor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class SubstrateConfig(BaseModel):
    shape: Tuple[int, int] = (256, 1024)
    wavelength: float = 1.55e-6
    dx: float = 0.2e-6

class CompilerConfig(BaseModel):
    learning_rate: float = 1e-3
    epochs: int = 50

class VPUConfigRequest(BaseModel):
    input_ports: List[int]
    output_ports: List[int]
    truth_table: List[Tuple[List[int], List[int]]]
    substrate_config: Optional[SubstrateConfig] = Field(default_factory=SubstrateConfig)
    compiler_config: Optional[CompilerConfig] = Field(default_factory=CompilerConfig)

class VPUCompilationResult(BaseModel):
    image_url: str
    verification: Dict[str, Any]
    verification_cases: List[Dict[str, Any]]
    loss_history: List[float]

def compile_and_visualize_sync(config: VPUConfigRequest) -> Dict[str, Any]:
    try:
        logger.info("Initializing VPU based on API request...")
        substrate = PhiSubstrate(**config.substrate_config.model_dump())
        compiler = NeuroOpticalCompiler(substrate, **config.compiler_config.model_dump())
        vpu = VirtualHolographicPU(
            substrate=substrate,
            compiler=compiler,
            input_ports=config.input_ports,
            output_ports=config.output_ports
        )

        delta_n, loss_history = vpu.learn_function(config.truth_table)
        verification_summary, verification_cases = vpu.verify(config.truth_table)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.patch.set_facecolor('#050505')
        im = ax.imshow(delta_n.T, cmap='viridis', aspect='auto', origin='lower')
        ax.set_title(f"Compiled Geometry (Verified: {verification_summary['all_correct']})", color='white')
        for port_idx in vpu.input_ports:
            ax.axvline(x=port_idx, color='lime', linestyle='--', label='Input Port')
        for port_idx in vpu.output_ports:
            ax.axvline(x=port_idx, color='red', linestyle='--', label='Output Port')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        fig.colorbar(im, ax=ax, label="Δn")
        plt.tight_layout()

        image_filename = f"{os.urandom(16).hex()}.png"
        output_path = STATIC_PATH / image_filename
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

        return {
            "image_url": f"/static/{image_filename}",
            "verification": verification_summary,
            "verification_cases": verification_cases,
            "loss_history": loss_history
        }
    except Exception as e:
        logger.error(f"Error during VPU compilation: {e}", exc_info=True)
        return None

@app.post("/compile-hologram", response_model=VPUCompilationResult)
async def compile_hologram(
    request: VPUConfigRequest,
    x_api_key: str = Header(...)
):
    verify_api_key(x_api_key)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, compile_and_visualize_sync, request)
    if result:
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=500, detail="Failed to generate hologram visualization.")

@app.on_event("shutdown")
async def on_shutdown():
    if STORAGE_PATH.exists():
        shutil.rmtree(STORAGE_PATH)
    logger.info("Cleaned up storage directory.")
