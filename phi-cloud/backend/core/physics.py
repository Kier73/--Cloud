import numpy as np
from scipy import fft
import os
from pathlib import Path
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HolographicCore:
    def __init__(self, n_size: int, job_id: str, storage_path: str = "backend/storage"):
        if n_size > 16384:
            raise ValueError("Matrix size N cannot exceed 16,384")
        self.n_size = n_size
        self.job_id = job_id
        self.storage_path = Path(storage_path) / job_id
        self.storage_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"HolographicCore initialized for job {job_id} with N={n_size} and storage at {self.storage_path}")

    def _get_memmap_path(self, name: str) -> Path:
        return self.storage_path / f"{name}.mmap"

    def _create_memmap(self, name: str, dtype=np.float32, shape=None):
        if shape is None:
            shape = (self.n_size, self.n_size)
        path = self._get_memmap_path(name)
        if path.exists():
            path.unlink()
        return np.memmap(path, dtype=dtype, mode='w+', shape=shape)

    def multiply(self, a_path: str, b_path: str) -> Path:
        logger.info(f"[{self.job_id}] Starting multiplication for {a_path} and {b_path}")

        timings = {}

        try:
            a = np.loadtxt(a_path, delimiter=',')
            b = np.loadtxt(b_path, delimiter=',')

            # --- Standard CPU Time (Sample) ---
            sample_size = min(self.n_size, 256)
            a_sample = a[:sample_size, :sample_size]
            b_sample = b[:sample_size, :sample_size]
            start_time = time.time()
            np.dot(a_sample, b_sample)
            end_time = time.time()
            # Extrapolate to full size: O(N^3) complexity
            timings['standard_cpu_time_est'] = (end_time - start_time) * ((self.n_size / sample_size) ** 3)
            logger.info(f"[{self.job_id}] Standard CPU time (estimated): {timings['standard_cpu_time_est']:.2f}s")


            # --- Holographic Multiplication ---
            start_holo_time = time.time()

            # Pad matrices to the required size
            padded_a = np.zeros((self.n_size, self.n_size), dtype=np.float32)
            padded_a[:a.shape[0], :a.shape[1]] = a
            a_mem = self._create_memmap('A')
            a_mem[:] = padded_a
            del a, padded_a

            padded_b = np.zeros((self.n_size, self.n_size), dtype=np.float32)
            padded_b[:b.shape[0], :b.shape[1]] = b
            b_mem = self._create_memmap('B')
            b_mem[:] = padded_b
            del b, padded_b

        except Exception as e:
            logger.error(f"[{self.job_id}] Error loading input matrices: {e}")
            raise

        logger.info(f"[{self.job_id}] Transforming matrices to frequency domain...")
        fft_shape = (self.n_size, self.n_size // 2 + 1)
        a_fft = self._create_memmap('A_fft', dtype=np.complex64, shape=fft_shape)
        a_fft[:] = fft.rfft2(a_mem)
        del a_mem

        b_fft = self._create_memmap('B_fft', dtype=np.complex64, shape=fft_shape)
        b_fft[:] = fft.rfft2(b_mem)
        del b_mem

        logger.info(f"[{self.job_id}] Performing spectral rupture (interference)...")
        c_fft = self._create_memmap('C_fft', dtype=np.complex64, shape=fft_shape)
        np.multiply(a_fft, b_fft, out=c_fft)

        self.generate_preview_slice(c_fft, 'preview')

        del a_fft, b_fft

        logger.info(f"[{self.job_id}] Relaxing back to spatial domain...")
        c_mem = self._create_memmap('C')
        c_mem[:] = fft.irfft2(c_fft)
        del c_fft

        end_holo_time = time.time()
        timings['holographic_time'] = end_holo_time - start_holo_time
        logger.info(f"[{self.job_id}] Holographic time: {timings['holographic_time']:.2f}s")

        # Save timings
        with open(self.storage_path / "timings.json", "w") as f:
            json.dump(timings, f)

        result_path = self.storage_path / "result.csv"
        np.savetxt(result_path, c_mem, delimiter=',')
        logger.info(f"[{self.job_id}] Multiplication complete. Result saved to {result_path}")

        # In a real scenario, you might keep the result memmap around
        del c_mem

        return result_path

    def generate_preview_slice(self, fft_data, slice_name: str, slice_size: int = 128):
        logger.info(f"[{self.job_id}] Generating preview slice of size {slice_size}x{slice_size}")
        magnitude_spectrum = np.abs(fft_data)

        step_x = max(1, magnitude_spectrum.shape[0] // slice_size)
        step_y = max(1, magnitude_spectrum.shape[1] // slice_size)
        downsampled = magnitude_spectrum[::step_x, ::step_y][:slice_size, :slice_size]

        # Log-scale for better visualization
        log_spectrum = np.log1p(downsampled)

        # Normalize to 0-255
        min_val, max_val = np.min(log_spectrum), np.max(log_spectrum)
        if max_val > min_val:
            heatmap = (255 * (log_spectrum - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            heatmap = np.zeros(log_spectrum.shape, dtype=np.uint8)

        preview_path = self._get_memmap_path(slice_name)
        # Use a temporary path for atomic write
        temp_preview_path = preview_path.with_suffix('.tmp')
        preview_mem = np.memmap(temp_preview_path, dtype=np.uint8, mode='w+', shape=(slice_size, slice_size))
        preview_mem[:] = heatmap
        del preview_mem # Flushes to disk
        os.rename(temp_preview_path, preview_path)

        logger.info(f"[{self.job_id}] Preview slice saved to {preview_path}")

    def cleanup(self):
        """Removes all files associated with the job."""
        import shutil
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)
            logger.info(f"[{self.job_id}] Cleaned up storage directory: {self.storage_path}")

def get_preview_slice(job_id: str, storage_path: str = "backend/storage", slice_size: int = 128) -> np.ndarray:
    """Retrieves a preview slice for a given job."""
    path = Path(storage_path) / job_id / "preview.mmap"
    if not path.exists():
        return None
    return np.memmap(path, dtype=np.uint8, mode='r', shape=(slice_size, slice_size))
