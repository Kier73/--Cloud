import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import tempfile
import os
import uuid

# ==============================================================================
# LAYER 1: THE PHYSICS KERNEL (The Substrate)
# ==============================================================================

class PhiSubstrate:
    """
    The Physical Layer.
    Represents a block of non-linear optical glass (Kerr Medium).
    """
    def __init__(self, size=128, resolution=0.1):
        self.size = size
        self.res = resolution
        self.shape = (size, size)

        # 1. REFRACTIVE INDEX (The Hardware)
        # Initialize with random atomic noise (The "Potting Soil" for our logic)
        np.random.seed(42) # Deterministic hardware for this demo
        self.n_base = np.ones(self.shape, dtype=np.float32)
        self.n_base += np.random.normal(0, 0.02, self.shape)

        # The Mutable Layer (The Learned Logic)
        self.n_learned = np.zeros(self.shape, dtype=np.float32)

        # 2. CACHED OPERATORS (Optimization)
        kx = scipy.fft.fftfreq(size, resolution)
        ky = scipy.fft.fftfreq(size, resolution)
        KX, KY = np.meshgrid(kx, ky)
        self.kinetic_op = np.exp(-1j * (KX**2 + KY**2) * 0.2) # dt=0.2

    def propagate(self, field, backward=False, steps=50):
        """
        Runs the wave equation.
        forward=True: Input -> Output
        backward=True: Error -> Input (Time Reversal)
        """
        psi = field.copy()
        n_total = self.n_base + self.n_learned

        accum_intensity = np.zeros(self.shape)

        for _ in range(steps):
            # Spatial Phase (Refraction)
            # If backward, we conjugate phase (Time Reversal)
            if backward:
                psi *= np.exp(-1j * (n_total**2) * 0.2)
            else:
                psi *= np.exp(1j * (n_total**2) * 0.2)

            # Spectral Diffraction
            spec = scipy.fft.fft2(psi)
            psi = scipy.fft.ifft2(spec * self.kinetic_op)

            # Boundary Damping (prevent wrap-around)
            psi[:5,:] *= 0.9; psi[-5:,:] *= 0.9
            psi[:,:5] *= 0.9; psi[:,-5:] *= 0.9

            accum_intensity += np.abs(psi)**2

        return psi, accum_intensity

# ==============================================================================
# LAYER 2: THE COMPILER (Genesis Engine)
# ==============================================================================

class NeuroOpticalCompiler:
    """
    The 'Software'.
    Uses Optical Backpropagation to carve logic from examples.
    """
    def __init__(self, substrate: PhiSubstrate):
        self.sub = substrate

    def compile(self, inputs, targets, input_locs, output_locs, epochs=50):
        """
        inputs: List of values [0, 1, ...]
        targets: List of values [0, 1, ...]
        input_locs: List of (x,y) tuples
        output_locs: List of (x,y) tuples
        """
        print(f"[COMPILER] compiling_logic_gate via Hebbian Interference...")
        print(f"           |- Samples: {len(inputs)}")
        print(f"           |- Epochs:  {epochs}")

        history = []

        for ep in range(epochs):
            batch_update = np.zeros(self.sub.shape)

            # Iterate through Truth Table examples
            for i, val_in in enumerate(inputs):
                target_val = targets[i]

                # 1. ENCODE INPUT (The Question)
                psi_fwd = np.zeros(self.sub.shape, dtype=np.complex64)
                if val_in > 0.5: # Logic High
                    loc = input_locs[0] # Single input for now
                    self._inject_gaussian(psi_fwd, loc)

                # 2. ENCODE TARGET (The Answer)
                psi_bwd = np.zeros(self.sub.shape, dtype=np.complex64)
                if target_val > 0.5: # Logic High
                    loc = output_locs[0]
                    self._inject_gaussian(psi_bwd, loc)

                # 3. PHYSICS
                # Run Forward
                _, I_fwd = self.sub.propagate(psi_fwd, backward=False)
                # Run Backward (Time Reversal of Desired Outcome)
                _, I_bwd = self.sub.propagate(psi_bwd, backward=True)

                # 4. INTERFERENCE (The Gradient)
                # Constructive interference between Cause and Effect
                # implies a valid causal path.
                overlap = I_fwd * I_bwd

                # Accumulate update for this batch
                batch_update += overlap

            # 5. ETCHING (Update Hardware)
            # Smooth the waveguides
            etch = gaussian_filter(batch_update, sigma=1.5)
            if np.max(etch) > 0: etch /= np.max(etch)

            # Apply learning rate
            self.sub.n_learned += etch * 0.05

            # Saturation (Glass Limit)
            self.sub.n_learned = np.clip(self.sub.n_learned, 0, 0.7)

            if ep % 10 == 0:
                print(f"           |- Epoch {ep}: Waveguides hardening...")

        print("[COMPILER] Compilation Complete. Hardware Logic Frozen.")
        return self.sub.n_learned

    def _inject_gaussian(self, field, pos, sigma=3.0, intensity=10.0):
        y, x = np.ogrid[:self.sub.size, :self.sub.size]
        dist = (x - pos[0])**2 + (y - pos[1])**2
        field += np.exp(-dist / (2*sigma**2)) * intensity

# ==============================================================================
# LAYER 3: THE VIRTUAL PROCESSOR (User API)
# ==============================================================================

class VirtualHolographicPU:
    """
    The Public API.
    Call this from your Python scripts.
    """
    def __init__(self, size=128):
        self.substrate = PhiSubstrate(size=size)
        self.compiler = NeuroOpticalCompiler(self.substrate)
        # Define Ports (Fixed Hardware Locations)
        mid = size // 2
        self.PORT_IN_A  = (30, mid)   # Left
        self.PORT_OUT_Y = (size-30, mid) # Right

    def learn_function(self, truth_table):
        """
        truth_table: list of (input, output) tuples.
        Example: [(1, 1), (0, 0)] (Identity)
        """
        inputs = [x[0] for x in truth_table]
        targets = [x[1] for x in truth_table]

        # Invoke Compiler
        self.compiler.compile(
            inputs, targets,
            [self.PORT_IN_A], [self.PORT_OUT_Y]
        )

    def execute(self, input_val):
        """
        Runs the compiled logic on new data.
        Returns the analog intensity at the output port.
        """
        # Encode
        psi_in = np.zeros(self.substrate.shape, dtype=np.complex64)
        if input_val > 0.5:
            self.compiler._inject_gaussian(psi_in, self.PORT_IN_A)

        # Run Physics (The "Calculation")
        _, accum = self.substrate.propagate(psi_in, steps=60)

        # Decode (Read Sensor)
        out_x, out_y = self.PORT_OUT_Y
        # Sample a small area around output port
        output_energy = np.sum(accum[out_y-5:out_y+5, out_x-5:out_x+5])

        return output_energy, accum

# ==============================================================================
# DEMONSTRATION: "HELLO WORLD" (Identity Gate)
# ==============================================================================

def main():
    print("="*60)
    print("PHI-OS: VIRTUAL HOLOGRAPHIC PROCESSING UNIT")
    print("Mode: Self-Compiling Logic")
    print("="*60)

    # 1. Initialize VPU
    vpu = VirtualHolographicPU(size=200)

    # 2. Define the Logic by Example
    # Let's teach it to be a "Wire" (Identity Gate)
    # Input 1 -> Output 1
    # Input 0 -> Output 0
    truth_table = [(1, 1), (0, 0)]

    # 3. Compile (The Genesis)
    vpu.learn_function(truth_table)

    # 4. Execute (The Inference)
    print("\n[VPU] Executing Kernel...")

    # Test Logic High (Should pass)
    val_high, flow_high = vpu.execute(1)
    print(f" -> Input 1, Output Energy: {val_high:.2f}")

    # Test Logic Low (Should be near zero)
    val_low, flow_low = vpu.execute(0)
    print(f" -> Input 0, Output Energy: {val_low:.2f}")

    # 5. Visualize
    print("\n[VPU] Generating Hardware Schematic...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#050505')

    # The Grown Hardware
    hardware = vpu.substrate.n_base + vpu.substrate.n_learned
    axes[0].imshow(hardware, cmap='gray')
    axes[0].set_title("1. Compiled Geometry\n(Self-Organized Waveguide)", color='white')
    axes[0].scatter(*vpu.PORT_IN_A, c='lime', label='In')
    axes[0].scatter(*vpu.PORT_OUT_Y, c='red', label='Out')
    axes[0].legend()

    # The Logic High Flow
    axes[1].imshow(flow_high, cmap='inferno')
    axes[1].set_title(f"2. Execution (Input=1)\nSignal reaches Output", color='#00ff00')

    # The Logic Low Flow
    axes[2].imshow(flow_low, cmap='inferno', vmin=0, vmax=np.max(flow_high))
    axes[2].set_title(f"3. Execution (Input=0)\nSilence", color='gray')

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()