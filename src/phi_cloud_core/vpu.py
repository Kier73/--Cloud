import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from typing import List, Tuple, Dict, Any

# ==============================================================================
# LAYER 1: THE PHYSICS KERNEL (The Substrate)
# ==============================================================================

class PhiSubstrate:
    """
    Implements the physical simulation of light propagation through the substrate.

    This class models the evolution of a complex scalar field ψ(x, z) governed by
    the Paraxial Helmholtz Equation, using the Split-Step Fourier Method.

    The substrate is defined by a learnable refractive index field n(x, z) = n₀ + Δn(x, z).
    """
    def __init__(self, shape: Tuple[int, int] = (128, 256), wavelength: float = 1.55e-6,
                 dx: float = 0.1e-6, dz: float = 0.5e-6, n0: float = 1.45):
        """
        Initializes the substrate and pre-computes the diffraction operator.

        :param shape: A tuple (ny, nz) defining the number of grid points in x and z.
        :param wavelength: The vacuum wavelength (λ) of the light.
        :param dx: The spatial grid spacing in the x-dimension.
        :param dz: The spatial grid spacing in the z-dimension (propagation step).
        :param n0: The constant background refractive index of the substrate.
        """
        self.shape = shape
        self.ny, self.nz = shape
        self.dx = dx
        self.dz = dz
        self.n0 = n0
        self.k0 = 2 * np.pi / wavelength  # Vacuum wavenumber k₀ = 2π/λ

        # The learnable structure function Δn(x, z)
        np.random.seed(42)
        self.delta_n = np.random.normal(0, 1e-4, self.shape).astype(np.float32)

        # Pre-compute the diffraction operator in Fourier space, corresponding to:
        # exp(-i * kx² * Δz / (2 * k₀ * n₀))
        kx = 2 * np.pi * scipy.fft.fftfreq(self.ny, self.dx)
        self.diffraction_op = np.exp(-1j * (kx**2 * self.dz) / (2 * self.k0 * self.n0))

    def propagate(self, field_in: np.ndarray, store_fields: bool = False) -> np.ndarray:
        """
        Propagates a field ψ(x, 0) from z=0 to z=L using the Split-Step Fourier method.

        :param field_in: The initial complex field ψ(x, 0) at the input plane.
        :param store_fields: If True, stores the field at each Δz step for gradient calculation.
        :return: The final complex field ψ(x, L) at the output plane.
        """
        psi = field_in.copy().astype(np.complex64)
        if store_fields: self.forward_fields = [psi]

        for z_idx in range(self.nz):
            # Refraction: ψ' = exp(i * k₀ * Δn * Δz) * ψ
            psi *= np.exp(1j * self.k0 * self.delta_n[:, z_idx] * self.dz)
            # Diffraction: ψ(z+Δz) = F⁻¹{ exp(-i*kx²*Δz/(2k₀n₀)) * F{ψ'} }
            psi = scipy.fft.ifft(scipy.fft.fft(psi) * self.diffraction_op)
            if store_fields: self.forward_fields.append(psi)
        return psi

    def backpropagate(self, field_in: np.ndarray) -> np.ndarray:
        """
        Propagates a field backward from z=L to z=0 using the adjoint of the propagation operator.
        """
        psi = field_in.copy().astype(np.complex64)
        self.backward_fields = [psi]
        for z_idx in range(self.nz - 1, -1, -1):
            # Adjoint Diffraction (conjugate of the operator)
            psi = scipy.fft.ifft(scipy.fft.fft(psi) * np.conj(self.diffraction_op))
            # Adjoint Refraction (conjugate of the operator)
            psi *= np.exp(-1j * self.k0 * self.delta_n[:, z_idx] * self.dz)
            self.backward_fields.append(psi)
        self.backward_fields.reverse()
        return psi

# ==============================================================================
# LAYER 2: THE COMPILER (Genesis Engine)
# ==============================================================================

class NeuroOpticalCompiler:
    """
    Implements the gradient-based optimization to find the optimal substrate Δn*.

    This class solves the inverse problem: given a target logical function (truth table),
    it finds the substrate structure that implements it by minimizing a loss function L(Δn).
    """
    def __init__(self, substrate: PhiSubstrate, learning_rate: float = 1e-3, epochs: int = 50):
        self.sub = substrate
        self.lr = learning_rate
        self.epochs = epochs

    def compile(self, vpu: 'VirtualHolographicPU', truth_table: List[Tuple[List[int], List[int]]],
                energy_high: float = 1.0, energy_low: float = 0.0) -> Tuple[np.ndarray, List[float]]:
        """
        Runs the optimization loop to compile the logic gate.

        The process follows: Δn* = argmin(L(Δn))
        """
        loss_history = []
        for epoch in range(self.epochs):
            total_loss = 0
            grad_accum = np.zeros_like(self.sub.delta_n)

            for input_vec, target_vec in truth_table:
                # 1. Forward Pass: Calculate ψ(x, L)
                initial_field = vpu._encode_input(input_vec)
                final_field = self.sub.propagate(initial_field, store_fields=True)

                # 2. Calculate Loss L and the error signal at the output plane
                output_energies = vpu._measure_output(final_field)
                target_energies = [energy_high if bit == 1 else energy_low for bit in target_vec]
                loss = np.sum([(out - tar)**2 for out, tar in zip(output_energies, target_energies)])
                total_loss += loss

                error_signal = np.zeros(self.sub.ny, dtype=np.complex64)
                for i, port_idx in enumerate(vpu.output_ports):
                    # The derivative of the energy loss w.r.t. the field ψ
                    err = 2 * (output_energies[i] - target_energies[i])
                    error_signal[vpu._get_port_slice(port_idx)] += err * final_field[vpu._get_port_slice(port_idx)]

                # 3. Backward Pass: Propagate the error signal back to z=0
                self.sub.backpropagate(error_signal)

                # 4. Gradient Calculation: The gradient ∂L/∂Δn is computed via the adjoint method
                grad = np.imag(np.conj(np.array(self.sub.forward_fields[:-1])) * np.array(self.sub.backward_fields[1:]))
                grad_accum += grad.T * self.sub.k0 * self.sub.dz

            # 5. Update Δn using gradient descent
            smoothed_grad = gaussian_filter(grad_accum, sigma=1.0) # Apply regularization
            self.sub.delta_n -= self.lr * smoothed_grad
            self.sub.delta_n = np.clip(self.sub.delta_n, 0, 0.05) # Enforce physical constraints

            avg_loss = total_loss / len(truth_table)
            loss_history.append(avg_loss)
            if epoch % 10 == 0: print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4e}")

        return self.sub.delta_n, loss_history

# ==============================================================================
# LAYER 3: THE VIRTUAL PROCESSOR (User API)
# ==============================================================================

class VirtualHolographicPU:
    """
    The main user-facing API for defining, compiling, and executing optical circuits.
    """
    def __init__(self, substrate: PhiSubstrate, compiler: NeuroOpticalCompiler,
                 input_ports: List[int], output_ports: List[int], port_width: int = 5):
        self.substrate = substrate
        self.compiler = compiler
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.port_width = port_width
        self.x_coords = np.arange(self.substrate.ny) * self.substrate.dx

    def _get_port_slice(self, port_pos_idx: int) -> slice:
        start = max(0, port_pos_idx - self.port_width // 2)
        end = min(self.substrate.ny, port_pos_idx + self.port_width // 2)
        return slice(start, end)

    def _encode_input(self, input_vector: List[int], amplitude: float = 1.0, sigma_x: float = None) -> np.ndarray:
        """
        Implements the Input Encoding Function E_in(b).
        Maps a binary vector b to a superposition of Gaussian beams.
        """
        if sigma_x is None: sigma_x = self.port_width * self.substrate.dx
        initial_field = np.zeros(self.substrate.ny, dtype=np.complex64)
        for i, bit in enumerate(input_vector):
            if bit == 1:
                port_pos = self.input_ports[i] * self.substrate.dx
                initial_field += amplitude * np.exp(-(self.x_coords - port_pos)**2 / (2 * sigma_x**2))
        return initial_field

    def _measure_output(self, final_field: np.ndarray) -> List[float]:
        """
        Implements the Output Measurement Function E_out(Δn, b).
        Calculates the energy ∫|ψ(x,L)|²dx at each detector region.
        """
        output_energies = []
        for port_pos_idx in self.output_ports:
            intensity = np.abs(final_field[self._get_port_slice(port_pos_idx)])**2
            energy = np.sum(intensity) * self.substrate.dx
            output_energies.append(energy)
        return output_energies

    def execute(self, input_vector: List[int], store_fields: bool = False) -> List[float]:
        """
        Executes a forward pass for a given input vector.
        """
        initial_field = self._encode_input(input_vector)
        final_field = self.substrate.propagate(initial_field, store_fields=store_fields)
        return self._measure_output(final_field)

    def learn_function(self, truth_table: List[Tuple[List[int], List[int]]]) -> Tuple[np.ndarray, List[float]]:
        """
        High-level function to invoke the compiler for a given truth table.
        """
        print("[VPU] Invoking gradient-based compiler...")
        return self.compiler.compile(self, truth_table)

    def verify(self, truth_table: List[Tuple[List[int], List[int]]], energy_high: float = 1.0,
             energy_low: float = 0.0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Performs digital verification of the compiled hardware.
        Compares the thresholded output y(s) against the target b_target(s).
        """
        print("\n[VERIFY] Verifying compiled logic...")
        e_thresh = (energy_high + energy_low) / 2.0
        correct_cases = 0
        total_cases = len(truth_table)
        case_results = []

        for i, (input_vec, target_vec) in enumerate(truth_table):
            output_energies = self.execute(input_vec)
            # Apply Heaviside step function H(E - E_thresh)
            result_vec = [1 if energy >= e_thresh else 0 for energy in output_energies]
            is_correct = (result_vec == target_vec)
            if is_correct: correct_cases += 1

            result = { "case": i + 1, "input": input_vec, "target": target_vec, "output": result_vec,
                       "energies": [float(e) for e in output_energies], "is_correct": is_correct }
            case_results.append(result)
            print(f"  Case #{result['case']}: Input={result['input']}, Target={result['target']}, Result={result['output']} -> {'PASS' if is_correct else 'FAIL'}")

        accuracy = (correct_cases / total_cases) * 100
        summary = { "accuracy": accuracy, "correct_cases": correct_cases,
                    "total_cases": total_cases, "all_correct": accuracy == 100.0 }
        print(f"[VERIFY] Verification complete. Accuracy: {summary['accuracy']:.2f}%")
        return summary, case_results

# ==============================================================================
# LAYER 4: LOGIC FABRIC for Circuit Composition
# ==============================================================================

class LogicFabric:
    """
    Manages a large substrate and the placement of pre-compiled components.
    This class is the foundation for building complex circuits by composition.
    """
    def __init__(self, shape=(1024, 4096), **kwargs):
        self.substrate = PhiSubstrate(shape=shape, **kwargs)
        # Mask to "freeze" the refractive index of placed components
        self.frozen_mask = np.zeros(shape, dtype=bool)
        self.components = []

    def place(self, component_delta_n: np.ndarray, position: Tuple[int, int]):
        """
        Places a pre-compiled component's Δn structure onto the fabric.

        :param component_delta_n: The 2D numpy array of the component's Δn.
        :param position: A tuple (y_pos, z_pos) for the top-left corner.
        """
        y_pos, z_pos = position
        ny, nz = component_delta_n.shape
        if y_pos + ny > self.substrate.ny or z_pos + nz > self.substrate.nz:
            raise ValueError("Component placement is out of fabric bounds.")

        self.substrate.delta_n[y_pos:y_pos+ny, z_pos:z_pos+nz] = component_delta_n
        self.frozen_mask[y_pos:y_pos+ny, z_pos:z_pos+nz] = True
        self.components.append({'position': position, 'shape': (ny, nz)})
        print(f"Placed component of shape {component_delta_n.shape} at {position}.")

    def visualize(self):
        """Displays the current state of the fabric's refractive index."""
        plt.figure(figsize=(15, 7))
        plt.imshow(self.substrate.delta_n.T, cmap='viridis', aspect='auto', origin='lower')
        plt.title("Logic Fabric with Placed Components")
        plt.xlabel("x position")
        plt.ylabel("z position (propagation)")
        plt.colorbar(label="Δn")
        plt.show()

# ==============================================================================
# DEMONSTRATION
# ==============================================================================

def main():
    print("="*60); print("PHI-OS: Compilation and Verification"); print("="*60)

    substrate = PhiSubstrate(shape=(256, 1024), dx=0.2e-6)
    compiler = NeuroOpticalCompiler(substrate)
    vpu = VirtualHolographicPU(substrate=substrate, compiler=compiler, input_ports=[64], output_ports=[192], port_width=10)

    truth_table = [([1], [1]), ([0], [0])]
    delta_n_final, loss_history = vpu.learn_function(truth_table)
    vpu.verify(truth_table)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(loss_history)
    axes[0].set_title("Loss History"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Mean Squared Error"); axes[0].set_yscale('log')
    im = axes[1].imshow(delta_n_final.T, aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title("Compiled Substrate (Δn)")
    fig.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()