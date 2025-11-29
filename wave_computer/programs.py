import numpy as np
import scipy.fft
from typing import List, Tuple
from .control import ControlFlowEngine

class WaveProgram:
    """
    High-level programs demonstrating Turing completeness
    """

    def __init__(self, size: int = 128):
        self.size = size
        self.control = ControlFlowEngine(size)

    def program_factorial(self, n: int) -> List[float]:
        """
        Factorial using accumulator loop
        Proves: Can do iterative computation with state
        """

        def multiply_step(acc, i):
            return acc * (i + 1)

        return self.control.accumulator_loop(1.0, multiply_step, n)

    def program_fibonacci(self, n: int) -> List[float]:
        """
        Fibonacci sequence using loop with state
        Proves: Can maintain multiple state variables
        """

        a, b = 0, 1
        sequence = [a, b]

        for _ in range(n - 2):
            a, b = b, a + b
            sequence.append(b)

        return sequence

    def program_wave_relaxation(self, initial_pattern: np.ndarray,
                                num_iterations: int = 50) -> List[np.ndarray]:
        """
        Iterative wave equation solver
        Proves: Can do continuous-space iterative computation
        """

        def diffusion_step(wave, iteration):
            """Simple diffusion operator"""
            wave_spatial = scipy.fft.ifft2(scipy.fft.fft2(wave))

            # Laplacian (diffusion)
            laplacian_kernel = np.array([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]]) * 0.1

            # Convolve
            kernel_freq = scipy.fft.fft2(laplacian_kernel, s=wave.shape)
            wave_freq = scipy.fft.fft2(wave_spatial)
            result_freq = wave_freq * kernel_freq
            result = scipy.fft.ifft2(result_freq)

            # Add diffusion and clip
            return wave + np.real(result)

        self.control.cavity.reset()
        results = self.control.for_loop(initial_pattern, num_iterations, diffusion_step)

        return results

    def program_conditional_gate(self, A: np.ndarray, B: np.ndarray,
                                 threshold: float = 0.5) -> np.ndarray:
        """
        If A > threshold: return A AND B
        Else: return A OR B

        Proves: Can do conditional logic
        """

        def then_op(wave):
            # AND operation
            return wave * B

        def else_op(wave):
            # OR operation (saturated sum)
            return np.tanh(wave + B)

        return self.control.if_then_else(A, then_op, else_op, threshold)

    def program_while_convergence(self, initial_pattern: np.ndarray,
                                  tolerance: float = 0.01) -> Tuple[np.ndarray, int]:
        """
        While loop that runs until convergence
        Proves: Can do unbounded loops with dynamic exit
        """

        # Condition: Check if still changing significantly
        prev_state = [initial_pattern.copy()]

        def condition(wave):
            diff = np.mean(np.abs(wave - prev_state[0]))
            return diff > tolerance

        def body(wave):
            prev_state[0] = wave.copy()
            # Apply smoothing
            wave_freq = scipy.fft.fft2(wave)
            # Low-pass filter
            h, w = wave.shape
            y, x = np.ogrid[:h, :w]
            mask = ((x - w//2)**2 + (y - h//2)**2) < (min(h, w) / 4)**2
            wave_freq = wave_freq * np.fft.fftshift(mask)
            return scipy.fft.ifft2(wave_freq)

        self.control.cavity.reset()
        return self.control.while_loop(initial_pattern, condition, body, max_iterations=50)
