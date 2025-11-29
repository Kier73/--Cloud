import numpy as np
from typing import Callable, List, Tuple, Optional

class SpatialLightModulator:
    """
    Virtual SLM - controls wave transmission based on control signal
    This is the key to feedback and control flow
    """

    def __init__(self, size: int = 256):
        self.size = size
        self.shape = (size, size)
        self.state = np.zeros((size, size))  # Internal state memory

    def transmit(self, input_wave: np.ndarray, control_signal: np.ndarray,
                mode: str = 'amplitude') -> np.ndarray:
        """
        Modulate input based on control signal

        Modes:
        - 'amplitude': Control_signal modulates transmission (0=block, 1=pass)
        - 'phase': Control_signal modulates phase shift
        - 'bistable': Two-state switch (hysteresis)
        """

        if mode == 'amplitude':
            # Simple amplitude modulation
            return input_wave * control_signal

        elif mode == 'phase':
            # Phase modulation
            phase_shift = control_signal * np.pi
            return input_wave * np.exp(1j * phase_shift)

        elif mode == 'bistable':
            # Bistable element with hysteresis
            intensity = np.abs(control_signal) ** 2

            # Schmitt trigger thresholds
            high_threshold = 0.7
            low_threshold = 0.3

            # Update state with hysteresis
            self.state = np.where(intensity > high_threshold, 1.0, self.state)
            self.state = np.where(intensity < low_threshold, 0.0, self.state)

            return input_wave * self.state

    def get_state(self) -> np.ndarray:
        """Return current internal state"""
        return self.state.copy()

    def reset(self):
        """Reset to zero state"""
        self.state = np.zeros(self.shape)

class ResonantCavity:
    """
    Optical cavity with feedback
    The foundation for loops and control flow
    """

    def __init__(self, size: int = 256, feedback_strength: float = 0.9):
        self.size = size
        self.shape = (size, size)
        self.feedback_strength = feedback_strength

        # Cavity components
        self.slm = SpatialLightModulator(size)

        # Current cavity field (spatial domain)
        self.cavity_field = np.zeros((size, size), dtype=np.complex128)

        # History for visualization
        self.history = []

    def inject_input(self, input_wave: np.ndarray):
        """Inject new input into cavity"""
        self.cavity_field = input_wave

    def iterate(self, logic_operation: Optional[Callable] = None,
               nonlinear_element: Optional[Callable] = None,
               record_history: bool = True) -> np.ndarray:
        """
        One round-trip through the cavity

        1. Apply logic operation (if any)
        2. Apply nonlinear element (creates bistability)
        3. Feedback through SLM (controlled by intensity)
        4. Mix with cavity field
        """

        # Step 1: Apply logic operation
        if logic_operation is not None:
            processed = logic_operation(self.cavity_field)
        else:
            processed = self.cavity_field

        # Step 2: Nonlinear element (creates bistability)
        if nonlinear_element is not None:
            processed = nonlinear_element(processed)
        else:
            # Default: Kerr nonlinearity
            intensity = np.abs(processed) ** 2
            phase_shift = 0.5 * intensity
            processed = processed * np.exp(1j * phase_shift)

        # Step 3: Extract control signal from intensity
        control_signal = np.abs(processed) ** 2
        control_signal = control_signal / (np.max(control_signal) + 1e-10)  # Normalize

        # Step 4: Feedback through SLM
        feedback = self.slm.transmit(processed, control_signal, mode='bistable')

        # Step 5: Mix with existing cavity field
        self.cavity_field = (self.feedback_strength * feedback +
                            (1 - self.feedback_strength) * self.cavity_field)

        # Record history
        if record_history:
            self.history.append(np.abs(self.cavity_field.copy()))

        return np.abs(self.cavity_field)

    def run_until_stable(self, max_iterations: int = 100,
                        tolerance: float = 1e-4,
                        logic_operation: Optional[Callable] = None) -> Tuple[np.ndarray, int]:
        """
        Iterate until convergence (stable state)
        Returns: (final_state, num_iterations)
        """

        prev_state = np.abs(self.cavity_field)

        for i in range(max_iterations):
            current_state = self.iterate(logic_operation=logic_operation,
                                        record_history=True)

            # Check convergence
            diff = np.mean(np.abs(current_state - prev_state))
            if diff < tolerance:
                return current_state, i + 1

            prev_state = current_state

        return current_state, max_iterations

    def get_history(self) -> List[np.ndarray]:
        """Return recorded history"""
        return self.history

    def reset(self):
        """Reset cavity to zero state"""
        self.cavity_field = np.zeros(self.shape, dtype=np.complex128)
        self.slm.reset()
        self.history = []
