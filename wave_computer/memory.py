import numpy as np
from .components import ResonantCavity

class WaveFlipFlop:
    """
    SR Latch implemented with resonant cavity
    This is the fundamental memory element
    """

    def __init__(self, size: int = 256):
        self.size = size
        self.cavity = ResonantCavity(size, feedback_strength=0.95)
        self.state = 0  # Binary state: 0 or 1

    def set(self):
        """Set flip-flop to 1"""
        # Inject strong signal
        set_signal = np.ones((self.size, self.size), dtype=np.complex128)
        self.cavity.inject_input(set_signal)

        # Let cavity stabilize
        final_state, _ = self.cavity.run_until_stable(max_iterations=50)

        # Read state from average intensity
        avg_intensity = np.mean(final_state)
        self.state = 1 if avg_intensity > 0.5 else 0

    def reset(self):
        """Reset flip-flop to 0"""
        # Inject weak/zero signal
        reset_signal = np.zeros((self.size, self.size), dtype=np.complex128)
        self.cavity.inject_input(reset_signal)

        # Let cavity stabilize
        final_state, _ = self.cavity.run_until_stable(max_iterations=50)

        avg_intensity = np.mean(final_state)
        self.state = 1 if avg_intensity > 0.5 else 0

    def read(self) -> int:
        """Read current state"""
        return self.state

    def toggle(self):
        """Toggle state"""
        if self.state == 0:
            self.set()
        else:
            self.reset()
