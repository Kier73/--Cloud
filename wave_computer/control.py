import numpy as np
from typing import Callable, List, Tuple
from .components import ResonantCavity

class ControlFlowEngine:
    """
    Implements loops and conditionals using resonant cavities
    """

    def __init__(self, size: int = 256):
        self.size = size
        self.shape = (size, size)

        # Execution state
        self.program_counter = 0
        self.registers = {}  # Named wave registers
        self.flags = {}  # Boolean flags (flip-flops)

        # Cavity for iterative operations
        self.cavity = ResonantCavity(size, feedback_strength=0.85)

    def if_then_else(self, condition_wave: np.ndarray,
                    then_operation: Callable,
                    else_operation: Callable,
                    threshold: float = 0.5) -> np.ndarray:
        """
        Conditional execution based on wave intensity

        If average(|condition_wave|²) > threshold:
            return then_operation(condition_wave)
        Else:
            return else_operation(condition_wave)
        """

        # Evaluate condition
        intensity = np.abs(condition_wave) ** 2
        avg_intensity = np.mean(intensity)

        if avg_intensity > threshold:
            return then_operation(condition_wave)
        else:
            return else_operation(condition_wave)

    def while_loop(self, initial_wave: np.ndarray,
                  condition_func: Callable[[np.ndarray], bool],
                  body_operation: Callable[[np.ndarray], np.ndarray],
                  max_iterations: int = 100) -> Tuple[np.ndarray, int]:
        """
        While loop using cavity feedback

        wave = initial_wave
        iterations = 0
        while condition_func(wave) and iterations < max_iterations:
            wave = body_operation(wave)
            iterations += 1
        return wave, iterations
        """

        self.cavity.inject_input(initial_wave)
        iterations = 0

        while iterations < max_iterations:
            current_wave = self.cavity.cavity_field

            # Check condition
            if not condition_func(current_wave):
                break

            # Execute body with feedback
            self.cavity.iterate(logic_operation=body_operation,
                               record_history=True)

            iterations += 1

        final_state = np.abs(self.cavity.cavity_field)
        return final_state, iterations

    def for_loop(self, initial_wave: np.ndarray,
                num_iterations: int,
                body_operation: Callable[[np.ndarray, int], np.ndarray]) -> List[np.ndarray]:
        """
        For loop with fixed iteration count
        Each iteration can access the loop counter
        """

        self.cavity.inject_input(initial_wave)
        results = []

        for i in range(num_iterations):
            # Body operation with loop counter
            def operation_with_counter(wave):
                return body_operation(wave, i)

            result = self.cavity.iterate(logic_operation=operation_with_counter,
                                        record_history=True)
            results.append(result)

        return results

    def accumulator_loop(self, initial_value: float,
                        operation: Callable[[float], float],
                        num_iterations: int) -> List[float]:
        """
        Classic accumulator loop (e.g., factorial, fibonacci)

        acc = initial_value
        for i in range(num_iterations):
            acc = operation(acc, i)
        """

        acc = initial_value
        history = [acc]

        for i in range(num_iterations):
            acc = operation(acc, i)
            history.append(acc)

        return history
